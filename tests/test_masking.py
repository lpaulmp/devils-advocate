"""Masking filter: every secret pattern is hidden, ordinary text is left alone.

All values are fake.
"""
import time

import pytest

import devils_advocate as da

SECRETS = [
    ("AKIA" + "ABCDEFGHIJKLMNOP", "[AWS_KEY]"),
    ("ASIA" + "ABCDEFGHIJKLMNOP", "[AWS_KEY]"),
    ("sk-ant-api03-" + "x" * 24, "[TOKEN]"),
    ("sk-proj-" + "abcdefghijklmnopuv-", "[TOKEN]"),  # ends in '-'
    ("sk-proj-" + "abcdefghijklmnopuv_", "[TOKEN]"),  # ends in '_'
    ("xai-" + "a1b2c3d4e5f6g7h8i9j0", "[TOKEN]"),
    ("10.0.0.1", "[IP]"),
    ("2001:db8::1", "[IP]"),
    ("::1", "[IP]"),
    ("fe80::1ff:fe23:4567:890a", "[IP]"),
    ("arn:aws:iam::123456789012:role/Example", "[ARN]"),
    ("123456789012", "[ACCOUNT_ID]"),
    ("a.b+c@sub.example.co", "[EMAIL]"),
    ("o'brien@example.com", "[EMAIL]"),
    ("first%last@example.com", "[EMAIL]"),
    ("::ffff:10.0.0.1", "[IP]"),          # IPv4-mapped: no octet may leak
    ("1234-5678-9012", "[ACCOUNT_ID]"),   # AWS console format
]


@pytest.mark.parametrize("secret,placeholder", SECRETS)
def test_secret_is_masked(secret, placeholder):
    text, n = da.mask(f"before {secret} after")
    assert secret not in text
    assert text == f"before {placeholder} after"
    assert n == 1


@pytest.mark.parametrize("text,expected", [
    ("addr 2001:db8::1: down", "addr [IP]: down"),   # IPv6 followed by ':'
    ("host:2001:db8::1", "host:[IP]"),               # IPv6 preceded by ':'
    ("KEY_" + "sk-" + "a" * 20, "KEY_[TOKEN]"),       # glued to a word character
    ("x" + "AKIA" + "ABCDEFGHIJKLMNOP", "x[AWS_KEY]"),
])
def test_secret_in_context_is_masked(text, expected):
    assert da.mask(text) == (expected, 1)


@pytest.mark.parametrize("text", [
    "sk-abc123",              # too short to be a token
    "1234567890123",          # 13 digits, not an account id
    "plain text with no secrets",
    "meet at 10:30:45",       # time, not IPv6
    "std::vector and a::b::c",
    "aa:bb:cc:dd:ee:ff",      # MAC address, not IPv6
])
def test_ordinary_text_is_untouched(text):
    assert da.mask(text) == (text, 0)


def test_custom_term_any_case():
    text, n = da.mask("Project Falcon and project falcon", ["project falcon"])
    assert text == "[TERM] and [TERM]"
    assert n == 2


@pytest.mark.parametrize("terms", [[""], ["   "], []])
def test_empty_terms_are_ignored(terms):
    assert da.mask("x", terms) == ("x", 0)


def test_terms_do_not_corrupt_placeholders():
    key = "AKIA" + "ABCDEFGHIJKLMNOP"
    text, n = da.mask(f"{key} 10.0.0.1 term", ["key", "ip", "term", "aws"])
    assert text == "[AWS_KEY] [IP] [TERM]"
    assert n == 3


def test_overlapping_terms_counted_once():
    text, n = da.mask("acme corp", ["acme", "acme corp"])
    assert (text, n) == ("[TERM]", 1)


@pytest.mark.parametrize("text,term", [
    ("Falcon-123456789012 rocks", "Falcon-123456789012"),  # term contains an account id
    ("Falcon 10.0.0.1", "Falcon 10.0.0.1"),                # term contains an IP
])
def test_term_containing_a_pattern_is_hidden_whole(text, term):
    masked, n = da.mask(text, [term])
    assert "Falcon" not in masked
    assert n == 1


def test_term_inside_a_secret_keeps_one_placeholder():
    assert da.mask("bob@acme.com", ["acme"]) == ("[EMAIL]", 1)


@pytest.mark.parametrize("text", ["a." * 50000, "a-" * 50000, "1." * 50000, "a:" * 50000])
def test_long_input_without_spaces_is_fast(text):
    start = time.perf_counter()
    da.mask(text, ["term"])
    assert time.perf_counter() - start < 1


def test_load_terms_from_config(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[mask]\nterms = ["Falcon", "internal-db"]\n', encoding="utf-8")
    assert da.load_mask_terms(cfg) == ["Falcon", "internal-db"]


def test_missing_config_means_no_terms(tmp_path):
    assert da.load_mask_terms(tmp_path / "absent.toml") == []


@pytest.mark.parametrize("content", ["[mask\nterms = 1", '[mask]\nterms = "Falcon"', "[mask]\nterms = [1, 2]", 'mask = "x"', "mask = 1"])
def test_invalid_config_fails_closed(tmp_path, content):
    cfg = tmp_path / "config.toml"
    cfg.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        da.load_mask_terms(cfg)
