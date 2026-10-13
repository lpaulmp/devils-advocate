# %% [markdown]
# # Devil's Advocate — Claude + Grok
# Takes an idea and looks for reasons why it would **not** be a good one.
# Stage 1: skeleton, schemas, masking and `--dry-run`.

# %%
import argparse
import ipaddress
import math
import re
import sys
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

console = Console()
CONFIG_PATH = "~/.config/devils-advocate/config.toml"

# %% [markdown]
# ## JSON schemas (pydantic)

# %%
Kind = Literal["premise", "failure", "edge_case", "cost", "security"]


class Objection(BaseModel):
    id: str
    kind: Kind
    text: str
    severity: int = Field(ge=1, le=5)
    probability: int = Field(ge=1, le=5)

    @property
    def score(self) -> int:
        return self.severity * self.probability


class CriticResponse(BaseModel):
    objections: list[Objection] = Field(max_length=5)


class Verdict(BaseModel):
    id: str
    verdict: Literal["valid", "exaggerated", "incorrect"]
    reason: str


class AuditorResponse(BaseModel):
    verdicts: list[Verdict]
    new: list[Objection] = Field(default_factory=list, max_length=3)


class JudgeResponse(BaseModel):
    consensus: list[str]
    disagreements: list[str]
    steelman: str
    questions: list[str]
    global_verdict: Literal["proceed", "adjust", "drop"]


# %% [markdown]
# ## Masking filter
# Applied to the idea before it is sent to any model.
# Over-masking is accepted (e.g. a version like `1.2.3.4` becomes `[IP]`): it is safe.
#
# Known limits:
# - A secret glued to a preceding letter, digit or `_` can leak in part or whole, because the
#   patterns need a boundary to know where it starts. Examples: `x2001:db8::1`, `id_fe80::1`,
#   `host10.0.0.1`, `acct123456789012`, an `AKIA…` key immediately followed by an `xai-…` token.
#   Keep secrets separated by spaces or punctuation, or add them to `[mask] terms`.
# - An email whose local part contains one of ``!#$&*/=?^`{|}~`` is masked only after the last
#   such character (`a!b@example.com` → `a![EMAIL]`). Add the address to `[mask] terms`.
# - IPv6 in non-standard forms, such as a full address followed by `::port`, may not be masked.


# %%
def _ipv6_span(s: str) -> tuple[int, int] | None:
    """Longest valid IPv6 address inside a candidate.

    Tries dropping stray colons at either end (one or all of them), a leading `group:` (as in
    `ab:2001:…`) and a trailing `:group` (an unbracketed port, as in `…:7334:443` from a log line).
    Few combinations, so the cost stays linear.
    """
    lead, tail = len(s) - len(s.lstrip(":")), len(s.rstrip(":"))
    starts = {0, 1, lead, s.find(":", lead) + 1}
    ends = {len(s), len(s) - 1, tail, s.rfind(":", 0, tail)}
    for start, end in sorted(((a, b) for a in starts for b in ends if b > a), key=lambda ab: ab[0] - ab[1]):
        try:
            ipaddress.IPv6Address(s[start:end])
        except ValueError:
            continue
        return start, end
    return None


# (name, regex, optional span finder for candidates that need validation).
# All patterns run on the original text; overlapping matches are merged (see `mask`).
PATTERNS = [
    ("AWS_KEY", r"(?<![A-Z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Z0-9])", None),
    ("TOKEN", r"(?<![A-Za-z0-9])(?:sk|xai)-[A-Za-z0-9_\-]{16,}", None),
    ("ARN", r"\barn:aws[a-z\-]*:[A-Za-z0-9\-]+:[a-z0-9\-]*:\d{0,12}:[^\s\"']+", None),
    # The lookbehind makes matching start only at the beginning of a run (no quadratic backtracking).
    ("EMAIL", r"(?<![\w.+\-'%])[\w.+\-'%]+@[\w\-]+(?:\.[\w\-]+)+\b", None),
    ("IP", r"(?<![\w.])[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,10}(?:(?:\.\d{1,3}){3})?(?!\w)", _ipv6_span),
    ("IP", r"\b(?:\d{1,3}\.){3}\d{1,3}\b", None),
    ("ACCOUNT_ID", r"\b\d{12}\b|\b\d{4}-\d{4}-\d{4}\b", None),
]


def _spans(text: str, terms: list[str]) -> list[tuple[int, int, str]]:
    spans = []
    for name, pattern, find_span in PATTERNS:
        for m in re.finditer(pattern, text):
            span = find_span(m.group()) if find_span else (0, len(m.group()))
            if span:
                spans.append((m.start() + span[0], m.start() + span[1], name))
    if terms:
        # The lookahead reports a match at every position, so overlapping occurrences are all found
        # (longest term first at each position); the merge in `mask` then joins terms that overlap,
        # e.g. "Falcon X" and "X Corp" in "Falcon X Corp".
        # Any run of whitespace matches the spaces in a term ("Project Falcon" across a line break).
        alternatives = "|".join(r"\s+".join(map(re.escape, t.split())) for t in sorted(terms, key=len, reverse=True))
        spans += [(*m.span(1), "TERM") for m in re.finditer(f"(?=({alternatives}))", text, re.IGNORECASE)]
    return spans


def mask(text: str, terms: list[str] | None = None) -> tuple[str, int]:
    """Return (masked text, number of replacements).

    Patterns and custom terms are all matched on the original text, then overlapping matches are
    merged into one placeholder named after the longest of them. A term that contains an IP or an
    account ID is therefore hidden whole, and a term never matches inside a placeholder.
    """
    terms = {t.strip() for t in terms or [] if t.strip()}
    merged: list[list] = []  # [start, end, name, length of the longest match]
    for start, end, name in sorted(_spans(text, terms)):
        if merged and start < merged[-1][1]:
            last = merged[-1]
            if end - start > last[3]:
                last[2], last[3] = name, end - start
            last[1] = max(last[1], end)
        else:
            merged.append([start, end, name, end - start])
    out, pos = [], 0
    for start, end, name, _ in merged:
        out += [text[pos:start], f"[{name}]"]
        pos = end
    out.append(text[pos:])
    return "".join(out), len(merged)


def load_mask_terms(path: str | Path | None = None) -> list[str]:
    """Read `[mask] terms` from config.toml. Missing file → no terms; invalid file → error (fail closed)."""
    path = Path(path or CONFIG_PATH).expanduser()
    if not path.exists():
        return []
    try:
        mask_cfg = tomllib.loads(path.read_text(encoding="utf-8")).get("mask", {})
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        raise ValueError(f"cannot read {path.name}: {e}") from e
    if not isinstance(mask_cfg, dict):
        raise ValueError(f"{path.name}: [mask] must be a table")
    terms = mask_cfg.get("terms", [])
    if not isinstance(terms, list) or not all(isinstance(t, str) for t in terms):
        raise ValueError(f"{path.name}: [mask] terms must be a list of strings")
    return terms


# %% [markdown]
# ## Simulated responses (`--dry-run`)

# %%
def simulate_critic(idea: str) -> CriticResponse:
    return CriticResponse(objections=[
        Objection(id="O1", kind="premise", text="[simulated] Assumes users actually want this.", severity=4, probability=3),
        Objection(id="O2", kind="failure", text="[simulated] Single point of failure.", severity=5, probability=2),
        Objection(id="O3", kind="cost", text="[simulated] Operating cost grows faster than value.", severity=3, probability=4),
    ])


def simulate_auditor(idea: str, critic: CriticResponse) -> AuditorResponse:
    return AuditorResponse(
        verdicts=[
            Verdict(id="O1", verdict="valid", reason="[simulated] No evidence of demand."),
            Verdict(id="O2", verdict="exaggerated", reason="[simulated] Mitigable with redundancy."),
            Verdict(id="O3", verdict="valid", reason="[simulated] Plausible at scale."),
        ],
        new=[Objection(id="G1", kind="edge_case", text="[simulated] What happens with empty input.", severity=2, probability=4)],
    )


def simulate_judge(idea: str, critic: CriticResponse, auditor: AuditorResponse) -> JudgeResponse:
    return JudgeResponse(
        consensus=["[simulated] The demand premise is unproven."],
        disagreements=["[simulated] Severity of the single point of failure."],
        steelman="[simulated] If demand exists, the solution is simple and cheap to iterate on.",
        questions=["[simulated] Who is the first real user?"],
        global_verdict="adjust",
    )


# %% [markdown]
# ## CLI

# %%
def positive_float(value: str) -> float:
    x = float(value)
    if not (math.isfinite(x) and x > 0):
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return x


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="devils-advocate", description="Devil's advocate agent (Claude + Grok)")
    source = p.add_mutually_exclusive_group()
    source.add_argument("idea", nargs="?", help="idea text")
    source.add_argument("-f", "--file", help="read the idea from a file")
    p.add_argument("--rounds", type=int, default=2, choices=[1, 2, 3])
    p.add_argument("--local", action="store_true", help="Ollama only; nothing leaves the machine")
    p.add_argument("--dry-run", action="store_true", help="no API calls (simulated responses)")
    p.add_argument("--budget", type=positive_float, default=0.10, help="USD cap for this run (> 0)")
    p.add_argument("--configure", action="store_true", help="store API keys in the keychain")
    return p.parse_args(argv)


def fail(message: str, code: int = 2):
    console.print(f"[red]{escape(message)}[/red]")
    sys.exit(code)


def read_idea(args) -> str:
    if args.file:
        try:
            idea = Path(args.file).expanduser().read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            fail(f"Cannot read idea file: {e.strerror if isinstance(e, OSError) else e}")
    else:
        idea = args.idea or ""
    if not idea.strip():
        fail("Missing idea: pass text or use -f file.")
    return idea


def render(critic: CriticResponse, auditor: AuditorResponse, judge: JudgeResponse):
    # Model text goes in as Text: no rich markup and no :emoji: codes are interpreted.
    table = Table(title="Objections (simulated)")
    for col in ("id", "kind", "text", "sev×prob"):
        table.add_column(col)
    for o in [*critic.objections, *auditor.new]:
        table.add_row(Text(o.id), o.kind, Text(o.text), f"{o.severity}×{o.probability}={o.score}")
    console.print(table)
    console.print(f"[bold]Global verdict (simulated):[/bold] {judge.global_verdict}")


def run(idea: str, rounds: int, budget: float, dry_run: bool, local: bool, terms=None):
    if not dry_run:
        fail("Real calls are not implemented yet (stage 2+). Use --dry-run.")
    masked, n = mask(idea, terms)
    console.print(f"[dim]Masking: {n} replacement(s).[/dim]")
    max_calls = 2 * rounds + 1
    console.print(f"[dim]dry-run · rounds={rounds} · max calls={max_calls} · budget=${budget:.2f}[/dim]")

    # Only the masked text reaches the roles.
    critic = simulate_critic(masked)
    auditor = simulate_auditor(masked, critic)
    judge = simulate_judge(masked, critic, auditor)
    render(critic, auditor, judge)
    return critic, auditor, judge


def main(argv=None):
    args = parse_args(argv)
    if args.configure:
        console.print("[yellow]--configure is implemented in stage 5.[/yellow]")
        return
    idea = read_idea(args)
    try:
        terms = load_mask_terms()
    except ValueError as e:
        fail(str(e))
    run(idea, args.rounds, args.budget, args.dry_run, args.local, terms)


if __name__ == "__main__":
    main()
