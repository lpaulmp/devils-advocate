import pytest

import devils_advocate as da


@pytest.fixture(autouse=True)
def isolated_history(monkeypatch, tmp_path):
    """Reports from any test go to a temporary folder, never to the real ~/devils-advocate/history."""
    history = tmp_path / "history"
    monkeypatch.setattr(da, "HISTORY_DIR", str(history))
    return history
