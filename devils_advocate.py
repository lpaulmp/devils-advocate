# %% [markdown]
# # Devil's Advocate — Claude + Grok
# Takes an idea and looks for reasons why it would **not** be a good one.
# Stage 1: skeleton, schemas, masking and `--dry-run`.

# %%
import argparse
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

console = Console()

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

# %%
PATTERNS = [
    ("AWS_KEY", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    ("TOKEN", r"\b(?:sk|xai)-[A-Za-z0-9_\-]{16,}\b"),
    ("ARN", r"\barn:aws[a-z\-]*:[A-Za-z0-9\-]+:[a-z0-9\-]*:\d{0,12}:[^\s\"']+"),
    ("EMAIL", r"\b[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+\b"),
    ("IP", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ("ACCOUNT_ID", r"\b\d{12}\b"),
]


def mask(text: str, terms: list[str] | None = None) -> tuple[str, int]:
    """Return (masked text, number of replacements)."""
    total = 0
    for name, pattern in PATTERNS:
        text, n = re.subn(pattern, f"[{name}]", text)
        total += n
    for term in terms or []:
        text, n = re.subn(re.escape(term), "[TERM]", text, flags=re.IGNORECASE)
        total += n
    return text, total


# %% [markdown]
# ## Simulated responses (`--dry-run`)

# %%
def simulate_critic() -> CriticResponse:
    return CriticResponse(objections=[
        Objection(id="O1", kind="premise", text="[simulated] Assumes users actually want this.", severity=4, probability=3),
        Objection(id="O2", kind="failure", text="[simulated] Single point of failure.", severity=5, probability=2),
        Objection(id="O3", kind="cost", text="[simulated] Operating cost grows faster than value.", severity=3, probability=4),
    ])


def simulate_auditor() -> AuditorResponse:
    return AuditorResponse(
        verdicts=[
            Verdict(id="O1", verdict="valid", reason="[simulated] No evidence of demand."),
            Verdict(id="O2", verdict="exaggerated", reason="[simulated] Mitigable with redundancy."),
            Verdict(id="O3", verdict="valid", reason="[simulated] Plausible at scale."),
        ],
        new=[Objection(id="G1", kind="edge_case", text="[simulated] What happens with empty input.", severity=2, probability=4)],
    )


def simulate_judge() -> JudgeResponse:
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
def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="devils-advocate", description="Devil's advocate agent (Claude + Grok)")
    p.add_argument("idea", nargs="?", help="idea text")
    p.add_argument("-f", "--file", help="read the idea from a file")
    p.add_argument("--rounds", type=int, default=2, choices=[1, 2, 3])
    p.add_argument("--local", action="store_true", help="Ollama only; nothing leaves the machine")
    p.add_argument("--dry-run", action="store_true", help="no API calls (simulated responses)")
    p.add_argument("--budget", type=float, default=0.10, help="USD cap for this run")
    p.add_argument("--configure", action="store_true", help="store API keys in the keychain")
    return p.parse_args(argv)


def read_idea(args) -> str:
    if args.file:
        return Path(args.file).expanduser().read_text(encoding="utf-8")
    if args.idea:
        return args.idea
    console.print("[red]Missing idea: pass text or use -f file.[/red]")
    sys.exit(2)


def run(idea: str, rounds: int, budget: float, dry_run: bool, local: bool, terms=None):
    _masked, n = mask(idea, terms)
    console.print(f"[dim]Masking: {n} replacement(s).[/dim]")
    if not dry_run:
        console.print("[yellow]Real calls not implemented yet (stage 2+). Use --dry-run.[/yellow]")
        return None
    max_calls = 2 * rounds + 1
    console.print(f"[dim]dry-run · rounds={rounds} · max calls={max_calls} · budget=${budget:.2f}[/dim]")

    critic, auditor, judge = simulate_critic(), simulate_auditor(), simulate_judge()

    table = Table(title="Objections (simulated)")
    for col in ("id", "kind", "text", "sev×prob"):
        table.add_column(col)
    for o in [*critic.objections, *auditor.new]:
        table.add_row(o.id, o.kind, o.text, f"{o.severity}×{o.probability}={o.score}")
    console.print(table)
    console.print(f"[bold]Global verdict (simulated):[/bold] {judge.global_verdict}")
    return critic, auditor, judge


def main(argv=None):
    args = parse_args(argv)
    if args.configure:
        console.print("[yellow]--configure is implemented in stage 5.[/yellow]")
        return
    idea = read_idea(args)
    run(idea, args.rounds, args.budget, args.dry_run, args.local)


if __name__ == "__main__":
    main()
