"""Contracts for the Combinator seam.

A Combinator reduces an ordered list of per-stage Verdicts to one cascade
Verdict. It mirrors the Decision Rule seam: new operators are added by dropping a
module here and registering it, with no change to the pipeline or the API.

Each Combinator also declares which single Verdict is *decisive* -- one that
settles the outcome regardless of the stages still to run -- so a cascade can
short-circuit: with AND a single NOK is decisive, with OR a single OK is. See
docs/adr/0006.
"""
from typing import Protocol, runtime_checkable

OK = "ok"
NOK = "not_ok"


@runtime_checkable
class Combinator(Protocol):
    name: str
    label: str

    def decisive(self, verdict: str) -> bool:
        """True if `verdict` alone settles the cascade (enables short-circuit)."""
        ...

    def combine(self, verdicts: list[str]) -> str:
        """Reduce the stage verdicts seen so far to a cascade verdict.

        Called with the verdicts actually evaluated; a short-circuited run passes
        only the stages that ran, which is why each operator must define its
        result on a partial list consistently with its decisive() rule.
        """
        ...
