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

# Single spelling authority for Verdict strings lives in inference.verdict.
from inference.verdict import NOT_OK as NOK
from inference.verdict import OK


@runtime_checkable
class Combinator(Protocol):
    """Reduces per-stage Verdicts to one cascade Verdict (see docs/adr/0006)."""

    name: str
    label: str

    def decisive(self, verdict: str) -> bool:
        """True if `verdict` alone settles the cascade (enables short-circuit).

        Args:
            verdict: A stage Verdict that has just been evaluated.

        Returns:
            True when no later stage can change the outcome.
        """
        ...

    def combine(self, verdicts: list[str]) -> str:
        """Reduce the stage verdicts seen so far to a cascade verdict.

        Called with the verdicts actually evaluated; a short-circuited run passes
        only the stages that ran, which is why each operator must define its
        result on a partial list consistently with its decisive() rule.

        Args:
            verdicts: Verdicts of the stages that ran, in order.

        Returns:
            The combined cascade Verdict ("ok" or "not_ok").
        """
        ...
