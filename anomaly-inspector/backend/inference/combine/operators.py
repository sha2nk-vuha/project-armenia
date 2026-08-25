"""Built-in Combinators: AND and OR."""
from inference.combine.base import NOK, OK
from inference.combine.registry import register


class AndCombinator:
    """Cascade passes only when every stage passes; the first NOK decides."""

    name = "and"
    label = "All must pass (AND)"

    def decisive(self, verdict: str) -> bool:
        """One NOK fails the whole cascade; nothing later can rescue it.

        Args:
            verdict: A stage Verdict that has just been evaluated.

        Returns:
            True when `verdict` is NOK.
        """
        return verdict == NOK

    def combine(self, verdicts: list[str]) -> str:
        """OK only if every stage evaluated was OK.

        On a short-circuited run the decisive NOK is the last entry, so this
        still returns NOK.

        Args:
            verdicts: Verdicts of the stages that ran, in order.

        Returns:
            "ok" only for a non-empty all-OK list; otherwise "not_ok".
        """
        return OK if verdicts and all(v == OK for v in verdicts) else NOK


class OrCombinator:
    """Cascade passes as soon as any stage passes; the first OK decides."""

    name = "or"
    label = "Any may pass (OR)"

    def decisive(self, verdict: str) -> bool:
        """One OK passes the whole cascade; nothing later can spoil it.

        Args:
            verdict: A stage Verdict that has just been evaluated.

        Returns:
            True when `verdict` is OK.
        """
        return verdict == OK

    def combine(self, verdicts: list[str]) -> str:
        """OK as soon as any evaluated stage passed.

        Args:
            verdicts: Verdicts of the stages that ran, in order.

        Returns:
            "ok" when any entry is "ok"; "not_ok" otherwise.
        """
        return OK if any(v == OK for v in verdicts) else NOK


register(AndCombinator())
register(OrCombinator())
