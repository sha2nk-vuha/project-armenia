"""Built-in Combinators: AND and OR."""
from inference.combine.base import NOK, OK, Combinator
from inference.combine.registry import register


class AndCombinator:
    name = "and"
    label = "All must pass (AND)"

    def decisive(self, verdict: str) -> bool:
        # One NOK fails the whole cascade; nothing later can rescue it.
        return verdict == NOK

    def combine(self, verdicts: list[str]) -> str:
        # OK only if every stage evaluated was OK. On a short-circuited run the
        # decisive NOK is the last entry, so this still returns NOK.
        return OK if verdicts and all(v == OK for v in verdicts) else NOK


class OrCombinator:
    name = "or"
    label = "Any may pass (OR)"

    def decisive(self, verdict: str) -> bool:
        # One OK passes the whole cascade; nothing later can spoil it.
        return verdict == OK

    def combine(self, verdicts: list[str]) -> str:
        return OK if any(v == OK for v in verdicts) else NOK


register(AndCombinator())
register(OrCombinator())
