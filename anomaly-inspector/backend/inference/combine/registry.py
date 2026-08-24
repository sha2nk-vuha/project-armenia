"""Registry of available Combinators."""
from inference.combine.base import Combinator

_COMBINATORS: dict[str, Combinator] = {}


def register(combinator: Combinator) -> Combinator:
    if combinator.name in _COMBINATORS:
        raise ValueError(f"Combinator already registered: {combinator.name!r}")
    _COMBINATORS[combinator.name] = combinator
    return combinator


def get(name: str) -> Combinator:
    try:
        return _COMBINATORS[name]
    except KeyError:
        known = ", ".join(sorted(_COMBINATORS)) or "<none>"
        raise ValueError(f"Unknown Combinator: {name!r} (known: {known})")


def all_combinators() -> list[Combinator]:
    return list(_COMBINATORS.values())


def describe(combinator: Combinator) -> dict:
    return {"name": combinator.name, "label": combinator.label}
