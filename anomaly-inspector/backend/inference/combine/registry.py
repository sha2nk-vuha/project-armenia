"""Registry of available Combinators."""
from inference.combine.base import Combinator

_COMBINATORS: dict[str, Combinator] = {}


def register(combinator: Combinator) -> Combinator:
    """Add a Combinator to the registry.

    Args:
        combinator: Instance whose `name` becomes its registry key.

    Returns:
        The same combinator, for use as a decorator-style call.

    Raises:
        ValueError: If a Combinator with this name is already registered.
    """
    if combinator.name in _COMBINATORS:
        raise ValueError(f"Combinator already registered: {combinator.name!r}")
    _COMBINATORS[combinator.name] = combinator
    return combinator


def get(name: str) -> Combinator:
    """The registered Combinator with this name.

    Args:
        name: Registry key, e.g. "and" or "or".

    Returns:
        The registered Combinator instance.

    Raises:
        ValueError: If the name is unknown (message lists known names).
    """
    try:
        return _COMBINATORS[name]
    except KeyError:
        known = ", ".join(sorted(_COMBINATORS)) or "<none>"
        raise ValueError(f"Unknown Combinator: {name!r} (known: {known})") from None


def all_combinators() -> list[Combinator]:
    """Every registered Combinator, in registration order.

    Returns:
        List of Combinator instances.
    """
    return list(_COMBINATORS.values())


def describe(combinator: Combinator) -> dict:
    """UI metadata for one Combinator.

    Args:
        combinator: The Combinator to describe.

    Returns:
        Dict with its "name" and operator-facing "label".
    """
    return {"name": combinator.name, "label": combinator.label}
