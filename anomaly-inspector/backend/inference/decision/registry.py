"""Registry of available Decision Rules.

Rules self-register at import. `compatible_with` filters by the output kinds a
decode advertises, which is how the GUI offers only rules that can actually run
against the loaded model.
"""
from inference.decision.base import DecisionRule

_RULES: dict[str, DecisionRule] = {}


def register(rule: DecisionRule) -> DecisionRule:
    """Add a Decision Rule to the registry.

    Args:
        rule: Instance whose `name` becomes its registry key.

    Returns:
        The same rule, for use as a decorator-style call.

    Raises:
        ValueError: If a rule with this name is already registered.
    """
    if rule.name in _RULES:
        raise ValueError(f"Decision Rule already registered: {rule.name!r}")
    _RULES[rule.name] = rule
    return rule


def get(name: str) -> DecisionRule:
    """The registered Decision Rule with this name.

    Args:
        name: Registry key, e.g. "expected_classes".

    Returns:
        The registered Decision Rule instance.

    Raises:
        ValueError: If the name is unknown (message lists known rules).
    """
    try:
        return _RULES[name]
    except KeyError:
        known = ", ".join(sorted(_RULES)) or "<none>"
        raise ValueError(f"Unknown Decision Rule: {name!r} (known: {known})") from None


def all_rules() -> list[DecisionRule]:
    """Every registered Decision Rule, in registration order.

    Returns:
        List of DecisionRule instances.
    """
    return list(_RULES.values())


def compatible_with(kinds: frozenset[str]) -> list[DecisionRule]:
    """Rules whose every consumed kind is present in `kinds`.

    Args:
        kinds: Output kinds the active decode advertises.

    Returns:
        Registered Decision Rules that can run against those outputs.
    """
    return [r for r in _RULES.values() if r.consumes <= kinds]


def describe(rule: DecisionRule) -> dict:
    """UI metadata for one rule: identity plus its parameter schema.

    Args:
        rule: The Decision Rule to describe.

    Returns:
        Dict with name/label/consumes, the ParamSpec schemas, and the
        per-SKU calibration declaration (or None).
    """
    calibration = getattr(rule, "calibration", None)
    return {
        "name": rule.name,
        "label": rule.label,
        "consumes": sorted(rule.consumes),
        "params": [p.as_dict() for p in rule.params],
        "calibration": (
            {
                "param": calibration.param,
                "metric": calibration.metric,
                "label": calibration.label,
            }
            if calibration
            else None
        ),
    }


def resolve_params(rule: DecisionRule, *override_layers: dict | None) -> dict:
    """Layer overrides onto the rule's declared defaults, lowest layer first.

    Layers are applied in order, so a caller passes (sidecar defaults, request
    overrides). Applying them in one pass matters: resolving each layer
    separately would re-seed every unspecified param from its declared default
    and silently discard the layer beneath it.

    Unknown keys are ignored — a stale param from a previously-selected rule
    must not leak into this one.

    Args:
        rule: Rule whose declared defaults seed the result.
        *override_layers: Param dicts applied lowest-first; each key only
            overrides when it matches a declared param name. None layers are
            skipped.

    Returns:
        Fully resolved params for the rule.
    """
    resolved = {p.name: p.default for p in rule.params}
    for layer in override_layers:
        for key, value in (layer or {}).items():
            if key in resolved:
                resolved[key] = value
    return resolved
