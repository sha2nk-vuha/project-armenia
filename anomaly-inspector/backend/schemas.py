"""Wire shapes crossing the HTTP seam.

The GUI posts multipart forms whose structured payloads arrive as
JSON-in-a-form-field (`rule_params`, `cascade_spec`). These models are the
single description of those shapes: `main.py` validates through them instead
of hand-rolling isinstance checks per route, and the cascade stage shape has
one authoritative definition shared with the GUI contract.

Responses deliberately stay plain dicts shaped exactly as today — declaring
response_model would filter/normalise fields and change the wire format the
existing GUI depends on.
"""
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter

# Rule params are open-ended by design: each Decision Rule declares its own
# param names (see inference.decision.base.ParamSpec), so the boundary only
# guarantees "a JSON object".
RuleParamsAdapter = TypeAdapter(dict[str, Any])


class CascadeStage(BaseModel):
    """One ordered stage of a Cascade inspection request."""

    feature: str
    rule: str | None = None
    threshold: float = 0.5
    params: dict[str, Any] = Field(default_factory=dict)


class CascadeSpec(BaseModel):
    """The per-request description of a Cascade: which stages, combined how.

    Mirrors what `features.build_cascade_pipeline` consumes; semantic checks
    (known Feature, uploaded model, rule compatibility) stay there because
    they need runtime state this schema deliberately does not have.
    """

    combinator: str = "and"
    short_circuit: bool = True
    stages: list[CascadeStage] = Field(min_length=1)
