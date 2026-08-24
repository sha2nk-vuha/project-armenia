"""Combinators: reduce per-stage Verdicts to one cascade Verdict.

Importing the package registers the built-ins. See docs/adr/0006.
"""
from inference.combine.base import NOK, OK, Combinator  # noqa: F401
from inference.combine.registry import (  # noqa: F401
    all_combinators,
    describe,
    get,
    register,
)
from inference.combine import operators  # noqa: F401  (registration side effect)
