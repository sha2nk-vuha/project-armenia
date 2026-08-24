"""Combinator seam: truth tables and short-circuit decisiveness."""
import pytest

from inference import combine


def test_builtins_registered():
    assert {c.name for c in combine.all_combinators()} == {"and", "or"}


def test_unknown_combinator_raises_with_known_listed():
    with pytest.raises(ValueError, match="and"):
        combine.get("xor")


@pytest.mark.parametrize(
    "verdicts,expected",
    [
        (["ok", "ok"], "ok"),
        (["ok", "not_ok"], "not_ok"),
        (["not_ok", "ok"], "not_ok"),
        (["not_ok"], "not_ok"),
        (["ok"], "ok"),
    ],
)
def test_and_truth_table(verdicts, expected):
    assert combine.get("and").combine(verdicts) == expected


@pytest.mark.parametrize(
    "verdicts,expected",
    [
        (["not_ok", "not_ok"], "not_ok"),
        (["not_ok", "ok"], "ok"),
        (["ok", "not_ok"], "ok"),
        (["ok"], "ok"),
        (["not_ok"], "not_ok"),
    ],
)
def test_or_truth_table(verdicts, expected):
    assert combine.get("or").combine(verdicts) == expected


def test_and_empty_is_not_ok():
    # A cascade that evaluated nothing cannot be a pass.
    assert combine.get("and").combine([]) == "not_ok"


def test_decisiveness_enables_short_circuit():
    # AND: a NOK settles it; OK does not. OR: the reverse.
    assert combine.get("and").decisive("not_ok") is True
    assert combine.get("and").decisive("ok") is False
    assert combine.get("or").decisive("ok") is True
    assert combine.get("or").decisive("not_ok") is False
