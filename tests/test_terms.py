from pathlib import Path

import pytest

from playground.spec import load_spec
from playground.terms import (
    Atom,
    Interaction,
    allowed,
    format_short_rhs,
    label,
    parse_short_rhs,
    to_patsy,
    to_short,
)

EXAMPLE = Path(__file__).resolve().parent.parent / "example"


@pytest.fixture(scope="module")
def spec():
    return load_spec(EXAMPLE / "app.toml")


CASES = [
    (Atom("educ"), "educ", "educ"),
    (Atom("wage", log=True), "np.log(wage)", "ln wage"),
    (Atom("exper", square=True), "I(exper**2)", "exper²"),
    (Atom("wage", log=True, square=True), "I(np.log(wage)**2)", "(ln wage)²"),
    (Atom("region", categorical=True), "C(region)", "C(region)"),
    (Interaction(Atom("educ"), Atom("female")), "educ:female", "educ × female"),
]


@pytest.mark.parametrize("term,patsy,_label", CASES)
def test_to_patsy(term, patsy, _label):
    assert to_patsy(term) == patsy


@pytest.mark.parametrize("term,_patsy,text", CASES)
def test_label(term, _patsy, text):
    assert label(term) == text


def test_short_roundtrip(spec):
    s = "educ + ln(wage) + sq(exper) + sq(ln(wage)) + C(region) + educ:female - 1"
    terms, constant, notices = parse_short_rhs(s, spec)
    assert constant is False
    assert notices == []
    assert format_short_rhs(terms, constant) == s


def test_space_and_plus_both_separate(spec):
    terms, constant, notices = parse_short_rhs("educ  exper+female", spec)
    assert [to_short(t) for t in terms] == ["educ", "exper", "female"]
    assert constant is True and notices == []


def test_constant_default(spec):
    assert format_short_rhs(*parse_short_rhs("educ", spec)[:2]) == "educ"


def test_category_type_gets_C(spec):
    terms, _, _ = parse_short_rhs("sector", spec)
    assert terms == [Atom("sector", categorical=True)]


def test_reject_log_of_logged(spec):
    terms, _, notices = parse_short_rhs("ln(lwage)", spec)
    assert terms == []
    assert len(notices) == 1 and "lwage" in notices[0]


@pytest.mark.parametrize("s", ["ln(C(region))", "sq(C(region))", "ln(sector)"])
def test_reject_log_or_sq_of_categorical(spec, s):
    terms, _, notices = parse_short_rhs(s, spec)
    assert terms == []
    assert len(notices) == 1


def test_unknown_var(spec):
    terms, _, notices = parse_short_rhs("nosuch + educ", spec)
    assert terms == [Atom("educ")]
    assert len(notices) == 1 and "nosuch" in notices[0]


def test_garbage_never_raises(spec):
    terms, _, notices = parse_short_rhs("((; educ + sq() a:b:c educ:", spec)
    assert Atom("educ") in terms
    assert notices


def test_interaction_with_unknown_dropped(spec):
    terms, _, notices = parse_short_rhs("educ:nosuch", spec)
    assert terms == [] and len(notices) == 1


def test_allowed(spec):
    assert "ln" not in allowed(Atom("lwage"), spec)
    a = allowed(Atom("region", categorical=True), spec)
    assert "ln" not in a and "sq" not in a
    assert allowed(Atom("educ"), spec) == {"ln", "sq", "C", "interact"}
