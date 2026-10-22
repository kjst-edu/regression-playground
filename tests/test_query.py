from pathlib import Path

import pytest

from playground.query import state_from_query
from playground.spec import load_spec
from playground.state import initial_state
from playground.terms import Atom

EXAMPLE = Path(__file__).resolve().parent.parent / "example"


@pytest.fixture(scope="module")
def spec():
    return load_spec(EXAMPLE / "app.toml")


def test_query_basic(spec):
    st, notices = state_from_query(
        "?y=ln(wage)&m1=educ&m2=educ%2Bexper&s2=urban", spec
    )
    assert st.y == Atom("wage", log=True)
    assert len(st.models) == 2
    assert st.models[0].sample == "all"
    assert st.models[1].sample == "urban"
    assert st.models[1].terms == (Atom("educ"), Atom("exper"))
    assert notices == []


def test_query_space_as_plus(spec):
    st, _ = state_from_query("?y=lwage&m1=educ exper", spec)
    assert len(st.models[0].terms) == 2


def test_query_unknown_and_double_log(spec):
    st, notices = state_from_query(
        "?y=lwage&m1=educ+nosuch+ln(lwage)&s1=nosuch", spec
    )
    assert st.models[0].terms == (Atom("educ"),)
    assert st.models[0].sample == "all"
    assert len(notices) == 3


def test_query_gap_in_numbers(spec):
    st, _ = state_from_query("?y=lwage&m1=educ&m3=exper", spec)
    assert len(st.models) == 2
    assert st.models[1].terms == (Atom("exper"),)


def test_query_empty_uses_defaults(spec):
    assert state_from_query("", spec) == (initial_state(spec), [])


def test_query_bad_y_falls_back(spec):
    st, notices = state_from_query("?y=nosuch&m1=educ", spec)
    assert st.y == initial_state(spec).y
    assert len(notices) == 2
