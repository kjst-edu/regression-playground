from pathlib import Path

import pytest

from playground.spec import load_spec
from playground.state import (
    Model,
    State,
    add_model,
    add_terms,
    duplicate_model,
    formula,
    initial_state,
    interaction_terms,
    remove_model,
    remove_term,
    set_constant,
    set_sample,
)
from playground.terms import Atom, Interaction

EXAMPLE = Path(__file__).resolve().parent.parent / "example"
educ, exper, female = Atom("educ"), Atom("exper"), Atom("female")


@pytest.fixture(scope="module")
def spec():
    return load_spec(EXAMPLE / "app.toml")


def base():
    return State(Atom("wage", log=True), (Model((educ,), True, "all"),))


def test_add_model_copies_last():
    st = add_model(base())
    assert st.models[1] == st.models[0]
    assert add_model(State(educ, ())).models == (Model(()),)


def test_duplicate_and_remove():
    st = add_terms(base(), 0, [exper])
    st = duplicate_model(st, 0)
    assert len(st.models) == 2 and st.models[1] == st.models[0]
    st = remove_model(st, 0)
    assert len(st.models) == 1
    assert remove_term(st, 0, 0).models[0].terms == (exper,)


def test_add_terms_dedup():
    st = add_terms(base(), 0, [exper])
    st = add_terms(st, 0, [exper, educ])
    assert st.models[0].terms == (educ, exper)


def test_interaction_adds_three():
    st = add_terms(base(), 0, interaction_terms(female, exper))
    assert st.models[0].terms == (educ, female, exper, Interaction(female, exper))
    st = add_terms(st, 0, interaction_terms(educ, female))
    assert st.models[0].terms[-1] == Interaction(educ, female)
    assert st.models[0].terms.count(educ) == 1
    # order-insensitive duplicate
    st = add_terms(st, 0, [Interaction(exper, female)])
    assert len(st.models[0].terms) == 5


def test_interaction_with_itself():
    assert interaction_terms(educ, educ) == [educ]


def test_constant_toggle_formula():
    st = add_terms(base(), 0, [Atom("exper", square=True)])
    assert formula(st, 0) == "np.log(wage) ~ educ + I(exper**2)"
    st = set_constant(st, 0, False)
    assert formula(st, 0) == "np.log(wage) ~ educ + I(exper**2) - 1"


def test_empty_formula():
    st = State(Atom("lwage"), (Model(()),))
    assert formula(st, 0) == "lwage ~ 1"
    assert formula(set_constant(st, 0, False), 0) == "lwage ~ 0"


def test_set_sample():
    assert set_sample(base(), 0, "urban").models[0].sample == "urban"


def test_initial_state_from_defaults(spec):
    st = initial_state(spec)
    assert len(st.models) == 2
    assert formula(st, 1) == "lwage ~ educ + exper + I(exper**2)"
    assert st.models[0].sample == "all"
    assert st.cov_type == "HC1"
