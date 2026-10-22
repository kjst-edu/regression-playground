from dataclasses import replace
from pathlib import Path

import pytest
import statsmodels.formula.api as smf

from playground.fit import fit_all, param_labels, regression_table_html
from playground.spec import Sample, load_data, load_spec
from playground.state import Model, State, formula, set_cov_type
from playground.terms import Atom, Interaction

EXAMPLE = Path(__file__).resolve().parent.parent / "example"
lwage, educ, exper, female = Atom("lwage"), Atom("educ"), Atom("exper"), Atom("female")


@pytest.fixture(scope="module")
def spec():
    return load_spec(EXAMPLE / "app.toml")


@pytest.fixture
def df(spec):
    return load_data(spec, EXAMPLE)


def one(model, y=lwage, cov_type="nonrobust"):
    return State(y, (model,), cov_type)


def test_matches_statsmodels(spec, df):
    st = one(Model((educ, exper)))
    [f] = fit_all(st, spec, df)
    ref = smf.ols(formula(st, 0), df).fit()
    assert f.error is None and f.n == 200
    assert f.result.params.values == pytest.approx(ref.params.values)
    assert f.result.bse.values == pytest.approx(ref.bse.values)


def test_sample_where(spec, df):
    [f] = fit_all(one(Model((educ,), True, "urban")), spec, df)
    assert f.n == (df.urban == 1).sum()
    assert f.sample_label == "Urban only"


def test_sample_none_means_first(spec, df):
    [f] = fit_all(one(Model((educ,))), spec, df)
    assert f.sample_label == "All workers"


def test_cov_type_hc1(spec, df):
    st = set_cov_type(one(Model((educ, exper))), "HC1")
    [f] = fit_all(st, spec, df)
    ref = smf.ols(formula(st, 0), df).fit(cov_type="HC1")
    assert f.result.bse.values == pytest.approx(ref.bse.values)


def test_log_of_nonpositive_drops_rows(spec, df):
    df.loc[0, "wage"] = 0.0
    [f] = fit_all(one(Model((educ,)), y=Atom("wage", log=True)), spec, df)
    assert f.error is None
    assert f.n == 199


def test_empty_model(spec, df):
    st = State(lwage, (Model((), constant=False), Model((educ,))))
    bad, good = fit_all(st, spec, df)
    assert bad.error == "no regressors" and bad.result is None
    assert good.error is None and good.result is not None


def test_too_few_observations(spec, df):
    tiny = replace(spec, samples=[Sample("tiny", "Tiny", "id in ['p001', 'p002', 'p003']")])
    [f] = fit_all(one(Model((educ, exper, female), True, "tiny")), tiny, df)
    assert "too few observations" in f.error
    assert f.result is None


def test_y_on_both_sides(spec, df):
    [f] = fit_all(one(Model((educ, lwage))), spec, df)
    assert "dependent variable" in f.error


def test_param_labels(spec, df):
    sq_ln_wage = Atom("wage", log=True, square=True)
    model = Model((Atom("region", categorical=True), Interaction(educ, female), sq_ln_wage))
    [f] = fit_all(one(model), spec, df)
    labels = set(param_labels(model, f.result).values())
    for want in ["C(region): 2", "educ × female", "(ln wage)²", "Constant"]:
        assert any(want in s for s in labels), (want, labels)


def test_table_html(spec, df):
    st = State(lwage, (Model((), constant=False), Model((educ,))))
    fitted = fit_all(st, spec, df)
    html = regression_table_html(st, spec, fitted)
    coef = fitted[1].result.params["educ"]
    assert f"{coef:.3f}" in html
    assert "Sample" in html and "All workers" in html
    assert ">N<" in html
    assert "(2)" in html
    assert "(1) no regressors" in html
    assert "Standard errors: nonrobust" in html


def test_table_html_all_failed(spec, df):
    st = State(lwage, (Model((), constant=False),))
    html = regression_table_html(st, spec, fit_all(st, spec, df))
    assert "(1) no regressors" in html
