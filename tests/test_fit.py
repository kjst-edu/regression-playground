from dataclasses import replace
from pathlib import Path

import pytest
import statsmodels.formula.api as smf

from playground.fit import fit_all, param_labels, regression_table_html
from playground.spec import Sample, load_data, load_spec
from playground.state import Model, State, formula, set_cov_type, set_digits
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


def test_collinear_dummy_and_complement(spec, df):
    df["rural"] = 1 - df["urban"]
    st = State(lwage, (Model((educ, Atom("urban"), Atom("rural"))), Model((educ,))))
    bad, good = fit_all(st, spec, df)
    assert bad.error == "perfect collinearity" and bad.result is None
    assert good.error is None


def test_collinear_constant_regressor_in_sample(spec, df):
    [f] = fit_all(one(Model((educ, Atom("urban")), True, "urban")), spec, df)
    assert f.error == "perfect collinearity"
    # Without a constant the same regressor is fine.
    [g] = fit_all(one(Model((educ, Atom("urban")), False, "urban")), spec, df)
    assert g.error is None


def row(html, label):
    start = html.index(f">{label}</td>")
    return html[start:html.index("</tr>", start)]


def test_same_label_is_one_row(spec, df):
    region = Atom("region", categorical=True)
    st = State(lwage, (Model((educ, region)), Model((educ, region), constant=False)))
    fitted = fit_all(st, spec, df)
    html = regression_table_html(st, spec, fitted)
    assert html.count(">C(region): 2</td>") == 1
    cells = row(html, "C(region): 2").count("<sup>")
    assert cells == 2  # a coefficient in both columns
    assert f"{fitted[1].result.params['C(region)[2]']:.3f}" in row(html, "C(region): 2")


def test_labels_escaped_once(spec, df):
    df["sector"] = df["sector"].astype(str).replace("service", "service & <x>")
    odd = replace(spec, samples=[Sample("all", "Urban & rural", None)])
    st = one(Model((educ, Atom("sector", categorical=True))))
    html = regression_table_html(st, odd, fit_all(st, odd, df))
    assert "C(sector): service &amp; &lt;x&gt;" in html
    assert "Urban &amp; rural" in html
    assert "&amp;amp;" not in html and "<x>" not in html


def test_table_digits(spec, df):
    st = State(lwage, (Model((educ,)),))
    fitted = fit_all(st, spec, df)
    coef = fitted[0].result.params["educ"]
    html3 = regression_table_html(st, spec, fitted)
    assert f"{coef:.3f}" in html3 and f"{coef:.4f}" not in html3
    st4 = set_digits(st, 4)
    assert f"{coef:.4f}" in regression_table_html(st4, spec, fitted)


region_c = Atom("region", categorical=True)


def test_level_only_in_rows_missing_a_regressor(spec, df):
    df["exper"] = df["exper"].astype(float)
    df.loc[df.region == 4, "exper"] = float("nan")
    [f] = fit_all(one(Model((educ, exper, region_c))), spec, df)
    assert f.error is None
    assert f.n == (df.region != 4).sum()
    assert not any("4" in c for c in f.result.params.index)


def test_level_only_in_rows_with_nonfinite_y(spec, df):
    df.loc[df.region == 4, "wage"] = 0.0
    [f] = fit_all(one(Model((educ, region_c)), y=Atom("wage", log=True)), spec, df)
    assert f.error is None
    assert f.n == (df.region != 4).sum()


def test_category_level_absent_from_sample(spec, df):
    no_public = replace(spec, samples=[Sample("np", "No public", "sector != 'public'")])
    assert "public" in df["sector"].cat.categories
    [f] = fit_all(one(Model((educ, Atom("sector", categorical=True)))), no_public, df)
    assert f.error is None
    assert f.n == (df.sector != "public").sum()


def test_collinear_still_fails_after_dropping_rows(spec, df):
    df["exper"] = df["exper"].astype(float)
    df.loc[df.region == 4, "exper"] = float("nan")
    df["rural"] = 1 - df["urban"]
    st = one(Model((exper, region_c, Atom("urban"), Atom("rural"))))
    [f] = fit_all(st, spec, df)
    assert f.error == "perfect collinearity"


def test_nonfinite_y_drops_are_noted(spec, df):
    df.loc[df.index[:7], "wage"] = 0.0
    st = State(Atom("wage", log=True), (Model((educ,)), Model((educ, exper))))
    fitted = fit_all(st, spec, df)
    assert [f.dropped for f in fitted] == [7, 7]
    assert fitted[0].drop_reason == "ln of non-positive values"
    html = regression_table_html(st, spec, fitted)
    assert "(1) 7 rows dropped: ln of non-positive values" in html
    assert "(2) 7 rows dropped: ln of non-positive values" in html


def test_nonfinite_regressor_drop_noted_for_that_column_only(spec, df):
    df.loc[df.index[:3], "exper"] = 0
    zeros = int((df.exper == 0).sum())
    ln_exper = Atom("exper", log=True)
    st = State(lwage, (Model((educ,)), Model((educ, ln_exper))))
    fitted = fit_all(st, spec, df)
    assert [f.dropped for f in fitted] == [0, zeros]
    assert fitted[1].n == 200 - zeros
    html = regression_table_html(st, spec, fitted)
    assert f"(2) {zeros} rows dropped: ln of non-positive values" in html
    assert "(1) " not in html.split("Standard errors")[1]


def test_missing_in_data_is_not_noted(spec, df):
    df["exper"] = df["exper"].astype(float)
    df.loc[df.index[:5], "exper"] = float("nan")
    st = one(Model((educ, exper)))
    [f] = fit_all(st, spec, df)
    assert f.n == 195 and f.dropped == 0 and f.drop_reason is None
    assert "dropped" not in regression_table_html(st, spec, [f])


@pytest.mark.parametrize(
    "rhs",
    [
        Atom("wage"),
        Atom("wage", log=True),
        Atom("wage", square=True),
        Atom("wage", log=True, square=True),
        Interaction(educ, Atom("wage")),
        Interaction(Atom("wage", log=True), female),
    ],
)
def test_y_in_any_form_on_the_right(spec, df, rhs):
    [f] = fit_all(one(Model((educ, rhs)), y=Atom("wage", log=True)), spec, df)
    assert f.error == "dependent variable appears on the right-hand side"
    assert f.result is None


def test_y_on_the_right_in_interaction(spec, df):
    [f] = fit_all(one(Model((Interaction(educ, lwage),))), spec, df)
    assert "dependent variable" in f.error
