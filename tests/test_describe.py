from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from playground.describe import (
    categorical_counts,
    format_stat,
    ln_warning,
    nonpositive_count,
    numeric_summary,
    y_ln_warnings,
)
from playground.spec import load_data, load_spec
from playground.state import Model, State
from playground.terms import Atom

EXAMPLE = Path(__file__).resolve().parent.parent / "example"


@pytest.fixture(scope="module")
def spec():
    return load_spec(EXAMPLE / "app.toml")


@pytest.fixture
def df(spec):
    return load_data(spec, EXAMPLE)


def test_numeric_summary(spec, df):
    out = numeric_summary(spec, df, "all")
    assert list(out.columns) == ["N", "Mean", "SD", "Min", "Max"]
    assert "educ" in out.index
    assert "sector" not in out.index and "female" not in out.index
    assert out.loc["educ", "Mean"] == pytest.approx(df["educ"].mean())
    assert out.loc["educ", "N"] == df["educ"].notna().sum()


def test_numeric_summary_excludes_missing(spec, df):
    df = df.astype({"educ": "float64"})
    df.loc[:9, "educ"] = np.nan
    out = numeric_summary(spec, df, "all")
    assert out.loc["educ", "N"] == len(df) - 10
    assert out.loc["educ", "Mean"] == pytest.approx(df["educ"].dropna().mean())


def test_categorical_counts(spec, df):
    out = categorical_counts(spec, df, "all")
    assert list(out.columns) == ["Variable", "Level", "N"]
    assert out[out["Variable"] == "sector"]["N"].sum() == len(df)
    assert set(out["Variable"]) == {"female", "sector"}


def test_sample_applied(spec, df):
    n_urban = int((df["urban"] == 1).sum())
    assert numeric_summary(spec, df, "urban").loc["educ", "N"] == n_urban
    sector = categorical_counts(spec, df, "urban").query("Variable == 'sector'")
    assert sector["N"].sum() == n_urban


def test_none_means_first_sample(spec, df):
    assert numeric_summary(spec, df, None).equals(numeric_summary(spec, df, "all"))


def test_undeclared_type_follows_dtype(spec, df):
    variables = {k: replace(v, type=None) for k, v in spec.variables.items()}
    sp = replace(spec, variables=variables)
    assert "educ" in numeric_summary(sp, df, "all").index
    assert "female" in set(categorical_counts(sp, df, "all")["Variable"])


@pytest.mark.parametrize(
    "x, digits, want",
    [
        (24230.4567, 3, "24,230.457"),
        (24230.4567, 2, "24,230.46"),
        (1234567.0, 2, "1,234,567.00"),
        (0.000123, 4, "0.0001"),
        (-1500.5, 2, "-1,500.50"),
        (float("nan"), 3, ""),
    ],
)
def test_format_stat(x, digits, want):
    assert format_stat(x, digits) == want


def test_summary_html_has_no_scientific_notation(spec, df):
    df["wage"] = df["wage"] * 1000
    out = numeric_summary(spec, df, "all")
    html = out.to_html(float_format=lambda x: format_stat(x, 3))
    assert "e+" not in html
    assert f"{df['wage'].max():,.3f}" in html


def test_nonpositive_count(spec, df):
    # example data: exper is 0 for 5 workers, wage is always positive.
    assert nonpositive_count(spec, df, "exper", "all") == (df.exper <= 0).sum() == 5
    urban = df[df.urban == 1]
    assert nonpositive_count(spec, df, "exper", "urban") == (urban.exper <= 0).sum()
    assert nonpositive_count(spec, df, "wage", None) == 0


def test_nonpositive_count_ignores_missing_and_counts_negatives(spec, df):
    df = df.astype({"exper": "float64"})
    df.loc[0, "exper"] = np.nan
    df.loc[1, "exper"] = -2
    expected = int((df.exper.fillna(1) <= 0).sum())
    assert nonpositive_count(spec, df, "exper", "all") == expected


def test_nonpositive_count_bool(spec, df):
    assert nonpositive_count(spec, df, "female", "all") == (~df.female).sum()


def test_ln_warning():
    assert ln_warning(0, "exper") is None
    assert ln_warning(1, "exper") == "exper has 1 non-positive value; ln drops 1 row"
    assert (
        ln_warning(7, "y1960", "All")
        == "y1960 has 7 non-positive values in All; ln drops 7 rows"
    )


def test_y_ln_warnings(spec, df):
    df.loc[df.index[:3], "wage"] = 0.0
    n_urban = int((df.wage[df.urban == 1] <= 0).sum())
    educ = Atom("educ")
    models = (Model((educ,)), Model((educ,), True, "urban"), Model((educ,), True, "all"))
    st = State(Atom("wage", log=True), models)
    out = y_ln_warnings(st, spec, df)
    assert out[0] == "wage has 3 non-positive values in All workers; ln drops 3 rows"
    assert len(out) == (2 if n_urban else 1)
    assert y_ln_warnings(State(Atom("wage"), models), spec, df) == []
    assert y_ln_warnings(State(Atom("wage", log=True), ()), spec, df)[0].startswith("wage has 3")
