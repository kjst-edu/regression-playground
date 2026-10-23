from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from playground.describe import categorical_counts, format_stat, numeric_summary
from playground.spec import load_data, load_spec

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
