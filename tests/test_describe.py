from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from playground.describe import categorical_counts, numeric_summary
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
