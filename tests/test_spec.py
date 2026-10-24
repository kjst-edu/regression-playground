from pathlib import Path

import pandas as pd
import pytest

from playground.spec import SpecError, check, load_data, load_spec

EXAMPLE = Path(__file__).resolve().parent.parent / "example"


def write(tmp_path, toml, csv="a,b\n1,2\n3,4\n"):
    (tmp_path / "data.csv").write_text(csv)
    p = tmp_path / "app.toml"
    p.write_text(toml)
    return p


BASE = """
[data]
file = "data.csv"
[variables.a]
label = "A"
[variables.b]
label = "B"
[defaults]
y = "b"
"""


def test_load_example():
    spec = load_spec(EXAMPLE / "app.toml")
    assert spec.title == "Wage regression playground"
    assert list(spec.variables) == [
        "wage", "lwage", "educ", "exper", "female", "region", "sector", "urban",
    ]
    assert spec.variables["lwage"].logged is True
    assert spec.variables["wage"].logged is False
    assert [sample.key for sample in spec.samples] == ["all", "urban"]
    assert spec.samples[1].where == "urban == 1"
    assert spec.id == "id"
    assert spec.defaults["y"] == "lwage"


def test_types_applied():
    spec = load_spec(EXAMPLE / "app.toml")
    df = load_data(spec, EXAMPLE)
    assert df["sector"].dtype == "category"
    assert df["female"].dtype == bool
    assert pd.api.types.is_integer_dtype(df["region"].dtype)
    check(spec, df)


def test_unlisted_columns_dropped(tmp_path):
    p = write(tmp_path, BASE, "a,b,extra\n1,2,3\n")
    df = load_data(load_spec(p), tmp_path)
    assert list(df.columns) == ["a", "b"]


def test_id_column_kept_not_listed(tmp_path):
    toml = BASE.replace('file = "data.csv"', 'file = "data.csv"\nid = "k"')
    p = write(tmp_path, toml, "k,a,b\nx,1,2\n")
    spec = load_spec(p)
    assert "k" not in spec.variables
    assert "k" in load_data(spec, tmp_path).columns


def test_unknown_column_is_error(tmp_path):
    p = write(tmp_path, BASE + '[variables.zzz]\nlabel = "Z"\n')
    spec = load_spec(p)
    with pytest.raises(SpecError, match="zzz"):
        load_data(spec, tmp_path)


def test_bad_where_is_error(tmp_path):
    p = write(tmp_path, BASE + '[samples]\nbroken_sample = { label = "Broken", where = "nosuch == 1" }\n')
    spec = load_spec(p)
    df = load_data(spec, tmp_path)
    with pytest.raises(SpecError, match="broken_sample"):
        check(spec, df)


def test_non_identifier_name_is_error(tmp_path):
    toml = BASE + '[variables."gdp growth"]\nlabel = "G"\n'
    with pytest.raises(SpecError, match="gdp growth"):
        load_spec(write(tmp_path, toml))


def test_no_samples_section(tmp_path):
    assert load_spec(write(tmp_path, BASE)).samples == []


def test_parquet_by_extension(tmp_path):
    pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]}).to_parquet(tmp_path / "data.parquet")
    p = tmp_path / "app.toml"
    p.write_text(BASE.replace("data.csv", "data.parquet"))
    df = load_data(load_spec(p), tmp_path)
    assert list(df.columns) == ["a", "b"]


def test_defaults_unknown_y_is_error(tmp_path):
    toml = BASE.replace('y = "b"', 'y = "nope"')
    with pytest.raises(SpecError, match="nope"):
        load_spec(write(tmp_path, toml))


def test_defaults_y_is_required(tmp_path):
    toml = BASE.replace('y = "b"\n', "")
    with pytest.raises(SpecError, match=r"^\[defaults\] y: required$"):
        load_spec(write(tmp_path, toml))


def test_digits_default_and_value(tmp_path):
    assert load_spec(write(tmp_path, BASE)).digits == 3
    toml = "[app]\ndigits = 4\n" + BASE
    assert load_spec(write(tmp_path, toml)).digits == 4


@pytest.mark.parametrize("bad", ["0", "-1", "2.5", '"4"', "true"])
def test_bad_digits_is_error(tmp_path, bad):
    toml = f"[app]\ndigits = {bad}\n" + BASE
    with pytest.raises(SpecError, match="digits"):
        load_spec(write(tmp_path, toml))


def test_download_info():
    from types import SimpleNamespace

    from playground.spec import download_info

    assert download_info(SimpleNamespace(file="data.csv")) == ("data.csv", "text/csv")
    assert download_info(SimpleNamespace(file="sub/d.parquet"))[0] == "d.parquet"
