import pandas as pd
import pytest

import pack
from playground.spec import SpecError

TOML = """
[data]
file = "data.csv"
id = "id"
[variables.a]
label = "A"
type = "int"
[variables.b]
label = "B"
[samples]
all = { label = "All" }
[defaults]
y = "b"
models = ["a"]
"""
CSV = "id,a,b,secret\nx,1,2.5,s1\ny,3,4.5,s2\n"


def make(tmp_path, toml=TOML, csv=CSV):
    (tmp_path / "app.toml").write_text(toml)
    (tmp_path / "data.csv").write_text(csv)
    return tmp_path


def test_valid(tmp_path):
    spec, df = pack.validate(make(tmp_path))
    assert list(df.columns) == ["a", "b", "id"]


def test_malformed_toml(tmp_path):
    with pytest.raises(SpecError, match="app.toml"):
        pack.validate(make(tmp_path, toml="[data\nfile ="))


def test_missing_toml(tmp_path):
    with pytest.raises(SpecError, match="not found"):
        pack.validate(tmp_path)


def test_bad_dtype(tmp_path):
    with pytest.raises(SpecError, match="data.csv: cannot read data"):
        pack.validate(make(tmp_path, csv="id,a,b\nx,1,2\ny,,3\n"))
    with pytest.raises(SpecError, match="data.csv: cannot read data"):
        pack.validate(make(tmp_path, csv="id,a,b\nx,foo,2\n"))


def test_bad_defaults_model(tmp_path):
    with pytest.raises(SpecError, match=r"models\[1\].*nope"):
        pack.validate(make(tmp_path, toml=TOML.replace('["a"]', '["a + nope"]')))


def test_bad_cov_type(tmp_path):
    with pytest.raises(SpecError, match="cov_type"):
        pack.validate(make(tmp_path, toml=TOML + 'cov_type = "HC9"\n'))


def test_write_data_drops_undeclared_columns(tmp_path):
    _, df = pack.validate(make(tmp_path))
    out = tmp_path / "out.csv"
    pack.write_data(df, out)
    back = pd.read_csv(out)
    assert list(back.columns) == ["a", "b", "id"]
    assert "secret" not in out.read_text()


def test_main_exit_codes(tmp_path, capsys):
    d = tmp_path / "d"
    d.mkdir()
    (d / "app.toml").write_text("[data\n")
    assert pack.main([str(d), str(tmp_path / "out")]) == 1
    assert not (tmp_path / "out").exists()
    assert "error: app.toml" in capsys.readouterr().err
