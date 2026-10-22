"""Read and validate app.toml and its data file."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TYPES = {"float", "int", "str", "category", "bool"}
_DTYPES = {
    "float": "float64",
    "int": "int64",
    "str": str,
    "category": "category",
    "bool": "bool",
}


class SpecError(Exception):
    """A mistake in app.toml (or between app.toml and the data)."""


@dataclass(frozen=True)
class Variable:
    name: str
    label: str
    unit: str | None
    type: str | None
    logged: bool


@dataclass(frozen=True)
class Sample:
    key: str
    label: str
    where: str | None


@dataclass(frozen=True)
class Spec:
    title: str
    subtitle: str
    file: str
    id: str | None
    source: dict[str, str]
    variables: dict[str, Variable]
    samples: list[Sample]
    defaults: dict


def load_spec(toml_path: Path) -> Spec:
    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)

    app = raw.get("app", {})
    data = raw.get("data", {})
    if "file" not in data:
        raise SpecError("[data] file: required")

    variables: dict[str, Variable] = {}
    for name, v in raw.get("variables", {}).items():
        where = f"[variables.{name}]"
        if not name.isidentifier():
            raise SpecError(f"{where}: name must be a valid Python identifier")
        if "label" not in v:
            raise SpecError(f"{where} label: required")
        if v.get("type") is not None and v["type"] not in TYPES:
            raise SpecError(
                f"{where} type: {v['type']!r} is not one of {sorted(TYPES)}"
            )
        variables[name] = Variable(
            name=name,
            label=v["label"],
            unit=v.get("unit"),
            type=v.get("type"),
            logged=bool(v.get("logged", False)),
        )

    samples = []
    for key, entry in raw.get("samples", {}).items():
        if "label" not in entry:
            raise SpecError(f"[samples] {key}: label required")
        samples.append(Sample(key=key, label=entry["label"], where=entry.get("where")))

    defaults = raw.get("defaults", {})
    if "y" in defaults and defaults["y"] not in variables:
        raise SpecError(
            f"[defaults] y: {defaults['y']!r} is not declared in [variables]"
        )

    return Spec(
        title=app.get("title", ""),
        subtitle=app.get("subtitle", ""),
        file=data["file"],
        id=data.get("id"),
        source=dict(raw.get("source", {})),
        variables=variables,
        samples=samples,
        defaults=defaults,
    )


def load_data(spec: Spec, base: Path) -> pd.DataFrame:
    """Read the data file; keep only declared columns (plus id), with types applied."""
    path = Path(base) / spec.file
    wanted = list(spec.variables) + ([spec.id] if spec.id else [])
    dtypes = {
        name: _DTYPES[v.type] for name, v in spec.variables.items() if v.type is not None
    }

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        _require_columns(spec, wanted, df.columns)
        df = df[wanted].astype(dtypes)
    else:
        header = pd.read_csv(path, nrows=0).columns
        _require_columns(spec, wanted, header)
        df = pd.read_csv(path, usecols=wanted, dtype=dtypes)[wanted]
    return df


def _require_columns(spec: Spec, wanted: list[str], columns) -> None:
    missing = [c for c in wanted if c not in columns]
    if missing:
        raise SpecError(f"{spec.file}: column(s) not found: {', '.join(missing)}")


def sample_rows(
    spec: Spec, df: pd.DataFrame, wanted: str | None
) -> tuple[pd.DataFrame, Sample | None]:
    """Rows of the sample `wanted` (None = first sample, or all rows if none declared)."""
    if wanted is None:
        if not spec.samples:
            return df, None
        wanted = spec.samples[0].key
    sample = next((x for x in spec.samples if x.key == wanted), None)
    if sample is None:
        raise ValueError(f"unknown sample {wanted!r}")
    return (df.query(sample.where) if sample.where else df), sample


def check(spec: Spec, df: pd.DataFrame) -> None:
    """Cross-check the spec against the data."""
    for sample in spec.samples:
        if sample.where is None:
            continue
        try:
            df.query(sample.where)
        except Exception as e:
            raise SpecError(
                f"[samples] {sample.key}: bad where {sample.where!r}: {e}"
            ) from e
    keys = {sample.key for sample in spec.samples}
    for k in spec.defaults.get("samples", []):
        if k not in keys:
            raise SpecError(f"[defaults] samples: {k!r} is not declared in [samples]")
