"""Descriptive statistics for the About tab."""
from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt

from .spec import Spec
from .state import first_sample


def _rows(spec: Spec, df: pd.DataFrame, sample_key: str | None) -> pd.DataFrame:
    wanted = sample_key if sample_key is not None else first_sample(spec)
    if wanted is None:
        return df
    sample = next((x for x in spec.samples if x.key == wanted), None)
    if sample is None:
        raise ValueError(f"unknown sample {wanted!r}")
    return df.query(sample.where) if sample.where else df


def _is_numeric(spec: Spec, name: str, df: pd.DataFrame) -> bool:
    t = spec.variables[name].type
    if t is not None:
        return t in ("float", "int")
    dtype = df[name].dtype
    return pdt.is_numeric_dtype(dtype) and not pdt.is_bool_dtype(dtype)


def numeric_summary(spec: Spec, df: pd.DataFrame, sample_key: str | None) -> pd.DataFrame:
    rows = _rows(spec, df, sample_key)
    names = [n for n in spec.variables if _is_numeric(spec, n, df)]
    return pd.DataFrame(
        {
            "N": [int(rows[n].count()) for n in names],
            "Mean": [rows[n].mean() for n in names],
            "SD": [rows[n].std() for n in names],
            "Min": [rows[n].min() for n in names],
            "Max": [rows[n].max() for n in names],
        },
        index=pd.Index(names, name="Variable"),
    )


def categorical_counts(spec: Spec, df: pd.DataFrame, sample_key: str | None) -> pd.DataFrame:
    rows = _rows(spec, df, sample_key)
    parts = []
    for n in spec.variables:
        if _is_numeric(spec, n, df):
            continue
        counts = rows[n].value_counts(sort=False).sort_index()
        parts.append(
            pd.DataFrame(
                {"Variable": n, "Level": counts.index.astype(str), "N": counts.to_numpy()}
            )
        )
    if not parts:
        return pd.DataFrame(columns=["Variable", "Level", "N"])
    return pd.concat(parts, ignore_index=True)
