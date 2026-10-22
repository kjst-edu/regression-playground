"""Descriptive statistics for the About tab."""
from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt

from .spec import Spec, sample_rows


def _rows(spec: Spec, df: pd.DataFrame, sample_key: str | None) -> pd.DataFrame:
    return sample_rows(spec, df, sample_key)[0]


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
