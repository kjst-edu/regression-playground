"""Descriptive statistics for the About tab, and data warnings for the Model tab."""
from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt

from .spec import Spec, sample_rows
from .state import State, first_sample


def format_stat(x: float, digits: int) -> str:
    """Fixed point with thousands separators, ``digits`` decimals (as in the table)."""
    if pd.isna(x):
        return ""
    return f"{x:,.{digits}f}"


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


def nonpositive_count(spec: Spec, df: pd.DataFrame, var: str, sample_key: str | None) -> int:
    """Rows of the sample where ``var`` is present and <= 0 (ln is not finite there)."""
    try:
        rows = _rows(spec, df, sample_key)
        x = pd.to_numeric(rows[var].astype("float64"), errors="coerce")
    except (KeyError, ValueError, TypeError):
        return 0
    return int((x <= 0).sum())


def ln_warning(n: int, var: str, sample_label: str | None = None) -> str | None:
    """'wage has 3 non-positive values[ in Urban only]; ln drops 3 rows', or None if n == 0."""
    if n <= 0:
        return None
    values = "value" if n == 1 else "values"
    rows = "row" if n == 1 else "rows"
    where = f" in {sample_label}" if sample_label else ""
    return f"{var} has {n} non-positive {values}{where}; ln drops {n} {rows}"


def y_ln_warnings(st: State, spec: Spec, df: pd.DataFrame) -> list[str]:
    """When y is ln of a variable: one ln_warning per distinct sample the models use.

    With no models, the first sample is checked.
    """
    if not st.y.log:
        return []
    first = first_sample(spec)
    keys = list(dict.fromkeys(m.sample or first for m in st.models)) or [first]
    labels = {s.key: s.label for s in spec.samples}
    out = []
    for k in keys:
        w = ln_warning(nonpositive_count(spec, df, st.y.var, k), st.y.var, labels.get(k))
        if w:
            out.append(w)
    return out
