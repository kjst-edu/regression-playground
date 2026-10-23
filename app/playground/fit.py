"""Fit each screen model with statsmodels and render the comparison table."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from pandas.api import types as pdt
from stargazer.stargazer import LineLocation, Stargazer
from statsmodels.regression.linear_model import RegressionResultsWrapper

from .spec import Spec, sample_rows
from .state import Model, State, formula
from .terms import Atom, Interaction, Term, label, to_patsy


@dataclass
class Fitted:
    index: int
    result: RegressionResultsWrapper | None
    error: str | None
    n: int | None
    sample_label: str | None
    # Rows present in the data but dropped because a transformed value (ln,
    # square) is not finite; rows already missing in the data are not counted.
    dropped: int = 0
    drop_reason: str | None = None


class FitError(Exception):
    """A reason, shown to students, why a model was not estimated."""


def _sample_rows(model: Model, spec: Spec, df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    try:
        rows, sample = sample_rows(spec, df, model.sample)
    except ValueError as e:
        raise FitError(str(e)) from e
    return rows, sample.label if sample else None


def _atoms(st: State, model: Model) -> list[Atom]:
    """The dependent variable followed by every atom on the right-hand side."""
    out = [st.y]
    for t in model.terms:
        out += [t.left, t.right] if isinstance(t, Interaction) else [t]
    return out


def _usable_rows(atoms: list[Atom], rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Masks over ``rows``: present in the data, and finite after transformation.

    A row is present when every variable the formula uses is there (not NaN,
    and finite for numeric variables). It is usable when, in addition, every
    transformed value (ln, square) is finite.
    """
    present = np.ones(len(rows), dtype=bool)
    usable = np.ones(len(rows), dtype=bool)
    for a in atoms:
        col = rows[a.var]
        if a.categorical or not pdt.is_numeric_dtype(col.dtype):
            ok = col.notna().to_numpy()
            present &= ok
            usable &= ok
            continue
        x = col.to_numpy(dtype="float64", na_value=np.nan)
        present &= np.isfinite(x)
        with np.errstate(all="ignore"):
            if a.log:
                x = np.log(x)
            if a.square:
                x = x**2
        usable &= np.isfinite(x)
    return present, usable


def _drop_reason(atoms: list[Atom]) -> str:
    transformed = [a for a in atoms if a.log or a.square]
    if transformed and all(a.log for a in transformed):
        return "ln of non-positive values"
    return "non-finite transformed values"


def _fit_one(st: State, i: int, spec: Spec, df: pd.DataFrame):
    model = st.models[i]
    if not model.terms and not model.constant:
        raise FitError("no regressors")
    atoms = _atoms(st, model)
    # Any form of y on the right (ln y, y², inside an interaction) is refused.
    if any(a.var == st.y.var for a in atoms[1:]):
        raise FitError("dependent variable appears on the right-hand side")
    rows, sample_label = _sample_rows(model, spec, df)
    f = formula(st, i)
    # Reduce to the rows the model can use (complete, and finite after ln or
    # sq) before patsy builds the design: patsy takes categorical levels from
    # every row it is given, so a level seen only in a dropped row would
    # become an all-zero column and look like perfect collinearity.
    present, usable = _usable_rows(atoms, rows)
    dropped = int((present & ~usable).sum())
    used = rows[usable].copy()
    for name in {a.var for a in atoms}:
        if isinstance(used[name].dtype, pd.CategoricalDtype):
            used[name] = used[name].cat.remove_unused_categories()
    with np.errstate(all="ignore"):
        _, X = patsy.dmatrices(f, used, NA_action="raise", return_type="dataframe")
    n, k = len(used), X.shape[1]
    if n <= k:
        raise FitError(f"too few observations (n={n}, k={k})")
    # statsmodels uses a pseudo-inverse and would return arbitrary
    # coefficients for a rank-deficient design.
    if np.linalg.matrix_rank(X.to_numpy()) < k:
        raise FitError("perfect collinearity")
    result = smf.ols(f, used, missing="raise").fit(cov_type=st.cov_type)
    reason = _drop_reason(atoms) if dropped else None
    return result, int(result.nobs), sample_label, dropped, reason


def fit_all(st: State, spec: Spec, df: pd.DataFrame) -> list[Fitted]:
    out = []
    for i in range(len(st.models)):
        try:
            result, n, sample_label, dropped, reason = _fit_one(st, i, spec, df)
            out.append(Fitted(i, result, None, n, sample_label, dropped, reason))
        except FitError as e:
            out.append(Fitted(i, None, str(e), None, None))
        except Exception as e:  # never let one model break the table
            msg = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            out.append(Fitted(i, None, msg, None, None))
    return out


def _patsy_name(t: Term) -> str:
    """The term name patsy gives ``to_patsy(t)`` (with its own spacing)."""
    return patsy.ModelDesc.from_formula(to_patsy(t)).rhs_termlist[-1].name()


_LEVEL = re.compile(r"\[(?:T\.)?([^\]]*)\]")


def param_labels(model: Model, result) -> dict[str, str]:
    """statsmodels coefficient name -> display label."""
    by_name = {_patsy_name(t): t for t in model.terms}
    info = result.model.data.design_info
    out: dict[str, str] = {}
    for name, sl in info.term_name_slices.items():
        for col in info.column_names[sl]:
            if name == "Intercept":
                out[col] = "Constant"
                continue
            term = by_name.get(name)
            base = label(term) if term is not None else name
            levels = _LEVEL.findall(col)
            out[col] = f"{base}: {', '.join(levels)}" if levels else base
    return out


_PER_COVARIATE = (
    "cov_values",
    "p_values",
    "cov_std_err",
    "conf_int_low_values",
    "conf_int_high_values",
)


def _relabel(md: dict, labels: dict[str, str]) -> None:
    """Rename one Stargazer model's covariates in place."""
    md["cov_names"] = [labels.get(c, c) for c in md["cov_names"]]
    for key in _PER_COVARIATE:
        md[key] = md[key].rename(lambda c: labels.get(c, c))


def f_cell(result, constant: bool, digits: int) -> str:
    """'12.345 (p = 0.012)' for the table's F statistic row, or '' when undefined.

    statsmodels computes ``fvalue`` under the covariance the model was fitted
    with (a Wald F for robust types). It tests the slopes against a model with
    only a constant, so without a constant it is left blank, as it is when
    there are no slopes (nan).
    """
    if not constant:
        return ""
    try:
        f, p = float(result.fvalue), float(result.f_pvalue)
    except Exception:
        return ""
    if not (np.isfinite(f) and np.isfinite(p)):
        return ""
    floor = 10.0**-digits
    pv = f"p < {floor:.{digits}f}" if p < floor else f"p = {p:.{digits}f}"
    return f"{f:.{digits}f} ({pv})"


def regression_table_html(st: State, spec: Spec, fitted: list[Fitted]) -> str:
    ok = [f for f in fitted if f.result is not None]
    failed = [f for f in fitted if f.result is None]
    parts = []
    if ok:
        table = Stargazer([f.result for f in ok])
        table.dependent_variable_name(html.escape(label(st.y)))
        table.show_model_numbers(False)
        table.significant_digits(st.digits)
        table.custom_columns([f"({f.index + 1})" for f in ok])
        # Key rows on the display label, not on statsmodels' names: patsy
        # names a level C(region)[T.2] with a constant and C(region)[2]
        # without one, and both must land in the same row. Stargazer's HTML
        # renderer does not escape, so labels are escaped here, once.
        order: list[str] = []
        for f, md in zip(ok, table.model_data):
            labels = {
                c: html.escape(s)
                for c, s in param_labels(st.models[f.index], f.result).items()
            }
            _relabel(md, labels)
            order += [c for c in md["cov_names"] if c not in order]
        constant = html.escape("Constant")
        table.cov_names = [c for c in order if c != constant] + [
            c for c in order if c == constant
        ]
        table.show_degrees_of_freedom(False)
        table.show_n = False  # replaced by our own "N" line next to "Sample"
        # Stargazer's own F row prints "nan***" when F is undefined; ours is
        # blank then (see f_cell).
        table.show_f_statistic = False
        table.add_line(
            "F statistic",
            [html.escape(f_cell(f.result, st.models[f.index].constant, st.digits)) for f in ok],
            LineLocation.FOOTER_BOTTOM,
        )
        samples = [html.escape(f.sample_label or "") for f in ok]
        table.add_line("Sample", samples, LineLocation.FOOTER_TOP)
        table.add_line("N", [str(f.n) for f in ok], LineLocation.FOOTER_TOP)
        notes = [f"Standard errors: {st.cov_type}"]
        for f in ok:
            if f.dropped:
                rows = "row" if f.dropped == 1 else "rows"
                notes.append(
                    html.escape(f"({f.index + 1}) {f.dropped} {rows} dropped: {f.drop_reason}")
                )
        table.add_custom_notes(notes)
        parts.append(table.render_html())
    else:
        parts.append(f"<p>Standard errors: {html.escape(st.cov_type)}</p>")
    if failed:
        items = "".join(
            f"<li>({f.index + 1}) {html.escape(f.error or '')}</li>" for f in failed
        )
        parts.append(f'<ul class="fit-errors">{items}</ul>')
    return "\n".join(parts)
