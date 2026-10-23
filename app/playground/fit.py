"""Fit each screen model with statsmodels and render the comparison table."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from stargazer.stargazer import LineLocation, Stargazer
from statsmodels.regression.linear_model import RegressionResultsWrapper

from .spec import Spec, sample_rows
from .state import Model, State, formula
from .terms import Term, label, to_patsy


@dataclass
class Fitted:
    index: int
    result: RegressionResultsWrapper | None
    error: str | None
    n: int | None
    sample_label: str | None


class FitError(Exception):
    """A reason, shown to students, why a model was not estimated."""


def _sample_rows(model: Model, spec: Spec, df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    try:
        rows, sample = sample_rows(spec, df, model.sample)
    except ValueError as e:
        raise FitError(str(e)) from e
    return rows, sample.label if sample else None


def _fit_one(st: State, i: int, spec: Spec, df: pd.DataFrame):
    model = st.models[i]
    if not model.terms and not model.constant:
        raise FitError("no regressors")
    if st.y in model.terms:
        raise FitError("dependent variable appears on the right-hand side")
    rows, sample_label = _sample_rows(model, spec, df)
    f = formula(st, i)
    # Evaluate once to find rows where y or any regressor is NaN or ±inf
    # (e.g. the log of a non-positive value), then fit on the rest.
    with np.errstate(all="ignore"):
        y, X = patsy.dmatrices(f, rows, NA_action="drop", return_type="dataframe")
    finite = np.isfinite(y.to_numpy()).all(axis=1) & np.isfinite(X.to_numpy()).all(axis=1)
    keep = y.index[finite]
    n, k = len(keep), X.shape[1]
    if n <= k:
        raise FitError(f"too few observations (n={n}, k={k})")
    # statsmodels uses a pseudo-inverse and would return arbitrary
    # coefficients for a rank-deficient design.
    if np.linalg.matrix_rank(X.to_numpy()[finite]) < k:
        raise FitError("perfect collinearity")
    result = smf.ols(f, rows[rows.index.isin(keep)], missing="drop").fit(cov_type=st.cov_type)
    return result, int(result.nobs), sample_label


def fit_all(st: State, spec: Spec, df: pd.DataFrame) -> list[Fitted]:
    out = []
    for i in range(len(st.models)):
        try:
            result, n, sample_label = _fit_one(st, i, spec, df)
            out.append(Fitted(i, result, None, n, sample_label))
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


def regression_table_html(st: State, spec: Spec, fitted: list[Fitted]) -> str:
    ok = [f for f in fitted if f.result is not None]
    failed = [f for f in fitted if f.result is None]
    parts = []
    if ok:
        table = Stargazer([f.result for f in ok])
        table.dependent_variable_name(html.escape(label(st.y)))
        table.show_model_numbers(False)
        table.significant_digits(spec.digits)
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
        # Without a constant the F statistic can be nan, which Stargazer
        # prints as "nan***".
        table.show_f_statistic = False
        samples = [html.escape(f.sample_label or "") for f in ok]
        table.add_line("Sample", samples, LineLocation.FOOTER_TOP)
        table.add_line("N", [str(f.n) for f in ok], LineLocation.FOOTER_TOP)
        table.add_custom_notes([f"Standard errors: {st.cov_type}"])
        parts.append(table.render_html())
    else:
        parts.append(f"<p>Standard errors: {html.escape(st.cov_type)}</p>")
    if failed:
        items = "".join(
            f"<li>({f.index + 1}) {html.escape(f.error or '')}</li>" for f in failed
        )
        parts.append(f'<ul class="fit-errors">{items}</ul>')
    return "\n".join(parts)
