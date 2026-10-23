"""Screen state: models, terms, constant, sample, standard-error type.

Everything here is a pure function; each update returns a new ``State``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .spec import Spec
from .terms import Atom, Interaction, Term, parse_short_rhs, to_patsy

COV_TYPES = ("nonrobust", "HC0", "HC1", "HC2", "HC3")
DIGITS_CHOICES = (2, 3, 4, 5, 6)


@dataclass(frozen=True)
class Model:
    terms: tuple[Term, ...]
    constant: bool = True
    sample: str | None = None


@dataclass(frozen=True)
class State:
    y: Atom
    models: tuple[Model, ...]
    cov_type: str = "nonrobust"
    digits: int = 3


def _with_model(st: State, i: int, model: Model) -> State:
    models = list(st.models)
    models[i] = model
    return replace(st, models=tuple(models))


def set_y(st: State, atom: Atom) -> State:
    return replace(st, y=atom)


def add_model(st: State) -> State:
    new = st.models[-1] if st.models else Model(terms=())
    return replace(st, models=st.models + (new,))


def duplicate_model(st: State, i: int) -> State:
    models = list(st.models)
    models.insert(i + 1, st.models[i])
    return replace(st, models=tuple(models))


def remove_model(st: State, i: int) -> State:
    return replace(st, models=st.models[:i] + st.models[i + 1:])


def _same(a: Term, b: Term) -> bool:
    if a == b:
        return True
    return (
        isinstance(a, Interaction)
        and isinstance(b, Interaction)
        and a.left == b.right
        and a.right == b.left
    )


def add_terms(st: State, i: int, terms: list[Term]) -> State:
    kept = list(st.models[i].terms)
    for t in terms:
        if not any(_same(t, k) for k in kept):
            kept.append(t)
    return _with_model(st, i, replace(st.models[i], terms=tuple(kept)))


def remove_term(st: State, i: int, j: int) -> State:
    m = st.models[i]
    return _with_model(st, i, replace(m, terms=m.terms[:j] + m.terms[j + 1:]))


def set_constant(st: State, i: int, on: bool) -> State:
    return _with_model(st, i, replace(st.models[i], constant=on))


def set_sample(st: State, i: int, name: str | None) -> State:
    return _with_model(st, i, replace(st.models[i], sample=name))


def set_cov_type(st: State, ct: str) -> State:
    return replace(st, cov_type=ct)


def set_digits(st: State, n: int) -> State:
    return replace(st, digits=n)


def interaction_terms(a: Atom, b: Atom) -> list[Term]:
    if a == b:
        return [a]
    return [a, b, Interaction(a, b)]


def formula(st: State, i: int) -> str:
    m = st.models[i]
    rhs = " + ".join(to_patsy(t) for t in m.terms)
    if not rhs:
        rhs = "1" if m.constant else "0"
    elif not m.constant:
        rhs += " - 1"
    return f"{to_patsy(st.y)} ~ {rhs}"


def first_sample(spec: Spec) -> str | None:
    return getattr(spec.samples[0], "key") if spec.samples else None


def initial_state(spec: Spec) -> State:
    d = spec.defaults
    first = first_sample(spec)
    chosen = d.get("samples") or []
    models = []
    for n, rhs in enumerate(d.get("models", [])):
        terms, constant, _ = parse_short_rhs(rhs, spec)
        sample = chosen[n] if n < len(chosen) else first
        models.append(Model(tuple(terms), constant, sample))
    y_terms, _, _ = parse_short_rhs(d["y"], spec)
    y = y_terms[0] if y_terms and isinstance(y_terms[0], Atom) else Atom(d["y"])
    return State(y, tuple(models), d.get("cov_type", "nonrobust"), spec.digits)
