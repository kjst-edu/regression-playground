"""Read an initial screen state from a URL query: ``?y=ln(wage)&m1=educ&s1=urban``."""
from __future__ import annotations

import re
from urllib.parse import parse_qs

from .spec import Spec
from .state import COV_TYPES, DIGITS_CHOICES, Model, State, first_sample, initial_state
from .terms import Atom, parse_short_rhs


def _parse_y(value: str, spec: Spec) -> tuple[Atom | None, list[str]]:
    terms, _, notices = parse_short_rhs(value, spec)
    if len(terms) == 1 and isinstance(terms[0], Atom):
        a = terms[0]
        if not a.square and not a.categorical:
            return a, notices
    return None, notices + [f"Ignored y={value!r}: use a variable or ln(variable)"]


def state_from_query(search: str, spec: Spec) -> tuple[State, list[str]]:
    q = parse_qs(search.lstrip("?"), keep_blank_values=True)
    base = initial_state(spec)
    nums = sorted(int(m.group(1)) for k in q if (m := re.fullmatch(r"m(\d+)", k)))
    if "y" not in q and not nums:
        return base, []

    notices: list[str] = []
    y = base.y
    if "y" in q:
        atom, ns = _parse_y(q["y"][0], spec)
        notices += ns
        if atom is not None:
            y = atom

    models = base.models
    if nums:
        first = first_sample(spec)
        valid = {getattr(s, "key") for s in spec.samples}
        built = []
        for n in nums:
            terms, constant, ns = parse_short_rhs(q[f"m{n}"][0], spec)
            notices += ns
            sample = first
            if f"s{n}" in q:
                name = q[f"s{n}"][0]
                if name in valid:
                    sample = name
                else:
                    notices.append(f"Ignored sample {name!r}: unknown sample")
            built.append(Model(tuple(terms), constant, sample))
        models = tuple(built)

    cov = base.cov_type
    if "cov" in q and q["cov"][0] in COV_TYPES:
        cov = q["cov"][0]
    digits = base.digits
    if "digits" in q:
        try:
            n = int(q["digits"][0])
        except ValueError:
            n = None
        if n in DIGITS_CHOICES:
            digits = n
        else:
            notices.append(f"Ignored digits={q['digits'][0]!r}: use 2 to 6")
    return State(y, models, cov, digits), notices
