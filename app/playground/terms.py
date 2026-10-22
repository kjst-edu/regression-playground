"""Regression terms and their three notations: short, patsy, display label.

Short syntax (URL queries, app.toml defaults): ``educ + ln(wage) + sq(exper)
+ C(region) + educ:female - 1``. Terms are separated by ``+`` or whitespace
(a ``+`` in a URL query can arrive as a space). User input never raises:
invalid pieces are dropped and reported in a notice.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .spec import Spec


@dataclass(frozen=True)
class Atom:
    var: str
    log: bool = False
    square: bool = False
    categorical: bool = False


@dataclass(frozen=True)
class Interaction:
    left: Atom
    right: Atom


Term = Atom | Interaction


def to_patsy(t: Term) -> str:
    if isinstance(t, Interaction):
        return f"{to_patsy(t.left)}:{to_patsy(t.right)}"
    if t.categorical:
        return f"C({t.var})"
    inner = f"np.log({t.var})" if t.log else t.var
    return f"I({inner}**2)" if t.square else inner


def to_short(t: Term) -> str:
    if isinstance(t, Interaction):
        return f"{to_short(t.left)}:{to_short(t.right)}"
    if t.categorical:
        return f"C({t.var})"
    inner = f"ln({t.var})" if t.log else t.var
    return f"sq({inner})" if t.square else inner


def label(t: Term) -> str:
    if isinstance(t, Interaction):
        return f"{label(t.left)} × {label(t.right)}"
    if t.categorical:
        return f"C({t.var})"
    if t.log and t.square:
        return f"(ln {t.var})²"
    if t.log:
        return f"ln {t.var}"
    return f"{t.var}²" if t.square else t.var


def format_short_rhs(terms: list[Term], constant: bool) -> str:
    s = " + ".join(to_short(t) for t in terms)
    if not constant:
        s = f"{s} - 1" if s else "-1"
    return s


_NAME = r"[A-Za-z_]\w*"
_NO_CONSTANT = re.compile(r"(?<![\w)])-\s*1(?!\w)")


def _split(s: str) -> list[str]:
    """Split on ``+`` and whitespace outside parentheses."""
    parts, buf, depth = [], "", 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        if depth == 0 and (ch == "+" or ch.isspace()):
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    parts.append(buf)
    return [p for p in parts if p]


def _parse_atom(piece: str, spec: Spec) -> tuple[Atom | None, str | None]:
    """Return (atom, None) or (None, reason)."""
    m = re.fullmatch(rf"(sq|ln|C)\((.*)\)", piece)
    func, inner = (m.group(1), m.group(2)) if m else (None, piece)
    log = square = categorical = False
    if func == "sq":
        square = True
        m2 = re.fullmatch(rf"ln\(({_NAME})\)", inner)
        if m2:
            log, inner = True, m2.group(1)
    elif func == "ln":
        log = True
    elif func == "C":
        categorical = True
    if not re.fullmatch(_NAME, inner):
        return None, "not understood"
    v = spec.variables.get(inner)
    if v is None:
        return None, f"unknown variable {inner!r}"
    if v.type == "category":
        categorical = True
    if categorical and (log or square):
        return None, f"cannot apply ln or sq to categorical {inner!r}"
    if log and v.logged:
        return None, f"{inner!r} is already logged"
    return Atom(inner, log=log, square=square, categorical=categorical), None


def parse_short_rhs(s: str, spec: Spec) -> tuple[list[Term], bool, list[str]]:
    constant = not _NO_CONSTANT.search(s)
    s = _NO_CONSTANT.sub(" ", s)
    terms: list[Term] = []
    notices: list[str] = []
    for piece in _split(s):
        halves = piece.split(":")
        atoms, reason = [], None
        if len(halves) > 2:
            reason = "interaction takes exactly two terms"
        else:
            for h in halves:
                atom, reason = _parse_atom(h, spec)
                if atom is None:
                    break
                atoms.append(atom)
        if reason is not None:
            notices.append(f"Dropped {piece!r}: {reason}")
            continue
        term = atoms[0] if len(atoms) == 1 else Interaction(atoms[0], atoms[1])
        if term not in terms:
            terms.append(term)
    return terms, constant, notices


def allowed(atom: Atom, spec: Spec) -> set[str]:
    """Helper buttons the UI may enable for this atom in its current state."""
    v = spec.variables.get(atom.var)
    out = {"interact"}
    if not atom.categorical:
        if not atom.log and not (v and v.logged):
            out.add("ln")
        if not atom.square:
            out.add("sq")
        if not atom.log and not atom.square:
            out.add("C")
    return out
