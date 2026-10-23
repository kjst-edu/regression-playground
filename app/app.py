# Regression playground: build OLS models on a phone and compare them.
#
# The screen is thin: every change goes through the pure functions in
# playground.state. Chips and frame buttons are plain HTML whose clicks one
# JavaScript handler sends as input.action. (No module docstring: Shiny
# Express would print it on the page.)
from dataclasses import replace
from pathlib import Path

from htmltools import HTML, TagList, tags
from shiny import reactive, render
from shiny import ui as core_ui
from shiny.express import input, ui

from playground.describe import (
    categorical_counts,
    format_stat,
    ln_warning,
    nonpositive_count,
    numeric_summary,
    y_ln_warnings,
)
from playground.fit import fit_all, regression_table_html
from playground.query import state_from_query
from playground.spec import load_data, load_spec
from playground.state import (
    COV_TYPES,
    DIGITS_CHOICES,
    add_model,
    add_terms,
    duplicate_model,
    formula,
    initial_state,
    interaction_terms,
    new_terms,
    remove_model,
    remove_term,
    set_constant,
    set_cov_type,
    set_digits,
    set_sample,
    set_y,
)
from playground.terms import Atom, allowed, label, to_patsy

HERE = Path(__file__).parent
spec = load_spec(HERE / "app.toml")
df = load_data(spec, HERE)

state = reactive.value(initial_state(spec))
# The term being built in the "+" dialog: {"model", "atom", "mode", "partner"}
# or None. mode is "build" or "partner"; partner is the chosen interaction
# partner (an Atom) until Add is pressed.
draft = reactive.value(None)
query_applied = [False]
# Bumped on every state change and stamped on the rendered controls, so an
# action sent from controls drawn before the last change (a double tap on ×)
# is ignored instead of hitting another model or term.
rev = [0]


def commit(new) -> None:
    rev[0] += 1
    state.set(new)

Y_TYPES = {None, "float", "int"}


def plain_atom(var: str) -> Atom:
    return Atom(var, categorical=spec.variables[var].type == "category")


def sample_key(m) -> str | None:
    return m.sample if m.sample is not None else (spec.samples[0].key if spec.samples else None)


def btn(text, action, cls="", disabled=False, title=None, **data):
    attrs = {f"data-{k}": str(v) for k, v in data.items()}
    return tags.button(
        text,
        type="button",
        class_=f"pg-btn {cls}".strip(),
        disabled=disabled or None,
        title=title,
        aria_label=title,
        **{"data-action": action},
        **attrs,
    )


ui.page_opts(window_title=spec.title, lang="en")

ui.head_content(
    tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
    tags.script(
        HTML(
            """
$(document).on('shiny:connected', function () {
  var q = '';
  try { q = window.parent.location.search || ''; } catch (e) { q = ''; }
  Shiny.setInputValue('page_search', q);
});
// Crossing into the wide layout: the Table pane was hidden, so Shiny had
// suspended its output. Visit the tabs once so it renders.
try {
  window.matchMedia('(min-width: 992px)').addEventListener('change', function (e) {
    if (!e.matches) return;
    var cur = document.querySelector('.nav-tabs .nav-link.active');
    var t = document.querySelector('.nav-tabs .nav-link[data-value="Table"]');
    if (t && cur && cur.dataset.value !== 'About') { t.click(); cur.click(); }
  });
} catch (e) {}
function pgSend(el, value) {
  var d = el.dataset;
  var r = el.closest('[data-rev]');
  Shiny.setInputValue('action', {
    kind: d.action, model: d.model ?? null, term: d.term ?? null,
    var: d.var ?? null, value: value, rev: r ? r.dataset.rev : null
  }, {priority: 'event'});
}
document.addEventListener('click', function (ev) {
  var el = ev.target.closest('[data-action]');
  if (!el || el.tagName === 'SELECT' || el.disabled) return;
  pgSend(el, el.dataset.value ?? null);
});
document.addEventListener('change', function (ev) {
  var el = ev.target;
  if (el.tagName === 'SELECT' && el.dataset.action) pgSend(el, el.value);
});
"""
        )
    ),
    tags.style(
        HTML(
            """
html, body { overflow-x: hidden; }
body { background: #fff; }
.container, .container-fluid { max-width: 760px; padding-left: 12px; padding-right: 12px; }
/* Wide screens: Model and Table side by side, About as the other tab. Narrow
   screens keep three tabs. The table is rendered once; CSS only places it. */
@media (min-width: 992px) {
  .container, .container-fluid { max-width: 1240px; }
  .tab-content { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); column-gap: 28px; align-items: start; }
  .tab-content > .tab-pane[data-value="Model"],
  .tab-content > .tab-pane[data-value="Table"] { display: block !important; opacity: 1 !important; }
  .tab-content > .tab-pane[data-value="Table"] { position: sticky; top: 8px; }
  .tab-content > .tab-pane[data-value="About"] { grid-column: 1 / -1; }
  .tab-content:has(> .tab-pane[data-value="About"].active) > .tab-pane[data-value="Model"],
  .tab-content:has(> .tab-pane[data-value="About"].active) > .tab-pane[data-value="Table"] { display: none !important; }
  .nav-tabs .nav-item:has(> .nav-link[data-value="Table"]) { display: none; }
  .navset-tab:not(:has(.tab-pane[data-value="About"].active)) .nav-link[data-value="Model"] {
    color: #495057; background: #fff; border-color: #dee2e6 #dee2e6 #fff; }
}
.pg-header { display: flex; align-items: flex-start; gap: 8px; margin: 12px 0 8px; }
.pg-header .pg-titles { flex: 1; min-width: 0; }
.pg-header h1 { font-size: 1.35rem; margin: 0; }
.pg-header p { margin: 2px 0 0; color: #666; font-size: .9rem; }
.pg-btn { min-height: 44px; min-width: 44px; padding: 0 12px; border: 1px solid #ccc;
  border-radius: 8px; background: #f7f7f7; font-size: 1rem; touch-action: manipulation; }
.pg-btn:disabled { opacity: .35; }
.pg-btn.on { background: #0d6efd; border-color: #0d6efd; color: #fff; }
.pg-btn.primary { background: #0d6efd; border-color: #0d6efd; color: #fff; }
/* Data (variables, terms) are blue; operations (ln, sq, C, × interact) are
   outlined in orange with a monospace label. */
.pg-btn.var { background: #e8f0fe; border-color: #b6ccf5; color: #0b3d91; }
.pg-btn.var.on { background: #0d6efd; border-color: #0d6efd; color: #fff; }
.pg-btn.op { background: #fff; border-color: #c2410c; color: #9a3412;
  font-family: ui-monospace, monospace; font-weight: 600; }
.pg-btn.op.on { background: #9a3412; border-color: #9a3412; color: #fff; }
.pg-btn.icon { padding: 0; background: transparent; border-color: transparent; font-size: 1.2rem; }
.nav-tabs .nav-link { min-height: 44px; display: flex; align-items: center; }
.pg-select { min-height: 44px; font-size: 1rem; border: 1px solid #ccc; border-radius: 8px;
  padding: 0 8px; background: #fff; max-width: 100%; }
.pg-yrow { display: flex; gap: 8px; align-items: center; margin-bottom: 16px; }
.pg-yrow .pg-select { flex: 1; min-width: 0; }
.pg-section-label { font-weight: 600; margin: 12px 0 6px; }
.pg-frame { border: 1px solid #ddd; border-radius: 10px; padding: 8px 10px 10px; margin-bottom: 12px; }
.pg-frame-head { display: flex; align-items: center; }
.pg-frame-head h3 { font-size: 1.05rem; margin: 0; flex: 1; }
.pg-row { display: flex; gap: 8px; align-items: center; margin: 6px 0; }
.pg-row > span { min-width: 4.5em; color: #555; }
.pg-row .pg-select { flex: 1; min-width: 0; }
.pg-chips { display: flex; flex-wrap: wrap; gap: 6px; }
.pg-chip { display: inline-flex; align-items: center; min-height: 44px; border-radius: 22px;
  background: #e8f0fe; padding-left: 14px; max-width: 100%; }
.pg-chip > span { overflow-wrap: anywhere; }
.pg-chip .pg-btn { background: transparent; border: none; border-radius: 22px; }
.pg-chip.off { background: transparent; border: 1px dashed #aaa; padding-left: 0; color: #666; }
.pg-add-model { width: 100%; margin-bottom: 16px; }
.pg-formula { font-family: ui-monospace, monospace; font-size: .95rem; background: #f6f6f6;
  border-radius: 8px; padding: 10px 12px; overflow-wrap: anywhere; white-space: pre-wrap; margin: 0; }
.pg-vars { display: flex; flex-direction: column; gap: 4px; max-height: 40vh; overflow-y: auto; }
.pg-var { display: flex; gap: 4px; }
.pg-var .pg-btn:first-child { flex: 1; text-align: left; }
.pg-helpers { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }
.pg-preview { margin: 8px 0; min-height: 1.5em; overflow-wrap: anywhere; }
.pg-warn { margin: 8px 0; padding: 6px 10px; border-radius: 8px; background: #fff4e5;
  color: #7a3300; font-size: .9rem; overflow-wrap: anywhere; }
.pg-yrow + .pg-warn { margin: -8px 0 16px; }
.pg-preview code { font-size: .85rem; }
.pg-actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 8px; }
.table-box { overflow-x: auto; max-width: 100%; -webkit-overflow-scrolling: touch; }
.table-box table { white-space: nowrap; }
.table-box td, .table-box th { padding: 2px 8px; }
.fit-errors { color: #a00; }
.pg-about table { font-size: .9rem; }
.pg-about dt { font-weight: 600; }
.pg-about dd { margin-bottom: 6px; overflow-wrap: anywhere; }
.pg-about .form-group, .pg-about .shiny-input-container { width: 100%; }
.pg-about select { min-height: 44px; }
.modal-footer .btn, .modal .btn-close { min-height: 44px; min-width: 44px; }
#shiny-notification-panel .shiny-notification { padding-right: 48px; min-height: 48px; }
#shiny-notification-panel .shiny-notification-close { width: 44px; height: 44px; }
.modal-body .shiny-input-radiogroup label { min-height: 44px; display: flex; align-items: center; gap: 8px; }
.modal-body .shiny-input-container-inline label, .modal-body .form-check-inline label { min-width: 44px; }
"""
        )
    ),
)

with ui.div(class_="pg-header"):
    with ui.div(class_="pg-titles"):
        ui.h1(spec.title)
        if spec.subtitle:
            ui.p(spec.subtitle)
    btn("⚙", "gear", cls="icon", title="Settings")

with ui.navset_tab(id="tab"):
    with ui.nav_panel("Model"):

        @render.ui
        def y_row():
            st = state()
            v = st.y.var
            can_ln = "ln" in allowed(Atom(v), spec) and not spec.variables[v].logged
            choices = [n for n, var in spec.variables.items() if var.type in Y_TYPES]
            if v not in choices:
                choices.insert(0, v)
            select = tags.select(
                *[tags.option(n, value=n, selected=(n == v) or None) for n in choices],
                class_="pg-select",
                aria_label="Dependent variable",
                **{"data-action": "y_var"},
            )
            return TagList(
                tags.div("Dependent variable", class_="pg-section-label"),
                tags.div(
                    btn("ln", "y_ln", cls="op on" if st.y.log else "op", disabled=not can_ln),
                    select,
                    class_="pg-yrow",
                    data_rev=str(rev[0]),
                ),
                warn(*y_ln_warnings(st, spec, df)),
            )

        @render.ui
        def frames():
            st = state()
            out = []
            for i, m in enumerate(st.models):
                rows = [
                    tags.div(
                        tags.h3(f"Model {i + 1}"),
                        btn("ƒ", "show_formula", cls="icon", title=f"Formula of model {i + 1}", model=i),
                        btn("⧉", "dup", cls="icon", title=f"Duplicate model {i + 1}", model=i),
                        btn("×", "del", cls="icon", title=f"Remove model {i + 1}", model=i),
                        class_="pg-frame-head",
                    )
                ]
                if spec.samples:
                    current = sample_key(m)
                    rows.append(
                        tags.div(
                            tags.span("Sample"),
                            tags.select(
                                *[
                                    tags.option(s.label, value=s.key, selected=(s.key == current) or None)
                                    for s in spec.samples
                                ],
                                class_="pg-select",
                                aria_label=f"Sample of model {i + 1}",
                                **{"data-action": "sample", "data-model": str(i)},
                            ),
                            class_="pg-row",
                        )
                    )
                chips = []
                if m.constant:
                    chips.append(
                        tags.span(
                            tags.span("Constant"),
                            btn("×", "const_off", title="Remove constant", model=i),
                            class_="pg-chip",
                        )
                    )
                else:
                    chips.append(
                        tags.span(
                            btn("＋ Constant", "const_on", title="Add constant", model=i),
                            class_="pg-chip off",
                        )
                    )
                for j, t in enumerate(m.terms):
                    chips.append(
                        tags.span(
                            tags.span(label(t)),
                            btn("×", "rm_term", title=f"Remove {label(t)}", model=i, term=j),
                            class_="pg-chip",
                        )
                    )
                chips.append(btn("＋", "open_add", title="Add a term", model=i))
                rows.append(tags.div("Regressors", class_="pg-section-label"))
                rows.append(tags.div(*chips, class_="pg-chips"))
                out.append(tags.div(*rows, class_="pg-frame"))
            out.append(btn("＋ Add model", "add_model", cls="pg-add-model"))
            return tags.div(*out, data_rev=str(rev[0]))

    with ui.nav_panel("Table"):

        @render.ui
        def table():
            st = state()
            if not st.models:
                return tags.p("No models. Add one on the Model tab.")
            html = regression_table_html(st, spec, fitted())
            return tags.div(HTML(html), class_="table-box")

    with ui.nav_panel("About"):
        with ui.div(class_="pg-about"):
            ui.h4("Source", class_="mt-3")

            @render.ui
            def source():
                s = spec.source
                items = []
                for key, title in [
                    ("name", "Name"),
                    ("citation", "Citation"),
                    ("license", "License"),
                    ("url", "URL"),
                    ("note", "Note"),
                ]:
                    value = s.get(key)
                    if not value:
                        continue
                    body = tags.a(value, href=value, target="_blank", rel="noopener") if key == "url" else value
                    items += [tags.dt(title), tags.dd(body)]
                return tags.dl(*items)

            ui.h4("Variables")
            with ui.div(class_="table-box"):

                @render.ui
                def variables():
                    head = tags.tr(*[tags.th(h) for h in ["Name", "Label", "Unit", "Type", "Logged"]])
                    rows = [
                        tags.tr(
                            tags.td(v.name),
                            tags.td(v.label),
                            tags.td(v.unit or ""),
                            tags.td(v.type or ""),
                            tags.td("yes" if v.logged else ""),
                        )
                        for v in spec.variables.values()
                    ]
                    return tags.table(tags.thead(head), tags.tbody(*rows), class_="table table-sm")

            ui.h4("Descriptive statistics")
            if spec.samples:
                ui.input_select(
                    "about_sample",
                    "Sample",
                    choices={s.key: s.label for s in spec.samples},
                )

            @render.ui
            def describe():
                key = input.about_sample() if spec.samples else None
                num = numeric_summary(spec, df, key)
                cat = categorical_counts(spec, df, key)
                parts = []
                if len(num):
                    parts.append(
                        tags.div(
                            HTML(num.to_html(float_format=lambda x: format_stat(x, state().digits), classes="table table-sm")),
                            class_="table-box",
                        )
                    )
                if len(cat):
                    parts.append(
                        tags.div(
                            HTML(cat.to_html(index=False, classes="table table-sm")),
                            class_="table-box",
                        )
                    )
                return TagList(*parts)


def join_labels(terms):
    """Term labels in bold, separated by commas."""
    out = []
    for k, t in enumerate(terms):
        out += [", "] if k else []
        out.append(tags.strong(label(t)))
    return out


def warn(*texts):
    return tags.div(*[tags.div(t) for t in texts], class_="pg-warn", role="status") if texts else None


def ln_note(atom, i: int):
    """Warning for ln of a variable with non-positive values in model i's sample."""
    st = state()
    if atom is None or not atom.log or i >= len(st.models):
        return None
    w = ln_warning(nonpositive_count(spec, df, atom.var, st.models[i].sample), atom.var)
    return warn(w) if w else None


# Dialog bodies are outputs held off the page and placed inside modals.
with ui.hold():

    @render.ui
    def add_body():
        d = draft()
        if d is None:
            return None
        atom = d["atom"]
        if d["mode"] == "partner":
            partner = d.get("partner")
            rows = [
                tags.div(
                    btn(n, "partner", cls="var on" if partner is not None and partner.var == n else "var", var=n),
                    class_="pg-var",
                )
                for n in spec.variables
                if n != atom.var
            ]
            adds = []
            if partner is not None:
                adds = new_terms(state(), d["model"], interaction_terms(atom, partner))
                preview = (
                    tags.div("Adds ", *join_labels(adds), class_="pg-preview")
                    if adds
                    else tags.div(f"Already in model {d['model'] + 1}.", class_="pg-preview text-muted")
                )
            else:
                preview = tags.div("Pick a partner.", class_="pg-preview text-muted")
            return TagList(
                tags.p("Interact ", tags.strong(label(atom)), " with:"),
                tags.div(*rows, class_="pg-vars"),
                preview,
                ln_note(atom, d["model"]) if adds else None,
                tags.div(
                    btn("Back", "partner_back"),
                    btn("Cancel", "cancel"),
                    btn("Add", "add_interaction", cls="primary", disabled=not adds),
                    class_="pg-actions",
                ),
            )
        rows = [
            tags.div(
                btn(n, "pick", cls="var on" if atom is not None and atom.var == n else "var", var=n),
                btn("ⓘ", "info", cls="icon", title=f"About {n}", var=n),
                class_="pg-var",
            )
            for n in spec.variables
        ]
        can = allowed(atom, spec) if atom is not None else set()
        forced_c = atom is not None and spec.variables[atom.var].type == "category"

        def helper(text, op, active, ok):
            return btn(text, "mod", cls="op on" if active else "op", disabled=not ok, value=op)

        helpers = tags.div(
            helper("ln", "ln", atom is not None and atom.log, atom is not None and (atom.log or "ln" in can)),
            helper("sq", "sq", atom is not None and atom.square, atom is not None and (atom.square or "sq" in can)),
            helper(
                "C",
                "C",
                atom is not None and atom.categorical,
                atom is not None and not forced_c and (atom.categorical or "C" in can),
            ),
            btn("× interact", "interact", cls="op", disabled=atom is None or "interact" not in can),
            class_="pg-helpers",
        )
        preview = (
            tags.div(tags.strong(label(atom)), " ", tags.code(f"({to_patsy(atom)})"), class_="pg-preview")
            if atom is not None
            else tags.div("Pick a variable.", class_="pg-preview text-muted")
        )
        return TagList(
            tags.div(*rows, class_="pg-vars"),
            helpers,
            preview,
            ln_note(atom, d["model"]),
            tags.div(
                btn("Cancel", "cancel"),
                btn("Add", "add_term", cls="primary", disabled=atom is None),
                class_="pg-actions",
            ),
        )


@reactive.calc
def fitted():
    return fit_all(state(), spec, df)


def centered_modal(*args, **kwargs):
    """ui.modal, centred vertically in the viewport (Bootstrap's modal-dialog-centered)."""
    m = ui.modal(*args, **kwargs)
    for child in m.children:
        if "modal-dialog" in str(child.attrs.get("class", "")).split():
            child.add_class("modal-dialog-centered")
    return m


def show_add_modal():
    ui.modal_show(
        centered_modal(core_ui.output_ui("add_body"), title="Add a term", footer=None, easy_close=True)
    )


def show_info_modal(var: str):
    v = spec.variables[var]
    items = [tags.dt("Label"), tags.dd(v.label)]
    if v.unit:
        items += [tags.dt("Unit"), tags.dd(v.unit)]
    items += [tags.dt("Type"), tags.dd(v.type or "(from data)")]
    if v.logged:
        items += [tags.dt("Logged"), tags.dd("yes (ln is not available)")]
    ui.modal_show(
        centered_modal(
            tags.dl(*items),
            tags.div(btn("Back", "back"), class_="pg-actions"),
            title=var,
            footer=None,
            easy_close=True,
        )
    )


def show_formula_modal(i: int):
    ui.modal_show(
        centered_modal(
            tags.p("patsy formula", class_="text-muted mb-2"),
            tags.pre(formula(state(), i), class_="pg-formula"),
            title=f"Model {i + 1}",
            footer=ui.modal_button("Close"),
            easy_close=True,
        )
    )


def show_gear_modal():
    ui.modal_show(
        centered_modal(
            ui.input_radio_buttons(
                "cov_type",
                "Standard errors (all columns)",
                choices={c: "nonrobust (usual OLS)" if c == "nonrobust" else c for c in COV_TYPES},
                selected=state().cov_type,
            ),
            ui.input_radio_buttons(
                "digits",
                "Decimal places in the table",
                choices={str(n): str(n) for n in DIGITS_CHOICES},
                selected=str(state().digits),
                inline=True,
            ),
            title="Settings",
            footer=ui.modal_button("Close"),
            easy_close=True,
        )
    )


MODEL_KINDS = {"sample", "show_formula", "dup", "del", "rm_term", "const_off", "const_on", "open_add"}


def _index(raw, n: int) -> int | None:
    """An index sent by the page, or None if it is malformed or out of range."""
    try:
        k = int(raw)
    except (TypeError, ValueError):
        return None
    return k if 0 <= k < n else None


@reactive.effect
@reactive.event(input.action)
def _on_action():
    a = input.action()
    kind = a.get("kind")
    st = state()
    d = draft()
    # Actions from controls drawn before the last state change are stale.
    if a.get("rev") not in (None, str(rev[0])):
        return
    i = _index(a.get("model"), len(st.models))
    if kind in MODEL_KINDS and i is None:
        return
    if a.get("var") is not None and a["var"] not in spec.variables:
        return

    if kind == "y_var":
        var = a["value"]
        if var not in spec.variables:
            return
        log = st.y.log and "ln" in allowed(Atom(var), spec)
        commit(set_y(st, Atom(var, log=log)))
    elif kind == "y_ln":
        if st.y.log or "ln" in allowed(Atom(st.y.var), spec):
            commit(set_y(st, Atom(st.y.var, log=not st.y.log)))
    elif kind == "sample":
        if a["value"] in {s.key for s in spec.samples}:
            commit(set_sample(st, i, a["value"]))
    elif kind == "show_formula":
        show_formula_modal(i)
    elif kind == "dup":
        commit(duplicate_model(st, i))
    elif kind == "del":
        commit(remove_model(st, i))
    elif kind == "add_model":
        commit(add_model(st))
    elif kind == "rm_term":
        j = _index(a.get("term"), len(st.models[i].terms))
        if j is not None:
            commit(remove_term(st, i, j))
    elif kind == "const_off":
        commit(set_constant(st, i, False))
    elif kind == "const_on":
        commit(set_constant(st, i, True))
    elif kind == "gear":
        show_gear_modal()
    elif kind == "open_add":
        draft.set({"model": i, "atom": None, "mode": "build"})
        show_add_modal()
    elif kind == "cancel":
        draft.set(None)
        ui.modal_remove()
    elif d is None or d["model"] >= len(st.models):
        return
    elif kind == "pick":
        draft.set({**d, "atom": plain_atom(a["var"])})
    elif kind == "mod" and d["atom"] is not None:
        atom = d["atom"]
        op = a["value"]
        field = {"ln": "log", "sq": "square", "C": "categorical"}.get(op)
        if field is not None and (getattr(atom, field) or op in allowed(atom, spec)):
            draft.set({**d, "atom": replace(atom, **{field: not getattr(atom, field)})})
    elif kind == "interact" and d["atom"] is not None:
        draft.set({**d, "mode": "partner", "partner": None})
    elif kind == "partner_back":
        draft.set({**d, "mode": "build", "partner": None})
    elif kind == "partner" and d["atom"] is not None and a["var"] is not None:
        draft.set({**d, "partner": plain_atom(a["var"])})
    elif kind == "add_interaction" and d["atom"] is not None and d.get("partner") is not None:
        commit(add_terms(st, d["model"], interaction_terms(d["atom"], d["partner"])))
        draft.set(None)
        ui.modal_remove()
    elif kind == "info":
        show_info_modal(a["var"])
    elif kind == "back":
        show_add_modal()
    elif kind == "add_term" and d["atom"] is not None:
        commit(add_terms(st, d["model"], [d["atom"]]))
        draft.set(None)
        ui.modal_remove()


@reactive.effect
@reactive.event(input.cov_type)
def _on_cov_type():
    ct = input.cov_type()
    if ct in COV_TYPES and ct != state().cov_type:
        commit(set_cov_type(state(), ct))


@reactive.effect
@reactive.event(input.digits)
def _on_digits():
    try:
        n = int(input.digits())
    except (TypeError, ValueError):
        return
    if n in DIGITS_CHOICES and n != state().digits:
        commit(set_digits(state(), n))


@reactive.effect
@reactive.event(input.page_search)
def _on_page_search():
    if query_applied[0]:
        return
    query_applied[0] = True
    st, notices = state_from_query(input.page_search() or "", spec)
    commit(st)
    for n in notices:
        ui.notification_show(n, type="warning", duration=10)
