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

from playground.describe import categorical_counts, numeric_summary
from playground.fit import fit_all, regression_table_html
from playground.query import state_from_query
from playground.spec import load_data, load_spec
from playground.state import (
    COV_TYPES,
    add_model,
    add_terms,
    duplicate_model,
    formula,
    initial_state,
    interaction_terms,
    remove_model,
    remove_term,
    set_constant,
    set_cov_type,
    set_sample,
    set_y,
)
from playground.terms import Atom, allowed, label, to_patsy

HERE = Path(__file__).parent
spec = load_spec(HERE / "app.toml")
df = load_data(spec, HERE)

state = reactive.value(initial_state(spec))
# The term being built in the "+" dialog: {"model", "atom", "mode"} or None.
draft = reactive.value(None)
query_applied = [False]

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
function pgSend(el, value) {
  var d = el.dataset;
  Shiny.setInputValue('action', {
    kind: d.action, model: d.model ?? null, term: d.term ?? null,
    var: d.var ?? null, value: value
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
.pg-header { display: flex; align-items: flex-start; gap: 8px; margin: 12px 0 8px; }
.pg-header .pg-titles { flex: 1; min-width: 0; }
.pg-header h1 { font-size: 1.35rem; margin: 0; }
.pg-header p { margin: 2px 0 0; color: #666; font-size: .9rem; }
.pg-btn { min-height: 44px; min-width: 44px; padding: 0 12px; border: 1px solid #ccc;
  border-radius: 8px; background: #f7f7f7; font-size: 1rem; touch-action: manipulation; }
.pg-btn:disabled { opacity: .35; }
.pg-btn.on { background: #0d6efd; border-color: #0d6efd; color: #fff; }
.pg-btn.primary { background: #0d6efd; border-color: #0d6efd; color: #fff; }
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
.pg-formulas { font-family: ui-monospace, monospace; font-size: .85rem; background: #f6f6f6;
  border-radius: 8px; padding: 8px 10px; overflow-wrap: anywhere; }
.pg-formulas div { margin: 2px 0; }
.pg-vars { display: flex; flex-direction: column; gap: 4px; max-height: 40vh; overflow-y: auto; }
.pg-var { display: flex; gap: 4px; }
.pg-var .pg-btn:first-child { flex: 1; text-align: left; }
.pg-helpers { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }
.pg-preview { margin: 8px 0; min-height: 1.5em; overflow-wrap: anywhere; }
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
.modal-body .shiny-input-radiogroup label { min-height: 44px; display: flex; align-items: center; gap: 8px; }
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
                    btn("ln", "y_ln", cls="on" if st.y.log else "", disabled=not can_ln),
                    select,
                    class_="pg-yrow",
                ),
            )

        @render.ui
        def frames():
            st = state()
            out = []
            for i, m in enumerate(st.models):
                rows = [
                    tags.div(
                        tags.h3(f"Model {i + 1}"),
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
            if st.models:
                out.append(
                    tags.div(
                        *[tags.div(f"({i + 1}) {formula(st, i)}") for i in range(len(st.models))],
                        class_="pg-formulas",
                    )
                )
            return TagList(*out)

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
                            HTML(num.to_html(float_format=lambda x: f"{x:.4g}", classes="table table-sm")),
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


# Dialog bodies are outputs held off the page and placed inside modals.
with ui.hold():

    @render.ui
    def add_body():
        d = draft()
        if d is None:
            return None
        atom = d["atom"]
        if d["mode"] == "partner":
            rows = [
                tags.div(btn(n, "partner", var=n), class_="pg-var")
                for n in spec.variables
                if n != atom.var
            ]
            return TagList(
                tags.p("Interact ", tags.strong(label(atom)), " with:"),
                tags.div(*rows, class_="pg-vars"),
                tags.div(btn("Back", "partner_back"), class_="pg-actions"),
            )
        rows = [
            tags.div(
                btn(n, "pick", cls="on" if atom is not None and atom.var == n else "", var=n),
                btn("ⓘ", "info", cls="icon", title=f"About {n}", var=n),
                class_="pg-var",
            )
            for n in spec.variables
        ]
        can = allowed(atom, spec) if atom is not None else set()
        forced_c = atom is not None and spec.variables[atom.var].type == "category"

        def helper(text, op, active, ok):
            return btn(text, "mod", cls="on" if active else "", disabled=not ok, value=op)

        helpers = tags.div(
            helper("ln", "ln", atom is not None and atom.log, atom is not None and (atom.log or "ln" in can)),
            helper("sq", "sq", atom is not None and atom.square, atom is not None and (atom.square or "sq" in can)),
            helper(
                "C",
                "C",
                atom is not None and atom.categorical,
                atom is not None and not forced_c and (atom.categorical or "C" in can),
            ),
            btn("× interact", "interact", disabled=atom is None or "interact" not in can),
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
            tags.div(
                btn("Cancel", "cancel"),
                btn("Add", "add_term", cls="primary", disabled=atom is None),
                class_="pg-actions",
            ),
        )


@reactive.calc
def fitted():
    return fit_all(state(), spec, df)


def show_add_modal():
    ui.modal_show(
        ui.modal(core_ui.output_ui("add_body"), title="Add a term", footer=None, easy_close=True)
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
        ui.modal(
            tags.dl(*items),
            tags.div(btn("Back", "back"), class_="pg-actions"),
            title=var,
            footer=None,
            easy_close=True,
        )
    )


def show_gear_modal():
    ui.modal_show(
        ui.modal(
            ui.input_radio_buttons(
                "cov_type",
                "Standard errors (all columns)",
                choices={c: "nonrobust (usual OLS)" if c == "nonrobust" else c for c in COV_TYPES},
                selected=state().cov_type,
            ),
            title="Settings",
            footer=ui.modal_button("Close"),
            easy_close=True,
        )
    )


@reactive.effect
@reactive.event(input.action)
def _on_action():
    a = input.action()
    kind = a.get("kind")
    i = int(a["model"]) if a.get("model") not in (None, "") else None
    st = state()
    d = draft()

    if kind == "y_var":
        var = a["value"]
        log = st.y.log and not spec.variables[var].logged
        state.set(set_y(st, Atom(var, log=log)))
    elif kind == "y_ln":
        state.set(set_y(st, Atom(st.y.var, log=not st.y.log)))
    elif kind == "sample":
        state.set(set_sample(st, i, a["value"]))
    elif kind == "dup":
        state.set(duplicate_model(st, i))
    elif kind == "del":
        state.set(remove_model(st, i))
    elif kind == "add_model":
        state.set(add_model(st))
    elif kind == "rm_term":
        state.set(remove_term(st, i, int(a["term"])))
    elif kind == "const_off":
        state.set(set_constant(st, i, False))
    elif kind == "const_on":
        state.set(set_constant(st, i, True))
    elif kind == "gear":
        show_gear_modal()
    elif kind == "open_add":
        draft.set({"model": i, "atom": None, "mode": "build"})
        show_add_modal()
    elif kind == "cancel":
        draft.set(None)
        ui.modal_remove()
    elif d is None:
        return
    elif kind == "pick":
        draft.set({**d, "atom": plain_atom(a["var"])})
    elif kind == "mod" and d["atom"] is not None:
        atom = d["atom"]
        field = {"ln": "log", "sq": "square", "C": "categorical"}[a["value"]]
        draft.set({**d, "atom": replace(atom, **{field: not getattr(atom, field)})})
    elif kind == "interact" and d["atom"] is not None:
        draft.set({**d, "mode": "partner"})
    elif kind == "partner_back":
        draft.set({**d, "mode": "build"})
    elif kind == "partner":
        terms = interaction_terms(d["atom"], plain_atom(a["var"]))
        state.set(add_terms(st, d["model"], terms))
        draft.set(None)
        ui.modal_remove()
    elif kind == "info":
        show_info_modal(a["var"])
    elif kind == "back":
        show_add_modal()
    elif kind == "add_term" and d["atom"] is not None:
        state.set(add_terms(st, d["model"], [d["atom"]]))
        draft.set(None)
        ui.modal_remove()


@reactive.effect
@reactive.event(input.cov_type)
def _on_cov_type():
    ct = input.cov_type()
    if ct in COV_TYPES and ct != state().cov_type:
        state.set(set_cov_type(state(), ct))


@reactive.effect
@reactive.event(input.page_search)
def _on_page_search():
    if query_applied[0]:
        return
    query_applied[0] = True
    st, notices = state_from_query(input.page_search() or "", spec)
    state.set(st)
    for n in notices:
        ui.notification_show(n, type="warning", duration=10)
