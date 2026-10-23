# regression-playground

A phone-friendly page where students build OLS regressions on your data, add
models side by side, and compare them in a regression table. It is a Shiny for
Python app exported with [shinylive](https://shinylive.io/py/): the result is a
static site that runs entirely in the browser (Pyodide). No server is needed.

## Instructor steps

1. Shallow-clone this template (you do not need its history):

   ```
   git clone --depth 1 https://github.com/kjst-edu/regression-playground.git
   cd regression-playground
   ```

2. Make a data directory with two files, `data.csv` (or `.parquet`) and
   `app.toml` (see below). `example/` is a working one.

3. Pack it (needs [uv](https://docs.astral.sh/uv/); the first run downloads the
   shinylive assets and, once, the stargazer wheel):

   ```
   uv run pack.py my-data site
   ```

   `pack.py` checks `app.toml` against the data first (missing columns, a
   `where` expression that does not parse, a malformed toml, a column that does
   not fit its `type`, unknown names or bad terms in `[defaults]`). On a
   mistake it prints a one-line reason, exits with status 1 and writes
   nothing. The exported site carries only the declared variables and the `id`
   column, in the same file name and format; other columns of your data file
   are not published.

4. Host the `site/` directory on any static web host (for example GitHub
   Pages). To try it locally:
   `python -m http.server --directory site --bind localhost 8008`.

The `?query` parameters (below) read the address of the page that contains the
app, so the app must be served from the same site as the page you link to,
which is the case when you host `site/` as is.

stargazer (pure Python, GPLv2) is bundled into the site, so the page does not
contact PyPI when it loads.

## `app.toml`

```toml
[app]
title    = "Wage regression playground"   # page title and heading
subtitle = "Synthetic data, 200 workers"  # optional
digits   = 3                              # optional: initial decimals in the regression table (default 3; students can change it in Settings)

[data]
file = "data.csv"      # relative to the data directory; .parquet is read as parquet
id   = "id"            # optional: column used to name rows (kept even if not declared below)

[source]               # shown on the About tab; every key optional
name     = "Synthetic wage data"
citation = "..."
license  = "CC0"
url      = "https://example.org"
note     = "Not real data."

[variables.educ]       # one table per variable; only declared columns reach the app
label  = "Years of education"   # required
unit   = "years"                # optional
type   = "int"                  # optional: float | int | str | category | bool
logged = false                  # optional: true if the column is already a log

[samples]              # in this order; the first is the default
all   = { label = "All workers" }                      # no where = all rows
urban = { label = "Urban only", where = "urban == 1" } # DataFrame.query expression

[defaults]             # the initial screen when there is no query
y        = "lwage"
models   = ["educ", "educ + exper + sq(exper)"]  # short syntax, see below
samples  = ["all", "urban"]                      # one key per model; missing = first sample
cov_type = "HC1"                                 # nonrobust | HC0 | HC1 | HC2 | HC3
```

Notes:

- `[data] file` is required. `type` is passed to the reader (`read_csv(dtype=...)`);
  columns without a type are inferred by pandas. A `category` column is always
  entered as a set of dummies. Variable names must be valid Python identifiers.
- `logged = true` declares a column that already holds logs, so `ln` is not
  offered for it.
- `where` is evaluated with `DataFrame.query`, so any expression your data
  supports works. The data are yours, so nothing is sandboxed.
- Columns not declared in `[variables]` are not loaded.
- Parquet loads more slowly on a phone; prefer CSV unless the data are large.

## Short syntax and patsy

Models in `[defaults] models` and in the `m1`, `m2`, ... query parameters use a
short syntax. The screen shows the equivalent patsy formula.

| Short syntax | patsy | Meaning |
| --- | --- | --- |
| `x` | `x` | variable |
| `ln(x)` | `np.log(x)` | natural log |
| `sq(x)` | `I(x**2)` | square |
| `sq(ln(x))` | `I(np.log(x)**2)` | square of the log |
| `C(x)` | `C(x)` | categorical (dummies); automatic for `type = "category"` |
| `a:b` | `a:b` | interaction of two terms |
| `-1` | `- 1` | drop the constant |

Terms are joined with `+`. `ln` is refused on a column declared `logged`, and
`ln`/`sq` on a categorical column.

## Query parameters

```
?y=lwage&m1=educ&m2=educ%2Bexper%2Bsq(exper)&s2=urban&cov=HC1&digits=4
```

- `y`: dependent variable, `x` or `ln(x)`.
- `m1`, `m2`, ...: right-hand side of each model, in short syntax.
- `s1`, `s2`, ...: sample key of each model (a key from `[samples]`); a model
  without one uses the first sample.
- `cov`: standard errors for all columns: `nonrobust`, `HC0`, `HC1`, `HC2` or
  `HC3`. Other values are ignored.
- `digits`: decimal places in the regression table, 2 to 6. Other values are
  ignored with a notice. Needs `y` or `m1` as well, like `cov`.
- A query replaces `[defaults]`. Parts that cannot be read are dropped and a
  short notice appears on screen.

In a URL a `+` means a space, so write it as `%2B`. The app also treats a
space as `+` when one arrives, so a pasted link with a bare `+` still works.

## Load time

Measured on localhost with this repository's `example` export, headless
Chromium (Playwright), until the Model tab is drawn:

- first load in a fresh browser session: about 13.6 s
- reload: about 11.8 s

An earlier measurement (a cold browser cache, before stargazer was bundled)
gave about 20 s and 17 s. The cost is Pyodide plus pandas, scipy and
statsmodels. Over a real network the first load also downloads these packages,
so allow longer on a phone.
