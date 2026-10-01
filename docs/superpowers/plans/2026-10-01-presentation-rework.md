# Presentation Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `analysis.ipynb` into a lesson-structured, presentable report whose code lives in focused modules and whose quoted numbers are computed from the data and checked against the prose.

**Architecture:** `analysis/support.py` splits into `data.py` (loading), `style.py` (theme + display), `figures.py` (one function per chart → `Figure`), and `findings.py` (every number the prose quotes). The notebook is generated once from a scratch builder script, executed with outputs saved, and thereafter is the source of truth. Tests check the modules, the prose/findings agreement, the notebook's structure, and end-to-end execution.

**Tech Stack:** Python 3.14 (free-threaded venv), polars, Altair 6, Jupyter/nbclient, pytest, vl-convert-python (PNG QA).

**Spec:** `docs/superpowers/specs/2026-10-01-presentation-rework-design.md`

## Global Constraints

- House style: US spelling; thousands separators as commas (`12,800`); spaced units (`9 µs`, `27.7 GB/s`); dtype and op names in code font.
- No changelog-style text anywhere ("an earlier version…", "this supersedes…").
- Notebook code cells: no `def`/`class`, no dict literal with ≥8 keys, ≤3 non-blank lines each; `metadata.jupyter.source_hidden = true`.
- Every lesson: `## N · <claim>` heading, plain-language text, figure cell(s), `> **What this means in practice**` blockquote, `<details><summary>Details</summary>…</details>`.
- Charting stays Altair; no matplotlib/seaborn/plotly.
- Benchmarks, job scripts and `data/` are not modified.
- Hardware wording: "AMD EPYC 7763 (Perlmutter CPU node)", "NVIDIA A100-40GB", "NVIDIA A100-80GB".

## Review Focus

1. **Number drift between prose and data** — a value edited in Markdown or recomputed in `findings.py` must fail a test, not silently disagree. Pinned in Task 6 (`test_prose_quotes_every_finding`).
2. **Notebook opened without running** — saved outputs must include every chart; an unexecuted notebook would show nothing in the room. Pinned in Task 6 (`test_every_figure_cell_has_saved_output`).
3. **`cpu_pure` loader on implementation labels with no itemsize** (`py-list-float` etc.) — `ITEMSIZE.replace_strict` would raise; the pure loader must not go through byte accounting. Pinned in Task 1.
4. **Ladder rows from different sizes** — quad/longdouble come from 256k elements, everything else from 4M; the chart must label this and the size-insensitivity must be checked. Pinned in Task 3.
5. **Charts unreadable at projector distance / clipped labels** — verified by PNG render and inspection in Task 7.

---

## File Structure

| File | Responsibility |
|---|---|
| `analysis/data.py` (create) | `load_analysis()`, `AnalysisData`, constants for loading, `pure` frame + memory metadata |
| `analysis/style.py` (create) | palette `P`, Altair theme registration, `Figure`, `show()`, `stat_tiles()`, `configure_notebook()` |
| `analysis/figures.py` (create) | one public function per figure, each `(data: AnalysisData) -> Figure` |
| `analysis/findings.py` (create) | `values(data) -> dict[str, float]`, `compute(data) -> dict[str, str]`, `jax_quality(data) -> pl.DataFrame` |
| `analysis/support.py` (delete) | replaced by the above |
| `analysis.ipynb` (rewrite) | the report |
| `tests/test_data.py`, `tests/test_findings.py`, `tests/test_figures.py`, `tests/test_analysis_notebook.py` | tests |
| `pyproject.toml` | add `jupyterlab`; `[dependency-groups] dev` |

Scratch (not committed): `$SCRATCH/build_notebook.py`, `$SCRATCH/render_pngs.py`.

---

### Task 1: Dependencies and `data.py`

**Files:**
- Modify: `pyproject.toml`
- Create: `analysis/data.py`, `tests/test_data.py`

**Interfaces:**
- Produces: `data.load_analysis(root=DATA_ROOT) -> AnalysisData` with fields `raw, meta, runs, parallel, everything, pure, pure_memory`. `pure`: cell-level frame (`n, n_elem, operation, implementation, cache_state, us, us_lo, us_hi, ns_elem, samples`). `pure_memory: dict[str, float]` = deep-sizeof bytes per element per Python implementation (from the largest size). Constants `OUT_COLD`, `OUT_HOT`, `OP_ORDER`, `ITEMSIZE`, `RUNS` re-exported from `data`.

- [ ] **Step 1:** Add to `pyproject.toml`: `"jupyterlab>=4.4"` in `dependencies`; and
```toml
[dependency-groups]
dev = ["pytest>=8", "nbclient>=0.10", "nbformat>=5.10", "vl-convert-python>=1.7"]
```
Run `uv sync` — expect success.

- [ ] **Step 2: Failing tests** in `tests/test_data.py`:
```python
import polars as pl
import pytest
from analysis import data

@pytest.fixture(scope="session")
def d():
    return data.load_analysis()

def test_runs_loaded(d):
    assert set(data.RUNS) <= set(d.runs)

def test_pure_cells(d):
    assert set(d.pure["implementation"].unique()) == {"py-list-float", "py-list-f32", "py-obj-array"}
    assert d.pure["samples"].min() == 30
    assert d.pure.filter(pl.col("n") == 1_000_000)["n_elem"].unique().to_list() == [4_000_000]

def test_pure_memory(d):
    assert 47 < d.pure_memory["py-list-float"] < 49
    assert set(d.pure_memory) == {"py-list-float", "py-list-f32", "py-obj-array"}
```
Run `uv run pytest tests/test_data.py -q` — expect ImportError.

- [ ] **Step 3: Implement** `analysis/data.py`: move `ITEMSIZE, CANONICAL_DTYPE, UNARY, OP_ORDER, RUNS, KEYS, MIB, OUT_COLD, OUT_HOT, AnalysisData, _load_run, _cells, load_analysis, allocation_penalty, gpu_ceiling_rows, jax_timer_series` verbatim from `support.py`. Add `pure`, `pure_memory` to `AnalysisData` and:
```python
PURE_KEYS = ["n", "n_elem", "operation", "implementation", "cache_state"]

def _load_pure(root: Path) -> tuple[pl.DataFrame, dict[str, float]]:
    """Load the pure-Python run, which has no byte accounting."""
    run_dir = root / "cpu_pure"
    frame = pl.read_parquet(run_dir / "cpu_pure.parquet")
    metadata = json.loads((run_dir / "cpu_pure.json").read_text())
    counts = {int(s): b["n_elem"] for s, b in metadata["per_size"].items()}
    cells = (
        frame.with_columns(pl.col("n").replace_strict(counts).alias("n_elem"))
        .with_columns((pl.col("time_us") * 1e3 / pl.col("n_elem")).alias("ns_elem"))
        .group_by(PURE_KEYS)
        .agg(pl.col("time_us").median().alias("us"),
             pl.col("time_us").quantile(0.25).alias("us_lo"),
             pl.col("time_us").quantile(0.75).alias("us_hi"),
             pl.col("ns_elem").median(),
             pl.col("time_us").is_not_null().sum().alias("samples"))
        .sort(PURE_KEYS)
    )
    largest = metadata["per_size"][str(max(metadata["sizes"]))]["memory"]
    memory = {impl: m["operand_sizeof_B_per_elem"] for impl, m in largest.items()}
    return cells, memory
```
- [ ] **Step 4:** `uv run pytest tests/test_data.py -q` — PASS.
- [ ] **Step 5:** Commit `pyproject.toml uv.lock analysis/data.py tests/test_data.py` — "Add data module with pure-Python run loading".

### Task 2: `style.py`

**Files:** Create `analysis/style.py`; test in `tests/test_figures.py`.

**Interfaces:**
- Produces: `P` (light palette dict, same keys as today), `FONT`, `Figure(NamedTuple: chart, table: pl.DataFrame | None, label: str)`, `show(*figs: Figure) -> None`, `configure_notebook() -> None`, `stat_tiles(tiles: list[tuple[str, str, str]]) -> None`.

- [ ] **Step 1: Failing test** `tests/test_figures.py`:
```python
import altair as alt
import polars as pl
from analysis import style

def test_theme_registered_and_enabled():
    assert alt.theme.active == "quadbench"

def test_show_accepts_figure(capsys):
    fig = style.Figure(alt.Chart(pl.DataFrame({"x": [1]})).mark_point().encode(x="x:Q"), pl.DataFrame({"x": [1]}), "t")
    style.show(fig)
```
Run — expect ImportError.
- [ ] **Step 2: Implement:** move `FONT`, `PALETTE`, `P`, the theme function (axis/legend label sizes 11→12.5, title 15→17, subtitle 11.5→12.5, header label 11.5→13, text 11→12) and `configure_notebook`, `_table_view` from `support.py`. `show(*figs)` displays each chart then its `<details>` table view. `stat_tiles` takes `(label, value, note)` tuples and renders the existing tile strip HTML.
- [ ] **Step 3:** Tests PASS. Commit "Add style module".

### Task 3: `figures.py`

**Files:** Create `analysis/figures.py`; extend `tests/test_figures.py`.

**Interfaces:**
- Consumes: `AnalysisData`, `style.Figure`, `style.P`, `data.*` constants.
- Produces these functions, each `(d: AnalysisData) -> Figure` (source = current notebook cell it is ported from):

| function | lesson | source |
|---|---|---|
| `representation_ladder` | 1 | new |
| `bytes_per_element` | 1 | new |
| `boxed_vs_unboxed_by_size` | 1 | new |
| `gpu_call_floor` | 2 | cell 23 |
| `cpu_vs_gpu` | 2 | cell 45 |
| `cache_curve` | 3 | cell 6 |
| `cpu_bandwidth` | 4 | cell 20 |
| `gpu_two_ceilings` | 4 | cell 25 |
| `float16_curve` | 5 | cell 8 |
| `float16_mechanism` | 5 | cell 10 |
| `dtype_grid` | 5 | cell 13 |
| `gpu_precision_ratios` | 5 | cell 27 |
| `destination_decomposition` | 6 | cell 16 |
| `allocation_cliff` | 6 | cell 18 |
| `jax_vs_cupy` | 7 | cell 30 |
| `jit_fusion` | 7 | cell 33 |
| `jax_vs_numpy_cpu` | 7 | cell 39 |
| `timing_protocol` | 8 | cell 36 |

- [ ] **Step 1: Failing test** (parametrized over every name above):
```python
import pytest
from analysis import data, figures, style

NAMES = ["representation_ladder", "bytes_per_element", "boxed_vs_unboxed_by_size",
         "gpu_call_floor", "cpu_vs_gpu", "cache_curve", "cpu_bandwidth",
         "gpu_two_ceilings", "float16_curve", "float16_mechanism", "dtype_grid",
         "gpu_precision_ratios", "destination_decomposition", "allocation_cliff",
         "jax_vs_cupy", "jit_fusion", "jax_vs_numpy_cpu", "timing_protocol"]

@pytest.fixture(scope="module")
def d():
    return data.load_analysis()

@pytest.mark.parametrize("name", NAMES)
def test_figure_builds_and_validates(d, name):
    fig = getattr(figures, name)(d)
    assert isinstance(fig, style.Figure)
    fig.chart.to_dict()  # raises on an invalid Vega-Lite spec
    assert fig.table is None or fig.table.height > 0

def test_ladder_size_insensitivity_of_cpu_sweep_rows(d):
    # quad/longdouble come from 256k elements; check 16k vs 256k agree within 15%
    import polars as pl
    s = d.runs["cpu_sweep"].filter(data.OUT_HOT & (pl.col("operation") == "add")
        & pl.col("implementation").is_in(["quad-sleef", "longdouble64"]) & pl.col("n").is_in([4000, 64000]))
    wide = s.pivot("n", index="implementation", values="ns_elem")
    assert ((wide["64000"] / wide["4000"]) - 1).abs().max() < 0.15
```
Run — FAIL (ImportError).
- [ ] **Step 2: Port** each existing cell's chart code into its function verbatim, replacing notebook globals with `d.runs[...]`, `d.meta[...]`, and fixing these literals:
  - `gpu_call_floor`: band `x2` = largest `n_elem` whose `us` is within 15% of the minimum; band label text uses the computed min/max µs.
  - `cpu_vs_gpu`: crossover = log-linear interpolation of the first size where GPU `ns_elem` < CPU `ns_elem`; label from it.
  - `gpu_two_ceilings`: note text uses computed element rate (`limits["elem"]` mean).
  - `cpu_bandwidth`: unchanged logic (already computed).
  - Titles become claims in plain words; subtitles keep the run/condition line.
- [ ] **Step 3: New lesson-1 figures.**
  - `representation_ladder`: rows built as: `cpu_pure` (`add`, cold, n=1,000,000) for three Python forms labeled `list[float]`, `list[np.float32]`, `NDArray[object]`; `cpu_dense` (`add`, cold, `alloc`, n=1,000,000) for `float32`, `float64`; `cpu_f16` (`add`, cold, `alloc`, n=1,000,000, implementation `float16`); `cpu_sweep` (`add`, cold, `alloc`, n=64,000) for `quad-sleef`, `longdouble64` labeled `quad (software)`/`longdouble (x87)` with a `†` and subtitle note "† measured at 256k elements"; `gpu_dense80` (`add`, cold, `alloc`, n=1,000,000, `GPU fp32`) labeled `CuPy float32 (A100)`. Group column: `Python objects` / `NumPy array` / `GPU array`. Horizontal dot plot, log x `ns per element`, y sorted by value descending, text label with value right of each dot, color by group (`P["series"]` order: orange for Python objects, blue NumPy, green GPU). Table: label, group, ns/element, "× float64".
  - `bytes_per_element`: bars from `d.pure_memory` (3 rows) plus `float64`=8, `float32`=4, `float16`=2; same group colors; x "bytes per element"; value labels.
  - `boxed_vs_unboxed_by_size`: lines of `ns_elem` vs `n_elem` (log-log) for `add`, cold: three `cpu_pure` forms and `cpu_dense` `alloc` `float64`/`float32`; color by group, strokeDash by series within group; end labels.
- [ ] **Step 4:** `uv run pytest tests/test_figures.py -q` — PASS.
- [ ] **Step 5:** Commit "Add figures module".

### Task 4: `findings.py`

**Files:** Create `analysis/findings.py`, `tests/test_findings.py`.

**Interfaces:**
- Produces: `values(d) -> dict[str, float]` (raw), `compute(d) -> dict[str, str]` (formatted, house style), `jax_quality(d) -> pl.DataFrame` (the cell-42 table, unchanged columns), helpers `fmt_ratio(x) -> "26.4×"`, `fmt_int(x) -> "12,800"`.
- Keys (minimum set; add any further number the prose quotes): `pure_add_ns_min`, `pure_add_ns_max` (list[float] across sizes), `pure_vs_f64_min`, `pure_vs_f64_max` (list[float] / float64 alloc, matched sizes), `f32_list_vs_float_list`, `pure_bytes_float`, `pure_size_ratio_f64`, `pure_size_ratio_f32`, `gpu_floor_us_lo`, `gpu_floor_us_hi`, `gpu_floor_span`, `jax_floor_us`, `crossover_elems`, `gpu_advantage_max`, `l2_min_ns`, `l2_min_elems`, `hotcold_peak`, `hotcold_peak_elems`, `exp_ns`, `cpu_plateau_gbs`, `four_core_gbs`, `four_core_scale`, `gpu_elem_rate`, `hbm_plateau_80`, `hbm_plateau_40`, `crossover_bytes_40`, `crossover_bytes_80`, `f16_ns`, `f16_vs_f32_min`, `f16_vs_f32_max`, `f16_conv_ns`, `quad_vs_f64_min`, `quad_vs_f64_max`, `ld_vs_f64_min`, `ld_vs_f64_max`, `gpu_f32_f16_binary`, `gpu_f32_f16_unary`, `alloc_median_below`, `alloc_median_above`, `alloc_min_above`, `page_fault_ns`, `jit_muladd_min/max`, `jit_chain_min/max`, `jit_chain_fp32`, `jit_matmul_min/max`, `numpy_jax_cpu_min/max`, `numpy_jax_cpu_matmul`, `blocking_ratio_add`, `cupy_jax_add`. Every key in `values` is also in `compute` and must be quoted somewhere in the notebook prose (Task 6 enforces this), so keep the set to numbers the prose actually uses.

- [ ] **Step 1: Failing tests** `tests/test_findings.py` — port every numeric assertion from the current `tests/test_analysis_notebook.py` (lines 61–164) to call `findings`/`data` directly, e.g.:
```python
def test_jax_quality(d):
    q = findings.jax_quality(d)
    assert q["measured cells"].sum() == 2_474
    assert q["timing samples"].sum() == 74_220
    assert q["validations"].sum() == 1_237
    assert q["failed validations"].sum() == 0
    assert q["timed-loop recompiles"].sum() == 0
    assert 2.44 < q["maximum ULP error"].max() < 2.45
    assert 2.07 < q["maximum JAX ULP error"].max() < 2.08

def test_framework_ratios(d):
    v = findings.values(d)
    assert 0.98 < v["cupy_jax_add"] < 1.00
    assert 5.43 < v["jit_chain_fp32"] < 5.44
    assert 1.23 < v["blocking_ratio_add"] < 1.24
    assert 26.39 < v["numpy_jax_cpu_matmul"] < 26.40

def test_pure_headline(d):
    v = findings.values(d)
    assert v["pure_vs_f64_min"] > 10
    assert 47 < v["pure_bytes_float"] < 49
    assert v["f32_list_vs_float_list"] > 1  # list of np.float32 is slower

def test_formatting():
    assert findings.fmt_int(12800) == "12,800"
    assert findings.fmt_ratio(26.396) == "26.4×"

def test_every_key_formatted(d):
    assert set(findings.compute(d)) == set(findings.values(d))
```
plus the JAX metadata/dtype assertions (current test lines 64–101) against `d.raw`/`d.meta`. Run — FAIL.
- [ ] **Step 2: Implement** each value from the existing notebook logic (cells 6–45) and the new pure comparisons; `compute` formats per key with a small `FORMATS: dict[str, Callable]` table (ratios `.1f×` or `.2f×` under 10, ns `.2f`, ints with commas, GB/s `.1f`, µs `.1f`).
- [ ] **Step 3:** PASS. Commit "Add findings module".

### Task 5: Notebook

**Files:** Rewrite `analysis.ipynb`; delete `analysis/support.py`. Scratch: `$SCRATCH/build_notebook.py`.

- [ ] **Step 1:** Print `findings.compute(d)` and write the notebook prose using those exact strings.
- [ ] **Step 2:** Write `build_notebook.py` that emits cells with `nbformat.v4` in this order: title + "Eight things to remember" (numbered, one line each, lesson 1 first) + "How to read the charts" + `stat_tiles` cell; setup cell (`from analysis import data, figures, findings, style` / `style.configure_notebook()` / `d = data.load_analysis()`); lessons 1–8 per the spec table, each with the Global-Constraints shape; lesson 1 includes the list-vs-ndarray memory-layout explanation as a small inline HTML diagram in Markdown; Close (recap + limitations from current cell 46, minus changelog lines); Appendix A (run inventory table incl. `cpu_pure`, metric definitions from cell 2, `findings.jax_quality(d)` cell + its prose from cell 43). All code cells get `metadata["jupyter"] = {"source_hidden": True}`. Kernel spec `python3`.
- [ ] **Step 3:** Run builder, then `uv run jupyter nbconvert --to notebook --execute --inplace analysis.ipynb`. Expect success.
- [ ] **Step 4:** `git rm analysis/support.py`; `rg support` finds no references. Commit "Rebuild analysis notebook around eight lessons".

### Task 6: Notebook tests

**Files:** Rewrite `tests/test_analysis_notebook.py`.

- [ ] **Step 1:** Write:
```python
"""Structural, prose-consistency and execution checks for analysis.ipynb."""
import ast, re
from pathlib import Path
import nbformat, pytest
from nbclient import NotebookClient
from analysis import data, findings

ROOT = Path(__file__).resolve().parents[1]
NB = nbformat.read(ROOT / "analysis.ipynb", as_version=4)
CODE = [c for c in NB.cells if c.cell_type == "code"]
PROSE = "\n".join(c.source for c in NB.cells if c.cell_type == "markdown")

def test_code_cells_are_short_calls():
    for c in CODE:
        tree = ast.parse(c.source)
        assert not [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.Dict) and len(n.keys) >= 8]
        assert len([l for l in c.source.splitlines() if l.strip()]) <= 3, c.source
        assert c.metadata.get("jupyter", {}).get("source_hidden") is True

def test_prose_quotes_every_finding():
    missing = {k: v for k, v in findings.compute(data.load_analysis()).items() if v not in PROSE}
    assert not missing, missing

def test_no_changelog_language():
    assert not re.search(r"earlier (version|reading)|supersedes|previous version", PROSE, re.I)

def test_every_figure_cell_has_saved_output():
    for c in CODE:
        if "figures." in c.source:
            assert any("application/vnd.vegalite" in k for o in c.outputs for k in o.get("data", {})), c.source

def test_lessons_have_required_shape():
    for n in range(1, 9):
        assert re.search(rf"^## {n} · ", PROSE, re.M), n
    assert PROSE.count("**What this means in practice**") == 8
    assert PROSE.count("<summary>Details</summary>") >= 8

@pytest.mark.slow
def test_notebook_executes():
    nb = nbformat.read(ROOT / "analysis.ipynb", as_version=4)
    NotebookClient(nb, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
```
- [ ] **Step 2:** `uv run pytest -q` — all PASS; fix prose or findings until they agree (never weaken the test).
- [ ] **Step 3:** Commit "Rewrite notebook tests around findings and structure".

### Task 7: Visual QA

- [ ] **Step 1:** Scratch `render_pngs.py`: for each name in `NAMES`, `getattr(figures, name)(d).chart.save(f"$SCRATCH/png/{name}.png", scale_factor=2)`.
- [ ] **Step 2:** Read every PNG. Check: no clipped/overlapping labels, legend readable, annotations land where intended, colors consistent across lessons (orange = Python objects, blue = NumPy, green = GPU in lesson 1), titles are claims. Fix in `figures.py`, re-run tests, re-execute notebook.
- [ ] **Step 3:** Commit "Polish figures after visual review".

### Task 8: Final verification

- [ ] `uv run pytest -q` — all pass (paste output).
- [ ] `uv run --with ruff ruff check analysis tests` — clean.
- [ ] `git status` clean; summarize to user with timing estimate per section.
