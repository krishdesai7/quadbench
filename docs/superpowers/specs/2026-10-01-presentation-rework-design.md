# Presentation rework of `analysis.ipynb` — design

## Purpose and audience

`analysis.ipynb` becomes a report presented live, from the `.ipynb` itself, in a
60-minute slot with ~40 minutes of material and the rest left for questions.

- **Primary audience:** colleagues who work almost exclusively in Jupyter, are
  nominally Python-proficient, know little about computer architecture, and
  default to `list[float]` and pickle. Their platform is slow partly because they
  have no intuition for where time goes. The notebook must give them that
  intuition and concrete habits.
- **Secondary audience:** a deeply technical manager (ex-Meta SWE). Mechanism
  and caveats must remain available, but not in the way of the main narrative.

Success: someone from the primary audience can open the notebook cold, read it
top to bottom without running anything, and leave with the eight lessons and
what to do about each; the expert can expand any "Details" block and find the
mechanism and the evidence.

## Content decisions

- Remove all changelog-style material ("an earlier version of this notebook
  claimed…", superseded readings). State current findings only.
- Organise by **lesson**, not by experiment run. Each lesson is a claim the
  data supports.
- New lesson 1 from `data/cpu_pure` (not used by the current notebook): boxed
  Python floats vs unboxed arrays. This is the headline finding.
- Practical guidance stays general (batch small calls, count bytes, reuse
  large buffers, compile composite expressions, benchmark at your real size).
  Parallels to request/agent overhead may be noted, never overclaimed: the repo
  measures numeric kernels.
- Hardware: AMD EPYC 7763 (Perlmutter CPU node; all CPU runs, including
  `cpu_pure`, are from the same node type), NVIDIA A100-40GB and A100-80GB.
- Consistent house style: US spelling, thousands separators as commas
  (`12,800`), units spaced (`9 µs`, `27.7 GB/s`), dtype and op names in code
  font.

## Notebook outline (~41 min)

Every lesson section has the same shape:

1. `## N · <headline claim>` — the claim itself is the heading.
2. 2–4 sentences in plain language; jargon (cache, bandwidth, boxing,
   allocation, kernel, JIT) defined at first use.
3. One or two figures, each a one-line code cell.
4. A **What this means in practice** callout (blockquote, 1–3 bullets).
5. A collapsed `<details><summary>Details</summary>…</details>` block with the
   mechanism, exact numbers, and caveats.

Numbers in the headlines below are from a first pass over the data; the final
wording uses whatever `findings.py` computes.

| # | Section | Source runs | Figures | ~min |
|---|---|---|---|---|
| 0 | Opening: title, "Eight things to remember" summary, how to read the charts (ns/element, log axes, medians + IQR), hardware/measurement tiles | all | stat tiles | 3 |
| 1 | **Containers matter more than types.** A `list[float]` is 25–300× slower and ~6× larger than a `float64` array | `cpu_pure`, `cpu_dense`, `cpu_f16`, `cpu_sweep`, `gpu_dense80` | (a) the ladder; (b) bytes per element; (c) boxed vs unboxed cost across size | 7 |
| 2 | **Every call has a fixed cost.** ~9 µs per CuPy call, ~60 µs per JAX call; below ~20k elements one CPU core beats an A100 | `gpu_dense80`, `gpu_host80`, `jax_dense`, `cpu_dense` | GPU call floor; CPU vs GPU crossover | 5 |
| 3 | **Where data lives matters more than the arithmetic.** Cache curve; `exp` has none | `cpu_dense` | hot/cold cache curve | 4 |
| 4 | **Memory bandwidth is the speed limit.** 27.7 GB/s one core, 39.5 GB/s four; GPU kernels sit on one of two ceilings | `cpu_dram50`, `cpu_par0…3`, `gpu_sweep`, `gpu_sweep80` | CPU bandwidth bars; GPU two-ceiling chart | 6 |
| 5 | **Smaller numbers aren't automatically faster.** CPU `float16` 6–44× slower than `float32`; GPU halving precision buys ≤1.25× on elementwise ops; quad/longdouble far off-scale | `cpu_dense`, `cpu_f16`, `cpu_sweep`, `gpu_sweep80` | f16 curve; f16 mechanism; dtype grid; GPU precision ratios | 6 |
| 6 | **Allocation is free until it isn't.** glibc's 32 MiB mmap threshold | `cpu_alloc`, `cpu_dense`, `cpu_dram50` | destination decomposition; allocation cliff | 4 |
| 7 | **Let the compiler see the whole expression.** JIT fusion 1.8–15.8×; JAX vs NumPy on CPU 1.3–26× | `jax_sweep`, `jax_cpu`, `jax_dense`, `gpu_dense80` | fusion ratios; JAX vs NumPy CPU; JAX vs CuPy | 4 |
| 8 | **Measure the thing you mean.** Timing protocol and problem size change the answer | `jax_dense`, `jax_latency`, `gpu_host80`, `cpu_dense` | blocking vs pipelined | 2 |
| — | Close: recap (the eight lessons, one line each) and limitations | | | — |
| A | Appendix: run inventory, metric definitions, data quality and JAX validation table | | inventory table; quality table | — |

### Lesson 1 figures

- **(a) The ladder.** Horizontal dot plot, log x-axis, ns/element for `add` with
  cold operands, at `n = 1,000,000` (4M elements) wherever measured. Rows,
  slowest first: `list[np.float32]`, `list[float]`, `NDArray[object]`,
  `quad-sleef`, `longdouble`, `float16`, `float64`, `float32`, and CuPy fp32 on
  A100-80GB. Boxed representations are colored as one group, NumPy primitive
  dtypes as another, the GPU as a third. `quad-sleef` and `longdouble` come from
  `cpu_sweep` at its largest size (256k elements); they are marked as such and
  are used only after confirming their per-element cost is size-insensitive
  between `cpu_sweep` sizes. NumPy rows use the allocating form so all rows
  include allocation, as the Python containers necessarily do.
- **(b) Bytes per element.** Bars: Python containers from `cpu_pure` memory
  metadata (deep `sizeof`), arrays from itemsize.
- **(c) Size curve.** ns/element vs elements for the three boxed forms and NumPy
  `float64`/`float32`, showing the gap persists at every size.
- Prose includes a small explanation (text, optionally a simple inline HTML/SVG
  sketch) of "list = array of pointers to 24-byte heap objects; ndarray = one
  contiguous block of raw values".

## Code layout

`analysis/support.py` is replaced by:

| Module | Responsibility |
|---|---|
| `analysis/data.py` | Run loading and derived metrics (current `_load_run`, `_cells`, `load_analysis`, `AnalysisData`), plus `cpu_pure` and its memory metadata. Behavior of existing derivations unchanged. |
| `analysis/style.py` | Palette, Altair theme (fonts bumped ~1–2 px for projection), stat tiles, `show(figure)` which displays a chart and its collapsible table view. |
| `analysis/figures.py` | One public function per figure: `fn(data) -> Figure`, where `Figure` is a small named tuple of `(chart, table, table_label)`. All Altair code lives here. Annotation positions (crossover, call-floor band, element-rate ceiling, plateau lines) are computed from data, not literals. |
| `analysis/findings.py` | `compute(data) -> dict[str, str]`: every number quoted in the notebook's Markdown, computed from the data and formatted in house style. Also exposes the underlying raw values for tests. |
| `analysis/__init__.py` | Re-exports nothing heavy; notebook imports modules explicitly. |

Notebook code cells: an import/setup cell, a load cell, then one line per
figure (`show(figures.ladder(data))`). No function/class definitions or large
literals in the notebook (existing structural rule kept). Code cells carry
`jupyter.source_hidden: true` metadata so JupyterLab collapses them. Outputs are
saved in the committed `.ipynb`.

## Testing

`tests/test_analysis_notebook.py` (inline-script uv deps, as today) is rewritten:

1. **Structure:** no definitions or ≥8-key dict literals in code cells (kept);
   every code cell has at most a few lines.
2. **Findings:** the existing numeric assertions (JAX quality totals, ULP
   maxima, CuPy/JAX endpoint ratio, fp32 chain JIT speedup, blocking/pipelined
   ratio, NumPy/JAX matmul ratio, JAX metadata/dtype checks) are preserved,
   re-targeted at `data.py`/`findings.py` rather than notebook globals. New
   assertions cover the `cpu_pure` numbers.
3. **Prose consistency:** every formatted value in `findings.compute(data)`
   appears verbatim in the notebook's Markdown, and the notebook quotes no
   finding key that is absent.
4. **Execution:** the notebook executes end-to-end via nbclient.

Visual QA: every figure is rendered to PNG with `vl-convert-python` (ad hoc
`uv run --with`, not a project dependency) and inspected.

## Out of scope

Benchmark harnesses, job scripts, data files, and `pyproject.toml` dependencies
are unchanged. No HTML/PDF export pipeline.
