"""Every chart in the report, one function per figure.

Each function takes the loaded :class:`~analysis.data.AnalysisData` and returns
a :class:`~analysis.style.Figure`: the Altair chart plus a table carrying the
same numbers.  Annotation positions (thresholds, crossovers, ceilings) are
computed from the data, never typed in.
"""

from __future__ import annotations

import math

import altair as alt
import polars as pl

from analysis.data import (
    ALLOC_COLD,
    OP_ORDER,
    OUT_COLD,
    OUT_HOT,
    AnalysisData,
    allocation_penalty,
    gpu_ceiling_rows,
    jax_timer_series,
)
from analysis.style import P, Figure


ADD = pl.col("operation") == "add"
LADDER_N = 1_000_000

# Lesson 1 uses one color per kind of container, kept identical across its charts.
KINDS = ["Python objects", "NumPy array", "GPU array"]
KIND_SCALE = alt.Scale(domain=KINDS, range=[P["series"][1], P["series"][0], P["series"][2]])
PYTHON_LABELS = {
    "py-list-f32": "list[np.float32]",
    "py-list-float": "list[float]",
    "py-obj-array": "NDArray[object]",
}


def _note(x: float, text: str, y_px: float, *, align: str = "left", dx: float = 7) -> alt.Chart:
    """Return a small muted text annotation anchored at a data x and pixel y."""
    return alt.Chart(pl.DataFrame({"x": [x], "t": [text]})).mark_text(
        align=align, dx=dx, fontSize=12, color=P["ink2"]
    ).encode(x="x:Q", y=alt.value(y_px), text="t:N")


def _vrule(x: float, *, dash: list[int] | None = None) -> alt.Chart:
    """Return a vertical reference rule at a data x."""
    return alt.Chart(pl.DataFrame({"x": [x]})).mark_rule(
        color=P["ink2"], strokeWidth=1, strokeDash=dash or []
    ).encode(x="x:Q")


def _ns_label(ns: float) -> str:
    """Format a per-element cost with precision suited to its magnitude."""
    if ns >= 10:
        return f"{ns:,.0f} ns"
    if ns >= 0.1:
        return f"{ns:.2f} ns"
    return f"{ns:.3f} ns"


# ---------------------------------------------------------------------------
# Lesson 1 · containers
# ---------------------------------------------------------------------------


def _ladder_rows(d: AnalysisData) -> pl.DataFrame:
    """Per-element add cost for every way the repo stored the numbers."""
    python = (
        d.pure.filter(ADD & (pl.col("cache_state") == "cold") & (pl.col("n") == LADDER_N))
        .select(
            pl.col("implementation").replace_strict(PYTHON_LABELS).alias("representation"),
            pl.lit("Python objects").alias("kind"),
            "ns_elem",
            "n_elem",
        )
    )
    dense = d.runs["cpu_dense"].filter(ADD & ALLOC_COLD & pl.col("us").is_not_null())
    numpy_rows = dense.filter(
        (pl.col("n") == LADDER_N) & pl.col("implementation").is_in(["float16", "float32", "float64"])
    ).select(
        pl.format("NDArray[np.{}]", pl.col("implementation")).alias("representation"),
        pl.lit("NumPy array").alias("kind"),
        "ns_elem",
        "n_elem",
    )
    # Quad was skipped at 4M elements by the harness's time cap; its cost is
    # flat across sizes, so the largest measured size stands in.
    quad = (
        dense.filter(pl.col("implementation") == "quad-sleef")
        .sort("n")
        .tail(1)
        .select(
            pl.lit("NDArray[quad] †").alias("representation"),
            pl.lit("NumPy array").alias("kind"),
            "ns_elem",
            "n_elem",
        )
    )
    longdouble = (
        d.runs["cpu_sweep"]
        .filter(ADD & ALLOC_COLD & (pl.col("implementation") == "longdouble64"))
        .sort("n")
        .tail(1)
        .select(
            pl.lit("NDArray[np.longdouble] †").alias("representation"),
            pl.lit("NumPy array").alias("kind"),
            "ns_elem",
            "n_elem",
        )
    )
    gpu = d.runs["gpu_dense80"].filter(
        ADD & ALLOC_COLD & (pl.col("n") == LADDER_N) & (pl.col("implementation") == "GPU fp32")
    ).select(
        pl.lit("CuPy float32 (A100)").alias("representation"),
        pl.lit("GPU array").alias("kind"),
        "ns_elem",
        "n_elem",
    )
    rows = pl.concat([python, numpy_rows, quad, longdouble, gpu])
    float64 = rows.filter(pl.col("representation") == "NDArray[np.float64]")["ns_elem"][0]
    return rows.with_columns(
        (pl.col("ns_elem") / float64).alias("vs_f64"),
        pl.col("ns_elem").map_elements(_ns_label, return_dtype=pl.String).alias("label"),
    ).sort("ns_elem", descending=True)


def representation_ladder(d: AnalysisData) -> Figure:
    """Per-element cost of add for every container and dtype, on one axis."""
    rows = _ladder_rows(d)
    order = rows["representation"].to_list()
    x = alt.X(
        "ns_elem:Q",
        scale=alt.Scale(type="log", domain=[0.004, 400], nice=False),
        title="nanoseconds per element (log scale)",
        axis=alt.Axis(values=[0.01, 0.1, 1, 10, 100], format="~g"),
    )
    y = alt.Y("representation:N", sort=order, title=None, axis=alt.Axis(labelFontSize=13, labelLimit=260))
    color = alt.Color("kind:N", scale=KIND_SCALE, sort=KINDS, legend=alt.Legend(title=None))
    dots = alt.Chart(rows).mark_point(size=170).encode(
        x=x,
        y=y,
        color=color,
        tooltip=[
            alt.Tooltip("representation:N"),
            alt.Tooltip("ns_elem:Q", title="ns / element", format=".3f"),
            alt.Tooltip("vs_f64:Q", title="× float64 array", format=",.1f"),
            alt.Tooltip("n_elem:Q", title="elements", format=","),
        ],
    )
    labels = alt.Chart(rows).mark_text(align="left", dx=13, fontSize=12.5).encode(x=x, y=y, text="label:N")
    chart = (dots + labels).properties(
        width=560,
        height=340,
        title=alt.Title(
            "A list of Python floats is the slowest way to hold numbers",
            subtitle=[
                "add · 4 million elements · cold operands · result allocated on every call",
                "† flat-cost dtype shown at the largest size the run measured",
            ],
        ),
    )
    table = rows.select(
        "representation",
        "kind",
        pl.col("ns_elem").alias("ns per element"),
        pl.col("vs_f64").alias("× float64 array"),
        pl.col("n_elem").alias("elements"),
    )
    return Figure(chart, table, "Table view · cost per element by representation")


def bytes_per_element(d: AnalysisData) -> Figure:
    """Memory per stored number: boxed Python objects against raw arrays."""
    rows = pl.concat(
        [
            pl.DataFrame(
                {
                    "representation": [PYTHON_LABELS[k] for k in PYTHON_LABELS],
                    "kind": ["Python objects"] * 3,
                    "bytes": [d.pure_memory[k] for k in PYTHON_LABELS],
                }
            ),
            pl.DataFrame(
                {
                    "representation": ["NDArray[np.float64]", "NDArray[np.float32]", "NDArray[np.float16]"],
                    "kind": ["NumPy array"] * 3,
                    "bytes": [8.0, 4.0, 2.0],
                }
            ),
        ]
    ).sort("bytes", descending=True)
    rows = rows.with_columns(pl.format("{} B", pl.col("bytes").round(0).cast(pl.Int64)).alias("label"))
    y = alt.Y("representation:N", sort=rows["representation"].to_list(), title=None,
              axis=alt.Axis(labelFontSize=13, labelLimit=260))
    x = alt.X("bytes:Q", title="bytes per stored number", scale=alt.Scale(domain=[0, 60]))
    bars = alt.Chart(rows).mark_bar(size=18).encode(
        x=x,
        y=y,
        color=alt.Color("kind:N", scale=KIND_SCALE, sort=KINDS, legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("representation:N"), alt.Tooltip("bytes:Q", format=".1f")],
    )
    labels = alt.Chart(rows).mark_text(align="left", dx=6, fontSize=12.5).encode(x=x, y=y, text="label:N")
    chart = (bars + labels).properties(
        width=520,
        height=230,
        title=alt.Title(
            "Each Python float costs about six times the memory of a float64",
            subtitle="measured with sys.getsizeof over the container and every element it holds",
        ),
    )
    return Figure(chart, rows.drop("label"), "Table view · bytes per element")


def boxed_vs_unboxed_by_size(d: AnalysisData) -> Figure:
    """Per-element add cost across problem size, boxed against unboxed."""
    python = d.pure.filter(ADD & (pl.col("cache_state") == "cold")).select(
        pl.col("implementation").replace_strict(PYTHON_LABELS).alias("representation"),
        pl.lit("Python objects").alias("kind"),
        "n_elem",
        "ns_elem",
    )
    arrays = d.runs["cpu_dense"].filter(
        ADD & ALLOC_COLD & pl.col("implementation").is_in(["float32", "float64"])
    ).select(
        pl.format("NDArray[np.{}]", pl.col("implementation")).alias("representation"),
        pl.lit("NumPy array").alias("kind"),
        "n_elem",
        "ns_elem",
    )
    rows = pl.concat([python, arrays]).sort("representation", "n_elem")
    series = ["list[np.float32]", "list[float]", "NDArray[object]", "NDArray[np.float64]", "NDArray[np.float32]"]
    x = alt.X("n_elem:Q", scale=alt.Scale(type="log", nice=False), title="elements (log scale)",
              axis=alt.Axis(format="~s", values=[1_000, 10_000, 100_000, 1_000_000]))
    y = alt.Y("ns_elem:Q", scale=alt.Scale(type="log", domain=[0.1, 300], nice=False),
              title="nanoseconds per element (log scale)", axis=alt.Axis(values=[0.1, 1, 10, 100], format="~g"))
    color = alt.Color("kind:N", scale=KIND_SCALE, sort=KINDS, legend=alt.Legend(title=None))
    dash = alt.StrokeDash("representation:N", sort=series, legend=None,
                          scale=alt.Scale(domain=series, range=[[6, 3], [1, 0], [2, 2], [1, 0], [6, 3]]))
    lines = alt.Chart(rows).mark_line(point=alt.OverlayMarkDef(size=40)).encode(
        x=x, y=y, color=color, strokeDash=dash,
        tooltip=[alt.Tooltip("representation:N"), alt.Tooltip("n_elem:Q", title="elements", format=","),
                 alt.Tooltip("ns_elem:Q", title="ns / element", format=".3f")],
    )
    ends = rows.group_by("representation").agg(pl.all().sort_by("n_elem").last())
    labels = alt.Chart(ends).mark_text(align="left", dx=8, fontSize=12).encode(
        x=x, y=y, text="representation:N"
    )
    chart = (lines + labels).properties(
        width=520,
        height=320,
        title=alt.Title(
            "The gap holds at every size, and widens once data outgrows cache",
            subtitle="add · cold operands · result allocated on every call",
        ),
    ).configure_view(clip=False)
    table = rows.pivot("representation", index="n_elem", values="ns_elem").sort("n_elem")
    return Figure(chart, table, "Table view · ns per element by size")


# ---------------------------------------------------------------------------
# Lesson 2 · fixed per-call cost
# ---------------------------------------------------------------------------

GPU_ORDER = ["GPU fp64", "GPU fp32", "GPU fp16"]


def gpu_floor_extent(d: AnalysisData, tolerance: float = 0.15) -> tuple[int, float, float]:
    """Return (largest flat size, min µs, max µs) of the GPU per-call floor.

    The floor extends to the largest size at which every precision is still
    within ``tolerance`` of its own minimum call time.
    """
    rows = d.runs["gpu_dense80"].filter(ADD & OUT_COLD)
    edges = []
    for _, group in rows.group_by("implementation"):
        group = group.sort("n_elem")
        above = group["us"] > group["us"].min() * (1 + tolerance)
        # The floor is the contiguous run from the smallest size up to the
        # first size that leaves it.
        edges.append(group["n_elem"][above.arg_true()[0] - 1] if above.any() else group["n_elem"].max())
    edge = min(edges)
    within = rows.filter(pl.col("n_elem") <= edge)
    return edge, within["us"].min(), within["us"].max()


def gpu_call_floor(d: AnalysisData) -> Figure:
    """Microseconds per CuPy call against size: flat until real work appears."""
    rows = d.runs["gpu_dense80"].filter(ADD & OUT_COLD).select("implementation", "n_elem", "us")
    edge, lo, hi = gpu_floor_extent(d)
    x = alt.X("n_elem:Q", scale=alt.Scale(type="log", nice=False), title="elements (log scale)",
              axis=alt.Axis(format="~s", values=[1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]))
    band = alt.Chart(pl.DataFrame({"x1": [rows["n_elem"].min() * 0.9], "x2": [float(edge)]})).mark_rect(
        color=P["grid"], opacity=0.6
    ).encode(x="x1:Q", x2="x2:Q")
    lines = alt.Chart(rows).mark_line(point=alt.OverlayMarkDef(size=45)).encode(
        x=x,
        y=alt.Y("us:Q", scale=alt.Scale(type="log"), title="microseconds per call (log scale)",
                axis=alt.Axis(format="~g")),
        color=alt.Color("implementation:N", sort=GPU_ORDER, scale=alt.Scale(domain=GPU_ORDER, range=P["series"]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("implementation:N", title="dtype"), alt.Tooltip("n_elem:Q", title="elements", format=","),
                 alt.Tooltip("us:Q", title="µs / call", format=".2f")],
    )
    note = _note(math.sqrt(rows["n_elem"].min() * edge), f"{lo:.1f}–{hi:.1f} µs no matter the size", 18,
                 align="center", dx=0)
    chart = (band + lines + note).properties(
        width=540,
        height=300,
        title=alt.Title(
            "A small GPU call costs the same whether it does 1,000 or 400,000 elements of work",
            subtitle="CuPy add on an A100-80GB · cold operands, preallocated result · shaded = the fixed-cost floor",
        ),
    )
    table = rows.pivot("implementation", index="n_elem", values="us").sort("n_elem")
    return Figure(chart, table, "Table view · µs per call")


CROSS_SERIES = ["x86 CPU, 1 core (float64)", "A100-80GB (fp64)"]


def _cross_rows(d: AnalysisData) -> pl.DataFrame:
    """Matched CPU and GPU add curves for the crossover comparison."""
    return pl.concat(
        [
            d.runs["cpu_dense"]
            .filter(OUT_COLD & ADD & (pl.col("implementation") == "float64"))
            .with_columns(pl.lit(CROSS_SERIES[0]).alias("machine")),
            d.runs["gpu_dense80"]
            .filter(OUT_COLD & ADD & (pl.col("implementation") == "GPU fp64"))
            .with_columns(pl.lit(CROSS_SERIES[1]).alias("machine")),
        ]
    ).select("machine", pl.col("n_elem").alias("elements"), "ns_elem", "us", "gbs").sort("machine", "elements")


def cpu_gpu_crossover(d: AnalysisData) -> float:
    """Problem size at which the GPU's per-element cost drops below the CPU's.

    Both curves are interpolated linearly in log-log space at the CPU's sizes;
    the first sign change of log(cpu) - log(gpu) is interpolated again.
    """
    rows = _cross_rows(d)
    cpu = rows.filter(pl.col("machine") == CROSS_SERIES[0])
    gpu = rows.filter(pl.col("machine") == CROSS_SERIES[1])
    gx = [math.log(v) for v in gpu["elements"]]
    gy = [math.log(v) for v in gpu["ns_elem"]]

    def gpu_at(log_n: float) -> float:
        for i in range(len(gx) - 1):
            if gx[i] <= log_n <= gx[i + 1]:
                t = (log_n - gx[i]) / (gx[i + 1] - gx[i])
                return gy[i] + t * (gy[i + 1] - gy[i])
        raise ValueError("size outside the GPU sweep")

    points = [
        (math.log(n), math.log(ns) - gpu_at(math.log(n)))
        for n, ns in zip(cpu["elements"], cpu["ns_elem"])
        if gx[0] <= math.log(n) <= gx[-1]
    ]
    for (x0, g0), (x1, g1) in zip(points, points[1:]):
        if g0 < 0 <= g1:
            return math.exp(x0 + (x1 - x0) * (-g0) / (g1 - g0))
    raise ValueError("curves do not cross")


def cpu_vs_gpu(d: AnalysisData) -> Figure:
    """Per-element cost of the same add on one CPU core and on an A100."""
    rows = _cross_rows(d)
    crossover = cpu_gpu_crossover(d)
    x = alt.X("elements:Q", scale=alt.Scale(type="log", nice=False), title="elements (log scale)",
              axis=alt.Axis(format="~s", values=[1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]))
    y = alt.Y("ns_elem:Q", scale=alt.Scale(type="log"), title="nanoseconds per element (log scale)",
              axis=alt.Axis(values=[0.01, 0.1, 1, 10], format="~g"))
    lines = alt.Chart(rows).mark_line(point=alt.OverlayMarkDef(size=45)).encode(
        x=x, y=y,
        color=alt.Color("machine:N", sort=CROSS_SERIES, scale=alt.Scale(domain=CROSS_SERIES, range=P["series"][:2]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("machine:N"), alt.Tooltip("elements:Q", format=","),
                 alt.Tooltip("ns_elem:Q", title="ns / element", format=".4f"),
                 alt.Tooltip("us:Q", title="µs / call", format=".2f")],
    )
    chart = (lines + _vrule(crossover, dash=[4, 3]) + _note(crossover, f"crossover ≈ {crossover:,.0f} elements", 14)).properties(
        width=540,
        height=310,
        title=alt.Title(
            "Below about 20,000 elements, one CPU core beats an A100",
            subtitle="add · 64-bit float · cold operands, preallocated result · the GPU pays its fixed cost first",
        ),
    )
    return Figure(chart, rows, "Table view · CPU against GPU")


# ---------------------------------------------------------------------------
# Lesson 3 · caches
# ---------------------------------------------------------------------------

FEATURED = ["float32", "float64", "int64"]


def cache_curve(d: AnalysisData) -> Figure:
    """Per-element add cost against size, cache-resident against streamed operands."""
    curve = d.runs["cpu_dense"].filter(
        (pl.col("destination") == "out") & ADD & pl.col("implementation").is_in(FEATURED)
    ).select("implementation", "n_elem", "cache_state", "ns_elem", "us")
    base = alt.Chart().mark_line(point=alt.OverlayMarkDef(size=45)).encode(
        x=alt.X("n_elem:Q", scale=alt.Scale(type="log"), title="elements (log scale)",
                axis=alt.Axis(format="~s", values=[1_000, 10_000, 100_000, 1_000_000])),
        y=alt.Y("ns_elem:Q", title="ns per element", scale=alt.Scale(domainMin=0)),
        color=alt.Color("cache_state:N", sort=["hot", "cold"],
                        scale=alt.Scale(domain=["hot", "cold"], range=P["series"][:2]),
                        legend=alt.Legend(title=None, labelExpr="datum.label == 'hot' ? 'hot: same operands every call' : 'cold: fresh operands every call'")),
        tooltip=[alt.Tooltip("implementation:N", title="dtype"), alt.Tooltip("n_elem:Q", title="elements", format=","),
                 alt.Tooltip("cache_state:N", title="operands"), alt.Tooltip("ns_elem:Q", title="ns / element", format=".3f")],
    )
    chart = base.properties(width=230, height=240).facet(
        column=alt.Column("implementation:N", title=None, sort=FEATURED), data=curve
    ).resolve_scale(y="independent").properties(
        title=alt.Title(
            "The same add costs 3× more once the data no longer fits in cache",
            subtitle="NumPy add · preallocated result · each panel has its own y-scale",
        )
    )
    return Figure(chart, curve.sort("implementation", "cache_state", "n_elem"), "Table view · cache curve")


# ---------------------------------------------------------------------------
# Lesson 4 · bandwidth
# ---------------------------------------------------------------------------


def cpu_bandwidth(d: AnalysisData) -> Figure:
    """Effective memory bandwidth for every op and dtype past all caches."""
    band = (
        d.runs["cpu_dram50"].filter(OUT_COLD).drop_nulls("gbs")
        .with_columns(pl.format("{} · {}", pl.col("operation"), pl.col("implementation")).alias("cell"))
        .sort("gbs", descending=True)
    )
    plateau = band["gbs"].max()
    states = ["at the memory limit", "short of it: limited by arithmetic"]
    band = band.with_columns(
        pl.when(pl.col("gbs") > plateau * 0.9).then(pl.lit(states[0])).otherwise(pl.lit(states[1])).alias("state")
    )
    aggregate = cpu_four_core_bandwidth(d)
    bars = alt.Chart(band).mark_bar(size=9).encode(
        y=alt.Y("cell:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200, labelFontSize=11.5)),
        x=alt.X("gbs:Q", title="effective bandwidth, GB/s", scale=alt.Scale(domain=[0, aggregate * 1.05])),
        color=alt.Color("state:N", sort=states, scale=alt.Scale(domain=states, range=[P["series"][0], P["context"]]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("cell:N"), alt.Tooltip("result_dtype:N", title="computes in"),
                 alt.Tooltip("gbs:Q", title="GB/s", format=".1f"), alt.Tooltip("us:Q", title="µs / call", format=",.0f")],
    )
    marks = pl.DataFrame({"x": [plateau, aggregate],
                          "t": [f"one core: {plateau:.1f} GB/s", f"four cores together: {aggregate:.1f} GB/s"]})
    rules = alt.Chart(marks).mark_rule(color=P["ink2"], strokeWidth=1).encode(x="x:Q")
    text = alt.Chart(marks).mark_text(align="right", dx=-5, dy=-4, fontSize=12, color=P["ink2"]).encode(
        x="x:Q", y=alt.value(6), text="t:N")
    chart = (bars + rules + text).properties(
        width=460,
        height=440,
        title=alt.Title(
            "Every cheap operation hits the same memory speed limit",
            subtitle="NumPy · 8 million elements, past every cache · four-core figure: four pinned single-threaded processes",
        ),
    )
    table = band.select("operation", "implementation", "result_dtype", "gbs", "us")
    return Figure(chart, table, "Table view · effective bandwidth")


def cpu_four_core_bandwidth(d: AnalysisData) -> float:
    """Aggregate add bandwidth of four concurrent pinned processes."""
    return sum(d.parallel[i].filter(OUT_COLD & ADD)["gbs"][0] for i in range(4))


def gpu_bandwidth_by_size(d: AnalysisData) -> Figure:
    """Share of the A100's peak memory bandwidth reached, by size and precision."""
    peak = d.meta["gpu_dense80"]["peak_hbm_gbs"]
    rows = d.runs["gpu_dense80"].filter(OUT_COLD & ADD).with_columns(
        (pl.col("gbs") / peak * 100).alias("pct")
    ).select("implementation", "n_elem", "us", "gbs", "pct")
    chart = alt.Chart(rows).mark_line(point=alt.OverlayMarkDef(size=45)).encode(
        x=alt.X("n_elem:Q", scale=alt.Scale(type="log", nice=False), title="elements (log scale)",
                axis=alt.Axis(format="~s", values=[1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000])),
        y=alt.Y("pct:Q", title=f"% of the {peak:,.0f} GB/s peak", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("implementation:N", sort=GPU_ORDER, scale=alt.Scale(domain=GPU_ORDER, range=P["series"]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("implementation:N", title="dtype"), alt.Tooltip("n_elem:Q", title="elements", format=","),
                 alt.Tooltip("gbs:Q", title="GB/s", format=",.0f"), alt.Tooltip("pct:Q", title="% of peak", format=".1f")],
    ).properties(
        width=540,
        height=300,
        title=alt.Title(
            "fp16 never gets past half of the GPU's memory bandwidth",
            subtitle="CuPy add on an A100-80GB · cold operands, preallocated result",
        ),
    )
    return Figure(chart, rows.sort("implementation", "n_elem"), "Table view · share of peak bandwidth")


def gpu_ceilings(d: AnalysisData) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (cells, per-card limits) for the two-ceiling model."""
    cells = pl.concat([gpu_ceiling_rows(d.runs, d.meta, "gpu_sweep"), gpu_ceiling_rows(d.runs, d.meta, "gpu_sweep80")])
    limits = cells.group_by("card").agg(
        pl.col("gelem_s").filter(pl.col("gelem_s") > 150).mean().alias("elem"),
        pl.col("gbs").max().alias("bw"),
    ).sort("card")
    return cells, limits


def gpu_two_ceilings(d: AnalysisData) -> Figure:
    """Achieved GPU bandwidth against bytes moved per element, both A100 variants."""
    two, limits = gpu_ceilings(d)
    cards = limits["card"].to_list()
    rate = limits["elem"].mean()
    model = pl.DataFrame(
        [{"b_elem": b, "card": r["card"], "gbs": min(r["bw"], r["elem"] * b)}
         for r in limits.iter_rows(named=True) for b in range(2, 27)]
    )
    scale = alt.Scale(domain=cards, range=P["series"][:2])
    x = alt.X("b_elem:Q", title="bytes moved per element", scale=alt.Scale(domain=[2, 27]),
              axis=alt.Axis(values=[4, 6, 8, 12, 16, 24]))
    y = alt.Y("gbs:Q", title="achieved bandwidth, GB/s", scale=alt.Scale(domain=[0, 2100]))
    tips = [alt.Tooltip("operation:N", title="op"), alt.Tooltip("implementation:N", title="dtype"),
            alt.Tooltip("b_elem:Q", title="bytes / element"), alt.Tooltip("gbs:Q", title="GB/s", format=",.0f"),
            alt.Tooltip("gelem_s:Q", title="G elements/s", format=".1f")]
    # The 40GB card is drawn hollow: on the ramp the two cards land on top of
    # each other, and that coincidence is the finding.
    hollow = alt.Chart(two.filter(pl.col("card") == cards[0])).mark_point(size=160, filled=False, strokeWidth=2).encode(
        x=x, y=y, color=alt.Color("card:N", scale=scale, legend=None), tooltip=tips)
    solid = alt.Chart(two.filter(pl.col("card") == cards[1])).mark_point(size=85).encode(
        x=x, y=y, color=alt.Color("card:N", scale=scale, legend=None), tooltip=tips)
    dashed = alt.Chart(model).mark_line(strokeWidth=1.5, strokeDash=[5, 3]).encode(
        x=x, y=y, color=alt.Color("card:N", scale=scale, legend=None))
    labels = alt.Chart(limits.with_columns(pl.lit(17.5).alias("x"), (pl.col("bw") + 80).alias("y"))).mark_text(
        align="left", fontSize=12.5, fontWeight=500).encode(
        x="x:Q", y="y:Q", text="card:N", color=alt.Color("card:N", scale=scale, legend=None))
    note = alt.Chart(pl.DataFrame({"x": [2.4], "y": [1990.0],
                                   "t": [f"dashed: {rate:.0f} G elements/s × bytes, capped at the card's memory limit"]})).mark_text(
        align="left", fontSize=12, color=P["ink2"]).encode(x="x:Q", y="y:Q", text="t:N")
    chart = (dashed + hollow + solid + labels + note).properties(
        width=540,
        height=320,
        title=alt.Title(
            "Every GPU kernel hits one of two limits: elements per second, or memory bandwidth",
            subtitle="CuPy elementwise ops · 100 million elements · hollow = A100-40GB, filled = A100-80GB",
        ),
    )
    return Figure(chart, two.sort("card", "b_elem", "gbs"), "Table view · the two ceilings")


# ---------------------------------------------------------------------------
# Lesson 5 · precision
# ---------------------------------------------------------------------------

F16_ORDER = ["float16", "float32", "float64"]


def float16_curve(d: AnalysisData) -> Figure:
    """float16 against float32 and float64 add cost across size."""
    rows = d.runs["cpu_dense"].filter(OUT_HOT & ADD & pl.col("implementation").is_in(F16_ORDER)).select(
        "implementation", "n_elem", "ns_elem", "us")
    ends = rows.filter(pl.col("n_elem") == rows["n_elem"].max())
    x = alt.X("n_elem:Q", scale=alt.Scale(type="log", nice=False), title="elements (log scale)",
              axis=alt.Axis(format="~s", values=[1_000, 10_000, 100_000, 1_000_000]))
    y = alt.Y("ns_elem:Q", scale=alt.Scale(type="log"), title="ns per element (log scale)",
              axis=alt.Axis(values=[0.1, 0.3, 1, 3, 10], format="~g"))
    color = alt.Color("implementation:N", sort=F16_ORDER, legend=None,
                      scale=alt.Scale(domain=F16_ORDER, range=[P["series"][1], P["series"][0], P["shade"]]))
    lines = alt.Chart(rows).mark_line(strokeWidth=2.2).encode(x=x, y=y, color=color)
    dots = lines.mark_point(size=55).encode(
        tooltip=[alt.Tooltip("implementation:N", title="dtype"), alt.Tooltip("n_elem:Q", title="elements", format=","),
                 alt.Tooltip("ns_elem:Q", title="ns / element", format=".3f")])
    labels = alt.Chart(ends).mark_text(align="left", dx=10, fontSize=12.5, fontWeight=500).encode(
        x=x, y=y, color=color, text="implementation:N")
    chart = (lines + dots + labels).properties(
        width=480,
        height=300,
        title=alt.Title(
            "On this CPU, float16 is the slowest hardware float at every size",
            subtitle="NumPy add · same operands every call, preallocated result",
        ),
    ).configure_view(clip=False)
    table = rows.pivot("implementation", index="n_elem", values="ns_elem").sort("n_elem")
    return Figure(chart, table, "Table view · float16 against float32 and float64")


def float16_mechanism(d: AnalysisData) -> Figure:
    """Native float16 ufuncs against widening to float32 by hand."""
    paths = ["native ufunc", "widened by hand"]
    mech = (
        d.runs["cpu_f16"].filter((pl.col("destination") == "alloc") & (pl.col("cache_state") == "hot")
                                 & (pl.col("n_elem") == 256_000))
        .with_columns(pl.col("operation").str.replace("_via_f32", "").alias("op"),
                      pl.when(pl.col("operation").str.contains("via_f32")).then(pl.lit(paths[1]))
                      .otherwise(pl.lit(paths[0])).alias("path"))
        .select("op", "implementation", "path", "ns_elem")
    )
    span = mech.pivot("path", index=["op", "implementation"], values="ns_elem").rename(
        {paths[0]: "native", paths[1]: "via"})
    # One dataset for both layers: a faceted layer cannot mix sources.
    mech = mech.join(span, on=["op", "implementation"])
    scale = alt.Scale(type="log", domain=[0.1, 12], nice=False)
    axis = alt.Axis(values=[0.1, 0.3, 1, 3, 10], format="~g")
    y = alt.Y("op:N", title=None, sort=["add", "mul", "exp"])
    # A dumbbell, not bars: the values span 40×, so the axis is log.
    rule = alt.Chart().mark_rule(color=P["context"], strokeWidth=2.5).encode(
        x=alt.X("native:Q", scale=scale, title="ns per element (log scale)", axis=axis), x2="via:Q", y=y)
    dots = alt.Chart().mark_point(size=100).encode(
        x=alt.X("ns_elem:Q", scale=scale, title="ns per element (log scale)", axis=axis), y=y,
        color=alt.Color("path:N", sort=paths, scale=alt.Scale(domain=paths, range=P["series"][:2]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("op:N"), alt.Tooltip("implementation:N", title="dtype"), alt.Tooltip("path:N"),
                 alt.Tooltip("ns_elem:Q", title="ns / element", format=".3f")])
    chart = alt.layer(rule, dots).properties(width=235, height=150).facet(
        column=alt.Column("implementation:N", title=None, sort=F16_ORDER), data=mech
    ).properties(
        title=alt.Title(
            "Converting to float32 by hand costs the same as NumPy's float16: the conversion is the cost",
            subtitle="256,000 elements · 'widened by hand' = astype(float32), operate, astype back",
        )
    )
    return Figure(chart, span.sort("implementation", "op"), "Table view · native against hand-widened")


DTYPE_NAMED = ["quad (software, 128-bit)", "longdouble (x87, 80-bit)", "float16 (converted, no ALU)"]


def dtype_grid(d: AnalysisData) -> Figure:
    """Every op's cost for every dtype, relative to float64."""
    family = (
        pl.when(pl.col("implementation") == "quad-sleef").then(pl.lit(DTYPE_NAMED[0]))
        .when(pl.col("implementation") == "longdouble64").then(pl.lit(DTYPE_NAMED[1]))
        .when(pl.col("implementation") == "float16").then(pl.lit(DTYPE_NAMED[2]))
        .otherwise(pl.lit("the 10 hardware dtypes"))
    )
    sweep = d.runs["cpu_sweep"].filter(OUT_HOT & (pl.col("n") == 64_000))
    f64 = sweep.filter(pl.col("implementation") == "float64").select("operation", pl.col("ns_elem").alias("base"))
    tax = (
        sweep.join(f64, on="operation")
        .with_columns((pl.col("ns_elem") / pl.col("base")).alias("vs_f64"), family.alias("family"))
        .drop_nulls("vs_f64")
        .select("operation", "implementation", "family", "vs_f64", "ns_elem")
    )
    context, named = tax.filter(~pl.col("family").is_in(DTYPE_NAMED)), tax.filter(pl.col("family").is_in(DTYPE_NAMED))
    x = alt.X("vs_f64:Q", scale=alt.Scale(type="log", nice=False), title="cost relative to float64, same operation (log scale)",
              axis=alt.Axis(values=[0.25, 1, 4, 16, 64, 256], labelExpr="format(datum.value, '.3~g') + '×'"))
    y = alt.Y("operation:N", sort=OP_ORDER, title=None)
    unity = alt.Chart(pl.DataFrame({"x": [1.0]})).mark_rule(color=P["ink2"]).encode(x=x)
    unity_label = alt.Chart(pl.DataFrame({"x": [1.0], "t": ["float64"]})).mark_text(
        dy=-6, fontSize=12, color=P["ink2"]).encode(x=x, y=alt.value(0), text="t:N")
    grey = alt.Chart(context).mark_point(size=60, color=P["context"], opacity=0.85).encode(
        x=x, y=y, tooltip=[alt.Tooltip("implementation:N", title="dtype"), alt.Tooltip("vs_f64:Q", title="× float64", format=".2f")])
    dots = alt.Chart(named).mark_point(size=140).encode(
        x=x, y=y,
        color=alt.Color("family:N", sort=DTYPE_NAMED, scale=alt.Scale(domain=DTYPE_NAMED, range=P["series"]),
                        legend=alt.Legend(title=None, columns=1)),
        tooltip=[alt.Tooltip("implementation:N", title="dtype"), alt.Tooltip("vs_f64:Q", title="× float64", format=".1f"),
                 alt.Tooltip("ns_elem:Q", title="ns / element", format=".2f")])
    quad_labels = alt.Chart(named.filter(pl.col("implementation") == "quad-sleef")).mark_text(
        align="left", dx=11, fontSize=11.5, color=P["ink2"]).encode(x=x, y=y, text=alt.Text("vs_f64:Q", format=".0f"))
    chart = (unity + unity_label + grey + dots + quad_labels).properties(
        width=540,
        height=320,
        title=alt.Title(
            "Ten hardware dtypes cluster near float64; three outliers don't",
            subtitle="NumPy · 256,000 elements · grey = the ten hardware dtypes · numbers = quad's multiple of float64",
        ),
    ).configure_view(clip=False)
    return Figure(chart, named.sort("operation", "implementation"), "Table view · dtypes outside the cluster")


GPU_PAIRS = ["fp64 / fp32", "fp32 / fp16"]


def gpu_precision_table(d: AnalysisData) -> pl.DataFrame:
    """Time ratios between adjacent GPU precisions at 100 million elements."""
    return (
        d.runs["gpu_sweep80"].filter(OUT_HOT & (pl.col("n") == 25_000_000))
        .pivot("implementation", index="operation", values="us").drop_nulls()
        .with_columns((pl.col("GPU fp64") / pl.col("GPU fp32")).alias(GPU_PAIRS[0]),
                      (pl.col("GPU fp32") / pl.col("GPU fp16")).alias(GPU_PAIRS[1]))
        .sort(GPU_PAIRS[1], descending=True)
    )


def gpu_precision_ratios(d: AnalysisData) -> Figure:
    """How much halving precision actually saves on the GPU, per operation."""
    ratios = gpu_precision_table(d)
    long = ratios.unpivot(GPU_PAIRS, index="operation", variable_name="pair", value_name="ratio")
    bars = alt.Chart(long).mark_bar(size=10).encode(
        y=alt.Y("operation:N", title=None, sort=alt.EncodingSortField("ratio", op="max", order="descending")),
        yOffset=alt.YOffset("pair:N", sort=GPU_PAIRS),
        x=alt.X("ratio:Q", title="time ratio (2.0 = halving precision halves the time)",
                scale=alt.Scale(domain=[0, 2.3], nice=False)),
        color=alt.Color("pair:N", sort=GPU_PAIRS, scale=alt.Scale(domain=GPU_PAIRS, range=P["series"][:2]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("operation:N", title="op"), alt.Tooltip("pair:N", title="ratio"),
                 alt.Tooltip("ratio:Q", format=".2f")],
    )
    values = bars.mark_text(align="left", dx=4, fontSize=11, color=P["ink2"]).encode(text=alt.Text("ratio:Q", format=".2f"))
    chart = (bars + values + _vrule(2.0)).properties(
        width=440,
        height=340,
        title=alt.Title(
            "On the GPU, halving precision rarely halves the time",
            subtitle="CuPy on an A100-80GB · 100 million elements · same operands every call",
        ),
    )
    return Figure(chart, ratios, "Table view · precision ratios")


# ---------------------------------------------------------------------------
# Lesson 6 · allocation
# ---------------------------------------------------------------------------

COMPONENTS = ["cold destination (out_pool − out)", "allocation (alloc − out_pool)"]


def destination_decomposition(d: AnalysisData) -> Figure:
    """Split the cost of a fresh result buffer into cache and allocation parts."""
    three = (
        d.runs["cpu_alloc"].filter((pl.col("cache_state") == "cold") & ADD)
        .pivot("destination", index=["implementation", "n_elem"], values="us")
        .with_columns(((pl.col("out_pool") - pl.col("out")) / pl.col("out") * 100).alias(COMPONENTS[0]),
                      ((pl.col("alloc") - pl.col("out_pool")) / pl.col("out") * 100).alias(COMPONENTS[1]))
        .unpivot(COMPONENTS, index=["implementation", "n_elem"], variable_name="component", value_name="pct")
        .sort("implementation", "n_elem")
    )
    zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color=P["ink2"]).encode(y="y:Q")
    lines = alt.Chart().mark_line(point=alt.OverlayMarkDef(size=50)).encode(
        x=alt.X("n_elem:Q", scale=alt.Scale(type="log"), title="elements (log scale)",
                axis=alt.Axis(format="~s", values=[10_000, 100_000, 1_000_000])),
        y=alt.Y("pct:Q", title="% of the preallocated time"),
        color=alt.Color("component:N", sort=COMPONENTS, scale=alt.Scale(domain=COMPONENTS, range=P["series"][:2]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("implementation:N", title="dtype"), alt.Tooltip("n_elem:Q", title="elements", format=","),
                 alt.Tooltip("component:N"), alt.Tooltip("pct:Q", title="% of out=", format=".1f")],
    )
    chart = alt.layer(zero, lines).properties(width=215, height=230).facet(
        column=alt.Column("implementation:N", title=None, sort=["float32", "float64", "int32", "int64"]), data=three
    ).properties(
        title=alt.Title(
            "A result buffer that has left the cache costs up to 60% extra; allocating it costs nothing until 8M elements",
            subtitle="NumPy add · cold operands · out_pool = rotate through preallocated buffers too big to stay cached",
        )
    )
    return Figure(chart, three, "Table view · destination cost decomposition")


SIDES = ["at or below 32 MiB", "above 32 MiB"]
MMAP_THRESHOLD_MIB = 32


def allocation_cliff_rows(d: AnalysisData) -> pl.DataFrame:
    """Allocating-form penalty for every pooled cell, tagged by threshold side."""
    return pl.concat([
        allocation_penalty(d.runs, "cpu_dense"),
        allocation_penalty(d.runs, "cpu_alloc"),
        allocation_penalty(d.runs, "cpu_dram50", operations=("add", "mul", "div", "sqrt", "exp")),
    ]).with_columns(
        pl.when(pl.col("result_mib") > MMAP_THRESHOLD_MIB).then(pl.lit(SIDES[1])).otherwise(pl.lit(SIDES[0])).alias("side")
    )


def allocation_cliff(d: AnalysisData) -> Figure:
    """Allocation penalty against result-buffer size, across three runs."""
    cliff = allocation_cliff_rows(d)
    zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color=P["grid"]).encode(y="y:Q")
    dots = alt.Chart(cliff).mark_point(size=95, opacity=0.85).encode(
        x=alt.X("result_mib:Q", scale=alt.Scale(type="log"), title="result buffer size, MiB (log scale)",
                axis=alt.Axis(values=[0.05, 0.5, 5, 32, 61], format="~g")),
        y=alt.Y("pct:Q", title="allocating form, % slower than preallocated"),
        color=alt.Color("side:N", sort=SIDES, scale=alt.Scale(domain=SIDES, range=[P["context"], P["series"][1]]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("run:N"), alt.Tooltip("operation:N", title="op"), alt.Tooltip("implementation:N", title="dtype"),
                 alt.Tooltip("result_mib:Q", title="result MiB", format=".2f"), alt.Tooltip("pct:Q", title="% vs out=", format=".1f")],
    )
    threshold = _vrule(float(MMAP_THRESHOLD_MIB), dash=[4, 3]) + _note(
        float(MMAP_THRESHOLD_MIB), "glibc mmap threshold, 32 MiB", 12, align="right", dx=-7)
    chart = (zero + dots + threshold).properties(
        width=520,
        height=300,
        title=alt.Title(
            "Allocating a result is free up to 32 MiB, then suddenly is not",
            subtitle="NumPy · cold operands · three runs pooled · above the threshold every call maps and unmaps fresh memory",
        ),
    )
    return Figure(chart, cliff.sort("result_mib"), "Table view · allocation penalty by buffer size")


# ---------------------------------------------------------------------------
# Lesson 7 · compilation
# ---------------------------------------------------------------------------

FRAMEWORK_SERIES = ["JAX JIT · host pipeline", "CuPy alloc · CUDA events"]
COMMON_GPU_OPS = ["add", "exp", "matmul"]


def framework_endpoint(d: AnalysisData) -> pl.DataFrame:
    """CuPy and JAX fp32 per-element cost at the largest matched size."""
    rows = _framework_rows(d)
    return (
        rows.filter(pl.col("n_elem") == rows["n_elem"].max())
        .pivot("series", index="operation", values="ns_elem")
        .with_columns((pl.col(FRAMEWORK_SERIES[1]) / pl.col(FRAMEWORK_SERIES[0])).alias("CuPy / JAX time"))
        .sort(pl.col("operation").replace_strict({op: i for i, op in enumerate(COMMON_GPU_OPS)}))
    )


def _framework_rows(d: AnalysisData) -> pl.DataFrame:
    jax = d.runs["jax_dense"].filter(
        (pl.col("destination") == "jit") & (pl.col("cache_state") == "cold")
        & (pl.col("implementation") == "JAX fp32") & pl.col("operation").is_in(COMMON_GPU_OPS)
    ).select("operation", "n_elem", "us", "ns_elem", "samples").with_columns(pl.lit(FRAMEWORK_SERIES[0]).alias("series"))
    cupy = d.runs["gpu_dense80"].filter(
        ALLOC_COLD & (pl.col("implementation") == "GPU fp32") & pl.col("operation").is_in(COMMON_GPU_OPS)
    ).select("operation", "n_elem", "us", "ns_elem", "samples").with_columns(pl.lit(FRAMEWORK_SERIES[1]).alias("series"))
    return pl.concat([jax, cupy])


def jax_vs_cupy(d: AnalysisData) -> Figure:
    """Matched JAX and CuPy fp32 execution cost across size."""
    rows = _framework_rows(d)
    chart = alt.Chart(rows).mark_line(point=alt.OverlayMarkDef(size=48), strokeWidth=2.1).encode(
        x=alt.X("n_elem:Q", scale=alt.Scale(type="log", nice=False), title="elements (log scale)", axis=alt.Axis(format="~s")),
        y=alt.Y("ns_elem:Q", scale=alt.Scale(type="log"), title="ns per element (log scale)"),
        color=alt.Color("series:N", sort=FRAMEWORK_SERIES, scale=alt.Scale(domain=FRAMEWORK_SERIES, range=P["series"][:2]),
                        legend=alt.Legend(title=None)),
        strokeDash=alt.StrokeDash("series:N", sort=FRAMEWORK_SERIES, legend=None,
                                  scale=alt.Scale(domain=FRAMEWORK_SERIES, range=[[1, 0], [6, 3]])),
        tooltip=[alt.Tooltip("operation:N"), alt.Tooltip("series:N", title="framework and timer"),
                 alt.Tooltip("n_elem:Q", title="elements", format=","), alt.Tooltip("us:Q", title="µs / call", format=".2f"),
                 alt.Tooltip("ns_elem:Q", title="ns / element", format=".4f")],
    ).properties(width=215, height=230).facet(
        column=alt.Column("operation:N", sort=COMMON_GPU_OPS, title=None)
    ).properties(
        title=alt.Title(
            "JAX pays more per call, then matches or beats CuPy once arrays are large",
            subtitle="fp32 on an A100-80GB · cold operands · result allocated every call",
        )
    )
    return Figure(chart, framework_endpoint(d), "Table view · largest matched size")


FUSION_OPS = ["add", "muladd", "chain", "matmul_explicit"]
JAX_DTYPES = ["JAX fp16", "JAX bf16", "JAX fp32", "JAX fp64"]


def fusion_table(d: AnalysisData) -> pl.DataFrame:
    """Eager-to-JIT time ratios for the fusion comparison."""
    return (
        d.runs["jax_sweep"].filter((pl.col("n") == 25_000_000) & (pl.col("cache_state") == "hot")
                                   & pl.col("operation").is_in(FUSION_OPS) & pl.col("implementation").is_in(JAX_DTYPES))
        .pivot("destination", index=["operation", "implementation"], values="us")
        .with_columns((pl.col("eager") / pl.col("jit")).alias("JIT speedup"))
        .sort(pl.col("implementation").replace_strict({k: i for i, k in enumerate(JAX_DTYPES)}),
              pl.col("operation").replace_strict({k: i for i, k in enumerate(FUSION_OPS)}))
    )


def jit_fusion(d: AnalysisData) -> Figure:
    """How much compiling the whole expression saves, per op and dtype."""
    fusion = fusion_table(d)
    points = alt.Chart().mark_point(size=110, color=P["series"][0]).encode(
        x=alt.X("JIT speedup:Q", scale=alt.Scale(type="log", domain=[0.8, 128]), title="eager time ÷ JIT time (log scale)",
                axis=alt.Axis(values=[1, 2, 4, 8, 16, 32, 64, 128], format="~g")),
        y=alt.Y("operation:N", sort=FUSION_OPS, title=None),
        tooltip=[alt.Tooltip("implementation:N", title="dtype"), alt.Tooltip("operation:N"),
                 alt.Tooltip("eager:Q", title="eager µs", format=",.1f"), alt.Tooltip("jit:Q", title="JIT µs", format=",.1f"),
                 alt.Tooltip("JIT speedup:Q", title="speedup", format=".2f")],
    )
    unity = alt.Chart().mark_rule(color=P["ink2"]).encode(x=alt.datum(1.0))
    chart = alt.layer(unity, points).properties(width=150, height=150).facet(
        column=alt.Column("implementation:N", sort=JAX_DTYPES, title=None), data=fusion
    ).properties(
        title=alt.Title(
            "Compiling a whole expression is 2–16× faster than running it step by step",
            subtitle="JAX on an A100-80GB · 100 million elements · add is the control: one step, nothing to fuse",
        )
    )
    return Figure(chart, fusion, "Table view · JIT speedups")


CPU_OPS = ["add", "exp", "chain", "matmul", "matmul_explicit"]


def cpu_framework_table(d: AnalysisData) -> pl.DataFrame:
    """NumPy eager against JAX JIT on the same CPU node."""
    n = max(d.meta["jax_cpu"]["sizes"])
    return (
        d.runs["jax_cpu"].filter(
            (pl.col("n") == n) & (pl.col("cache_state") == "hot") & pl.col("operation").is_in(CPU_OPS)
            & (((pl.col("implementation") == "JAX fp32") & (pl.col("destination") == "jit"))
               | ((pl.col("implementation") == "NumPy fp32") & (pl.col("destination") == "eager")))
        )
        .with_columns(pl.when(pl.col("implementation") == "JAX fp32").then(pl.lit("JAX JIT"))
                      .otherwise(pl.lit("NumPy eager")).alias("runtime"))
        .pivot("runtime", index="operation", values="us")
        .with_columns((pl.col("NumPy eager") / pl.col("JAX JIT")).alias("NumPy / JAX time"))
        .sort(pl.col("operation").replace_strict({op: i for i, op in enumerate(CPU_OPS)}))
    )


def jax_vs_numpy_cpu(d: AnalysisData) -> Figure:
    """Same-node NumPy and JAX CPU time ratios."""
    table = cpu_framework_table(d)
    points = alt.Chart(table).mark_point(size=120, color=P["series"][0]).encode(
        x=alt.X("NumPy / JAX time:Q", scale=alt.Scale(type="log"), title="NumPy time ÷ JAX time (log scale)",
                axis=alt.Axis(format="~g")),
        y=alt.Y("operation:N", sort=CPU_OPS, title=None),
        tooltip=[alt.Tooltip("operation:N"), alt.Tooltip("NumPy eager:Q", title="NumPy µs", format=",.1f"),
                 alt.Tooltip("JAX JIT:Q", title="JAX µs", format=",.1f"), alt.Tooltip("NumPy / JAX time:Q", title="ratio", format=".2f")],
    )
    labels = points.mark_text(align="left", dx=10, fontSize=12, color=P["ink2"]).encode(
        text=alt.Text("NumPy / JAX time:Q", format=".1f"))
    chart = (_vrule(1.0) + points + labels).properties(
        width=500,
        height=220,
        title=alt.Title(
            "On the same CPU, compiled JAX beats NumPy on every operation measured",
            subtitle="10 million elements · fp32 · same operands every call · thread pools not constrained",
        ),
    )
    return Figure(chart, table, "Table view · same-node CPU comparison")


# ---------------------------------------------------------------------------
# Lesson 8 · measurement
# ---------------------------------------------------------------------------

TIMER_SERIES = ["JAX pipelined", "JAX blocking", "CuPy host timer"]
LATENCY_OPS = ["add", "exp"]


def timer_rows(d: AnalysisData) -> pl.DataFrame:
    """JAX pipelined, JAX blocking, and CuPy host-timed fp32 calls at matched sizes."""
    sizes = d.meta["jax_latency"]["sizes"]
    return pl.concat([
        jax_timer_series(d.runs, "jax_dense", TIMER_SERIES[0], sizes, LATENCY_OPS),
        jax_timer_series(d.runs, "jax_latency", TIMER_SERIES[1], sizes, LATENCY_OPS),
        d.runs["gpu_host80"].filter(ALLOC_COLD & (pl.col("implementation") == "GPU fp32") & pl.col("operation").is_in(LATENCY_OPS))
        .select("operation", "n_elem", "us", "ns_elem", "samples").with_columns(pl.lit(TIMER_SERIES[2]).alias("series")),
    ])


def timer_floor(d: AnalysisData) -> pl.DataFrame:
    """Smallest-size call times under each protocol."""
    rows = timer_rows(d)
    return (
        rows.filter(pl.col("n_elem") == rows["n_elem"].min())
        .pivot("series", index="operation", values="us")
        .with_columns((pl.col("JAX blocking") / pl.col("JAX pipelined")).alias("blocking / pipelined"))
        .sort(pl.col("operation").replace_strict({op: i for i, op in enumerate(LATENCY_OPS)}))
    )


def timing_protocol(d: AnalysisData) -> Figure:
    """Host-observed call time under three timing protocols."""
    rows = timer_rows(d)
    chart = alt.Chart(rows).mark_line(point=alt.OverlayMarkDef(size=52), strokeWidth=2).encode(
        x=alt.X("n_elem:Q", scale=alt.Scale(type="log", nice=False), title="elements (log scale)", axis=alt.Axis(format="~s")),
        y=alt.Y("us:Q", scale=alt.Scale(type="log"), title="µs per call, seen from Python (log scale)"),
        color=alt.Color("series:N", sort=TIMER_SERIES, legend=alt.Legend(title=None),
                        scale=alt.Scale(domain=TIMER_SERIES, range=[P["series"][0], P["shade"], P["series"][1]])),
        strokeDash=alt.StrokeDash("series:N", sort=TIMER_SERIES, legend=None,
                                  scale=alt.Scale(domain=TIMER_SERIES, range=[[1, 0], [6, 3], [2, 2]])),
        tooltip=[alt.Tooltip("operation:N"), alt.Tooltip("series:N", title="protocol"),
                 alt.Tooltip("n_elem:Q", title="elements", format=","), alt.Tooltip("us:Q", title="µs / call", format=".2f")],
    ).properties(width=300, height=250).facet(
        column=alt.Column("operation:N", sort=LATENCY_OPS, title=None)
    ).properties(
        title=alt.Title(
            "How you time a call changes the answer",
            subtitle="fp32 on an A100-80GB · cold operands · result allocated every call",
        )
    )
    return Figure(chart, timer_floor(d), "Table view · smallest-size call times")
