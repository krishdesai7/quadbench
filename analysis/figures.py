# ty: ignore[unresolved-attribute, unsupported-operator]
# pyrefly: ignore-errors[unsupported-operation]
"""Every chart in the report, one function per figure.

Each function takes the loaded :class:`~analysis.data.AnalysisData` and returns
a :class:`~analysis.style.Figure`: the Altair chart plus a table carrying the
same numbers.  Annotation positions (thresholds, crossovers, ceilings) are
computed from the data, never typed in.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, cast

import altair as alt
import numpy as np
import polars as pl
from altair.vegalite.v6.api import Chart, FacetChart, LayerChart
from altair.vegalite.v6.schema.channels import Tooltip, X
from polars.dataframe.frame import DataFrame
from polars.series.series import Series

from analysis.data import (
    ALLOC_COLD,
    OP_ORDER,
    OUT_COLD,
    OUT_HOT,
    UNARY,
    AnalysisData,
    allocation_penalty,
    gpu_ceiling_rows,
    jax_timer_series,
)
from analysis.style import Figure, P

if TYPE_CHECKING:
    from typing import Any, Final

    from numpy.typing import NDArray

ADD: pl.Expr = pl.col.operation == "add"
LADDER_N: Final[int] = 1_000_000

# Lesson 1 uses one color per kind of container, kept identical across its charts.
KINDS: Final[list[str]] = ["Python objects", "NumPy array", "CuPy array"]
KIND_COLORS: Final[dict[str, str]] = dict(
    zip(KINDS, [P["series"][1], P["series"][0], P["series"][2]])
)


def _kind_color(kinds: list[str]) -> alt.Color:
    """Color by container kind, with a legend listing only the kinds present."""
    return alt.Color(
        shorthand="kind:N",
        sort=kinds,
        legend=alt.Legend(title=None),
        scale=alt.Scale(domain=kinds, range=[KIND_COLORS[k] for k in kinds]),
    )


PYTHON_LABELS: Final[dict[str, str]] = {
    "py-list-f32": "list[np.float32]",
    "py-list-float": "list[float]",
    "py-obj-array": "NDArray[float]",
}


def _note(
    x: float, text: str, y_px: float, *, align: str = "left", dx: float = 7
) -> alt.Chart:
    """Return a small muted text annotation anchored at a data x and pixel y."""
    return (
        alt.Chart(data=pl.DataFrame(data={"x": [x], "t": [text]}))
        .mark_text(align=align, dx=dx, fontSize=12, color=P["ink2"])  # ty: ignore[invalid-argument-type]
        .encode(x="x:Q", y=alt.value(value=y_px), text="t:N")
    )


def _vrule(x: float, *, dash: list[int] | None = None) -> alt.Chart:
    """Return a vertical reference rule at a data x."""
    return (
        alt.Chart(data=pl.DataFrame(data={"x": [x]}))
        .mark_rule(color=P["ink2"], strokeWidth=1, strokeDash=dash or [])  # ty: ignore[invalid-argument-type]
        .encode(x="x:Q")
    )


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
    python: pl.DataFrame = d.pure.filter(
        ADD & (pl.col.cache_state == "cold") & (pl.col.n == LADDER_N)
    ).select(
        pl.col.implementation.replace_strict(old=PYTHON_LABELS).alias(
            name="representation"
        ),
        pl.lit(value="Python objects").alias(name="kind"),
        "ns_elem",
        "n_elem",
    )
    dense: pl.DataFrame = d.runs["cpu_dense"].filter(
        ADD & ALLOC_COLD & pl.col.us.is_not_null()
    )
    numpy_rows: pl.DataFrame = dense.filter(
        (pl.col.n == LADDER_N)
        & pl.col.implementation.is_in(other=["float16", "float32", "float64"])
    ).select(
        pl.format("NDArray[np.{}]", pl.col.implementation).alias(name="representation"),
        pl.lit(value="NumPy array").alias(name="kind"),
        "ns_elem",
        "n_elem",
    )
    # Quad was skipped at 4M elements by the harness's time cap; its cost is
    # flat across sizes, so the largest measured size stands in.
    float128: pl.DataFrame = (
        dense.filter(pl.col.implementation == "quad-sleef")
        .sort(by="n")
        .tail(n=1)
        .select(
            pl.lit(value="NDArray[np.float128]†").alias(name="representation"),
            pl.lit(value="NumPy array").alias(name="kind"),
            "ns_elem",
            "n_elem",
        )
    )
    float96: pl.DataFrame = (
        d.runs["cpu_sweep"]
        .filter(ADD & ALLOC_COLD & (pl.col.implementation == "longdouble64"))
        .sort(by="n")
        .tail(n=1)
        .select(
            pl.lit(value="NDArray[np.float96]†").alias(name="representation"),
            pl.lit(value="NumPy array").alias(name="kind"),
            "ns_elem",
            "n_elem",
        )
    )
    gpu: pl.DataFrame = (
        d.runs["gpu_dense80"]
        .filter(
            ADD
            & ALLOC_COLD
            & (pl.col.n == LADDER_N)
            & (pl.col.implementation == "GPU fp32")
        )
        .select(
            pl.lit(value="NDArray[cp.float32] (A100)").alias(name="representation"),
            pl.lit(value="CuPy array").alias(name="kind"),
            "ns_elem",
            "n_elem",
        )
    )
    rows: pl.DataFrame = pl.concat(items=[python, numpy_rows, float128, float96, gpu])
    float64: pl.Series = rows.filter(pl.col.representation == "NDArray[np.float64]")[
        "ns_elem"
    ][0]
    return rows.with_columns(
        (pl.col.ns_elem / float64).alias(name="vs_f64"),
        pl.col.ns_elem.map_elements(function=_ns_label, return_dtype=pl.String).alias(
            name="label"
        ),
    ).sort(by="ns_elem", descending=True)


def representation_ladder(d: AnalysisData) -> Figure:
    """Per-element cost of add for every container and dtype, on one axis."""
    rows: pl.DataFrame = _ladder_rows(d)
    order: list[str] = rows["representation"].to_list()
    borrowed: list[tuple[str, int]] = (
        rows.filter(pl.col.representation.str.ends_with(suffix="†"))
        .select(pl.col.representation, pl.col.n_elem)
        .rows()
    )
    x = alt.X(
        shorthand="ns_elem:Q",
        scale=alt.Scale(type="log", domain=[0.004, 400], nice=False),
        title="nanoseconds per element (log scale)",
        axis=alt.Axis(
            values=[0.01, 0.1, 1, 10, 100],
            format="~g",
            labelFontSize=16,
            titleFontSize=16,
        ),
    )
    y = alt.Y(
        shorthand="representation:N",
        sort=order,
        title=None,
        axis=alt.Axis(labelFontSize=16, labelLimit=260, labelFont="monospace"),
    )
    color: alt.Color = _kind_color(KINDS)
    dots: alt.Chart = (
        alt.Chart(data=rows)
        .mark_point(size=170)
        .encode(
            x=x,
            y=y,
            color=color,
            tooltip=[
                alt.Tooltip(shorthand="representation:N"),
                alt.Tooltip(shorthand="ns_elem:Q", title="ns / element", format=".3f"),
                alt.Tooltip(
                    shorthand="vs_f64:Q", title="× float64 array", format=",.1f"
                ),
                alt.Tooltip(shorthand="n_elem:Q", title="elements", format=","),
            ],
        )
    )
    labels: alt.Chart = (
        alt.Chart(data=rows)
        .mark_text(align="left", dx=13, fontSize=12.5)
        .encode(x=x, y=y, text="label:N")
    )
    chart: alt.FacetChart | alt.LayerChart = (dots + labels).properties(
        width=560,
        height=340,
        title=alt.Title(
            text="A list of Python floats is the slowest way to hold numbers",
            subtitle=[
                "add · 4 million elements · cold operands · result allocated on every call · Python rows: free-threaded CPython 3.14",
                "† flat cost per element, shown at the largest size measured: "
                + ", ".join(
                    f"{name.split(sep='[')[1].split(sep=']')[0]} at {n:,} elements"
                    for name, n in borrowed
                ),
            ],
        ),
    )
    table: pl.DataFrame = rows.select(
        "representation",
        "kind",
        pl.col.ns_elem.alias(name="ns per element"),
        pl.col.vs_f64.alias(name="× float64 array"),
        pl.col.n_elem.alias(name="elements"),
    )
    return Figure(chart, table, label="Table view · cost per element by representation")


def bytes_per_element(d: AnalysisData) -> Figure:
    """Memory per stored number: boxed Python objects against raw arrays."""
    rows: pl.DataFrame = pl.concat(
        items=[
            pl.DataFrame(
                data={
                    "representation": PYTHON_LABELS.values(),
                    "kind": ["Python objects"] * 3,
                    "bytes": (d.pure_memory[k] for k in PYTHON_LABELS),
                }
            ),
            pl.DataFrame(
                data={
                    "representation": [
                        "NDArray[np.float64]",
                        "NDArray[np.float32]",
                        "NDArray[np.float16]",
                    ],
                    "kind": ["NumPy array"] * 3,
                    "bytes": [8.0, 4.0, 2.0],
                }
            ),
        ]
    ).sort(by="bytes", descending=True)
    rows: pl.DataFrame = rows.with_columns(
        pl.format("{} B", pl.col.bytes.round(decimals=0).cast(dtype=pl.Int64)).alias(
            name="label"
        )
    )
    y = alt.Y(
        shorthand="representation:N",
        sort=rows["representation"].to_list(),
        title=None,
        axis=alt.Axis(labelFontSize=16, labelLimit=260, labelFont="monospace"),
    )
    x = alt.X(
        shorthand="bytes:Q",
        title="bytes per stored number",
        scale=alt.Scale(domain=[0, 60]),
        axis=alt.Axis(labelFontSize=16, titleFontSize=16),
    )
    bars: alt.Chart = (
        alt.Chart(data=rows)
        .mark_bar(size=18)
        .encode(
            x=x,
            y=y,
            color=_kind_color(KINDS[:2]),
            tooltip=[
                alt.Tooltip(shorthand="representation:N"),
                alt.Tooltip(shorthand="bytes:Q", format=".1f"),
            ],
        )
    )
    labels: alt.Chart = (
        alt.Chart(data=rows)
        .mark_text(align="left", dx=6, fontSize=12.5)
        .encode(x=x, y=y, text="label:N")
    )
    chart: alt.FacetChart | alt.LayerChart = (bars + labels).properties(
        width=520,
        height=230,
        title=alt.Title(
            text=f"A list of floats takes {32 / 8:.0f}–{d.pure_memory['py-list-float'] / 8:.0f}× the memory of a float64 array",
            subtitle="measured with sys.getsizeof over the container and every element, on free-threaded CPython 3.14",
        ),
    )
    return Figure(
        chart, table=rows.drop("label"), label="Table view · bytes per element"
    )


def boxed_vs_unboxed_by_size(d: AnalysisData) -> Figure:
    """Per-element add cost across problem size, boxed against unboxed."""
    python: pl.DataFrame = d.pure.filter(ADD & (pl.col.cache_state == "cold")).select(
        pl.col.implementation.replace_strict(old=PYTHON_LABELS).alias(
            name="representation"
        ),
        pl.lit(value="Python objects").alias(name="kind"),
        "n_elem",
        "ns_elem",
    )
    arrays: pl.DataFrame = (
        d.runs["cpu_dense"]
        .filter(
            ADD & ALLOC_COLD & pl.col.implementation.is_in(other=["float32", "float64"])
        )
        .select(
            pl.format("NDArray[np.{}]", pl.col.implementation).alias(
                name="representation"
            ),
            pl.lit(value="NumPy array").alias(name="kind"),
            "n_elem",
            "ns_elem",
        )
    )
    rows: pl.DataFrame = pl.concat(items=[python, arrays]).sort(
        "representation", "n_elem"
    )
    series: list[str] = [
        "list[np.float32]",
        "list[float]",
        "NDArray[float]",
        "NDArray[np.float64]",
        "NDArray[np.float32]",
    ]
    x = alt.X(
        shorthand="n_elem:Q",
        scale=alt.Scale(type="log", nice=False),
        title="elements (log scale)",
        axis=alt.Axis(
            format="~s",
            values=[1_000, 10_000, 100_000, 1_000_000],
            labelFontSize=16,
            titleFontSize=16,
        ),
    )
    y = alt.Y(
        shorthand="ns_elem:Q",
        scale=alt.Scale(type="log", domain=[0.1, 300], nice=False),
        title="nanoseconds per element (log scale)",
        axis=alt.Axis(
            values=[0.1, 1, 10, 100], format="~g", labelFontSize=16, titleFontSize=16
        ),
    )
    color: alt.Color = _kind_color(KINDS[:2])
    lines: alt.Chart = (
        alt.Chart(data=rows)
        .mark_line(point=alt.OverlayMarkDef(size=40))
        .encode(
            x=x,
            y=y,
            color=color,
            detail="representation:N",
            tooltip=[
                alt.Tooltip(shorthand="representation:N"),
                alt.Tooltip(shorthand="n_elem:Q", title="elements", format=","),
                alt.Tooltip(shorthand="ns_elem:Q", title="ns / element", format=".3f"),
            ],
        )
    )
    ends: pl.DataFrame = rows.group_by(by="representation").agg(
        pl.all().sort_by(by="n_elem").last()
    )
    # The two list curves end close together; nudge their labels apart.
    nudge: dict[str, int] = {
        "list[np.float32]": -6,
        "list[float]": 2,
        "NDArray[float]": 6,
    }
    labels: alt.LayerChart | alt.FacetChart = alt.layer(
        *[
            alt.Chart(data=ends.filter(pl.col.representation == name))
            .mark_text(
                align="left", dx=8, dy=nudge.get(name, 0), fontSize=12, font="monospace"
            )
            .encode(x=x, y=y, text="representation:N")
            for name in series
        ]
    )
    float64: pl.DataFrame = rows.filter(
        pl.col.representation == "NDArray[np.float64]"
    ).select("n_elem", pl.col.ns_elem.alias(name="f64"))
    ratio: pl.Series = (
        rows.filter(pl.col.representation == "list[float]")
        .join(other=float64, on="n_elem")
        .select(pl.col.ns_elem / pl.col.f64)["ns_elem"]
    )
    chart: alt.FacetChart | alt.LayerChart = (
        (lines + labels)
        .properties(
            width=520,
            height=320,
            title=alt.Title(
                text=f"A list of floats stays {ratio.min():.0f}–{ratio.max():.0f}× slower than a float64 array at every size",
                subtitle="add · cold operands · result allocated on every call · Python rows: free-threaded CPython 3.14",
            ),
        )
        .configure_view(clip=False)
    )
    table: pl.DataFrame = rows.pivot(
        on="representation", index="n_elem", values="ns_elem"
    ).sort(by="n_elem")
    return Figure(chart, table, label="Table view · ns per element by size")


# ---------------------------------------------------------------------------
# Lesson 2 · fixed per-call cost
# ---------------------------------------------------------------------------

GPU_ORDER: Final[dict[str, str]] = {
    "GPU fp64": "cp.float64",
    "GPU fp32": "cp.float32",
    "GPU fp16": "cp.float16",
}


def _gpu_legend_expr() -> str:
    """Vega ``labelExpr`` mapping each GPU_ORDER key to its display label."""
    *most, (_, last_label) = GPU_ORDER.items()
    chain: str = "".join(
        f"datum.label == '{key}' ? '{label}' : " for key, label in most
    )
    return f"{chain}'{last_label}'"


def gpu_floor_extent(
    d: AnalysisData, tolerance: float = 0.15
) -> tuple[int, float, float]:
    """Return (largest flat size, min µs, max µs) of the GPU per-call floor.

    The floor extends to the largest size at which every precision is still
    within ``tolerance`` of its own minimum call time.
    """
    rows: pl.DataFrame = d.runs["gpu_dense80"].filter(ADD & OUT_COLD)
    edges: list[Any] = []
    for _, group in rows.group_by("implementation"):
        group: pl.DataFrame = group.sort(by="n_elem")
        above: pl.Series = group["us"] > group["us"].min() * (1 + tolerance)
        # The floor is the contiguous run from the smallest size up to the
        # first size that leaves it.
        edges.append(
            group["n_elem"][above.arg_true()[0] - 1]
            if above.any()
            else group["n_elem"].max()
        )
    edge: int = min(edges)
    within: pl.DataFrame = rows.filter(pl.col.n_elem <= edge)
    return edge, within["us"].min(), within["us"].max()  # ty: ignore[invalid-return-type]


def gpu_call_floor(d: AnalysisData) -> Figure:
    """Microseconds per CuPy call against size: flat until real work appears."""
    rows: pl.DataFrame = (
        d.runs["gpu_dense80"]
        .filter(ADD & OUT_COLD)
        .select("implementation", "n_elem", "us")
    )
    edge, lo, hi = gpu_floor_extent(d)
    x = alt.X(
        shorthand="n_elem:Q",
        scale=alt.Scale(type="log", nice=False),
        title="elements (log scale)",
        axis=alt.Axis(
            format="~s",
            values=[1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000],
            labelFontSize=16,
            titleFontSize=16,
        ),
    )
    band: alt.Chart = (
        alt.Chart(
            data=pl.DataFrame(
                data={"x1": [rows["n_elem"].min() * 0.9], "x2": [float(edge)]}
            )
        )
        .mark_rect(color=P["grid"], opacity=0.6)  # ty: ignore[invalid-argument-type]
        .encode(x="x1:Q", x2="x2:Q")
    )
    lines: alt.Chart = (
        alt.Chart(data=rows)
        .mark_line(point=alt.OverlayMarkDef(size=45))
        .encode(
            x=x,
            y=alt.Y(
                shorthand="us:Q",
                scale=alt.Scale(type="log"),
                title="microseconds per call (log scale)",
                axis=alt.Axis(
                    values=[1, 10, 100, 1_000, 10_000],
                    format="~g",
                    labelFontSize=16,
                    titleFontSize=16,
                ),
            ),
            color=alt.Color(
                shorthand="implementation:N",
                sort=list(GPU_ORDER),
                scale=alt.Scale(domain=list(GPU_ORDER), range=P["series"]),
                legend=alt.Legend(
                    title=None, labelExpr=_gpu_legend_expr(), labelFont="monospace"
                ),
            ),
            tooltip=[
                alt.Tooltip(shorthand="implementation:N", title="dtype"),
                alt.Tooltip(shorthand="n_elem:Q", title="elements", format=","),
                alt.Tooltip(shorthand="us:Q", title="µs / call", format=".2f"),
            ],
        )
    )
    note: alt.Chart = _note(
        x=math.sqrt(rows["n_elem"].min() * edge),  # pyrefly: ignore[bad-argument-type]
        text=f"{lo:.1f}–{hi:.1f} µs no matter the size",
        y_px=18,
        align="center",
        dx=0,
    )
    chart: alt.FacetChart | alt.LayerChart = (band + lines + note).properties(
        width=540,
        height=300,
        title=alt.Title(
            text=f"A small GPU call costs the same whether it does {rows['n_elem'].min():,} or {edge:,} elements of work",
            subtitle="CuPy add on an A100-80GB · cold operands, preallocated result · shaded = the fixed-cost floor",
        ),
    )
    table: pl.DataFrame = rows.pivot(
        on="implementation", index="n_elem", values="us"
    ).sort(by="n_elem")
    return Figure(chart, table, label="Table view · µs per call")


CROSS_SERIES: Final[list[str]] = [
    "x86 CPU, 1 core (np.float64)",
    "A100-80GB (cp.float64)",
]


def _cross_rows(d: AnalysisData) -> pl.DataFrame:
    """Matched CPU and GPU add curves for the crossover comparison."""
    return (
        pl.concat(
            items=[
                d.runs["cpu_dense"]
                .filter(OUT_COLD & ADD & (pl.col.implementation == "float64"))
                .with_columns(pl.lit(value=CROSS_SERIES[0]).alias(name="machine")),
                d.runs["gpu_dense80"]
                .filter(OUT_COLD & ADD & (pl.col.implementation == "GPU fp64"))
                .with_columns(pl.lit(value=CROSS_SERIES[1]).alias(name="machine")),
            ]
        )
        .select("machine", pl.col.n_elem.alias(name="elements"), "ns_elem", "us", "gbs")
        .sort("machine", "elements")
    )


def cpu_gpu_crossover(d: AnalysisData) -> float:
    """Problem size at which the GPU's per-element cost drops below the CPU's.

    Both curves are interpolated linearly in log-log space at the CPU's sizes;
    the first sign change of log(cpu) - log(gpu) is interpolated again.
    """
    rows: pl.DataFrame = _cross_rows(d)
    cpu: pl.DataFrame = rows.filter(pl.col.machine == CROSS_SERIES[0])
    gpu: pl.DataFrame = rows.filter(pl.col.machine == CROSS_SERIES[1])
    gx: NDArray[np.double] = np.log(gpu["elements"])
    gy: NDArray[np.double] = np.log(gpu["ns_elem"])

    def gpu_at(log_n: float) -> float:
        for i in range(len(gx) - 1):
            if gx[i] <= log_n <= gx[i + 1]:
                t = (log_n - gx[i]) / (gx[i + 1] - gx[i])
                return gy[i] + t * (gy[i + 1] - gy[i])
        raise ValueError("size outside the GPU sweep")

    points: list[tuple[float, float]] = [
        (np.log(n), np.log(ns) - gpu_at(log_n=np.log(n)))
        for n, ns in zip(cpu["elements"], cpu["ns_elem"])
        if gx[0] <= np.log(n) <= gx[-1]
    ]
    for (x0, g0), (x1, g1) in itertools.pairwise(points):
        if g0 < 0 <= g1:
            return math.exp(x0 + (x1 - x0) * (-g0) / (g1 - g0))
    raise ValueError("curves do not cross")


def cpu_vs_gpu(d: AnalysisData) -> Figure:
    """Per-element cost of the same add on one CPU core and on an A100."""
    rows: pl.DataFrame = _cross_rows(d)
    crossover: float = cpu_gpu_crossover(d)
    x = alt.X(
        shorthand="elements:Q",
        scale=alt.Scale(type="log", nice=False),
        title="elements (log scale)",
        axis=alt.Axis(
            format="~s",
            values=[1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000],
            labelFontSize=16,
            titleFontSize=16,
        ),
    )
    y = alt.Y(
        shorthand="ns_elem:Q",
        scale=alt.Scale(type="log"),
        title="nanoseconds per element (log scale)",
        axis=alt.Axis(
            values=[0.01, 0.1, 1, 10], format="~g", labelFontSize=16, titleFontSize=16
        ),
    )
    lines: alt.Chart = (
        alt.Chart(data=rows)
        .mark_line(point=alt.OverlayMarkDef(size=45))
        .encode(
            x=x,
            y=y,
            color=alt.Color(
                shorthand="machine:N",
                sort=CROSS_SERIES,
                scale=alt.Scale(domain=CROSS_SERIES, range=P["series"][:2]),
                legend=alt.Legend(title=None, labelFont="monospace"),
            ),
            tooltip=[
                alt.Tooltip(shorthand="machine:N"),
                alt.Tooltip(shorthand="elements:Q", format=","),
                alt.Tooltip(shorthand="ns_elem:Q", title="ns / element", format=".4f"),
                alt.Tooltip(shorthand="us:Q", title="µs / call", format=".2f"),
            ],
        )
    )
    chart: alt.FacetChart | alt.LayerChart = (
        lines
        + _vrule(x=crossover, dash=[4, 3])
        + _note(
            x=crossover,
            text=f"crossover ≈ {round(number=crossover, ndigits=-2):,.0f} elements",
            y_px=14,
        )
    ).properties(
        width=540,
        height=310,
        title=alt.Title(
            text=f"Below about {round(number=crossover, ndigits=-4):,.0f} elements, one CPU core beats an A100",
            subtitle="add · 64-bit float · cold operands, preallocated result · the GPU pays its fixed cost first",
        ),
    )
    return Figure(chart, table=rows, label="Table view · CPU against GPU")


# ---------------------------------------------------------------------------
# Lesson 3 · caches
# ---------------------------------------------------------------------------

FEATURED: Final[list[str]] = ["float32", "float64", "int64"]
FEATURED_DISPLAY: Final[list[str]] = [f"np.{dtype}" for dtype in FEATURED]


def cache_curve(d: AnalysisData) -> Figure:
    """Per-element add cost against size, cache-resident against streamed operands."""
    curve: pl.DataFrame = (
        d.runs["cpu_dense"]
        .filter(
            (pl.col.destination == "out")
            & ADD
            & pl.col.implementation.is_in(other=FEATURED)
        )
        .select("implementation", "n_elem", "cache_state", "ns_elem", "us")
        # Faceting groups data before the child chart's transforms run, so the
        # display label must be present in the data rather than calculated by
        # Altair below.
        .with_columns(
            pl.format("np.{}", pl.col.implementation).alias(
                name="implementation_display"
            )
        )
    )
    base: alt.Chart = (
        alt.Chart()
        .mark_line(point=alt.OverlayMarkDef(size=45))
        .encode(
            x=alt.X(
                shorthand="n_elem:Q",
                scale=alt.Scale(type="log"),
                title="elements (log scale)",
                axis=alt.Axis(
                    format="~s",
                    values=[1_000, 10_000, 100_000, 1_000_000],
                    labelFontSize=16,
                    titleFontSize=16,
                ),
            ),
            y=alt.Y(
                shorthand="ns_elem:Q",
                title="ns per element",
                scale=alt.Scale(domainMin=0),
                axis=alt.Axis(labelFontSize=16, titleFontSize=16),
            ),
            color=alt.Color(
                shorthand="cache_state:N",
                sort=["hot", "cold"],
                scale=alt.Scale(domain=["hot", "cold"], range=P["series"][:2]),
                legend=alt.Legend(
                    title=None,
                    labelExpr="datum.label == 'hot' ? 'hot: same operands every call' : 'cold: a different pair every call, from a set too big to cache'",
                    labelFontSize=15,
                ),
            ),
            tooltip=[
                alt.Tooltip(shorthand="implementation:N", title="dtype"),
                alt.Tooltip(shorthand="n_elem:Q", title="elements", format=","),
                alt.Tooltip(shorthand="cache_state:N", title="operands"),
                alt.Tooltip(shorthand="ns_elem:Q", title="ns / element", format=".3f"),
            ],
        )
    )
    chart: alt.FacetChart = (
        base.properties(width=230, height=240)
        .facet(
            column=alt.Column(
                shorthand="implementation_display:N",
                title=None,
                sort=FEATURED_DISPLAY,
                header=alt.Header(
                    labelFont="monospace",
                ),
            ),
            data=curve,
        )
        .resolve_scale(y="independent")
        .properties(
            title=alt.Title(
                text="The same add costs far more once the data no longer fits in cache",
                subtitle="NumPy add · preallocated result · each panel has its own y-scale",
            )
        )
    )
    return Figure(
        chart,
        table=curve.sort("implementation", "cache_state", "n_elem"),
        label="Table view · cache curve",
    )


# ---------------------------------------------------------------------------
# Lesson 4 · bandwidth
# ---------------------------------------------------------------------------


def cpu_bandwidth(d: AnalysisData) -> Figure:
    """Effective memory bandwidth for every op and dtype past all caches."""
    band: pl.DataFrame = (
        d.runs["cpu_dram50"]
        .filter(OUT_COLD)
        .drop_nulls(subset="gbs")
        .with_columns(
            pl.format("{} · {}", pl.col.operation, pl.col.implementation).alias(
                name="cell"
            )
        )
        .sort(by="gbs", descending=True)
    )
    plateau: float = cast(typ=float, val=band["gbs"].max())
    states: list[str] = ["at the memory limit", "short of it: limited by arithmetic"]
    band: pl.DataFrame = band.with_columns(
        pl.when(pl.col.gbs > plateau * 0.9)
        .then(statement=pl.lit(value=states[0]))
        .otherwise(statement=pl.lit(value=states[1]))
        .alias(name="state")
    )
    bars: alt.Chart = (
        alt.Chart(data=band)
        .mark_bar(size=9)
        .encode(
            y=alt.Y(
                shorthand="cell:N",
                sort="-x",
                title=None,
                axis=alt.Axis(
                    labelLimit=200,
                    labelFontSize=11,
                    labelFont="monospace",
                ),
            ),
            x=alt.X(
                shorthand="gbs:Q",
                title="Effective bandwidth, GB/s",
                axis=alt.Axis(labelFontSize=16, titleFontSize=16),
            ),
            color=alt.Color(
                shorthand="state:N",
                sort=states,
                scale=alt.Scale(domain=states, range=[P["series"][0], P["context"]]),  # ty: ignore[invalid-argument-type]
                legend=alt.Legend(title=None),
            ),
            tooltip=[
                alt.Tooltip(shorthand="cell:N"),
                alt.Tooltip(shorthand="result_dtype:N", title="computes in"),
                alt.Tooltip(shorthand="gbs:Q", title="GB/s", format=".1f"),
                alt.Tooltip(shorthand="us:Q", title="µs / call", format=",.0f"),
            ],
        )
    )
    marks: pl.DataFrame = pl.DataFrame(
        data={
            "x": plateau,
            "t": f"One CPU core: {plateau:.1f} GB/s",
        }
    )
    rules: alt.Chart = (
        alt.Chart(data=marks)
        .mark_rule(color=P["ink2"], strokeWidth=1)  # ty: ignore[invalid-argument-type]
        .encode(x=alt.X(shorthand="x:Q"))
    )
    marks: pl.DataFrame = marks.with_columns(pl.Series(name="y", values=[400.0]))
    text: alt.Chart = (
        alt.Chart(data=marks)
        .mark_text(align="right", dx=-6, fontSize=16, color=P["ink2"])  # ty: ignore[invalid-argument-type]
        .encode(x="x:Q", y=alt.Y(shorthand="y:Q", scale=None), text="t:N")
    )
    chart: alt.FacetChart | alt.LayerChart = (bars + rules + text).properties(
        width=460,
        height=440,
        title=alt.Title(
            text="Every cheap operation hits the same memory speed limit",
            subtitle="NumPy · 8 million elements, past every cache",
        ),
    )
    table: pl.DataFrame = band.select(
        "operation", "implementation", "result_dtype", "gbs", "us"
    )
    return Figure(chart, table, label="Table view · effective bandwidth")


def gpu_bandwidth_by_size(d: AnalysisData) -> Figure:
    """Share of the A100's peak memory bandwidth reached, by size and precision."""
    peak: float = d.meta["gpu_dense80"]["peak_hbm_gbs"]
    rows: pl.DataFrame = (
        d.runs["gpu_dense80"]
        .filter(OUT_COLD & ADD)
        .with_columns((pl.col.gbs / peak * 100).alias(name="pct"))
        .select("implementation", "n_elem", "us", "gbs", "pct")
    )
    chart: alt.Chart = (
        alt.Chart(data=rows)
        .mark_line(point=alt.OverlayMarkDef(size=45))
        .encode(
            x=alt.X(
                shorthand="n_elem:Q",
                scale=alt.Scale(type="log", nice=False),
                title="elements (log scale)",
                axis=alt.Axis(
                    format="~s",
                    values=[1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000],
                    labelFontSize=16,
                    titleFontSize=16,
                ),
            ),
            y=alt.Y(
                shorthand="pct:Q",
                title=f"% of the {peak:,.0f} GB/s peak",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(labelFontSize=16, titleFontSize=16),
            ),
            color=alt.Color(
                shorthand="implementation:N",
                sort=list(GPU_ORDER),
                scale=alt.Scale(domain=list(GPU_ORDER), range=P["series"]),
                legend=alt.Legend(
                    title=None, labelExpr=_gpu_legend_expr(), labelFont="monospace"
                ),
            ),
            tooltip=[
                alt.Tooltip(shorthand="implementation:N", title="dtype"),
                alt.Tooltip(shorthand="n_elem:Q", title="elements", format=","),
                alt.Tooltip(shorthand="gbs:Q", title="GB/s", format=",.0f"),
                alt.Tooltip(shorthand="pct:Q", title="% of peak", format=".1f"),
            ],
        )
        .properties(
            width=540,
            height=300,
            title=alt.Title(
                text="fp16 never gets past half of the GPU's memory bandwidth",
                subtitle="CuPy add on an A100-80GB · cold operands, preallocated result",
            ),
        )
    )
    return Figure(
        chart,
        table=rows.sort("implementation", "n_elem"),
        label="Table view · share of peak bandwidth",
    )


def gpu_ceilings(d: AnalysisData) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (cells, per-card limits) for the two-ceiling model."""
    cells: pl.DataFrame = pl.concat(
        items=[
            gpu_ceiling_rows(d.runs, metadata=d.meta, run="gpu_sweep"),
            gpu_ceiling_rows(d.runs, metadata=d.meta, run="gpu_sweep80"),
        ]
    )
    limits: pl.DataFrame = (
        cells.group_by("card")
        .agg(
            pl.col.gelem_s.filter(pl.col.gelem_s > 150).mean().alias(name="elem"),
            pl.col.gbs.max().alias(name="bw"),
        )
        .sort(by="card")
    )
    return cells, limits


def gpu_two_ceilings(d: AnalysisData) -> Figure:
    """Achieved GPU bandwidth against bytes moved per element, both A100 variants."""
    two, limits = gpu_ceilings(d)
    cards: list[str] = limits["card"].to_list()
    rate: float = cast(typ=float, val=limits["elem"].mean())
    model: pl.DataFrame = pl.DataFrame(
        data=[
            {"b_elem": b, "card": r["card"], "gbs": min(r["bw"], r["elem"] * b)}
            for r in limits.iter_rows(named=True)
            for b in range(2, 27)
        ]
    )
    # Bytes per element fixes the dtype and the number of inputs, and every op in
    # such a group lands within a few percent of the others. Draw one dot per
    # group at the median, with a rule from the slowest op to the fastest.
    groups: pl.DataFrame = (
        two.sort(by="gbs")
        .group_by("card", "b_elem")
        .agg(
            pl.col.implementation.first()
            .str.replace(pattern="GPU ", value="")
            .alias(name="dtype"),
            pl.col.operation.is_in(other=UNARY).first().alias(name="unary"),
            pl.col.gbs.median(),
            pl.col.gbs.min().alias(name="lo"),
            pl.col.gbs.max().alias(name="hi"),
            pl.col.gelem_s.median(),
            pl.format(
                "{} {}", "operation", pl.col.gbs.round(decimals=0).cast(dtype=pl.Int64)
            )
            .str.join(delimiter=" · ")
            .alias(name="ops"),
        )
        .with_columns(
            pl.when("unary")
            .then(statement=pl.lit(value="1 input"))
            .otherwise(statement=pl.lit(value="2 inputs"))
            .alias(name="inputs")
        )
    )
    scale = alt.Scale(domain=cards, range=P["series"][:2])
    x = alt.X(
        shorthand="b_elem:Q",
        title="bytes moved per element",
        scale=alt.Scale(domain=[2, 27]),
        axis=alt.Axis(labelFontSize=16, titleFontSize=16),
    )
    y = alt.Y(
        shorthand="gbs:Q",
        title="achieved bandwidth, GB/s",
        scale=alt.Scale(domain=[0, 2100]),
        axis=alt.Axis(labelFontSize=16, titleFontSize=16),
    )
    color = alt.Color(shorthand="card:N", scale=scale, legend=None)
    tips: list[Tooltip] = [
        alt.Tooltip(shorthand="card:N", title="card"),
        alt.Tooltip(shorthand="dtype:N", title="dtype"),
        alt.Tooltip(shorthand="b_elem:Q", title="bytes / element"),
        alt.Tooltip(shorthand="gbs:Q", title="median GB/s", format=",.0f"),
        alt.Tooltip(shorthand="gelem_s:Q", title="G elements/s", format=".1f"),
        alt.Tooltip(shorthand="ops:N", title="GB/s by op"),
    ]
    spread: alt.Chart = (
        alt.Chart(data=groups)
        .mark_rule(strokeWidth=2)
        .encode(
            x=x, y=alt.Y(shorthand="lo:Q"), y2=alt.Y2(shorthand="hi:Q"), color=color
        )
    )
    # The 40GB card is drawn hollow: on the ramp the two cards land on top of
    # each other, and that coincidence is the finding.
    hollow: alt.Chart = (
        alt.Chart(data=groups.filter(pl.col.card == cards[0]))
        .mark_point(size=160, filled=False, strokeWidth=2)
        .encode(x=x, y=y, color=color, tooltip=tips)
    )
    solid: alt.Chart = (
        alt.Chart(data=groups.filter(pl.col.card == cards[1]))
        .mark_point(size=85, filled=True, opacity=1)
        .encode(x=x, y=y, color=color, tooltip=tips)
    )
    # What produces each byte count, under the points it explains.
    kinds: alt.Chart = (
        alt.Chart(
            data=groups.unique(subset="b_elem").select(
                "b_elem",
                pl.concat_list("dtype", "inputs").alias(name="t"),
                pl.lit(value=170.0).alias(name="y"),
            )
        )
        .mark_text(fontSize=12, lineHeight=14, color=P["ink2"])  # ty: ignore[invalid-argument-type]
        .encode(x=x, y=alt.Y(shorthand="y:Q"), text=alt.Text(shorthand="t:N"))
    )
    dashed: alt.Chart = (
        alt.Chart(data=model)
        .mark_line(strokeWidth=1.5, strokeDash=[5, 3])
        .encode(x=x, y=y, color=alt.Color(shorthand="card:N", scale=scale, legend=None))
    )
    labels: alt.Chart = (
        alt.Chart(
            data=limits.with_columns(
                pl.lit(value=17.5).alias(name="x"), (pl.col.bw + 80).alias(name="y")
            )
        )
        .mark_text(align="left", fontSize=12.5, fontWeight=500)
        .encode(
            x="x:Q",
            y="y:Q",
            text="card:N",
            color=alt.Color(shorthand="card:N", scale=scale, legend=None),
        )
    )
    note: alt.Chart = (
        alt.Chart(
            data=pl.DataFrame(
                data=[
                    {
                        "x": 2.4,
                        "y": 2100.0,
                        "t": f"dashed: {rate:.0f} G elements/s × bytes, capped at the card's memory limit",
                    },
                    {
                        "x": 2.4,
                        "y": 1985.0,
                        "t": "dots: the median op at each byte count; bars: slowest to fastest op",
                    },
                ]
            )
        )
        .mark_text(align="left", fontSize=16, color=P["ink2"])  # ty: ignore[invalid-argument-type]
        .encode(
            x=alt.X(shorthand="x:Q"),
            y=alt.Y(shorthand="y:Q"),
            text=alt.Text(shorthand="t:N"),
        )
    )
    chart: alt.LayerChart | alt.FacetChart = (
        dashed + spread + hollow + solid + kinds + labels + note
    ).properties(
        width=540,
        height=320,
        title=alt.Title(
            text="CuPy elementwise kernels hit one of two limits: elements per second, or memory bandwidth",
            subtitle="CuPy elementwise ops · 100 million elements · hollow = A100-40GB, filled = A100-80GB",
        ),
    )
    return Figure(
        chart,
        table=two.sort("card", "b_elem", "gbs"),
        label="Table view · the two ceilings",
    )


# ---------------------------------------------------------------------------
# Lesson 5 · precision
# ---------------------------------------------------------------------------

F16_ORDER: list[str] = ["float16", "float32", "float64", "quad-sleef"]
F16_LABELS: Final[dict[str, str]] = {
    "float16": "np.float16",
    "float32": "np.float32",
    "float64": "np.float64",
    "quad-sleef": "np.float128",
}


def float16_curve(d: AnalysisData) -> Figure:
    rows: pl.DataFrame = (
        d.runs["cpu_dense"]
        .filter(OUT_HOT & ADD & pl.col.implementation.is_in(other=F16_ORDER))
        .select("implementation", "n_elem", "ns_elem", "us")
    )
    # Label each line at its own largest *measured* size, not the dataset's global max
    ends: pl.DataFrame = (
        rows.filter(pl.col.ns_elem.is_not_null())
        .filter(
            pl.col.n_elem == pl.col.n_elem.max().over(partition_by="implementation")
        )
        .with_columns(
            pl.col.implementation.replace_strict(old=F16_LABELS).alias(name="label")
        )
    )
    x = alt.X(
        shorthand="n_elem:Q",
        scale=alt.Scale(type="log", nice=False),
        title="elements (log scale)",
        axis=alt.Axis(
            format="~s",
            values=[1_000, 10_000, 100_000, 1_000_000],
            labelFontSize=16,
            titleFontSize=16,
        ),
    )
    y = alt.Y(
        shorthand="ns_elem:Q",
        scale=alt.Scale(type="log"),
        title="ns per element (log scale)",
        axis=alt.Axis(
            values=[0.1, 0.3, 1, 3, 10], labelFontSize=16, titleFontSize=16, format="~g"
        ),
    )
    color = alt.Color(
        shorthand="implementation:N",
        sort=F16_ORDER,
        legend=None,
        scale=alt.Scale(
            domain=F16_ORDER,
            range=[P["series"][i] for i in (2, 3, 1, 0)],
        ),
    )
    lines: alt.Chart = (
        alt.Chart(data=rows).mark_line(strokeWidth=2.2).encode(x=x, y=y, color=color)
    )
    dots: alt.Chart = lines.mark_point(size=55).encode(
        tooltip=[
            alt.Tooltip(shorthand="implementation:N", title="dtype"),
            alt.Tooltip(shorthand="n_elem:Q", title="elements", format=","),
            alt.Tooltip(shorthand="ns_elem:Q", title="ns / element", format=".3f"),
        ]
    )
    labels: alt.Chart = (
        alt.Chart(data=ends)
        .mark_text(align="left", dx=10, fontSize=12.5, fontWeight=500)
        .encode(x=x, y=y, color=color, text="label:N")
    )
    chart: alt.LayerChart | alt.FacetChart = (
        (lines + dots + labels)
        .properties(
            width=480,
            height=300,
            # Room for the end-of-line labels, which sit right of the plot area
            padding={"left": 6, "right": 100, "top": 6, "bottom": 6},
            title=alt.Title(
                text="On CPU, float16 never gets cheaper with size and stays far slower than float32",
                subtitle="NumPy add · same operands every call, preallocated result",
            ),
        )
        .configure_view(clip=False)
    )
    table: pl.DataFrame = rows.pivot(
        on="implementation", index="n_elem", values="ns_elem"
    ).sort(by="n_elem")
    return Figure(
        chart, table, label="Table view · float16 against float32, float64, quad-sleef"
    )


def float16_mechanism(d: AnalysisData) -> Figure:
    """Native float16 ufuncs against widening to float32 by hand."""
    paths: list[str] = ["native ufunc", "widened by hand"]
    mech: pl.DataFrame = (
        d.runs["cpu_f16"]
        .filter(
            (pl.col.destination == "alloc")
            & (pl.col.cache_state == "hot")
            & (pl.col.n_elem == 256_000)
        )
        .with_columns(
            pl.col.operation.str.replace(pattern="_via_f32", value="").alias(name="op"),
            pl.when(pl.col.operation.str.contains(pattern="via_f32"))
            .then(statement=pl.lit(value=paths[1]))
            .otherwise(statement=pl.lit(value=paths[0]))
            .alias(name="path"),
        )
        .select("op", "implementation", "path", "ns_elem")
    )
    span: pl.DataFrame = mech.pivot(
        on="path", index=["op", "implementation"], values="ns_elem"
    ).rename(mapping={paths[0]: "native", paths[1]: "via"})
    # One dataset for both layers: a faceted layer cannot mix sources.
    mech: pl.DataFrame = mech.join(other=span, on=["op", "implementation"])
    scale = alt.Scale(type="log", domain=[0.1, 12], nice=False)
    axis = alt.Axis(
        values=[0.1, 0.3, 1, 3, 10], format="~g", labelFontSize=16, titleFontSize=16
    )
    y = alt.Y(shorthand="op:N", title=None, sort=["add", "mul", "exp"])
    # A dumbbell, not bars: the values span 40×, so the axis is log.
    rule: alt.Chart = (
        alt.Chart()
        .mark_rule(color=P["context"], strokeWidth=2.5)  # ty: ignore[invalid-argument-type]
        .encode(
            x=alt.X(
                shorthand="native:Q",
                scale=scale,
                title="ns per element (log scale)",
                axis=axis,
            ),
            x2="via:Q",
            y=y,
        )
    )
    dots: alt.Chart = (
        alt.Chart()
        .mark_point(size=100)
        .encode(
            x=alt.X(
                shorthand="ns_elem:Q",
                scale=scale,
                title="ns per element (log scale)",
                axis=axis,
            ),
            y=y,
            color=alt.Color(
                shorthand="path:N",
                sort=paths,
                scale=alt.Scale(domain=paths, range=P["series"][:2]),
                legend=alt.Legend(title=None, labelFontSize=16, titleFontSize=16),
            ),
            tooltip=[
                alt.Tooltip(shorthand="op:N"),
                alt.Tooltip(shorthand="implementation:N", title="dtype"),
                alt.Tooltip(shorthand="path:N"),
                alt.Tooltip(shorthand="ns_elem:Q", title="ns / element", format=".3f"),
            ],
        )
    )
    chart: alt.LayerChart | alt.FacetChart = (
        alt.layer(rule, dots)
        .properties(width=235, height=150)
        .facet(
            column=alt.Column(
                shorthand="implementation:N",
                title=None,
                sort=F16_ORDER,
                header=alt.Header(labelFont="monospace"),
            ),
            data=mech,
        )
        .properties(
            title=alt.Title(
                text="Converting to float32 by hand costs the same as NumPy's float16: the conversion is the cost",
                subtitle="256,000 elements · 'widened by hand' = astype(float32), operate, astype back",
            )
        )
    )
    return Figure(
        chart,
        table=span.sort("implementation", "op"),
        label="Table view · native against hand-widened",
    )


DTYPE_NAMED: Final[list[str]] = [
    "np.float128 (software, 128-bit)",
    "np.float96 (x87, 80-bit)",
    "np.float32 (hardware)",
    "np.float16 (converted, no ALU)",
]
FLOATS: Final[list[str]] = [
    "float16",
    "float32",
    "float64",
    "longdouble64",
    "quad-sleef",
]
INT_BITS: Final[list[int]] = [8, 16, 32, 64]
# Integer ops in three groups: arithmetic NumPy does in the integer type, ops
# it converts to a float first, and the 2×2 matrix products.
INT_OP_ORDER: Final[list[str]] = [
    "add",
    "mul",
    "muladd",
    "muladd_accum",
    "div",
    "sqrt",
    "exp",
    "cos",
    "matmul_explicit",
    "matmul",
]


def vs_float64(d: AnalysisData) -> pl.DataFrame:
    """Every dtype's cost per element at 256,000 elements, relative to float64."""
    sweep: pl.DataFrame = d.runs["cpu_sweep"].filter(OUT_HOT & (pl.col.n == 64_000))
    f64: pl.DataFrame = sweep.filter(pl.col.implementation == "float64").select(
        "operation", pl.col.ns_elem.alias(name="base")
    )
    return (
        sweep.join(other=f64, on="operation")
        .with_columns((pl.col.ns_elem / pl.col.base).alias(name="vs_f64"))
        .drop_nulls(subset="vs_f64")
        .select("operation", "implementation", "result_dtype", "vs_f64", "ns_elem")
    )


def _vs_f64_x(values: list[float], domain: list[float]) -> alt.X:
    """The shared log x-axis of the dtype grids, with 1× named and emphasized."""
    unit = "datum.value == 1"
    return alt.X(
        shorthand="vs_f64:Q",
        scale=alt.Scale(type="log", domain=domain, nice=False),
        title="cost relative to float64, same operation (log scale)",
        axis=alt.Axis(
            values=values,
            labelExpr=f"format(datum.value, '.3~g') + '×' + ({unit} ? ' = float64' : '')",
            labelFontWeight=alt.condition(
                predicate=unit,
                if_true=alt.value(value=700),
                if_false=alt.value(value=400),
            ),
            labelColor=alt.condition(
                predicate=unit,
                if_true=alt.value(value=P["ink"]),
                if_false=alt.value(value=P["ink2"]),
            ),
            labelFontSize=16,
            titleFontSize=16,
        ),
    )


def _float64_rule() -> alt.Chart:
    """float64 itself: the reference every dot in a dtype grid is measured against."""
    return (
        alt.Chart()
        .mark_rule(color=P["ink"], strokeWidth=1.5)  # ty: ignore[invalid-argument-type]
        .encode(x=alt.datum(1.0))
    )


def dtype_grid(d: AnalysisData) -> Figure:
    """Every op's cost for every float dtype, relative to float64."""
    family: pl.Expr = pl.col.implementation.replace_strict(
        old={
            "quad-sleef": DTYPE_NAMED[0],
            "longdouble64": DTYPE_NAMED[1],
            "float32": DTYPE_NAMED[2],
            "float16": DTYPE_NAMED[3],
        }
    )
    tax: pl.DataFrame = (
        vs_float64(d)
        .filter(
            pl.col.implementation.is_in(other=FLOATS)
            & (pl.col.implementation != "float64")
        )
        .with_columns(family.alias(name="family"))
    )
    x: X = _vs_f64_x(values=[0.25, 1, 4, 16, 64, 256], domain=[0.1, 150])
    y = alt.Y(
        shorthand="operation:N",
        sort=OP_ORDER,
        title=None,
        axis=alt.Axis(labelFont="monospace"),
    )
    dots: alt.Chart = (
        alt.Chart(data=tax)
        .mark_point(size=140)
        .encode(
            x=x,
            y=y,
            color=alt.Color(
                shorthand="family:N",
                sort=DTYPE_NAMED,
                scale=alt.Scale(
                    domain=DTYPE_NAMED,
                    # float16 keeps the slot it has in Lesson 5's other charts.
                    range=[P["series"][i] for i in (0, 1, 3, 2)],
                ),
                legend=alt.Legend(
                    title=None, columns=1, labelFontSize=16, titleFontSize=16
                ),
            ),
            tooltip=[
                alt.Tooltip(shorthand="implementation:N", title="dtype"),
                alt.Tooltip(shorthand="vs_f64:Q", title="× float64", format=".1f"),
                alt.Tooltip(shorthand="ns_elem:Q", title="ns / element", format=".2f"),
            ],
        )
    )
    quad_labels: alt.Chart = (
        alt.Chart(data=tax.filter(pl.col.implementation == "quad-sleef"))
        .mark_text(dy=-13, fontSize=11.5, color=P["ink2"])  # ty: ignore[invalid-argument-type]
        .encode(x=x, y=y, text=alt.Text(shorthand="vs_f64:Q", format=".0f"))
    )
    chart: alt.LayerChart | alt.FacetChart = (
        (_float64_rule() + dots + quad_labels)
        .properties(
            width=540,
            height=320,
            title=alt.Title(
                text="float32 stays near float64; the three floats this CPU can't compute directly don't",
                subtitle="NumPy · 256,000 elements",
            ),
        )
        .configure_view(clip=False)
    )
    return Figure(
        chart,
        table=tax.sort("operation", "implementation").drop("result_dtype"),
        label="Table view · every float dtype against float64",
    )


def int_grid(d: AnalysisData) -> Figure:
    """Every op's cost for every signed integer dtype, relative to float64."""
    ints: pl.DataFrame = (
        vs_float64(d)
        .filter(pl.col.implementation.str.contains(pattern="int"))
        .with_columns(
            pl.col.implementation.str.extract(pattern=r"(\d+)")
            .cast(dtype=pl.Int64)
            .alias(name="bits"),
            pl.col.implementation.str.starts_with(prefix="u").alias(name="unsigned"),
        )
        .with_columns(pl.format("int{}", "bits").alias(name="width"))
    )
    signed: pl.DataFrame = ints.filter(~pl.col.unsigned)
    # Unsigned types track their signed twins except on div, so only signed are drawn.
    twins: pl.DataFrame = signed.join(
        other=ints.filter(pl.col.unsigned), on=["operation", "bits"], suffix="_u"
    ).filter(pl.col.operation != "div")
    twin_gap: float = cast(
        typ=float,
        val=((pl.col.vs_f64_u / pl.col.vs_f64) - 1)
        .abs()
        .max()
        .pipe(function=twins.select)
        .item(),
    )
    widths: list[str] = [f"int{b}" for b in INT_BITS]
    ramp: list[str] = [P["ramp"][1], P["ramp"][3], P["ramp"][5], P["ramp"][6]]
    x: X = _vs_f64_x(values=[0.125, 0.25, 0.5, 1, 2, 4, 8], domain=[1 / 16, 10])
    y = alt.Y(
        shorthand="operation:N",
        sort=INT_OP_ORDER,
        title=None,
        axis=alt.Axis(labelFont="monospace"),
    )
    dots: alt.Chart = (
        alt.Chart(data=signed)
        .mark_point(size=110, filled=True, opacity=1)
        .encode(
            x=x,
            y=y,
            color=alt.Color(
                shorthand="width:N",
                sort=widths,
                scale=alt.Scale(domain=widths, range=ramp),
                legend=alt.Legend(title=None, orient="top", labelFontSize=16),
            ),
            tooltip=[
                alt.Tooltip(shorthand="implementation:N", title="dtype"),
                alt.Tooltip(shorthand="result_dtype:N", title="computed in"),
                alt.Tooltip(shorthand="vs_f64:Q", title="× float64", format=".2f"),
                alt.Tooltip(shorthand="ns_elem:Q", title="ns / element", format=".2f"),
            ],
        )
    )
    chart: alt.LayerChart | alt.FacetChart = (
        (_float64_rule() + dots)
        .properties(
            width=540,
            height=320,
            title=alt.Title(
                text="Integers cost what their bytes cost, until NumPy converts them to a float",
                subtitle=f"NumPy · 256,000 elements · signed types (unsigned are within {twin_gap:.0%},"
                " except on div) · right: the type NumPy computes in",
            ),
        )
        .configure_view(clip=False)
    )
    return Figure(
        chart,
        table=ints.sort("operation", "bits", "unsigned").select(
            "operation", "implementation", "result_dtype", "vs_f64", "ns_elem"
        ),
        label="Table view · every integer dtype against float64",
    )


GPU_PAIRS: list[str] = ["cp.float64/cp.float32", "cp.float32/cp.float16"]


def gpu_precision_table(d: AnalysisData) -> pl.DataFrame:
    """Time ratios between adjacent GPU precisions at 100 million elements."""
    return (
        d.runs["gpu_sweep80"]
        .filter(OUT_HOT & (pl.col.n == 25_000_000))
        .pivot(on="implementation", index="operation", values="us")
        .drop_nulls()
        .with_columns(
            (pl.col(name="GPU fp64") / pl.col(name="GPU fp32")).alias(
                name=GPU_PAIRS[0]
            ),
            (pl.col(name="GPU fp32") / pl.col(name="GPU fp16")).alias(
                name=GPU_PAIRS[1]
            ),
        )
        .sort(by=GPU_PAIRS[1], descending=True)
    )


def gpu_precision_ratios(d: AnalysisData) -> Figure:
    """How much halving precision actually saves on the GPU, per operation."""
    ratios: DataFrame = gpu_precision_table(d)
    long: DataFrame = ratios.unpivot(
        on=GPU_PAIRS, index="operation", variable_name="pair", value_name="ratio"
    )
    bars: Chart = (
        alt.Chart(data=long)
        .mark_bar(size=10)
        .encode(
            y=alt.Y(
                shorthand="operation:N",
                title=None,
                sort=alt.EncodingSortField(field="ratio", op="max", order="descending"),
                axis=alt.Axis(labelFont="monospace"),
            ),
            yOffset=alt.YOffset(shorthand="pair:N", sort=GPU_PAIRS),
            x=alt.X(
                shorthand="ratio:Q",
                title="time ratio (2.0 = halving precision halves the time)",
                scale=alt.Scale(domain=[0, 2.3], nice=False),
                axis=alt.Axis(labelFontSize=16, titleFontSize=16),
            ),
            color=alt.Color(
                shorthand="pair:N",
                sort=GPU_PAIRS,
                scale=alt.Scale(domain=GPU_PAIRS, range=P["series"][:2]),
                legend=alt.Legend(title=None, labelFont="monospace"),
            ),
            tooltip=[
                alt.Tooltip(shorthand="operation:N", title="op"),
                alt.Tooltip(shorthand="pair:N", title="ratio"),
                alt.Tooltip(shorthand="ratio:Q", format=".2f"),
            ],
        )
    )
    values: Chart = bars.mark_text(align="left", dx=4, fontSize=11).encode(
        text=alt.Text(shorthand="ratio:Q", format=".2f"),
        color=alt.value(value=P["ink2"]),
    )
    chart: FacetChart | LayerChart = (bars + values + _vrule(x=2.0)).properties(
        width=440,
        height=340,
        title=alt.Title(
            text="On the GPU, halving precision rarely halves the time",
            subtitle="CuPy on an A100-80GB · 100 million elements · same operands every call",
        ),
    )
    return Figure(chart, table=ratios, label="Table view · precision ratios")


# ---------------------------------------------------------------------------
# Lesson 6 · allocation
# ---------------------------------------------------------------------------

COMPONENTS: list[str] = [
    "cold destination (out_pool − out)",
    "allocation (alloc − out_pool)",
]


def destination_decomposition(d: AnalysisData) -> Figure:
    """Split the cost of a fresh result buffer into cache and allocation parts."""
    three: DataFrame = (
        d.runs["cpu_alloc"]
        .filter((pl.col.cache_state == "cold") & ADD)
        .pivot(on="destination", index=["implementation", "n_elem"], values="us")
        .with_columns(
            ((pl.col.out_pool - pl.col.out) / pl.col.out * 100).alias(
                name=COMPONENTS[0]
            ),
            ((pl.col.alloc - pl.col.out_pool) / pl.col.out * 100).alias(
                name=COMPONENTS[1]
            ),
        )
        .unpivot(
            on=COMPONENTS,
            index=["implementation", "n_elem"],
            variable_name="component",
            value_name="pct",
        )
        .sort(by=["implementation", "n_elem"])
    )
    zero: Chart = (
        alt.Chart(data=pl.DataFrame(data={"y": [0.0]}))
        .mark_rule(color=P["ink2"])  # ty: ignore[invalid-argument-type]
        .encode(y="y:Q")
    )
    lines: Chart = (
        alt.Chart()
        .mark_line(point=alt.OverlayMarkDef(size=50))
        .encode(
            x=alt.X(
                shorthand="n_elem:Q",
                scale=alt.Scale(type="log"),
                title="elements (log scale)",
                axis=alt.Axis(
                    format="~s",
                    values=[10_000, 100_000, 1_000_000],
                    labelFontSize=16,
                    titleFontSize=16,
                ),
            ),
            y=alt.Y(
                shorthand="pct:Q",
                title="% of the preallocated time",
                axis=alt.Axis(labelFontSize=16, titleFontSize=16),
            ),
            color=alt.Color(
                shorthand="component:N",
                sort=COMPONENTS,
                scale=alt.Scale(domain=COMPONENTS, range=P["series"][:2]),
                legend=alt.Legend(title=None, labelFontSize=16, titleFontSize=16),
            ),
            tooltip=[
                alt.Tooltip(shorthand="implementation:N", title="dtype"),
                alt.Tooltip(shorthand="n_elem:Q", title="elements", format=","),
                alt.Tooltip(shorthand="component:N"),
                alt.Tooltip(shorthand="pct:Q", title="% of out=", format=".1f"),
            ],
        )
    )
    chart: FacetChart = (
        alt.layer(zero, lines)
        .properties(width=215, height=230)
        .facet(
            column=alt.Column(
                shorthand="implementation:N",
                title=None,
                sort=["float32", "float64", "int32", "int64"],
                header=alt.Header(labelFont="monospace"),
            ),
            data=three,
        )
        .properties(
            title=alt.Title(
                text="A result buffer that has left the cache costs extra; allocating a new one is free until the largest size",
                subtitle=[
                    "NumPy add · cold operands · out_pool = rotate through preallocated buffers too big to stay cached",
                    "below 32 MiB a fresh allocation reuses the block just freed, which is still in cache",
                ],
            )
        )
    )
    return Figure(
        chart, table=three, label="Table view · destination cost decomposition"
    )


SIDES: list[str] = ["at or below 32 MB", "above 32 MB"]
MMAP_THRESHOLD_MIB = 32


def allocation_cliff_rows(d: AnalysisData) -> pl.DataFrame:
    """Allocating-form penalty for every pooled cell, tagged by threshold side."""
    return pl.concat(
        [
            allocation_penalty(d.runs, run="cpu_dense"),
            allocation_penalty(d.runs, run="cpu_alloc"),
            allocation_penalty(
                d.runs,
                run="cpu_dram50",
                operations=("add", "mul", "div", "sqrt", "exp"),
            ),
        ]
    ).with_columns(
        pl.when(pl.col.result_mib > MMAP_THRESHOLD_MIB)
        .then(statement=pl.lit(value=SIDES[1]))
        .otherwise(statement=pl.lit(value=SIDES[0]))
        .alias(name="side")
    )


def allocation_cliff(d: AnalysisData) -> Figure:
    """Allocation penalty against result-buffer size, across three runs."""
    cliff: DataFrame = allocation_cliff_rows(d)
    zero: Chart = (
        alt.Chart(data=pl.DataFrame(data={"y": [0.0]}))
        .mark_rule(color=P["grid"])  # ty: ignore[invalid-argument-type]
        .encode(y="y:Q")
    )
    dots: Chart = (
        alt.Chart(data=cliff)
        .mark_point(size=95, opacity=0.85)
        .encode(
            x=alt.X(
                shorthand="result_mib:Q",
                scale=alt.Scale(type="log"),
                title="result buffer size, MB (log scale)",
                axis=alt.Axis(
                    values=[0.05, 0.5, 5, 32, 61],
                    format="~g",
                    labelFontSize=16,
                    titleFontSize=16,
                ),
            ),
            y=alt.Y(
                shorthand="pct:Q",
                title="allocating form, % slower than preallocated",
                axis=alt.Axis(labelFontSize=16, titleFontSize=16),
            ),
            color=alt.Color(
                shorthand="side:N",
                sort=SIDES,
                scale=alt.Scale(domain=SIDES, range=[P["context"], P["series"][1]]),  # ty: ignore[invalid-argument-type]
                legend=alt.Legend(title=None, labelFontSize=16, titleFontSize=16),
            ),
            tooltip=[
                alt.Tooltip(shorthand="run:N"),
                alt.Tooltip(shorthand="operation:N", title="op"),
                alt.Tooltip(shorthand="implementation:N", title="dtype"),
                alt.Tooltip(shorthand="result_mib:Q", title="result MiB", format=".2f"),
                alt.Tooltip(shorthand="pct:Q", title="% vs out=", format=".1f"),
            ],
        )
    )
    threshold: LayerChart | FacetChart = _vrule(
        x=float(MMAP_THRESHOLD_MIB), dash=[4, 3]
    ) + _note(
        x=float(MMAP_THRESHOLD_MIB),
        text="glibc mmap threshold, 32 MB",
        y_px=12,
        align="right",
        dx=-7,
    )
    chart: FacetChart | LayerChart = (zero + dots + threshold).properties(
        width=520,
        height=300,
        title=alt.Title(
            text="Allocating a result is free up to 32 MB, then suddenly is not",
            subtitle="NumPy · cold operands · three runs pooled · above the threshold every call maps and unmaps fresh memory",
        ),
    )
    return Figure(
        chart,
        table=cliff.sort(by="result_mib"),
        label="Table view · allocation penalty by buffer size",
    )


# ---------------------------------------------------------------------------
# Lesson 7 · compilation
# ---------------------------------------------------------------------------

FRAMEWORK_SERIES: Final[list[str]] = [
    "JAX JIT · host pipeline",
    "CuPy alloc · CUDA events",
]
COMMON_GPU_OPS: Final[list[str]] = ["add", "exp", "matmul"]


def framework_endpoint(d: AnalysisData) -> pl.DataFrame:
    """CuPy and JAX fp32 per-element cost at the largest matched size."""
    rows: DataFrame = framework_rows(d)
    return (
        rows.filter(pl.col.n_elem == rows["n_elem"].max())
        .pivot(on="series", index="operation", values="ns_elem")
        .with_columns(
            (pl.col(name=FRAMEWORK_SERIES[1]) / pl.col(name=FRAMEWORK_SERIES[0])).alias(
                name="CuPy / JAX time"
            )
        )
        .sort(
            by=pl.col.operation.replace_strict(
                old={op: i for i, op in enumerate(iterable=COMMON_GPU_OPS)}
            )
        )
    )


def framework_rows(d: AnalysisData) -> pl.DataFrame:
    jax: DataFrame = (
        d.runs["jax_dense"]
        .filter(
            (pl.col.destination == "jit")
            & (pl.col.cache_state == "cold")
            & (pl.col.implementation == "JAX fp32")
            & pl.col.operation.is_in(other=COMMON_GPU_OPS)
        )
        .select("operation", "n_elem", "us", "ns_elem", "samples")
        .with_columns(pl.lit(value=FRAMEWORK_SERIES[0]).alias(name="series"))
    )
    cupy: DataFrame = (
        d.runs["gpu_dense80"]
        .filter(
            ALLOC_COLD
            & (pl.col.implementation == "GPU fp32")
            & pl.col.operation.is_in(other=COMMON_GPU_OPS)
        )
        .select("operation", "n_elem", "us", "ns_elem", "samples")
        .with_columns(pl.lit(value=FRAMEWORK_SERIES[1]).alias(name="series"))
    )
    return pl.concat(items=[jax, cupy])


def jax_vs_cupy(d: AnalysisData) -> Figure:
    """Matched JAX and CuPy fp32 execution cost across size."""
    rows: DataFrame = framework_rows(d)
    chart: FacetChart = (
        alt.Chart(data=rows)
        .mark_line(point=True, strokeWidth=2.1)
        .encode(
            x=alt.X(
                shorthand="n_elem:Q",
                scale=alt.Scale(type="log", nice=False),
                title="elements (log scale)",
                axis=alt.Axis(
                    format="~s",
                    values=[1_000, 100_000, 10_000_000],
                    labelFontSize=16,
                    titleFontSize=16,
                ),
            ),
            y=alt.Y(
                shorthand="ns_elem:Q",
                scale=alt.Scale(type="log"),
                title="ns per element (log scale)",
                axis=alt.Axis(
                    values=[0.001, 0.01, 0.1, 1, 10, 100],
                    format="~g",
                    labelFontSize=16,
                    titleFontSize=16,
                ),
            ),
            color=alt.Color(
                shorthand="series:N",
                sort=FRAMEWORK_SERIES,
                scale=alt.Scale(domain=FRAMEWORK_SERIES, range=P["series"][:2]),
                legend=alt.Legend(title=None, labelFontSize=16, titleFontSize=16),
            ),
            tooltip=[
                alt.Tooltip(shorthand="operation:N"),
                alt.Tooltip(shorthand="series:N", title="framework and timer"),
                alt.Tooltip(shorthand="n_elem:Q", title="elements", format=","),
                alt.Tooltip(shorthand="us:Q", title="µs / call", format=".2f"),
                alt.Tooltip(shorthand="ns_elem:Q", title="ns / element", format=".4f"),
            ],
        )
        .properties(width=215, height=230)
        .facet(
            column=alt.Column(
                shorthand="operation:N",
                sort=COMMON_GPU_OPS,
                title=None,
                header=alt.Header(labelFont="monospace"),
            )
        )
        .properties(
            title=alt.Title(
                text="JAX pays more per call, then matches or beats CuPy once arrays are large",
                subtitle="fp32 on an A100-80GB · cold operands · result allocated every call",
            )
        )
    )
    return Figure(
        chart, table=framework_endpoint(d), label="Table view · largest matched size"
    )


FUSION_OPS: Final[list[str]] = ["add", "muladd", "chain", "matmul_explicit"]
JAX_DTYPES: Final[list[str]] = ["JAX fp16", "JAX bf16", "JAX fp32", "JAX fp64"]


def fusion_table(d: AnalysisData) -> pl.DataFrame:
    """Eager-to-JIT time ratios for the fusion comparison."""
    return (
        d.runs["jax_sweep"]
        .filter(
            (pl.col.n == 25_000_000)
            & (pl.col.cache_state == "hot")
            & pl.col.operation.is_in(other=FUSION_OPS)
            & pl.col.implementation.is_in(other=JAX_DTYPES)
        )
        .pivot(on="destination", index=["operation", "implementation"], values="us")
        .with_columns((pl.col.eager / pl.col.jit).alias(name="JIT speedup"))
        .sort(
            pl.col.implementation.replace_strict(
                old={k: i for i, k in enumerate(iterable=JAX_DTYPES)}
            ),
            pl.col.operation.replace_strict(
                old={k: i for i, k in enumerate(iterable=FUSION_OPS)}
            ),
        )
    )


def jit_fusion(d: AnalysisData) -> Figure:
    """How much compiling the whole expression saves, per op and dtype."""
    fusion: DataFrame = fusion_table(d)
    fused: Series = fusion.filter(pl.col.operation != "add")["JIT speedup"]
    points: Chart = (
        alt.Chart()
        .mark_point(size=110, color=P["series"][0])
        .encode(
            x=alt.X(
                shorthand="JIT speedup:Q",
                scale=alt.Scale(type="log", domain=[0.8, 128]),
                title="eager ÷ JIT time (log)",
                axis=alt.Axis(
                    values=[1, 4, 16, 64],
                    format="~g",
                    labelFontSize=16,
                    titleFontSize=16,
                ),
            ),
            y=alt.Y(
                shorthand="operation:N",
                sort=FUSION_OPS,
                title=None,
                axis=alt.Axis(labelFont="monospace"),
            ),
            tooltip=[
                alt.Tooltip(shorthand="implementation:N", title="dtype"),
                alt.Tooltip(shorthand="operation:N"),
                alt.Tooltip(shorthand="eager:Q", title="eager µs", format=",.1f"),
                alt.Tooltip(shorthand="jit:Q", title="JIT µs", format=",.1f"),
                alt.Tooltip(shorthand="JIT speedup:Q", title="speedup", format=".2f"),
            ],
        )
    )
    unity: Chart = alt.Chart().mark_rule(color=P["ink2"]).encode(x=alt.datum(datum=1.0))  # ty: ignore[invalid-argument-type]
    chart: FacetChart = (
        alt.layer(unity, points)
        .properties(width=185, height=150)
        .facet(
            column=alt.Column(
                shorthand="implementation:N",
                sort=JAX_DTYPES,
                title=None,
                header=alt.Header(labelFont="monospace"),
            ),
            data=fusion,
        )
        .properties(
            title=alt.Title(
                text=f"Compiling a whole expression is {fused.min():.0f}–{fused.max():.0f}× faster than running it step by step",
                subtitle="JAX on an A100-80GB · 100 million elements · add is the control: one step, nothing to fuse",
            )
        )
    )
    return Figure(chart, table=fusion, label="Table view · JIT speedups")


CPU_OPS: Final[list[str]] = ["add", "exp", "chain", "matmul", "matmul_explicit"]


def cpu_framework_table(d: AnalysisData) -> pl.DataFrame:
    """NumPy eager against JAX JIT on the same CPU node."""
    n: int = max(d.meta["jax_cpu"]["sizes"])
    return (
        d.runs["jax_cpu"]
        .filter(
            (pl.col.n == n)
            & (pl.col.cache_state == "hot")
            & pl.col.operation.is_in(other=CPU_OPS)
            & (
                ((pl.col.implementation == "JAX fp32") & (pl.col.destination == "jit"))
                | (
                    (pl.col.implementation == "NumPy fp32")
                    & (pl.col.destination == "eager")
                )
            )
        )
        .with_columns(
            pl.when(pl.col.implementation == "JAX fp32")
            .then(statement=pl.lit(value="JAX JIT"))
            .otherwise(statement=pl.lit(value="NumPy eager"))
            .alias(name="runtime")
        )
        .pivot(on="runtime", index="operation", values="us")
        .with_columns(
            (pl.col(name="NumPy eager") / pl.col(name="JAX JIT")).alias(
                name="NumPy / JAX time"
            )
        )
        .sort(
            by=pl.col.operation.replace_strict(
                old={op: i for i, op in enumerate(iterable=CPU_OPS)}
            )
        )
    )


def jax_vs_numpy_cpu(d: AnalysisData) -> Figure:
    """Same-node NumPy and JAX CPU time ratios."""
    table: DataFrame = cpu_framework_table(d)
    points: Chart = (
        alt.Chart(data=table)
        .mark_point(size=120, color=P["series"][0])
        .encode(
            x=alt.X(
                shorthand="NumPy / JAX time:Q",
                scale=alt.Scale(type="log"),
                title="NumPy time ÷ JAX time (log scale)",
                axis=alt.Axis(
                    values=[1, 2, 5, 10, 20, 50],
                    format="~g",
                    labelFontSize=16,
                    titleFontSize=16,
                ),
            ),
            y=alt.Y(
                shorthand="operation:N",
                sort=CPU_OPS,
                title=None,
                axis=alt.Axis(labelFont="monospace"),
            ),
            tooltip=[
                alt.Tooltip(shorthand="operation:N"),
                alt.Tooltip(shorthand="NumPy eager:Q", title="NumPy µs", format=",.1f"),
                alt.Tooltip(shorthand="JAX JIT:Q", title="JAX µs", format=",.1f"),
                alt.Tooltip(
                    shorthand="NumPy / JAX time:Q", title="ratio", format=".2f"
                ),
            ],
        )
    )
    labels: Chart = points.mark_text(
        align="left",
        dx=10,
        fontSize=16,
        color=P["ink2"],  # ty: ignore[invalid-argument-type]
    ).encode(text=alt.Text(shorthand="NumPy / JAX time:Q", format=".1f"))
    chart: FacetChart | LayerChart = (_vrule(x=1.0) + points + labels).properties(
        width=500,
        height=220,
        title=alt.Title(
            text="On the same CPU, compiled JAX beats NumPy on every operation measured",
            subtitle="10 million elements · fp32 · same operands every call · thread pools not constrained",
        ),
    )
    return Figure(chart, table, label="Table view · same-node CPU comparison")


# ---------------------------------------------------------------------------
# Lesson 8 · measurement
# ---------------------------------------------------------------------------

TIMER_SERIES: Final[list[str]] = ["JAX pipelined", "JAX blocking", "CuPy host timer"]
LATENCY_OPS: Final[list[str]] = ["add", "exp"]


def timer_rows(d: AnalysisData) -> pl.DataFrame:
    """JAX pipelined, JAX blocking, and CuPy host-timed fp32 calls at matched sizes."""
    sizes: Series = d.meta["jax_latency"]["sizes"]
    return pl.concat(
        items=[
            jax_timer_series(
                d.runs, "jax_dense", TIMER_SERIES[0], sizes, operations=LATENCY_OPS
            ),
            jax_timer_series(
                d.runs, "jax_latency", TIMER_SERIES[1], sizes, operations=LATENCY_OPS
            ),
            d.runs["gpu_host80"]
            .filter(
                ALLOC_COLD
                & (pl.col.implementation == "GPU fp32")
                & pl.col.operation.is_in(other=LATENCY_OPS)
            )
            .select("operation", "n_elem", "us", "ns_elem", "samples")
            .with_columns(pl.lit(value=TIMER_SERIES[2]).alias(name="series")),
        ]
    )


def timer_floor(d: AnalysisData) -> pl.DataFrame:
    """Smallest-size call times under each protocol."""
    rows: DataFrame = timer_rows(d)
    return (
        rows.filter(pl.col.n_elem == rows["n_elem"].min())
        .pivot(on="series", index="operation", values="us")
        .with_columns(
            (pl.col.JAX_blocking / pl.col.JAX_pipelined).alias(
                name="blocking / pipelined"
            )
        )
        .sort(
            by=pl.col.operation.replace_strict(
                old={op: i for i, op in enumerate(iterable=LATENCY_OPS)}
            )
        )
    )
