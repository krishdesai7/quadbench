"""Visual language for the report: palette, Altair theme, and display helpers.

Importing this module registers and enables the ``quadbench`` Altair theme, so
every chart in the notebook shares one look without per-chart configuration.
"""

from __future__ import annotations

from typing import NamedTuple, cast

import altair as alt
import polars as pl
from IPython.display import HTML, display

FONT = 'Cochineal, system-ui, -apple-system, "Segoe UI", sans-serif'

# Categorical slots are used in fixed order: blue, oxblood, green, cerulean.
# Muted to sit on the cream notebook surface; every pair of the four was checked
# for separation under protan, deutan, and tritan simulation (OKLab ΔE ≥ 9) and
# normal vision.
P: dict[str, list[str] | str] = {
    "transparent": "transparent",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series": ["#2a5aab", "#883327", "#3e8343", "#349ab6"],
    "context": "#c3c2b7",
    # Sequential: the series blue's hue (OKLCH 260), light to dark; step 4 is
    # series[0] itself.
    "ramp": [
        "#cfdff9",
        "#a4bfeb",
        "#759cdc",
        "#4c7bc8",
        "#2a5aab",
        "#1f4584",
        "#17315d",
    ],
}


class Figure(NamedTuple):
    """A chart plus the table that carries the same numbers in text form."""

    chart: alt.TopLevelMixin
    table: pl.DataFrame | None
    label: str = "Table view"


@alt.theme.register(name="quadbench", enable=True)  # ty: ignore[invalid-argument-type]
def _quadbench_theme() -> dict:
    """Return the shared presentation theme, sized for a projector."""
    return {
        "config": {
            "background": P["transparent"],
            "font": FONT,
            "padding": {"left": 6, "right": 18, "top": 6, "bottom": 6},
            "view": {
                "stroke": None,
                "continuousWidth": 620,
                "continuousHeight": 300,
            },
            "axis": {
                "labelFont": FONT,
                "titleFont": FONT,
                "labelColor": P["muted"],
                "titleColor": P["ink2"],
                "labelFontSize": 12.5,
                "titleFontSize": 12.5,
                "titleFontWeight": 500,
                "titlePadding": 10,
                "domainColor": P["axis"],
                "domainWidth": 1,
                "tickColor": P["axis"],
                "tickSize": 4,
                "gridColor": P["grid"],
                "gridWidth": 1,
                "gridDash": [],
            },
            "legend": {
                "labelFont": FONT,
                "titleFont": FONT,
                "labelColor": P["ink2"],
                "titleColor": P["ink2"],
                "labelFontSize": 12.5,
                "titleFontSize": 12.5,
                "titleFontWeight": 500,
                "symbolType": "circle",
                "symbolSize": 110,
                "orient": "top",
                "direction": "horizontal",
                "offset": 10,
                "titlePadding": 10,
                "labelLimit": 360,
            },
            "title": {
                "font": FONT,
                "color": P["ink"],
                "fontSize": 17,
                "fontWeight": 600,
                "subtitleFont": FONT,
                "subtitleColor": P["ink2"],
                "subtitleFontSize": 12.5,
                "subtitlePadding": 8,
                "anchor": "start",
                "offset": 16,
                "dy": -4,
            },
            "header": {
                "labelFont": FONT,
                "titleFont": FONT,
                "labelColor": P["ink2"],
                "titleColor": P["ink2"],
                "labelFontSize": 13,
                "labelFontWeight": 600,
                "titleFontSize": 12.5,
            },
            "range": {
                "category": P["series"],
                "heatmap": P["ramp"],
                "ramp": P["ramp"],
            },
            "bar": {"cornerRadiusEnd": 4, "discreteBandSize": 16},
            "point": {
                "size": 100,
                "filled": True,
                "stroke": P["transparent"],
                "strokeWidth": 2,
                "opacity": 1,
            },
            "line": {
                "strokeWidth": 2,
                "strokeCap": "round",
                "strokeJoin": "round",
            },
            "rule": {"strokeWidth": 1},
            "text": {"font": FONT, "fontSize": 12, "color": P["ink2"]},
        }
    }


def configure_notebook() -> None:
    """Apply display defaults used throughout the notebook."""
    pl.Config.set_tbl_rows(n=20)
    pl.Config.set_tbl_width_chars(width=160)
    pl.Config.set_tbl_hide_dataframe_shape(active=True)
    alt.data_transformers.enable(name="default", max_rows=20_000)


def _table_view(frame: pl.DataFrame, label: str) -> HTML:
    """Return a collapsed, textual counterpart to a chart."""
    return HTML(
        data=f"<details style='font:13px {FONT};color:{P['ink2']};margin:2px 0 22px'>"
        f"<summary style='cursor:pointer;padding:4px 0'>{label} "
        f"({frame.height:,} rows)</summary>{frame._repr_html_()}</details>"
    )


def _has_bars(spec: object) -> bool:
    """Whether a Vega-Lite spec (or any of its layers) draws bar marks."""
    if isinstance(spec, dict):
        mark = spec.get("mark")
        if mark == "bar" or (isinstance(mark, dict) and mark.get("type") == "bar"):
            return True
        return any(_has_bars(v) for v in spec.values())
    if isinstance(spec, list):
        return any(_has_bars(v) for v in spec)
    return False


def show(*figures: Figure) -> None:
    """Display each chart followed by its collapsed table view.

    Charts get zoom and pan, except bar charts: panning cuts bars off from
    their zero baseline, and the clipping that scale binding turns on makes
    Vega's canvas renderer skip bars with rounded ends entirely.
    """
    for figure in figures:
        chart = figure.chart
        display(chart if _has_bars(chart.to_dict()) else chart.interactive())  # ty: ignore[unresolved-attribute]
        if figure.table is not None:
            display(_table_view(figure.table, figure.label))


def memory_layout() -> HTML:
    """Diagram: a list of pointers to boxed floats against one block of raw values."""
    boxed: str = P["series"][1]
    raw: str = P["series"][0]
    ink: str = cast(typ=str, val=P["ink"])
    ink2: str = cast(typ=str, val=P["ink2"])
    grid: str = cast(typ=str, val=P["grid"])
    values: list[str] = ["1.5", "2.0", "0.7", "3.1", "4.2", "0.9"]
    cell: int = 64
    x0: int = 190
    # Boxed floats are scattered on the heap: deliberately irregular positions.
    heap: list[tuple[int, int]] = [
        (150, 150),
        (330, 196),
        (250, 232),
        (520, 160),
        (430, 226),
        (610, 206),
    ]
    parts: list[str] = [
        (
            "<svg xmlns='http://www.w3.org/2000/svg' width='760' height='392' viewBox='0 0 760 392' "
            f"font-family='{FONT}'>"
        ),
        f"<rect width='760' height='392' fill='{P['transparent']}'/>",
        (
            "<defs><marker id='arrow' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' "
            f"orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' fill='{ink2}'/></marker></defs>"
        ),
        f"<text x='20' y='36' font-size='15' font-weight='600' fill='{ink}'><tspan font-family='monospace'>list[float]</tspan></text>",
        f"<text x='20' y='56' font-size='12' fill='{ink2}'>a row of 8-byte pointers</text>",
    ]
    for i, (hx, hy) in enumerate(iterable=heap):
        x: int = x0 + i * cell
        parts.append(
            f"<rect x='{x}' y='26' width='{cell - 4}' height='34' rx='4' fill='none' stroke='{boxed}' stroke-width='2'/>"
        )
        parts.append(
            f"<circle cx='{x + (cell - 4) / 2}' cy='43' r='4' fill='{boxed}'/>"
        )
        parts.append(
            f"<path d='M{x + (cell - 4) / 2},47 L{hx + 45},{hy - 2}' stroke='{ink2}' stroke-width='1.2' "
            "fill='none' marker-end='url(#arrow)'/>"
        )
        parts.append(
            f"<rect x='{hx}' y='{hy}' width='120' height='30' rx='4' fill='{boxed}' fill-opacity='0.14' stroke='{boxed}' stroke-width='1.5'/>"
        )
        parts.append(
            f"<text x='{hx + 8}' y='{hy + 19}' font-size='11' fill='{ink2}'><tspan font-family='monospace'>ref · type · </tspan><tspan font-weight='600' fill='{ink}'>{values[i]}</tspan></text>"
        )
    parts += [
        f"<text x='20' y='76' font-size='12' fill='{ink2}'>to float objects</text>",
        f"<text x='20' y='93' font-size='12' fill='{ink2}'>scattered on the heap,</text>",
        f"<text x='20' y='110' font-size='12' fill='{ink2}'>each with its own</text>",
        f"<text x='20' y='127' font-size='12' fill='{ink2}'>header</text>",
        f"<line x1='20' y1='282' x2='740' y2='282' stroke='{grid}'/>",
        f"<text x='20' y='326' font-size='15' font-weight='600' fill='{ink}'><tspan font-family='monospace'>NDArray[np.float64]</tspan></text>",
        f"<text x='20' y='346' font-size='12' fill='{ink2}'>one contiguous block of raw numbers</text>",
    ]
    for i, value in enumerate(iterable=values):
        x: int = x0 + 60 + i * cell
        parts.append(
            f"<rect x='{x}' y='306' width='{cell}' height='40' fill='{raw}' fill-opacity='0.14' stroke='{raw}' stroke-width='1.5'/>"
        )
        parts.append(
            f"<text x='{x + cell / 2}' y='331' font-size='13' font-weight='600' text-anchor='middle' fill='{ink}'>{value}</text>"
        )
    parts += [
        f"<text x='{x0 + 60}' y='370' font-size='12' fill='{ink2}'>8 bytes each, side by side: the CPU streams through them in one compiled loop</text>",
        "</svg>",
    ]
    return HTML(data="".join(parts))


def _stat_tile(label: str, value: str, note: str) -> str:
    """Render one statistic in a summary strip."""
    return (
        f"<div style='flex:1 1 200px;min-width:180px;background:{P['transparent']};"
        f"border:1px solid {P['grid']};border-radius:10px;padding:14px 16px'>"
        f"<div style='font:500 12.5px {FONT};color:{P['muted']};"
        f"letter-spacing:.02em'>{label}</div>"
        f"<div style='font:600 26px {FONT};color:{P['ink']};margin:6px 0 2px'>"
        f"{value}</div>"
        f"<div style='font:400 12.5px {FONT};color:{P['ink2']}'>{note}</div></div>"
    )


def stat_tiles(tiles: list[tuple[str, str, str]]) -> None:
    """Display a strip of (label, value, note) statistic tiles."""
    display(
        HTML(
            data=f"<div style='display:flex;gap:12px;flex-wrap:wrap;"
            f"background:{P['transparent']};padding:14px;border-radius:12px'>"
            + "".join(_stat_tile(*tile) for tile in tiles)
            + "</div>"
        )
    )
