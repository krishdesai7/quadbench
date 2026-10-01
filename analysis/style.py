"""Visual language for the report: palette, Altair theme, and display helpers.

Importing this module registers and enables the ``quadbench`` Altair theme, so
every chart in the notebook shares one look without per-chart configuration.
"""

from __future__ import annotations

from typing import NamedTuple

import altair as alt
import polars as pl
from IPython.display import HTML, display


FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Categorical slots are used in fixed order: blue, orange, aqua.  They are the
# validated reference palette's first three slots.
P = {
    "surface": "#fcfcfb",
    "plane": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    "shade": "#9ec5f4",
    "context": "#c3c2b7",
    "ramp": [
        "#cde2fb",
        "#9ec5f4",
        "#6da7ec",
        "#3987e5",
        "#256abf",
        "#184f95",
        "#0d366b",
    ],
}


class Figure(NamedTuple):
    """A chart plus the table that carries the same numbers in text form."""

    chart: alt.TopLevelMixin
    table: pl.DataFrame | None
    label: str = "Table view"


@alt.theme.register("quadbench", enable=True)
def _quadbench_theme() -> dict:
    """Return the shared presentation theme, sized for a projector."""
    return {
        "config": {
            "background": P["surface"],
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
                "stroke": P["surface"],
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
    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_width_chars(160)
    pl.Config.set_tbl_hide_dataframe_shape(True)
    alt.data_transformers.enable("default", max_rows=20_000)


def _table_view(frame: pl.DataFrame, label: str) -> HTML:
    """Return a collapsed, textual counterpart to a chart."""
    return HTML(
        f"<details style='font:13px {FONT};color:{P['ink2']};margin:2px 0 22px'>"
        f"<summary style='cursor:pointer;padding:4px 0'>{label} "
        f"({frame.height:,} rows)</summary>{frame._repr_html_()}</details>"
    )


def show(*figures: Figure) -> None:
    """Display each chart followed by its collapsed table view."""
    for figure in figures:
        display(figure.chart)
        if figure.table is not None:
            display(_table_view(figure.table, figure.label))


def _stat_tile(label: str, value: str, note: str) -> str:
    """Render one statistic in a summary strip."""
    return (
        f"<div style='flex:1 1 200px;min-width:180px;background:{P['surface']};"
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
            f"<div style='display:flex;gap:12px;flex-wrap:wrap;"
            f"background:{P['plane']};padding:14px;border-radius:12px'>"
            + "".join(_stat_tile(*tile) for tile in tiles)
            + "</div>"
        )
    )
