"""Theme, display, and figure-construction checks."""

import altair as alt
import polars as pl

from analysis import style


def _tiny_figure() -> style.Figure:
    frame = pl.DataFrame({"x": [1.0, 2.0]})
    chart = alt.Chart(frame).mark_point().encode(x="x:Q")
    return style.Figure(chart, frame, "Table view")


def test_importing_style_enables_the_report_theme():
    assert alt.theme.active == "quadbench"


def test_show_renders_chart_then_table_view(monkeypatch):
    shown = []
    monkeypatch.setattr(style, "display", shown.append)
    style.show(_tiny_figure())
    assert isinstance(shown[0], alt.TopLevelMixin)
    assert "<details" in shown[1].data and "Table view" in shown[1].data


def test_show_skips_table_view_when_figure_has_none(monkeypatch):
    shown = []
    monkeypatch.setattr(style, "display", shown.append)
    style.show(_tiny_figure()._replace(table=None))
    assert len(shown) == 1


def test_stat_tiles_render_every_tile(monkeypatch):
    shown = []
    monkeypatch.setattr(style, "display", shown.append)
    style.stat_tiles([("CPU", "EPYC 7763", "1 thread"), ("GPU", "A100", "108 SMs")])
    html = shown[0].data
    assert "EPYC 7763" in html and "A100" in html and "108 SMs" in html
