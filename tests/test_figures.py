"""Theme, display, and figure-construction checks."""

import altair as alt
import polars as pl
import pytest

from analysis import data, figures, style


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


FIGURES = [
    "representation_ladder",
    "bytes_per_element",
    "boxed_vs_unboxed_by_size",
    "gpu_call_floor",
    "cpu_vs_gpu",
    "cache_curve",
    "cpu_bandwidth",
    "gpu_bandwidth_by_size",
    "gpu_two_ceilings",
    "float16_curve",
    "float16_mechanism",
    "dtype_grid",
    "int_grid",
    "gpu_precision_ratios",
    "destination_decomposition",
    "allocation_cliff",
    "jax_vs_cupy",
    "jit_fusion",
    "jax_vs_numpy_cpu",
    "timing_protocol",
]


@pytest.mark.parametrize("name", FIGURES)
def test_figure_builds_a_valid_spec_with_a_table(d, name):
    figure = getattr(figures, name)(d)
    assert isinstance(figure, style.Figure)
    figure.chart.to_dict(validate=True)
    assert figure.table is not None and figure.table.height > 0


def test_ladder_orders_every_representation_slowest_first(d):
    table = figures.representation_ladder(d).table
    assert table["representation"].to_list()[:3] == [
        "list[np.float32]",
        "list[float]",
        "NDArray[float]",
    ]
    assert table["representation"].to_list()[-1] == "NDArray[cp.float32] (A100)"
    assert table["ns per element"].is_sorted(descending=True)


def test_ladder_rows_borrowed_from_other_sizes_are_size_insensitive(d):
    quad = (
        d.runs["cpu_dense"]
        .filter(
            data.ALLOC_COLD
            & (pl.col("operation") == "add")
            & (pl.col("implementation") == "quad-sleef")
        )["ns_elem"]
        .drop_nulls()
    )
    assert quad.max() / quad.min() < 1.06
    longdouble = d.runs["cpu_sweep"].filter(
        data.ALLOC_COLD
        & (pl.col("operation") == "add")
        & (pl.col("implementation") == "longdouble64")
        & pl.col("n").is_in([4_000, 64_000])
    )["ns_elem"]
    assert longdouble.max() / longdouble.min() < 1.15


def test_crossover_annotation_is_computed_between_measured_sizes(d):
    table = figures.cpu_vs_gpu(d).table
    crossover = figures.cpu_gpu_crossover(d)
    cpu = table.filter(pl.col("machine").str.starts_with("x86"))
    assert cpu["elements"].min() < crossover < cpu["elements"].max()


def test_memory_layout_diagram_contrasts_list_and_array():
    svg = style.memory_layout().data
    assert svg.lstrip().startswith("<svg")
    assert "list[float]" in svg and "NDArray[np.float64]" in svg


def test_bytes_chart_shows_standard_cpython_layout_beside_measured(d):
    table = figures.bytes_per_element(d).table
    rows = dict(zip(table["representation"], table["bytes"]))
    assert rows["list[float] · standard CPython ‡"] == 32
    assert 47 < rows["list[float]"] < 49
