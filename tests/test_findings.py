"""The numbers the report quotes, and the JAX evidence behind Lesson 7."""

import polars as pl
import pytest

from analysis import findings


@pytest.fixture(scope="module")
def v(d):
    return findings.values(d)


# --- formatting ------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "text"),
    [(1.4213, "1.42×"), (26.396, "26.4×"), (126.81, "127×"), (12_056.4, "12,056×")],
)
def test_ratio_precision_scales_with_magnitude(value, text):
    assert findings.fmt_ratio(value) == text


def test_integers_use_comma_separators():
    assert findings.fmt_int(12_800) == "12,800"


def test_units_are_spaced():
    assert findings.fmt_ns(102.07) == "102 ns"
    assert findings.fmt_ns(0.2721) == "0.27 ns"
    assert findings.fmt_us(8.423) == "8.4 µs"
    assert findings.fmt_gbs(27.71) == "27.7 GB/s"
    assert findings.fmt_gbs(1767.2) == "1,767 GB/s"


def test_negative_zero_prints_as_zero():
    assert findings.fmt_pct(-0.01) == "0.0%"


def test_range_low_ends_keep_their_precision():
    assert findings.fmt_low(1.06) == "1.06"
    assert findings.fmt_low(10.4) == "10.4"


def test_large_microsecond_values_drop_decimals():
    assert findings.fmt_us(2431.5) == "2,432 µs"


def test_every_value_has_a_formatted_form(d, v):
    assert set(findings.compute(d)) == set(v)


# --- lesson 1: containers -------------------------------------------------


def test_python_list_is_two_orders_of_magnitude_slower_than_an_array(v):
    assert 120 < v["list_vs_f64"] < 135
    assert v["list_vs_f64_min"] > 20
    assert v["list_vs_f64_max"] >= v["list_vs_f64"]


def test_list_of_numpy_scalars_is_slower_than_list_of_floats(v):
    assert v["f32_list_vs_float_list"] > 1.2


def test_software_quad_array_beats_a_list_of_hardware_doubles(v):
    assert v["list_vs_quad"] > 3


def test_python_float_memory(v):
    assert 47 < v["list_bytes"] < 49
    assert v["list_size_vs_f64"] == pytest.approx(6, abs=0.1)


# --- lessons 2-6 ------------------------------------------------------------


def test_gpu_floor_and_crossover(v):
    assert 8.3 < v["gpu_floor_lo"] < v["gpu_floor_hi"] < 9.4
    assert v["gpu_floor_edge"] == 400_000
    assert 15_000 < v["crossover"] < 25_000


def test_cache_minimum_is_float64_at_12800(v):
    assert v["l2_min_elems"] == 12_800
    assert 0.27 < v["l2_min_ns"] < 0.28
    assert 1.49 < v["hotcold_peak"] < 1.50


def test_cpu_bandwidth_plateau(v):
    assert 27 < v["cpu_plateau_gbs"] < 28
    assert 39 < v["four_core_gbs"] < 40


def test_gpu_two_ceilings(v):
    assert 167 < v["gpu_elem_rate"] < 169
    assert v["hbm_plateau_80"] > v["hbm_plateau_40"]


def test_float16_is_flat_and_conversion_bound(v):
    assert 5.9 < v["f16_ns"] < 6.1
    assert v["f16_vs_f32_min"] < 7
    assert v["f16_vs_f32_max"] > 40


def test_allocation_cliff(v):
    assert abs(v["alloc_median_below"]) < 1
    assert 30 < v["alloc_median_above"] < 40


# --- lesson 7: JAX (ported from the original notebook test) -----------------


def test_jax_quality_totals(d):
    quality = findings.jax_quality(d)
    assert quality["measured cells"].sum() == 2_474
    assert quality["timing samples"].sum() == 74_220
    assert quality["validations"].sum() == 1_237
    assert quality["failed validations"].sum() == 0
    assert quality["timed-loop recompiles"].sum() == 0
    assert 2.44 < quality["maximum ULP error"].max() < 2.45
    assert 2.07 < quality["maximum JAX ULP error"].max() < 2.08


def test_framework_ratios(v):
    assert 0.98 < v["cupy_jax_add"] < 1.00
    assert 5.43 < v["jit_chain_fp32"] < 5.44
    assert 1.23 < v["blocking_ratio_add"] < 1.24
    assert 26.39 < v["numpy_jax_cpu_matmul"] < 26.40


def test_jax_runs_are_complete_and_validated(d):
    for run in ["jax_cpu", "jax_dense", "jax_latency", "jax_sweep"]:
        measured = d.runs[run].filter(pl.col("samples") > 0)
        assert measured.height > 0, run
        assert measured["samples"].min() == 30, run
        assert measured["us"].is_null().sum() == 0, run
        results = [r for size in d.meta[run]["per_size"].values() for r in size["validation"]]
        assert results and all(r["ok"] for r in results), run


def test_jax_metadata(d):
    assert set(d.runs["jax_dense"]["destination"].unique()) == {"eager", "jit"}
    assert set(d.runs["jax_cpu"]["implementation"].unique()) >= {
        "JAX fp16", "JAX bf16", "JAX fp32", "JAX fp64", "NumPy fp16", "NumPy fp32", "NumPy fp64",
    }
    assert d.meta["jax_dense"]["timer"] == "pipelined"
    assert d.meta["jax_latency"]["timer"] == "blocking"
    assert d.meta["jax_cpu"]["backend"]["platform"] == "cpu"
    assert d.meta["jax_dense"]["backend"]["device_kind"] == "NVIDIA A100-SXM4-80GB"
    assert set(d.runs["jax_dense"]["framework"].unique()) == {"JAX"}
    assert set(d.runs["jax_dense"]["execution_mode"].unique()) == {"eager", "jit"}
    assert set(d.runs["jax_dense"]["timer"].unique()) == {"pipelined"}


@pytest.mark.parametrize(
    ("implementation", "result_dtype", "width"),
    [("JAX fp16", "float16", 2), ("JAX bf16", "bfloat16", 2), ("JAX fp32", "float32", 4), ("JAX fp64", "float64", 8)],
)
def test_jax_byte_accounting(d, implementation, result_dtype, width):
    row = (
        d.raw["jax_dense"][0]
        .filter((pl.col("implementation") == implementation) & (pl.col("operation") == "add")
                & (pl.col("destination") == "jit") & (pl.col("cache_state") == "cold")
                & pl.col("time_us").is_not_null())
        .row(0, named=True)
    )
    assert row["result_dtype"] == result_dtype
    assert row["operand_bytes"] == row["result_bytes"] == width
    assert row["b_elem"] == 3 * width
    assert row["promoted"] is False


def test_summary_tiles_count_every_measurement(d):
    tiles = findings.summary_tiles(d)
    labels = [label for label, _, _ in tiles]
    assert labels == ["Cells measured", "Timing samples", "CPU", "GPU"]
    samples = int(tiles[1][1].replace(",", ""))
    assert samples > 74_220  # more than the JAX runs alone
    assert "EPYC 7763" in tiles[2][1]


def test_hardware_cluster_excludes_cells_numpy_computes_in_float16(v):
    assert v["hardware_vs_f64_max"] < 3
    assert v["int8_sqrt_vs_f64"] > 6


def test_standard_cpython_list_layout_is_pointer_plus_float(v):
    # The pure-Python run used free-threaded CPython; the audience uses the
    # standard build, where a float object is 24 bytes.
    assert v["std_list_bytes"] == 32
    assert v["std_list_vs_f64"] == 4


def test_allocation_residue_is_alloc_beating_preallocated(v):
    assert 20 < v["alloc_beats_out_max"] < 30
    assert "alloc_negative_max" not in v


def test_fresh_memory_costs_about_one_write_pass(v):
    assert 20 < v["fault_gbs"] < 30
    assert "fault_ns_per_page" not in v


def test_repeat_range_covers_every_run(v):
    assert v["repeats_lo"] == 10
    assert v["repeats_hi"] == 50


def test_quad_range_low_end_is_cos(v):
    assert v["quad_vs_f64_min"] < 11
    assert "quad_add_vs_f64" not in v
