#!/usr/bin/env -S uv run --python 3.13
# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = [
#   "altair>=6.2.2",
#   "ipykernel>=7.3.0",
#   "nbclient>=0.10.2",
#   "nbformat>=5.10.4",
#   "polars>=1.43.2",
# ]
# ///
"""Execute the analysis notebook and verify its JAX integration contract.

Run directly with ``uv run --python 3.13 tests/test_analysis_notebook.py``.
The function is also discoverable by pytest when these inline dependencies are
installed in the test environment.
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "analysis.ipynb"


def test_analysis_notebook() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            """
expected_jax_runs = {"jax_cpu", "jax_dense", "jax_latency", "jax_sweep"}
assert expected_jax_runs <= set(D), sorted(set(D))
assert expected_jax_runs <= set(meta), sorted(set(meta))

for run in expected_jax_runs:
    measured = D[run].filter(pl.col("samples") > 0)
    assert measured.height > 0, run
    assert measured["samples"].min() == 30, run
    assert measured["us"].is_null().sum() == 0, run

assert set(D["jax_dense"]["destination"].unique()) == {"eager", "jit"}
assert set(D["jax_cpu"]["implementation"].unique()) >= {
    "JAX fp16", "JAX bf16", "JAX fp32", "JAX fp64",
    "NumPy fp16", "NumPy fp32", "NumPy fp64",
}
assert meta["jax_dense"]["timer"] == "pipelined"
assert meta["jax_latency"]["timer"] == "blocking"
assert meta["jax_cpu"]["backend"]["platform"] == "cpu"
assert meta["jax_dense"]["backend"]["device_kind"] == "NVIDIA A100-SXM4-80GB"

for implementation, result_dtype, width in [
    ("JAX fp16", "float16", 2),
    ("JAX bf16", "bfloat16", 2),
    ("JAX fp32", "float32", 4),
    ("JAX fp64", "float64", 8),
]:
    row = (
        raw["jax_dense"][0]
        .filter((pl.col("implementation") == implementation)
                & (pl.col("operation") == "add")
                & (pl.col("destination") == "jit")
                & (pl.col("cache_state") == "cold")
                & pl.col("time_us").is_not_null())
        .row(0, named=True)
    )
    assert row["result_dtype"] == result_dtype
    assert row["operand_bytes"] == width
    assert row["result_bytes"] == width
    assert row["b_elem"] == 3 * width
    assert row["promoted"] is False

assert set(D["jax_dense"]["framework"].unique()) == {"JAX"}
assert set(D["jax_dense"]["execution_mode"].unique()) == {"eager", "jit"}
assert set(D["jax_dense"]["timer"].unique()) == {"pipelined"}
assert FRAMEWORK_SERIES == [
    "JAX JIT · host pipeline",
    "CuPy alloc · CUDA events",
]

for run in expected_jax_runs:
    validations = [
        result
        for size in meta[run]["per_size"].values()
        for result in size["validation"]
    ]
    assert validations, run
    assert all(result["ok"] for result in validations), run

assert jax_quality["measured cells"].sum() == 2_474
assert jax_quality["timing samples"].sum() == 74_220
assert jax_quality["validations"].sum() == 1_237
assert jax_quality["failed validations"].sum() == 0
assert jax_quality["timed-loop recompiles"].sum() == 0
assert 2.44 < jax_quality["maximum ULP error"].max() < 2.45
assert 2.07 < jax_quality["maximum JAX ULP error"].max() < 2.08

endpoint = framework_endpoint.row(by_predicate=pl.col("operation") == "add", named=True)
assert 0.98 < endpoint["CuPy / JAX time"] < 1.00
fp32_chain = fusion.row(
    by_predicate=(pl.col("operation") == "chain")
    & (pl.col("implementation") == "JAX fp32"),
    named=True,
)
assert 5.43 < fp32_chain["JIT speedup"] < 5.44
add_floor = timer_floor.row(by_predicate=pl.col("operation") == "add", named=True)
assert 1.23 < add_floor["blocking / pipelined"] < 1.24
cpu_matmul = cpu_framework.row(
    by_predicate=pl.col("operation") == "matmul", named=True
)
assert 26.39 < cpu_matmul["NumPy / JAX time"] < 26.40
"""
        )
    )

    NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()


if __name__ == "__main__":
    test_analysis_notebook()
