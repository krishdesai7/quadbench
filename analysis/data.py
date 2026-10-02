"""Load benchmark runs and derive the normalized measures the report uses.

``time_us`` is the only directly measured quantity.  Everything else (ns per
element, bytes per element, GB/s, elements per second) is derived here, once,
so that every figure and every quoted number agrees.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import polars as pl

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

# Storage widths are fixed to the x86 Linux environment used for collection.
# In particular, long double uses an 80-bit x87 value in a 16-byte slot.
ITEMSIZE = {
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "uint64": 8,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "float128": 16,
    "longdouble64": 16,
    "quad-sleef": 16,
    "QuadPrecDType(backend='sleef')": 16,
    "GPU fp16": 2,
    "GPU fp32": 4,
    "GPU fp64": 8,
    "JAX fp16": 2,
    "JAX bf16": 2,
    "JAX fp32": 4,
    "JAX fp64": 8,
    "NumPy fp16": 2,
    "NumPy fp32": 4,
    "NumPy fp64": 8,
    "bfloat16": 2,
}
CANONICAL_DTYPE = {
    "JAX fp16": "float16",
    "JAX bf16": "bfloat16",
    "JAX fp32": "float32",
    "JAX fp64": "float64",
    "NumPy fp16": "float16",
    "NumPy fp32": "float32",
    "NumPy fp64": "float64",
}
UNARY = ["sqrt", "exp", "cos"]
OP_ORDER = [
    "add",
    "mul",
    "div",
    "muladd",
    "chain",
    "muladd_accum",
    "muladd_fused",
    "sqrt",
    "exp",
    "cos",
    "matmul",
    "matmul_explicit",
    "matmul_explicit_fused",
]
RUNS = [
    "cpu_sweep",
    "cpu_dense",
    "cpu_alloc",
    "cpu_f16",
    "cpu_dram50",
    "gpu_sweep",
    "gpu_sweep80",
    "gpu_dense80",
    "gpu_host80",
    "jax_sweep",
    "jax_dense",
    "jax_latency",
    "jax_cpu",
]
KEYS = [
    "run",
    "n",
    "n_elem",
    "operation",
    "implementation",
    "framework",
    "execution_mode",
    "timer",
    "destination",
    "cache_state",
]
PURE_KEYS = ["n", "n_elem", "operation", "implementation", "cache_state"]
MIB = 2**20

OUT_COLD = (pl.col("destination") == "out") & (pl.col("cache_state") == "cold")
OUT_HOT = (pl.col("destination") == "out") & (pl.col("cache_state") == "hot")
ALLOC_COLD = (pl.col("destination") == "alloc") & (pl.col("cache_state") == "cold")


class AnalysisData(NamedTuple):
    """Loaded samples, metadata, cell aggregates, and combined tables."""

    raw: dict[str, tuple[pl.DataFrame, dict]]
    meta: dict[str, dict]
    runs: dict[str, pl.DataFrame]
    parallel: dict[int, pl.DataFrame]
    everything: pl.DataFrame
    pure: pl.DataFrame
    pure_memory: dict[str, float]


def _load_run(run: str, data_root: Path) -> tuple[pl.DataFrame, dict]:
    """Load repeat samples and derive element, byte, and rate measures."""
    run_dir = data_root / run
    frame = pl.read_parquet(run_dir / f"{run}.parquet")
    metadata = json.loads((run_dir / f"{run}.json").read_text())

    if "n" not in frame.columns:
        frame = frame.with_columns(
            pl.lit(metadata["sizes"][0], dtype=pl.Int64).alias("n")
        )

    element_counts = {
        int(size): block["n_elem"] for size, block in metadata["per_size"].items()
    }
    result_dtype_rows = []
    for size, block in metadata["per_size"].items():
        result_dtype_rows.extend(
            {
                "n": int(size),
                "operation": key.split("__")[0],
                "implementation": key.split("__")[1],
                "result_dtype": result_dtype,
            }
            for key, result_dtype in block.get("result_dtypes", {}).items()
        )
        result_dtype_rows.extend(
            {
                "n": int(size),
                "operation": report["op"],
                "implementation": report["dtype"],
                "result_dtype": report["res_dtype"],
            }
            for report in block.get("validation", [])
            if report.get("res_dtype")
        )
    promotions = pl.DataFrame(
        result_dtype_rows,
        schema={
            "n": pl.Int64,
            "operation": pl.String,
            "implementation": pl.String,
            "result_dtype": pl.String,
        },
    ).unique(["n", "operation", "implementation"])

    execution_mode = (
        pl.col("destination") if run.startswith("jax_") else pl.lit("eager")
    )
    base_operation = pl.col("operation").str.replace("_via_f32", "")

    return (
        frame.join(
            promotions,
            on=["n", "operation", "implementation"],
            how="left",
        )
        .with_columns(
            pl.lit(run).alias("run"),
            pl.when(pl.col("implementation").str.starts_with("JAX "))
            .then(pl.lit("JAX"))
            .when(pl.col("implementation").str.starts_with("GPU "))
            .then(pl.lit("CuPy"))
            .otherwise(pl.lit("NumPy"))
            .alias("framework"),
            pl.col("implementation").replace(CANONICAL_DTYPE).alias("operand_dtype"),
            execution_mode.alias("execution_mode"),
            pl.lit(metadata.get("timer", "host")).alias("timer"),
            pl.col("n").replace_strict(element_counts).alias("n_elem"),
            pl.col("result_dtype").fill_null(pl.col("implementation")),
        )
        .with_columns(
            pl.col("implementation").replace_strict(ITEMSIZE).alias("operand_bytes"),
            pl.col("result_dtype").replace_strict(ITEMSIZE).alias("result_bytes"),
            pl.when(base_operation.is_in(UNARY)).then(1).otherwise(2).alias("reads"),
            (pl.col("result_dtype") != pl.col("operand_dtype")).alias("promoted"),
        )
        .with_columns(
            (pl.col("reads") * pl.col("operand_bytes") + pl.col("result_bytes")).alias(
                "b_elem"
            )
        )
        .with_columns(
            (pl.col("time_us") * 1e3 / pl.col("n_elem")).alias("ns_elem"),
            (
                pl.col("b_elem") * pl.col("n_elem") / (pl.col("time_us") * 1e-6) / 1e9
            ).alias("gbs"),
            (pl.col("n_elem") / (pl.col("time_us") * 1e-6) / 1e9).alias("gelem_s"),
            (pl.col("n_elem") * pl.col("result_bytes") / MIB).alias("result_mib"),
        ),
        metadata,
    )


def _cells(frame: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-repeat observations to one row per measured cell."""
    return (
        frame.group_by(KEYS)
        .agg(
            pl.col("time_us").median().alias("us"),
            pl.col("time_us").quantile(0.25).alias("us_lo"),
            pl.col("time_us").quantile(0.75).alias("us_hi"),
            pl.col("ns_elem").median().alias("ns_elem"),
            pl.col("gbs").median().alias("gbs"),
            pl.col("gelem_s").median().alias("gelem_s"),
            pl.col("b_elem").first(),
            pl.col("result_mib").first(),
            pl.col("result_dtype").first(),
            pl.col("promoted").first(),
            pl.col("time_us").is_not_null().sum().alias("samples"),
        )
        .sort(KEYS)
    )


def _load_pure(data_root: Path) -> tuple[pl.DataFrame, dict[str, float]]:
    """Load the pure-Python run, which has no byte accounting.

    Python containers hold pointers to heap objects, so a fixed itemsize means
    nothing for them; the harness measured bytes per element directly instead.
    """
    run_dir = data_root / "cpu_pure"
    frame = pl.read_parquet(run_dir / "cpu_pure.parquet")
    metadata = json.loads((run_dir / "cpu_pure.json").read_text())

    element_counts = {
        int(size): block["n_elem"] for size, block in metadata["per_size"].items()
    }
    cells = (
        frame.with_columns(pl.col("n").replace_strict(element_counts).alias("n_elem"))
        .with_columns((pl.col("time_us") * 1e3 / pl.col("n_elem")).alias("ns_elem"))
        .group_by(PURE_KEYS)
        .agg(
            pl.col("time_us").median().alias("us"),
            pl.col("time_us").quantile(0.25).alias("us_lo"),
            pl.col("time_us").quantile(0.75).alias("us_hi"),
            pl.col("ns_elem").median(),
            pl.col("time_us").is_not_null().sum().alias("samples"),
        )
        .sort(PURE_KEYS)
    )
    largest = metadata["per_size"][str(max(metadata["sizes"]))]["memory"]
    memory = {
        implementation: block["operand_sizeof_B_per_elem"]
        for implementation, block in largest.items()
    }
    return cells, memory


def load_analysis(data_root: str | Path = DATA_ROOT) -> AnalysisData:
    """Load every benchmark run the report uses."""
    root = Path(data_root)
    raw = {run: _load_run(run, root) for run in RUNS}
    metadata = {run: run_meta for run, (_, run_meta) in raw.items()}
    runs = {run: _cells(frame) for run, (frame, _) in raw.items()}
    parallel = {
        index: _cells(_load_run(f"cpu_par{index}", root)[0]) for index in range(4)
    }
    everything = pl.concat([runs[run] for run in RUNS])
    pure, pure_memory = _load_pure(root)
    return AnalysisData(raw, metadata, runs, parallel, everything, pure, pure_memory)


def allocation_penalty(
    runs: dict[str, pl.DataFrame],
    run: str,
    operations: tuple[str, ...] = ("add", "mul"),
) -> pl.DataFrame:
    """Return allocation overhead relative to a preallocated destination."""
    return (
        runs[run]
        .filter(
            (pl.col("cache_state") == "cold") & pl.col("operation").is_in(operations)
        )
        .pivot(
            "destination",
            index=["operation", "implementation", "result_mib"],
            values="us",
        )
        .drop_nulls(["alloc", "out"])
        .with_columns(
            ((pl.col("alloc") - pl.col("out")) / pl.col("out") * 100).alias("pct"),
            pl.lit(run).alias("run"),
        )
        .select("run", "operation", "implementation", "result_mib", "pct")
    )


def gpu_ceiling_rows(
    runs: dict[str, pl.DataFrame], metadata: dict[str, dict], run: str
) -> pl.DataFrame:
    """Select the large-problem elementwise rows used in the ceiling model."""
    run_metadata = metadata[run]
    return (
        runs[run]
        .filter(
            OUT_HOT
            & (pl.col("n") == 25_000_000)
            & ~pl.col("operation").str.starts_with("matmul")
            & (pl.col("operation") != "muladd")
        )
        .select("operation", "implementation", "b_elem", "us", "gbs", "gelem_s")
        .with_columns(
            pl.lit(
                f"{run_metadata['device']['name'].replace('NVIDIA A100-SXM4-', '')}"
                f"  ({run_metadata['peak_hbm_gbs']:,.0f} GB/s)"
            ).alias("card")
        )
    )


def jax_timer_series(
    runs: dict[str, pl.DataFrame],
    run: str,
    label: str,
    sizes: pl.Series,
    operations: list[str],
) -> pl.DataFrame:
    """Select one JAX timing-protocol series for the matched comparison."""
    return (
        runs[run]
        .filter(
            (pl.col.destination == "jit")
            & (pl.col.cache_state == "cold")
            & (pl.col.implementation == "JAX fp32")
            & pl.col.operation.is_in(other=operations)
            & pl.col.n.is_in(other=sizes)
        )
        .select("operation", "n_elem", "us", "ns_elem", "samples")
        .with_columns(pl.lit(value=label).alias(name="series"))
    )
