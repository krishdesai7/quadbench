"""Every number the report quotes, computed from the data.

``values(d)`` returns the raw numbers; ``compute(d)`` returns them formatted in
house style, exactly as they appear in the notebook's prose.  A test checks
that every formatted value is quoted in the notebook, so the text cannot drift
from the data.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

from analysis import figures
from analysis.data import ALLOC_COLD, OUT_COLD, AnalysisData

Finding = tuple[float, Callable[[float], str]]
ADD = pl.col("operation") == "add"
# Standard (GIL) CPython on 64-bit: a list slot is an 8-byte pointer and a
# float object is 24 bytes.  The pure-Python run used the free-threaded build,
# whose object header makes a float 40 bytes, so these give the layout the
# audience's own interpreter has.
STANDARD_POINTER_BYTES = 8
STANDARD_FLOAT_BYTES = 24


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def fmt_ratio(x: float) -> str:
    """A multiple, with precision that shrinks as the number grows."""
    if x >= 100:
        return f"{x:,.0f}×"
    if x >= 10:
        return f"{x:.1f}×"
    return f"{x:.2f}×"


def fmt_int(x: float) -> str:
    return f"{x:,.0f}"


def fmt_ns(x: float) -> str:
    return f"{x:,.0f} ns" if x >= 10 else f"{x:.2f} ns"


def fmt_low(x: float) -> str:
    """The low end of a quoted range: a multiple without its × sign."""
    return fmt_ratio(x).removesuffix("×")


def fmt_us(x: float) -> str:
    return f"{x:,.0f} µs" if x >= 100 else f"{x:.1f} µs"


def fmt_ms(x: float) -> str:
    return f"{x:,.2f} ms"


def fmt_gbs(x: float) -> str:
    return f"{x:,.0f} GB/s" if x >= 100 else f"{x:.1f} GB/s"


def fmt_pct(x: float) -> str:
    return f"{x:.1f}%".replace("-0.0%", "0.0%")


def fmt_pct0(x: float) -> str:
    return f"{x:.0f}%"


def fmt_bytes(x: float) -> str:
    return f"{x:.0f} B"


def fmt_mib(x: float) -> str:
    return f"{x:.2f} MiB"


def fmt_dec(x: float) -> str:
    return f"{x:.1f}"


def fmt_hundreds(x: float) -> str:
    return f"{round(x, -2):,.0f}"


# ---------------------------------------------------------------------------
# Lesson 1 · containers
# ---------------------------------------------------------------------------


def _containers(d: AnalysisData) -> dict[str, Finding]:
    ladder = figures.representation_ladder(d).table
    ns = dict(zip(ladder["representation"], ladder["ns per element"]))
    lst, f64, f32 = (
        ns["list[float]"],
        ns["NDArray[np.float64]"],
        ns["NDArray[np.float32]"],
    )

    matched = (
        d.pure.filter(
            ADD
            & (pl.col("cache_state") == "cold")
            & (pl.col("implementation") == "py-list-float")
        )
        .select("n_elem", pl.col("ns_elem").alias("list"))
        .join(
            d.runs["cpu_dense"]
            .filter(ADD & ALLOC_COLD & (pl.col("implementation") == "float64"))
            .select("n_elem", pl.col("ns_elem").alias("f64")),
            on="n_elem",
        )
        .with_columns((pl.col("list") / pl.col("f64")).alias("ratio"))
        .sort("n_elem")
    )
    return {
        "list_ns": (lst, fmt_ns),
        "list_ns_small": (matched["list"][0], fmt_ns),
        "list_vs_f64": (lst / f64, fmt_ratio),
        "list_vs_f64_min": (matched["ratio"].min(), fmt_ratio),
        "list_vs_f64_max": (matched["ratio"].max(), fmt_ratio),
        "f32_list_vs_float_list": (ns["list[np.float32]"] / lst, fmt_ratio),
        "obj_array_vs_f64": (ns["NDArray[float]"] / f64, fmt_ratio),
        "list_vs_quad": (lst / ns["NDArray[np.float128]†"], fmt_ratio),
        "f64_vs_f32": (f64 / f32, fmt_ratio),
        "list_vs_gpu": (lst / ns["NDArray[cp.float32] (A100)"], fmt_ratio),
        "list_bytes": (d.pure_memory["py-list-float"], fmt_bytes),
        "list_size_vs_f64": (d.pure_memory["py-list-float"] / 8, lambda x: f"{x:.0f}×"),
        "list_size_vs_f32": (d.pure_memory["py-list-float"] / 4, lambda x: f"{x:.0f}×"),
        "std_list_bytes": (STANDARD_POINTER_BYTES + STANDARD_FLOAT_BYTES, fmt_bytes),
        "std_list_vs_f64": (
            (STANDARD_POINTER_BYTES + STANDARD_FLOAT_BYTES) / 8,
            lambda x: f"{x:.0f}×",
        ),
        "std_list_vs_f32": (
            (STANDARD_POINTER_BYTES + STANDARD_FLOAT_BYTES) / 4,
            lambda x: f"{x:.0f}×",
        ),
    }


# ---------------------------------------------------------------------------
# Lesson 2 · fixed per-call cost
# ---------------------------------------------------------------------------


def _call_cost(d: AnalysisData) -> dict[str, Finding]:
    edge, lo, hi = figures.gpu_floor_extent(d)
    smallest = d.runs["gpu_dense80"]["n_elem"].min()
    cross = figures.cpu_vs_gpu(d).table
    cpu = cross.filter(pl.col("machine") == figures.CROSS_SERIES[0]).select(
        "elements", pl.col("ns_elem").alias("cpu")
    )
    gpu = cross.filter(pl.col("machine") == figures.CROSS_SERIES[1]).select(
        "elements", pl.col("ns_elem").alias("gpu")
    )
    advantage = cpu.join(gpu, on="elements").with_columns(
        (pl.col("cpu") / pl.col("gpu")).alias("x")
    )
    return {
        "gpu_floor_lo": (lo, fmt_us),
        "gpu_floor_hi": (hi, fmt_us),
        "gpu_floor_edge": (edge, fmt_int),
        "gpu_floor_span": (edge / smallest, fmt_ratio),
        "crossover": (figures.cpu_gpu_crossover(d), fmt_hundreds),
        "gpu_advantage_max": (advantage["x"].max(), fmt_ratio),
    }


# ---------------------------------------------------------------------------
# Lesson 3 · caches
# ---------------------------------------------------------------------------


def _caches(d: AnalysisData) -> dict[str, Finding]:
    dense = d.runs["cpu_dense"].filter(
        (pl.col("destination") == "out") & (pl.col("implementation") == "float64")
    )
    add = (
        dense.filter(ADD)
        .pivot("cache_state", index="n_elem", values="ns_elem")
        .sort("n_elem")
        .with_columns((pl.col("cold") / pl.col("hot")).alias("ratio"))
    )
    minimum = add.row(by_predicate=pl.col("hot") == add["hot"].min(), named=True)
    peak = add.row(by_predicate=pl.col("ratio") == add["ratio"].max(), named=True)
    large = add.filter(pl.col("n_elem") >= 400_000)
    exp = dense.filter(
        (pl.col("operation") == "exp") & (pl.col("cache_state") == "hot")
    )["ns_elem"]
    return {
        "l2_min_ns": (minimum["hot"], fmt_ns),
        "l2_min_elems": (minimum["n_elem"], fmt_int),
        "hotcold_peak": (peak["ratio"], fmt_ratio),
        "hotcold_peak_elems": (peak["n_elem"], fmt_int),
        "hotcold_large_max": (large["ratio"].max(), fmt_ratio),
        "dram_ns_lo": (large["hot"].min(), fmt_ns),
        "dram_ns_hi": (large["hot"].max(), fmt_ns),
        "cache_penalty": (large["hot"].min() / minimum["hot"], lambda x: f"{x:.1f}×"),
        "exp_ns": (exp.median(), fmt_ns),
    }


# ---------------------------------------------------------------------------
# Lesson 4 · bandwidth
# ---------------------------------------------------------------------------


def _bandwidth(d: AnalysisData) -> dict[str, Finding]:
    dram = d.runs["cpu_dram50"].filter(OUT_COLD).drop_nulls("gbs")
    plateau = dram["gbs"].max()
    individual = [d.parallel[i].filter(OUT_COLD & ADD)["gbs"][0] for i in range(4)]
    aggregate = sum(individual)
    _, limits = figures.gpu_ceilings(d)
    by_card = {
        ("80" if "80GB" in card else "40"): row
        for card, row in zip(limits["card"], limits.iter_rows(named=True))
    }
    peaks = {
        "80": d.meta["gpu_sweep80"]["peak_hbm_gbs"],
        "40": d.meta["gpu_sweep"]["peak_hbm_gbs"],
    }
    saturation = figures.gpu_bandwidth_by_size(d).table
    top = saturation.filter(pl.col("n_elem") == saturation["n_elem"].max())
    pct = dict(zip(top["implementation"], top["pct"]))
    return {
        "cpu_plateau_gbs": (plateau, fmt_gbs),
        "plateau_cells": (dram.filter(pl.col("gbs") > plateau * 0.9).height, fmt_int),
        "four_core_gbs": (aggregate, fmt_gbs),
        "four_core_scale": (aggregate / plateau, fmt_ratio),
        "per_process_slowdown": (plateau / (aggregate / 4), fmt_ratio),
        "gpu_elem_rate": (
            limits["elem"].mean(),
            lambda x: f"{x:.0f} billion elements per second",
        ),
        "hbm_plateau_80": (by_card["80"]["bw"], fmt_gbs),
        "hbm_plateau_40": (by_card["40"]["bw"], fmt_gbs),
        "hbm_pct_80": (by_card["80"]["bw"] / peaks["80"] * 100, fmt_pct0),
        "hbm_pct_40": (by_card["40"]["bw"] / peaks["40"] * 100, fmt_pct0),
        "crossover_bytes_80": (by_card["80"]["bw"] / by_card["80"]["elem"], fmt_dec),
        "crossover_bytes_40": (by_card["40"]["bw"] / by_card["40"]["elem"], fmt_dec),
        "bandwidth_gain_80": (
            (by_card["80"]["bw"] / by_card["40"]["bw"] - 1) * 100,
            fmt_pct0,
        ),
        "gpu_fp16_pct": (pct["GPU fp16"], fmt_pct0),
        "gpu_fp32_pct": (pct["GPU fp32"], fmt_pct0),
        "gpu_fp64_pct": (pct["GPU fp64"], fmt_pct0),
    }


# ---------------------------------------------------------------------------
# Lesson 5 · precision
# ---------------------------------------------------------------------------


def _precision(d: AnalysisData) -> dict[str, Finding]:
    curve = figures.float16_curve(d).table.with_columns(
        (pl.col("float16") / pl.col("float32")).alias("ratio")
    )
    f16 = curve["float16"]
    lo_row = curve.row(by_predicate=pl.col("ratio") == curve["ratio"].min(), named=True)
    hi_row = curve.row(by_predicate=pl.col("ratio") == curve["ratio"].max(), named=True)

    mech = figures.float16_mechanism(d).table.filter(pl.col("op") == "add")
    native = dict(zip(mech["implementation"], mech["native"]))
    via = dict(zip(mech["implementation"], mech["via"]))

    grid = figures.dtype_grid(d).table

    def span(impl: str) -> pl.Series:
        return grid.filter(pl.col("implementation") == impl)["vs_f64"]

    ints = figures.int_grid(d).table.with_columns(
        pl.col("implementation").str.extract(r"(\d+)").cast(pl.Int64).alias("bits"),
        pl.col("implementation").str.starts_with("u").alias("unsigned"),
    )
    # Arithmetic NumPy does in the integer type itself: cost tracks width.
    arith = ints.filter(
        pl.col("operation").is_in(["add", "mul", "muladd", "muladd_accum"])
    )
    halving = (
        arith.filter(~pl.col("unsigned"))
        .sort("operation", "bits")
        .with_columns(
            (pl.col("vs_f64").shift(-1) / pl.col("vs_f64"))
            .over("operation")
            .alias("step")
        )["step"]
        .drop_nulls()
    )

    def ints_where(op: str | None = None, bits: int | None = None) -> pl.Series:
        rows = arith if op is None else ints.filter(pl.col("operation") == op)
        return (rows if bits is None else rows.filter(pl.col("bits") == bits))["vs_f64"]

    # NumPy computes sqrt of int8/uint8 in float16, so those cells pay the
    # float16 conversion cost.
    in_f16 = ints.filter(
        (pl.col("operation") == "sqrt") & (pl.col("result_dtype") == "float16")
    )
    quad = grid.filter(pl.col("implementation") == "quad-sleef")

    ratios = figures.gpu_precision_table(d)
    binary = ratios.filter(pl.col("operation").is_in(["add", "mul", "div"]))
    unary = ratios.filter(pl.col("operation").is_in(["sqrt", "exp", "cos"]))
    fused = ratios.row(
        by_predicate=pl.col("operation") == "matmul_explicit_fused", named=True
    )
    return {
        "f16_ns": (f16.median(), fmt_ns),
        "f16_spread": (
            (f16.max() - f16.min()) / 2 / f16.median() * 100,
            lambda x: f"±{x:.0f}%",
        ),
        "f16_vs_f32_min": (lo_row["ratio"], fmt_ratio),
        "f16_vs_f32_min_elems": (lo_row["n_elem"], fmt_int),
        "f16_vs_f32_max": (hi_row["ratio"], fmt_ratio),
        "f16_vs_f32_max_elems": (hi_row["n_elem"], fmt_int),
        "f16_native_ns": (native["float16"], fmt_ns),
        "f16_widened_ns": (via["float16"], fmt_ns),
        "f32_native_ns": (native["float32"], fmt_ns),
        "f32_widened_ns": (via["float32"], fmt_ns),
        "f32_vs_f64_min": (span("float32").min(), fmt_low),
        "f32_vs_f64_max": (span("float32").max(), fmt_ratio),
        "int64_arith_min": (ints_where(bits=64).min(), fmt_low),
        "int64_arith_max": (ints_where(bits=64).max(), fmt_ratio),
        "int8_arith_min": (ints_where(bits=8).min(), fmt_low),
        "int8_arith_max": (ints_where(bits=8).max(), fmt_ratio),
        "int_halving_min": (halving.min(), fmt_low),
        "int_halving_max": (halving.max(), fmt_ratio),
        "int_div_min": (ints_where(op="div").min(), fmt_low),
        "int_div_max": (ints_where(op="div").max(), fmt_ratio),
        "int8_sqrt_vs_f64": (in_f16["vs_f64"].max(), fmt_ratio),
        "int_matmul_min": (ints_where(op="matmul").min(), fmt_low),
        "int_matmul_max": (ints_where(op="matmul").max(), fmt_ratio),
        "quad_vs_f64_min": (span("quad-sleef").min(), fmt_low),
        "quad_vs_f64_max": (span("quad-sleef").max(), fmt_ratio),
        "quad_exp_vs_f64": (
            quad.filter(pl.col("operation") == "exp")["vs_f64"].item(),
            fmt_ratio,
        ),
        "ld_vs_f64_min": (span("longdouble64").min(), fmt_low),
        "ld_vs_f64_max": (span("longdouble64").max(), fmt_ratio),
        "f16grid_vs_f64_min": (span("float16").min(), fmt_low),
        "f16grid_vs_f64_max": (span("float16").max(), fmt_ratio),
        "gpu_f32_f16_binary_min": (binary["fp32 / fp16"].min(), fmt_low),
        "gpu_f32_f16_binary_max": (binary["fp32 / fp16"].max(), fmt_ratio),
        "gpu_f64_f32_binary_min": (binary["fp64 / fp32"].min(), fmt_low),
        "gpu_f64_f32_binary_max": (binary["fp64 / fp32"].max(), fmt_ratio),
        "gpu_f32_f16_unary": (unary["fp32 / fp16"].mean(), fmt_ratio),
        "gpu_fused_f32_f16": (fused["fp32 / fp16"], fmt_ratio),
    }


# ---------------------------------------------------------------------------
# Lesson 6 · allocation
# ---------------------------------------------------------------------------


def _allocation(d: AnalysisData) -> dict[str, Finding]:
    parts = figures.destination_decomposition(d).table
    cold = parts.filter(pl.col("component") == figures.COMPONENTS[0])
    alloc = parts.filter(pl.col("component") == figures.COMPONENTS[1])
    largest = alloc["n_elem"].max()

    cliff = figures.allocation_cliff_rows(d)
    below = cliff.filter(pl.col("side") == figures.SIDES[0])
    above = cliff.filter(pl.col("side") == figures.SIDES[1])

    dram = (
        d.runs["cpu_dram50"]
        .filter(
            (pl.col("cache_state") == "cold")
            & ADD
            & (pl.col("implementation") == "float64")
        )
        .pivot("destination", index="result_mib", values="us")
        .row(0, named=True)
    )
    fault_us = dram["alloc"] - dram["out"]
    # Where alloc beats even the preallocated `out=` form: the part of the
    # decomposition that cache state does not explain.
    vs_out = (
        d.runs["cpu_alloc"]
        .filter((pl.col("cache_state") == "cold") & ADD)
        .pivot("destination", index=["implementation", "n_elem"], values="us")
        .filter(pl.col("n_elem") < largest)
    )
    alloc_beats_out = ((1 - vs_out["alloc"] / vs_out["out"]) * 100).max()
    return {
        "cold_dest_max": (cold["pct"].max(), fmt_pct0),
        "alloc_beats_out_max": (alloc_beats_out, fmt_pct0),
        "alloc_jump": (
            alloc.filter(pl.col("n_elem") == largest)["pct"].max(),
            fmt_pct0,
        ),
        "alloc_cells_below": (below.height, fmt_int),
        "alloc_cells_above": (above.height, fmt_int),
        "alloc_largest_below_mib": (below["result_mib"].max(), fmt_mib),
        "alloc_size_above_mib": (above["result_mib"].min(), fmt_mib),
        "alloc_median_below": (below["pct"].median(), fmt_pct),
        "alloc_p90_below": (below["pct"].quantile(0.9), fmt_pct),
        "alloc_median_above": (above["pct"].median(), fmt_pct),
        "alloc_min_above": (above["pct"].min(), fmt_pct),
        "fault_us": (fault_us, fmt_us),
        "fault_gbs": (dram["result_mib"] * 2**20 / (fault_us * 1e-6) / 1e9, fmt_gbs),
    }


# ---------------------------------------------------------------------------
# Lesson 7 · compilation
# ---------------------------------------------------------------------------


def _compilation(d: AnalysisData) -> dict[str, Finding]:
    endpoint = figures.framework_endpoint(d)
    cupy_over_jax = dict(zip(endpoint["operation"], endpoint["CuPy / JAX time"]))
    matmul = endpoint.row(by_predicate=pl.col("operation") == "matmul", named=True)
    rows = figures.framework_rows(d)
    smallest = rows.filter(pl.col("n_elem") == rows["n_elem"].min())
    jax_small = smallest.filter(pl.col("series") == figures.FRAMEWORK_SERIES[0])["us"]
    cupy_small = smallest.filter(pl.col("series") == figures.FRAMEWORK_SERIES[1])["us"]

    fusion = figures.fusion_table(d)

    def speedup(op: str) -> pl.Series:
        return fusion.filter(pl.col("operation") == op)["JIT speedup"]

    chain32 = fusion.row(
        by_predicate=(pl.col("operation") == "chain")
        & (pl.col("implementation") == "JAX fp32"),
        named=True,
    )["JIT speedup"]

    cpu = figures.cpu_framework_table(d)
    cpu_ratio = dict(zip(cpu["operation"], cpu["NumPy / JAX time"]))
    found: dict[str, Finding] = {
        "cupy_jax_add": (cupy_over_jax["add"], fmt_ratio),
        "cupy_jax_exp": (cupy_over_jax["exp"], fmt_ratio),
        "cupy_jax_matmul": (cupy_over_jax["matmul"], fmt_ratio),
        "jax_matmul_ms": (
            matmul[figures.FRAMEWORK_SERIES[0]] * rows["n_elem"].max() / 1e6,
            fmt_ms,
        ),
        "cupy_matmul_ms": (
            matmul[figures.FRAMEWORK_SERIES[1]] * rows["n_elem"].max() / 1e6,
            lambda x: f"{x:,.0f} ms",
        ),
        "jax_small_lo": (jax_small.min(), fmt_us),
        "jax_small_hi": (jax_small.max(), fmt_us),
        "cupy_small_lo": (cupy_small.min(), fmt_us),
        "cupy_small_hi": (cupy_small.max(), fmt_us),
        "jit_add_deviation": ((speedup("add") - 1).abs().max() * 100, fmt_pct0),
        "jit_muladd_min": (speedup("muladd").min(), fmt_low),
        "jit_muladd_max": (speedup("muladd").max(), fmt_ratio),
        "jit_chain_min": (speedup("chain").min(), fmt_low),
        "jit_chain_max": (speedup("chain").max(), fmt_ratio),
        "jit_chain_fp32": (chain32, fmt_ratio),
        "jit_matmul_min": (speedup("matmul_explicit").min(), fmt_low),
        "jit_matmul_max": (speedup("matmul_explicit").max(), fmt_ratio),
    }
    for op in figures.CPU_OPS:
        found[f"numpy_jax_cpu_{op}"] = (cpu_ratio[op], fmt_ratio)
    return found


# ---------------------------------------------------------------------------
# Lesson 8 · measurement
# ---------------------------------------------------------------------------


def _measurement(d: AnalysisData) -> dict[str, Finding]:
    floor = figures.timer_floor(d)
    add = floor.row(by_predicate=pl.col("operation") == "add", named=True)
    exp = floor.row(by_predicate=pl.col("operation") == "exp", named=True)
    rows = figures.timer_rows(d)
    big = rows.filter(pl.col("n_elem") == rows["n_elem"].max()).pivot(
        "series", index="operation", values="us"
    )
    gap = ((big["JAX blocking"] / big["JAX pipelined"]) - 1).abs().max() * 100
    return {
        "pipelined_add_us": (add["JAX pipelined"], fmt_us),
        "blocking_add_us": (add["JAX blocking"], fmt_us),
        "blocking_ratio_add": (add["blocking / pipelined"], fmt_ratio),
        "blocking_ratio_exp": (exp["blocking / pipelined"], fmt_ratio),
        "blocking_extra_lo": (
            min(
                add["JAX blocking"] - add["JAX pipelined"],
                exp["JAX blocking"] - exp["JAX pipelined"],
            ),
            fmt_low,
        ),
        "blocking_extra_hi": (
            max(
                add["JAX blocking"] - add["JAX pipelined"],
                exp["JAX blocking"] - exp["JAX pipelined"],
            ),
            fmt_us,
        ),
        "protocol_gap_large": (gap, fmt_pct0),
    }


# ---------------------------------------------------------------------------
# Appendix · data quality
# ---------------------------------------------------------------------------

JAX_RUNS = ["jax_cpu", "jax_dense", "jax_latency", "jax_sweep"]


def jax_quality(d: AnalysisData) -> pl.DataFrame:
    """Planned and measured cells, samples, validations, and recompiles per JAX run."""
    rows = []
    for run in JAX_RUNS:
        frame, meta = d.raw[run]
        validations = [
            r for size in meta["per_size"].values() for r in size["validation"]
        ]
        samples = frame["time_us"].is_not_null().sum()
        rows.append(
            {
                "run": run,
                "completed sizes": f"{len(meta['sizes_completed'])}/{len(meta['sizes'])}",
                "planned cells": frame.height // meta["repeats"],
                "measured cells": samples // meta["repeats"],
                "timing samples": samples,
                "validations": len(validations),
                "failed validations": sum(not r["ok"] for r in validations),
                "maximum ULP error": max(r["ulp"] for r in validations),
                "maximum JAX ULP error": max(
                    r["ulp"] for r in validations if r["dtype"].startswith("JAX ")
                ),
                "timed-loop recompiles": sum(
                    len(s["compile_check"]["recompiled"])
                    for s in meta["per_size"].values()
                ),
            }
        )
    return pl.DataFrame(rows)


def repeat_range(d: AnalysisData) -> tuple[int, int]:
    """Fewest and most timing samples behind any measured cell."""
    counts = pl.concat(
        [
            d.everything["samples"],
            *(frame["samples"] for frame in d.parallel.values()),
            d.pure["samples"],
        ]
    )
    measured = counts.filter(counts > 0)
    return int(measured.min()), int(measured.max())


def _quality(d: AnalysisData) -> dict[str, Finding]:
    q = jax_quality(d)
    unmeasured = (q["planned cells"] - q["measured cells"]).sum()
    repeats_lo, repeats_hi = repeat_range(d)
    return {
        "repeats_lo": (repeats_lo, fmt_int),
        "repeats_hi": (repeats_hi, fmt_int),
        "jax_validations": (q["validations"].sum(), fmt_int),
        "jax_unmeasured_cells": (unmeasured, fmt_int),
        "jax_max_ulp": (q["maximum JAX ULP error"].max(), lambda x: f"{x:.2f} ULP"),
        "all_max_ulp": (q["maximum ULP error"].max(), lambda x: f"{x:.2f} ULP"),
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def summary_tiles(d: AnalysisData) -> list[tuple[str, str, str]]:
    """(label, value, note) tiles describing what was measured and on what."""
    cells: int = d.everything.filter(pl.col.us.is_not_null()).height + d.pure.height
    samples: float = (
        sum(frame["time_us"].is_not_null().sum() for frame, _ in d.raw.values())
        + d.pure["samples"].sum()  # ty: ignore[unsupported-operator]
    )
    repeats_lo, repeats_hi = repeat_range(d)
    small, large = d.meta["gpu_sweep"]["device"], d.meta["gpu_sweep80"]["device"]
    return [
        ("Cells measured", fmt_int(x=cells), "one operation × dtype × size × setting"),
        (
            "Timing samples",
            fmt_int(x=samples),
            f"{repeats_lo:.0f}–{repeats_hi:.0f} repeats per cell, medians throughout",
        ),
        ("CPU", "AMD EPYC 7763", "Perlmutter CPU node · one core · AVX2"),
        (
            "GPU",
            "NVIDIA A100",
            (
                f"40GB and 80GB · {large['sms']} SMs each · "
                f"{small['peak_hbm_gbs']:,.0f} / {large['peak_hbm_gbs']:,.0f} GB/s peak memory bandwidth"
            ),
        ),
    ]


def _registry(d: AnalysisData) -> dict[str, Finding]:
    return (
        _containers(d)
        | _call_cost(d)
        | _caches(d)
        | _bandwidth(d)
        | _precision(d)
        | _allocation(d)
        | _compilation(d)
        | _measurement(d)
        | _quality(d)
    )


def values(d: AnalysisData) -> dict[str, float]:
    """Every quoted number, unformatted."""
    return {key: float(value) for key, (value, _) in _registry(d).items()}


def compute(d: AnalysisData) -> dict[str, str]:
    """Every quoted number, formatted exactly as the prose prints it."""
    return {key: fmt(float(value)) for key, (value, fmt) in _registry(d).items()}
