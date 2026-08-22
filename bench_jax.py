#!/usr/bin/env python3
"""
jax.numpy counterpart to bench_v2.py (NumPy) and bench_gpu.py (CuPy).

Side experiment, not part of the presented analysis: the question is only how
jax.numpy lands next to the numpy and cupy numbers on the same kind of node.

Same skeleton as the other two -- hot/cold locality, an n sweep, per-op
validation against a float64 NumPy reference, run-by-run timings to .npz with
the same key layout, so clean_npz.py converts it unchanged.

Four things differ because it is JAX:

1. `eager` vs `jit` replaces the out= axis. Eager is one primitive at a time,
   each dispatched separately; jit hands the whole expression to XLA as one
   compiled program, which is the performance model JAX actually recommends.
   Both are measured for every op. The distinction is invisible for `add` (one
   primitive either way) and is the entire story for `muladd`,
   `matmul_explicit` and `chain` -- there the jit column is the fused kernel,
   the analogue of muladd_fused / matmul_explicit_fused in the CuPy script, so
   eager/jit is the HBM round trip for the intermediates plus the extra
   dispatches. `chain` exists only to make that gap big enough to read.

2. There is no equivalent of out=. Ordinary jit output is the alloc case;
   donate_argnums approximates storage reuse but consumes the donated operand,
   which would delete the operand pool after one pass and make hot/cold
   meaningless. It is deliberately not benchmarked here -- the jit column is
   alloc, and should be compared against the CuPy alloc column, not out=.

3. Timing is host-side, because there is no public CUDA-event hook. Compilation
   is forced out of the timed region (validate, probe, then an explicit warmup),
   and the jit cache size is checked before and after timing so a silent
   recompile cannot be mistaken for a slow sample. --timer picks what a sample
   measures, and both are reported side by side at the end:

     pipelined (default)  `inner` calls enqueued back to back, one
                          block_until_ready at the end, divided by inner. The
                          block means execution is never skipped; what you read
                          is steady-state max(dispatch, device) per call, i.e.
                          what a loop actually pays.
     blocking             one call per sample, block_until_ready on each. Honest
                          per-call latency, and charges every sample a full
                          dispatch and sync round trip.

4. dtypes include bfloat16, which has no NumPy equivalent, and float64 requires
   x64 mode (set below via env, before any array is constructed -- JAX is fp32
   by default and would otherwise quietly downgrade the fp64 rows).

Accuracy is not assumed from the op name: every (op, dtype, mode) is checked
against a float64 NumPy reference and the max error is printed in ULPs of the
result dtype, not just pass/fail. exp, cos and fp64 are where JAX and NumPy are
most likely to have made different approximation choices.

On a Perlmutter shared GPU node, note that JAX preallocates 75% of the card by
default; that is disabled here so it coexists with whatever else is on the GPU.

Usage:
    python bench_jax.py                                  # default n, auto backend
    python bench_jax.py --sweep 250000,2500000,25000000 -r 30 --with-numpy
    python bench_jax.py --timer blocking                 # per-call latency
    python bench_jax.py --platform cpu                   # force the CPU backend
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from time import perf_counter

import numpy as np

# Both of these have to be set before jax is imported, so peek at argv rather
# than waiting for argparse.
for _flag, _env in (("--platform", "JAX_PLATFORMS"),
                    ("--matmul-precision", "JAX_DEFAULT_MATMUL_PRECISION")):
    for _i, _a in enumerate(sys.argv):
        _v = (sys.argv[_i + 1] if _a == _flag and _i + 1 < len(sys.argv)
              else _a.split("=", 1)[1] if _a.startswith(_flag + "=") else None)
        if _v and _v != "auto":
            os.environ[_env] = _v

os.environ.setdefault("JAX_ENABLE_X64", "1")
# fp32 matmul on an A100 otherwise silently drops to TF32, which is a different
# experiment and would fail validation against the float64 reference.
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
# Shared node: take what we use, not 75% of the card up front.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    sys.exit("jax is not installed.  CPU:  pip install jax\n"
             "                        GPU:  pip install 'jax[cuda13]'  (or cuda12)")


# --------------------------------------------------------------------------
# dtypes
# --------------------------------------------------------------------------

JAX_DTYPES = {
    "JAX fp16": jnp.float16,
    "JAX bf16": jnp.bfloat16,
    "JAX fp32": jnp.float32,
    "JAX fp64": jnp.float64,
}

NUMPY_DTYPES = {
    "NumPy fp16": np.float16,
    "NumPy fp32": np.float32,
    "NumPy fp64": np.float64,
}


# --------------------------------------------------------------------------
# operations
# --------------------------------------------------------------------------


def build_ops(xp):
    """One definition, instantiated against numpy and against jax.numpy."""

    def matmul_2x2_explicit(a, b):
        c00 = a[:, 0, 0] * b[:, 0, 0] + a[:, 0, 1] * b[:, 1, 0]
        c01 = a[:, 0, 0] * b[:, 0, 1] + a[:, 0, 1] * b[:, 1, 1]
        c10 = a[:, 1, 0] * b[:, 0, 0] + a[:, 1, 1] * b[:, 1, 0]
        c11 = a[:, 1, 0] * b[:, 0, 1] + a[:, 1, 1] * b[:, 1, 1]
        return xp.stack([c00, c01, c10, c11], axis=-1).reshape(a.shape)

    return {
        "add": lambda a, b: a + b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,
        "sqrt": lambda a, b: xp.sqrt(a),
        "exp": lambda a, b: xp.exp(a),
        "cos": lambda a, b: xp.cos(a),
        "muladd": lambda a, b: a * b + a,
        # Seven primitives, one buffer under jit. This is the case that
        # separates primitive-at-a-time dispatch from whole-expression XLA.
        "chain": lambda a, b: xp.exp(-(a * b + a)) * xp.sqrt(b) + a,
        # Batched 2x2 GEMM vs the same product written out. `@` may lower to a
        # generic batched-GEMM path; the explicit form is what XLA can fuse.
        "matmul": lambda a, b: a @ b,
        "matmul_explicit": matmul_2x2_explicit,
    }


OPS_JAX = build_ops(jnp)
OPS_NP = build_ops(np)
OP_NAMES = list(OPS_JAX)
UNARY = ("sqrt", "exp", "cos")          # read one operand, not two


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def make_pairs(shape, dtype, n_pairs, seed, to_device):
    """Operands are built on the host and, for JAX, committed to the device
    before anything is timed; no transfer is ever inside the timer."""
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n_pairs):
        a = (1 + np.abs(rng.standard_normal(shape))).astype(dtype)
        b = (1 + np.abs(rng.standard_normal(shape))).astype(dtype)
        if to_device:
            a, b = jax.device_put(a), jax.device_put(b)
        pairs.append((a, b))
    if to_device:
        jax.block_until_ready(pairs)
    return pairs


def pairs_for_budget(shape, dtype, budget_bytes, cap):
    per_pair = 2 * int(np.prod(shape)) * np.dtype(dtype).itemsize
    return int(max(2, min(cap, budget_bytes // max(per_pair, 1))))


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def result_tolerance(dt):
    try:
        eps = float(jnp.finfo(dt).eps)      # handles bfloat16, np.finfo may not
    except (TypeError, ValueError):
        eps = 0.0
    return 32 * max(eps, float(np.finfo(np.float64).eps))


def mixed_error(got, ref):
    return float(np.max(np.abs(got - ref) / (np.abs(ref) + 1.0)))


def ulp_error(got, ref, dt):
    """Max error in ULPs of `dt`, so fp16/bf16/fp32/fp64 are on one scale.
    frexp puts |ref| at m * 2**e with m in [0.5, 1), so one ULP is eps * 2**(e-1)."""
    try:
        eps = float(jnp.finfo(dt).eps)
    except (TypeError, ValueError):
        return float("nan")
    _, exp2 = np.frexp(ref)
    ulp = np.ldexp(eps, exp2 - 1)
    return float(np.max(np.abs(got - ref) / np.maximum(ulp, np.finfo(np.float64).tiny)))


def validate(op_name, fn, a, b):
    """Against the float64 NumPy reference, exactly as the CuPy script does.
    The measured error is reported whether or not it passes: matching op names
    across libraries does not imply matching approximations."""
    out = {"op": op_name, "ok": True, "note": "", "err": 0.0, "ulp": 0.0,
           "res_dtype": None}
    try:
        got_dev = jax.block_until_ready(fn(a, b))
    except Exception as exc:
        out["ok"] = False
        out["note"] = f"raised {type(exc).__name__}: {exc}"
        return out

    out["res_dtype"] = str(getattr(got_dev, "dtype", "?"))
    got = np.asarray(got_dev).astype(np.float64)
    a_h, b_h = np.asarray(a).astype(np.float64), np.asarray(b).astype(np.float64)
    ref = OPS_NP[op_name](a_h, b_h)

    if got.shape != ref.shape:
        out["ok"] = False
        out["note"] = f"shape {got.shape} != reference {ref.shape}"
        return out
    if not np.all(np.isfinite(got)):
        out["ok"] = False
        out["note"] = "non-finite values in result"
        return out

    out["err"] = err = mixed_error(got, ref)
    out["ulp"] = ulp_error(got, ref, got_dev.dtype)
    tol = result_tolerance(got_dev.dtype)
    if err > tol:
        out["ok"] = False
        out["note"] = f"mismatch vs float64 reference, err {err:.3g} > tol {tol:.3g}"
    return out


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------


class Task:
    def __init__(self, op, dtype_name, mode, locality):
        self.op, self.dtype_name = op, dtype_name
        self.mode, self.locality = mode, locality
        self.key = f"{op}__{dtype_name}__{mode}__{locality}"
        self.fn = None
        self.pairs = []
        self.res_dtype = ""
        self.res_itemsize = 0
        self.op_itemsize = 0
        self.inner = 1
        self.est_us = 0.0
        self.est_sample_us = 0.0

    def _launch(self, base, inner):
        """Enqueue `inner` calls and return the last result, unblocked. Each
        iteration drops the previous result, so device memory stays bounded
        even when inner is large."""
        fn, npairs, out = self.fn, len(self.pairs), None
        if self.locality == "hot":
            a, b = self.pairs[0]
            for _ in range(inner):
                out = fn(a, b)
        else:
            for k in range(inner):
                a, b = self.pairs[(base + k) % npairs]
                out = fn(a, b)
        return out

    def sample(self, base):
        t0 = perf_counter()
        jax.block_until_ready(self._launch(base, self.inner))
        return (perf_counter() - t0) * 1e6 / self.inner

    def sample_blocking(self, base):
        """Per-call latency: block on every call instead of pipelining."""
        n, npairs = max(1, self.inner), len(self.pairs)
        t0 = perf_counter()
        for k in range(n):
            a, b = (self.pairs[0] if self.locality == "hot"
                    else self.pairs[(base + k) % npairs])
            jax.block_until_ready(self.fn(a, b))
        return (perf_counter() - t0) * 1e6 / n

    def calibrate(self, target_us, timer):
        self.inner = 1
        if timer == "blocking":          # one blocked call per sample, by definition
            self.est_us = self.est_sample_us = self.sample_blocking(0)
            return self.inner
        for _ in range(4):
            used = self.inner
            el = self.sample(0) * used
            self.est_us = el / max(used, 1)
            if el > target_us * 0.5:
                break
            self.inner = min(2_000,
                             max(used + 1, int(used * target_us / max(el, 0.05))))
        self.est_sample_us = self.est_us * self.inner
        return self.inner


def build_tasks(dtypes, op_names, shape, repeats, seed, budget_bytes, modes):
    """`budget_bytes` is a TOTAL across dtypes, split evenly, as in bench_gpu."""
    tasks, reports = [], []
    per_dtype_budget = max(1, budget_bytes // max(len(dtypes), 1))

    for dtype_name, dtype in dtypes.items():
        on_device = not dtype_name.startswith("NumPy")
        base_ops = OPS_JAX if on_device else OPS_NP
        n_pairs = pairs_for_budget(shape, dtype, per_dtype_budget, repeats)
        pairs = make_pairs(shape, dtype, n_pairs, seed, to_device=on_device)
        a0, b0 = pairs[0]

        for op in op_names:
            for mode in modes:
                if mode == "jit" and not on_device:
                    continue             # the NumPy rows are eager by definition
                fn = jax.jit(base_ops[op]) if mode == "jit" else base_ops[op]

                v = validate(op, fn, a0, b0)
                v["dtype"], v["mode"] = dtype_name, mode
                reports.append(v)
                if not v["ok"]:
                    continue

                probe = fn(a0, b0)
                for locality in ("hot", "cold"):
                    t = Task(op, dtype_name, mode, locality)
                    t.fn, t.pairs = fn, pairs
                    t.res_dtype = str(probe.dtype)
                    t.res_itemsize = np.dtype(probe.dtype).itemsize
                    t.op_itemsize = np.dtype(dtype).itemsize
                    tasks.append(t)

    random.Random(seed).shuffle(tasks)
    return tasks, reports


def jit_cache_sizes(tasks):
    """Compiled-executable count per jitted task. Comparing this before and
    after the timed loop is how we know compilation stayed out of the samples."""
    sizes = {}
    for t in tasks:
        probe = getattr(t.fn, "_cache_size", None)
        if callable(probe):
            try:
                sizes[t.key] = probe()
            except Exception:            # private API, absent on some versions
                pass
    return sizes


def run(tasks, repeats, seed, target_us, timer):
    times = {t.key: np.empty(repeats, dtype=np.float64) for t in tasks}
    order = list(range(len(tasks)))
    shuffler = random.Random(seed + 1)
    sampler = Task.sample_blocking if timer == "blocking" else Task.sample

    # Every jitted task has already been traced and compiled by validate() and
    # the probe call in build_tasks; this runs it again so the allocator, the
    # module load and any lazy device init are warm too.
    print("  warming up and calibrating ...", file=sys.stderr)
    for t in tasks:
        jax.block_until_ready(t._launch(0, min(4, len(t.pairs))))
    for t in tasks:
        t.calibrate(target_us, timer)
    before = jit_cache_sizes(tasks)

    est_s = sum(t.est_sample_us for t in tasks) * repeats / 1e6
    print(f"  {len(tasks)} tasks, estimated {est_s:.1f} s "
          f"({est_s / 60:.1f} min) of timed work", file=sys.stderr)

    for rep in range(repeats):
        shuffler.shuffle(order)
        for j in order:
            t = tasks[j]
            times[t.key][rep] = sampler(t, rep * t.inner)
        if repeats >= 10 and (rep + 1) % max(1, repeats // 10) == 0:
            print(f"  ... {rep + 1}/{repeats} repeats", file=sys.stderr)

    after = jit_cache_sizes(tasks)
    grew = [k for k, v in after.items() if v != before.get(k, v)]
    if grew:
        print(f"  WARNING: {len(grew)} task(s) recompiled during timing, so "
              f"those samples include compilation:", file=sys.stderr)
        for k in grew[:10]:
            print(f"    {k}: {before.get(k)} -> {after[k]} executables",
                  file=sys.stderr)
    return times, {"checked": len(before), "recompiled": grew}


def dispatch_comparison(tasks):
    """Pipelined vs blocking, same kernels: JAX's async dispatch showing up as
    latency instead of throughput."""
    picks = [t for t in tasks if t.locality == "hot"
             and t.op in ("add", "cos", "muladd", "matmul_explicit")]
    if not picks:
        return
    print(f"\n{'=' * 100}\npipelined vs blocking dispatch, same kernels "
          f"(median us per call)\n{'=' * 100}")
    print(f"{'op':20s} {'dtype':12s} {'mode':6s} {'pipelined':>10s} "
          f"{'blocking':>10s} {'delta':>10s} {'ratio':>8s}")
    for t in picks:
        saved, t.inner = t.inner, 1
        pi = float(np.median([t.sample(i) for i in range(20)]))
        bl = float(np.median([t.sample_blocking(i) for i in range(20)]))
        t.inner = saved
        print(f"{t.op:20s} {t.dtype_name:12s} {t.mode:6s} {pi:10.2f} "
              f"{bl:10.2f} {bl - pi:10.2f} {bl / max(pi, 1e-9):7.1f}x")
    print("  Blocking on every call charges each sample a full dispatch and")
    print("  synchronisation round trip; pipelined is what a loop actually pays.")


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def stat(x):
    return {"median": float(np.median(x)), "min": float(np.min(x))}


def report(tasks, times, dtypes, op_names, n_elem, timer="pipelined"):
    by = {(t.op, t.dtype_name, t.mode, t.locality): stat(times[t.key]) for t in tasks}
    meta = {(t.op, t.dtype_name): (t.res_dtype, t.res_itemsize, t.op_itemsize)
            for t in tasks}

    def get(op, dt, mode, loc):
        v = by.get((op, dt, mode, loc))
        return v["median"] if v else float("nan")

    for op in op_names:
        rows = [dt for dt in dtypes if (op, dt) in meta]
        if not rows:
            continue
        print(f"\n{'=' * 100}\n{op}   (median us per call, {timer})\n{'=' * 100}")
        print(f"{'dtype':12s} {'->result':10s} {'hot jit':>10s} {'cold jit':>10s} "
              f"{'hot eager':>10s} {'cold eager':>11s} {'eager/jit':>10s} "
              f"{'GB/s hot':>9s}")
        for dt in rows:
            res_dtype, res_size, op_size = meta[(op, dt)]
            hj, cj = get(op, dt, "jit", "hot"), get(op, dt, "jit", "cold")
            he, ce = get(op, dt, "eager", "hot"), get(op, dt, "eager", "cold")
            best = hj if not np.isnan(hj) else he
            n_read = 1 if op in UNARY else 2
            gbs = (n_read * op_size + res_size) * n_elem / (best * 1e-6) / 1e9
            ratio = he / hj if hj and not np.isnan(hj) else float("nan")
            print(f"{dt:12s} {res_dtype:10s} {hj:10.2f} {cj:10.2f} {he:10.2f} "
                  f"{ce:11.2f} {ratio:10.2f} {gbs:9.1f}")

    print(f"\n{'=' * 100}\nprecision scaling, hot, median us\n{'=' * 100}")
    have = [d for d in dtypes if any((op, d) in meta for op in op_names)]
    mode_of = {d: ("eager" if d.startswith("NumPy") else "jit") for d in have}
    print(f"{'op':20s} " + " ".join(f"{d:>12s}" for d in have))
    for op in op_names:
        row = [get(op, d, mode_of[d], "hot") for d in have]
        if all(np.isnan(v) for v in row):
            continue
        print(f"{op:20s} " + " ".join(f"{v:12.2f}" for v in row))
    print("\n  JAX columns are the jit form, NumPy columns are eager; bf16 has no")
    print("  NumPy counterpart. Bandwidth-bound ops should track bytes moved.")


def accuracy_report(reports, dtypes, op_names):
    """Measured error against the float64 NumPy reference, in ULPs of the result
    dtype. Printed unconditionally: `jnp.exp` and `np.exp` share a name, not an
    implementation, and the fp64 rows are where that shows up."""
    by = {(r["op"], r["dtype"], r["mode"]): r for r in reports}
    have = [d for d in dtypes if any((op, d, m) in by
                                     for op in op_names for m in ("eager", "jit"))]
    print(f"\n{'=' * 100}\nmax error vs float64 NumPy reference, in ULPs of the "
          f"result dtype\n{'=' * 100}")
    print(f"{'op':20s} {'mode':6s} " + " ".join(f"{d:>12s}" for d in have))
    for op in op_names:
        for mode in ("eager", "jit"):
            row, seen = [], False
            for d in have:
                r = by.get((op, d, mode))
                if r is None:
                    row.append(float("nan"))
                    continue
                seen = True
                row.append(r["ulp"] if r["ok"] else float("nan"))
            if seen:
                print(f"{op:20s} {mode:6s} " + " ".join(f"{v:12.1f}" for v in row))
    print("\n  0 ULP means bit-identical to NumPy at that precision. A fraction of")
    print("  a ULP is just the rounding of the float64 reference down to the")
    print("  result dtype; >1 ULP on exp/cos means XLA and NumPy picked different")
    print("  approximations, which is a result about the libraries rather than an")
    print("  error in the harness. The NumPy fp64 column is 0 by construction --")
    print("  it is the reference. nan entries are combinations that were not run,")
    print("  or that failed validation outright (listed above).")


def sweep_report(results, dtypes, op_names, modes):
    ns = [n for n, _ in results]
    for mode in modes:
        for locality in ("hot", "cold"):
            print(f"\n{'=' * 100}\nns per element, {locality}, {mode}  "
                  f"(elements per array in header)\n{'=' * 100}")
            print(f"{'op':20s} {'dtype':12s} "
                  + "  ".join(f"{n * 4:>11,d}" for n in ns))
            for op in op_names:
                for dt in dtypes:
                    row = []
                    for n, (_, times, _, _, _) in results:
                        key = f"{op}__{dt}__{mode}__{locality}"
                        row.append(np.median(times[key]) * 1e3 / (n * 4)
                                   if key in times else float("nan"))
                    if all(np.isnan(v) for v in row):
                        continue
                    print(f"{op:20s} {dt:12s} "
                          + "  ".join(f"{v:11.4f}" for v in row))
    print("\n  Falling steeply with size => dispatch/launch dominated at small n.")
    print("  Flattening => saturated, and you are reading real throughput.")


def backend_metadata():
    d = jax.devices()[0]
    md = {
        "jax": jax.__version__,
        "platform": d.platform,
        "device_kind": d.device_kind,
        "n_devices": len(jax.devices()),
        "x64": bool(jax.config.jax_enable_x64),
        "matmul_precision": os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        "numpy": np.__version__,
        "host": platform.platform(),
    }
    try:
        ms = d.memory_stats() or {}
        md["mem_limit_gb"] = ms.get("bytes_limit", 0) / 1e9
        md["mem_in_use_gb"] = ms.get("bytes_in_use", 0) / 1e9
    except Exception:
        pass
    return md


# --------------------------------------------------------------------------


def run_one(n, args, dtypes, op_names, modes, quiet=False):
    shape = (n, 2, 2)
    tasks, reports = build_tasks(dtypes, op_names, shape, args.repeats,
                                 args.seed, int(args.budget_gb * 1e9), modes)
    # One operand pool per dtype, shared by every task on it, so count pools.
    pools = {t.dtype_name: (len(t.pairs), t.op_itemsize) for t in tasks}
    footprint = sum(2 * npairs * n * 4 * isize
                    for npairs, isize in pools.values()) / 1e9
    failures = [r for r in reports if not r["ok"]]

    if not quiet:
        print(f"\n  shape {shape} = {n * 4:,} elements, operand footprint "
              f"{footprint:.2f} GB")
        print(f"\n{'=' * 100}\nvalidation: {len(reports) - len(failures)}/"
              f"{len(reports)} passed\n{'=' * 100}")
        for r in failures:
            print(f"  FAIL  {r['dtype']:12s} {r['mode']:6s} {r['op']:20s} {r['note']}")
        if not failures:
            print("  all ops match a float64 NumPy reference")

    print(f"  running {len(tasks)} tasks x {args.repeats} repeats (n={n}) ...",
          file=sys.stderr)
    times, compile_check = run(tasks, args.repeats, args.seed, args.target_us,
                               args.timer)
    return tasks, times, reports, footprint, compile_check


def save_results(path, results, args, md, sizes):
    """Written after every size so a killed sweep keeps its completed sizes.
    Key layout matches bench_gpu.py, so clean_npz.py converts this unchanged;
    the `destination` field carries eager/jit instead of alloc/out."""
    payload, meta_by_n = {}, {}
    for n, (tasks, times, reports, footprint, compile_check) in results:
        for t in tasks:
            payload[f"n{n}__{t.key}" if len(sizes) > 1 else t.key] = times[t.key]
        meta_by_n[str(n)] = {
            "n_elem": n * 4, "footprint_gb": footprint, "validation": reports,
            "inner_reps": {t.key: t.inner for t in tasks},
            "n_pairs": {t.key: len(t.pairs) for t in tasks},
            "compile_check": compile_check,
        }
    payload["__meta__"] = np.array(json.dumps({
        "argv": sys.argv, "sizes": sizes,
        "sizes_completed": [n for n, _ in results], "repeats": args.repeats,
        "timer": args.timer, "backend": md, "per_size": meta_by_n,
    }, default=str))
    np.savez_compressed(path, **payload)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", type=int, default=2_500_000, help="leading dim; (n,2,2)")
    p.add_argument("-r", "--repeats", type=int, default=30)
    p.add_argument("--sweep", default="", help="comma-separated n values")
    p.add_argument("--platform", default="auto", choices=("auto", "cpu", "gpu"),
                   help="force a JAX backend")
    p.add_argument("--matmul-precision", default="highest",
                   choices=("highest", "high", "default"),
                   help="'default' lets fp32 matmul use TF32 on an A100")
    p.add_argument("--timer", choices=("pipelined", "blocking"),
                   default="pipelined",
                   help="pipelined: amortised throughput per call; blocking: "
                        "block_until_ready on every single call, i.e. latency")
    p.add_argument("--budget-gb", type=float, default=8.0,
                   help="cap on the operand set, total across dtypes")
    p.add_argument("--target-us", type=float, default=200.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="jax")
    p.add_argument("--ops", default=",".join(OP_NAMES))
    p.add_argument("--modes", default="eager,jit")
    p.add_argument("--with-numpy", action="store_true",
                   help="also time the same ops through NumPy on this host, "
                        "for a same-node comparison")
    args = p.parse_args()

    op_names = [o.strip() for o in args.ops.split(",") if o.strip()]
    unknown = [o for o in op_names if o not in OP_NAMES]
    if unknown:
        p.error(f"unknown ops: {unknown}. available: {OP_NAMES}")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if any(m not in ("eager", "jit") for m in modes):
        p.error("--modes takes eager, jit, or eager,jit")

    dtypes = dict(JAX_DTYPES)
    if args.with_numpy:
        dtypes.update(NUMPY_DTYPES)

    md = backend_metadata()
    print(f"jax {md['jax']} on {md['platform']}: {md['device_kind']} "
          f"({md['n_devices']} device(s) visible)")
    if "mem_limit_gb" in md:
        print(f"  device memory limit {md['mem_limit_gb']:.1f} GB, "
              f"{md['mem_in_use_gb']:.2f} GB in use")
    print(f"  x64 {md['x64']}, matmul precision {md['matmul_precision']}, "
          f"timer {args.timer}")
    print(f"  {len(dtypes)} dtypes, {len(op_names)} ops, {len(modes)} mode(s), "
          f"{args.repeats} repeats")
    if not md["x64"]:
        print("  WARNING: x64 is off, so the fp64 rows are really fp32.")

    sizes = [int(s) for s in args.sweep.split(",") if s.strip()] or [args.n]
    out = f"bench_{args.tag}.npz"
    results = []
    for i, n in enumerate(sizes):
        results.append((n, run_one(n, args, dtypes, op_names, modes,
                                   quiet=len(sizes) > 1)))
        save_results(out, results, args, md, sizes)
        if i < len(sizes) - 1:
            # Release this size's operand pools before building the next one;
            # otherwise a long sweep holds every size's operands at once.
            for t in results[-1][1][0]:
                t.pairs = []
            print(f"  n={n} done, checkpointed to {out}", file=sys.stderr)

    tasks, times, _, _, _ = results[-1][1]
    if len(sizes) == 1:
        report(tasks, times, list(dtypes), op_names, sizes[0] * 4, args.timer)
    else:
        sweep_report(results, list(dtypes), op_names, modes)
    dispatch_comparison(tasks)              # last size, whose pools are still live
    # Accuracy is a property of the op, not of n; report it once.
    accuracy_report(results[0][1][2], list(dtypes), op_names)

    save_results(out, results, args, md, sizes)
    print(f"\nsaved run-by-run timings + metadata to {out}")
    print("keys are  " + ("n<N>__" if len(sizes) > 1 else "")
          + "<op>__<dtype>__<eager|jit>__<hot|cold>")


if __name__ == "__main__":
    main()
