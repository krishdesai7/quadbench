#!/usr/bin/env python3
"""
Revised numpy dtype/op benchmark.

Fixes over v1:
  * Preallocated `out=` buffers, so arithmetic is timed instead of malloc.
    Both allocating and out= variants are measured, so the allocation cost
    is a reported quantity rather than a confounder.
  * hot/cold axis. `hot` reuses one operand pair (cache-resident -> ALU/SIMD
    bound). `cold` cycles distinct pairs (streaming -> bandwidth bound).
    v1 only had `cold`, which is why add/mul/div all pinned to ~50 GB/s.
  * Round-robin interleaving with a shuffled task order, so drift and noisy
    neighbours smear across every (op, dtype) instead of landing entirely on
    whichever pass happened to be running.
  * No operand mutation. v1's muladd_inplace wrote back into `a`, so ops 7-10
    ran on different data than ops 1-6.
  * Operands are touched by a warm-up sweep before timing, so no pass eats a
    first-touch/page-fault cost that belongs to the memory system.
  * Correctness validation per (op, dtype) against a float64 reference, and
    `a @ b` cross-checked against the hand-rolled 2x2 product.
  * Result dtype recorded, so integer rows that silently promote to float
    (int8->float16, int16->float32, int32+->float64 for sqrt/exp/cos) are
    labelled instead of being read as integer performance.

To answer "how much does the vectorised path actually buy me", run twice on
the same box and diff:

    python bench_v2.py --tag baseline
    NPY_DISABLE_CPU_FEATURES="AVX2,FMA3" python bench_v2.py --tag nosimd

numpy honours NPY_DISABLE_CPU_FEATURES at import time, so the second run uses
the same binary with the AVX2/FMA loops switched off. The ratio between the
two is a direct SIMD-vs-scalar measurement on identical hardware, which is a
much cleaner signal than comparing dtypes against each other.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from time import perf_counter
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# dtypes
# --------------------------------------------------------------------------


def build_dtypes(want_quad: bool):
    dts = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
    }

    # np.float128 is not portable (it does not exist on arm64 macOS, and where
    # it does exist it is usually 80-bit extended padded to 16 bytes).
    ld = np.dtype(np.longdouble)
    if ld != np.dtype(np.float64):
        nmant = np.finfo(np.longdouble).nmant
        dts[f"longdouble{nmant + 1}"] = np.longdouble

    if want_quad:
        try:
            from numpy_quaddtype import QuadPrecDType

            dts["quad-sleef"] = QuadPrecDType(backend="sleef")
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"note: numpy_quaddtype unavailable ({exc}); skipping", file=sys.stderr)

    for n in (8, 16, 32, 64):
        dts[f"int{n}"] = getattr(np, f"int{n}")
        dts[f"uint{n}"] = getattr(np, f"uint{n}")
    return dts


# --------------------------------------------------------------------------
# operations
#
# Each entry is (alloc_fn, out_fn, needs_temp).
#   alloc_fn(a, b)          -> new array, the natural Python expression
#   out_fn(a, b, o, t)      -> writes into preallocated `o`, may use temp `t`
# --------------------------------------------------------------------------


def matmul_2x2_explicit(a, b):
    """v1's hand-rolled stacked-2x2 product, unchanged for comparability."""
    c00 = a[:, 0, 0] * b[:, 0, 0] + a[:, 0, 1] * b[:, 1, 0]
    c01 = a[:, 0, 0] * b[:, 0, 1] + a[:, 0, 1] * b[:, 1, 1]
    c10 = a[:, 1, 0] * b[:, 0, 0] + a[:, 1, 1] * b[:, 1, 0]
    c11 = a[:, 1, 0] * b[:, 0, 1] + a[:, 1, 1] * b[:, 1, 1]
    return np.stack([c00, c01, c10, c11], axis=-1).reshape(a.shape)


def matmul_2x2_explicit_out(a, b, o, t):
    """Same product with no allocation. `t` is a (5, N) scratch buffer."""
    a00, a01, a10, a11 = a[:, 0, 0], a[:, 0, 1], a[:, 1, 0], a[:, 1, 1]
    b00, b01, b10, b11 = b[:, 0, 0], b[:, 0, 1], b[:, 1, 0], b[:, 1, 1]
    c00, c01, c10, c11, s = t[0], t[1], t[2], t[3], t[4]

    np.multiply(a00, b00, out=c00); np.multiply(a01, b10, out=s); np.add(c00, s, out=c00)
    np.multiply(a00, b01, out=c01); np.multiply(a01, b11, out=s); np.add(c01, s, out=c01)
    np.multiply(a10, b00, out=c10); np.multiply(a11, b10, out=s); np.add(c10, s, out=c10)
    np.multiply(a10, b01, out=c11); np.multiply(a11, b11, out=s); np.add(c11, s, out=c11)

    np.copyto(o.reshape(-1, 4), t[:4].T)
    return o


OPS = {
    # name              alloc                          out=                                                    temp
    "add":              (lambda a, b: a + b,           lambda a, b, o, t: np.add(a, b, out=o),                 None),
    "mul":              (lambda a, b: a * b,           lambda a, b, o, t: np.multiply(a, b, out=o),            None),
    "div":              (lambda a, b: a / b,           lambda a, b, o, t: np.divide(a, b, out=o),              None),
    "sqrt":             (lambda a, b: np.sqrt(a),      lambda a, b, o, t: np.sqrt(a, out=o),                   None),
    "exp":              (lambda a, b: np.exp(a),       lambda a, b, o, t: np.exp(a, out=o),                    None),
    "cos":              (lambda a, b: np.cos(a),       lambda a, b, o, t: np.cos(a, out=o),                    None),
    # muladd needs a temporary in the allocating form; the out= form reuses one.
    "muladd":           (lambda a, b: a * b + a,
                         lambda a, b, o, t: np.add(np.multiply(a, b, out=t), a, out=o),                        "like_out"),
    # in-place accumulate onto the *temp*, never onto an operand.
    "muladd_accum":     (lambda a, b: np.add(a * b, a),
                         lambda a, b, o, t: np.add(np.multiply(a, b, out=t), a, out=t),                        "like_out"),
    "matmul":           (lambda a, b: a @ b,           lambda a, b, o, t: np.matmul(a, b, out=o),              None),
    "matmul_explicit":  (matmul_2x2_explicit,          matmul_2x2_explicit_out,                                "stack5"),
}


# --------------------------------------------------------------------------
# data generation
# --------------------------------------------------------------------------


def make_pairs(shape, dtype, n_pairs, rng):
    """n_pairs distinct operand pairs, values in roughly [1, 5.5]."""
    pairs = []
    for _ in range(n_pairs):
        a = (1 + np.abs(rng.standard_normal(shape))).astype(dtype)
        b = (1 + np.abs(rng.standard_normal(shape))).astype(dtype)
        pairs.append((a, b))
    return pairs


def as_f64(x):
    """Best-effort conversion to float64 for reference comparison."""
    try:
        return np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError):
        return x.astype(np.float64)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def result_tolerance(res_dtype):
    """Loosest of (result precision, float64 reference precision)."""
    try:
        eps = float(np.finfo(res_dtype).eps)
    except (TypeError, ValueError):
        eps = 0.0
    return 32 * max(eps, float(np.finfo(np.float64).eps))


def mixed_error(got, ref):
    """Blended relative/absolute error; the +1 keeps cos near its zeros sane."""
    return float(np.max(np.abs(got - ref) / (np.abs(ref) + 1.0)))


def validate(op_name, alloc_fn, a, b):
    """Run the op and check it against a float64 reference. Returns a dict."""
    out = {"op": op_name, "ok": True, "note": "", "err": 0.0, "res_dtype": None}
    try:
        got_raw = alloc_fn(a, b)
    except Exception as exc:
        out["ok"] = False
        out["note"] = f"raised {type(exc).__name__}: {exc}"
        return out

    out["res_dtype"] = str(getattr(got_raw, "dtype", "?"))
    a64, b64 = as_f64(a), as_f64(b)
    ref = alloc_fn(a64, b64)

    try:
        got = as_f64(got_raw)
    except Exception as exc:
        out["ok"] = False
        out["note"] = f"result not convertible to float64: {exc}"
        return out

    if got.shape != ref.shape:
        out["ok"] = False
        out["note"] = f"shape {got.shape} != reference {ref.shape}"
        return out

    if not np.all(np.isfinite(got)):
        out["ok"] = False
        out["note"] = "non-finite values in result"
        return out

    err = mixed_error(got, ref)
    out["err"] = err
    tol = result_tolerance(got_raw.dtype if hasattr(got_raw, "dtype") else np.float64)
    # integer results are exact under the float64 reference; floats get eps room
    if err > max(tol, 1e-6):
        out["ok"] = False
        out["note"] = f"mismatch vs float64 reference, max blended err {err:.3g} > {tol:.3g}"
    return out


def cross_check_matmul(a, b):
    """`a @ b` must agree with the hand-rolled 2x2 product. Catches the case
    where a custom dtype's matmul silently does not compute anything."""
    try:
        got = as_f64(a @ b)
        ref = as_f64(matmul_2x2_explicit(a, b))
    except Exception as exc:
        return False, f"raised {type(exc).__name__}: {exc}"
    err = mixed_error(got, ref)
    # 2x2 accumulates a handful of roundings, so allow a few dtype epsilons.
    # This still catches "matmul returned garbage / did nothing" by orders of
    # magnitude, which is the failure mode that matters.
    tol = result_tolerance(a.dtype)
    return err < tol, f"max blended err vs explicit product {err:.3g} (tol {tol:.3g})"


# --------------------------------------------------------------------------
# benchmark driver
# --------------------------------------------------------------------------


class Task:
    def __init__(self, op, dtype_name, mode, locality, key):
        self.op, self.dtype_name = op, dtype_name
        self.mode, self.locality, self.key = mode, locality, key
        self.fn: Any = None
        self.pairs: list = []
        self.out = None
        self.tmp = None
        self.n_elem = 0
        self.res_dtype = ""
        self.res_itemsize = 0
        self.op_itemsize = 0
        self.inner = 1
        self.est_us = 0.0
        self.est_sample_us = 0.0

    def call(self, i):
        a, b = self.pairs[0] if self.locality == "hot" else self.pairs[i % len(self.pairs)]
        if self.mode == "alloc":
            return self.fn(a, b)
        return self.fn(a, b, self.out, self.tmp)

    def sample(self, base):
        """One timed sample = `inner` back-to-back calls, so that per-call Python
        and perf_counter overhead (~0.5-2 us) does not dominate at small N."""
        fn, inner, npairs = self.fn, self.inner, len(self.pairs)
        if self.locality == "hot":
            a, b = self.pairs[0]
            if self.mode == "alloc":
                for _ in range(inner):
                    fn(a, b)
            else:
                o, t = self.out, self.tmp
                for _ in range(inner):
                    fn(a, b, o, t)
        else:
            if self.mode == "alloc":
                for k in range(inner):
                    a, b = self.pairs[(base + k) % npairs]
                    fn(a, b)
            else:
                o, t = self.out, self.tmp
                for k in range(inner):
                    a, b = self.pairs[(base + k) % npairs]
                    fn(a, b, o, t)

    def calibrate(self, target_us):
        """Pick `inner` so each sample lasts roughly target_us, and record the
        estimated cost of a single call so the driver can budget the run."""
        self.inner = 1
        for _ in range(4):
            used = self.inner
            t0 = perf_counter()
            self.sample(0)
            el = (perf_counter() - t0) * 1e6
            self.est_us = el / max(used, 1)
            if el > target_us * 0.5:
                break
            self.inner = min(100_000,
                             max(used + 1, int(used * target_us / max(el, 0.05))))
        self.est_sample_us = self.est_us * self.inner
        return self.inner


def pairs_for_budget(shape, dtype, budget_bytes, cap):
    """How many distinct operand pairs fit the memory budget for this dtype.

    Decoupled from `repeats` on purpose: at large N, one pair per repeat is what
    produced v1's 60.8 GB footprint. `cold` only needs the operand set to be
    comfortably larger than L3, not larger than RAM.
    """
    per_pair = 2 * int(np.prod(shape)) * np.dtype(dtype).itemsize
    return int(max(2, min(cap, budget_bytes // max(per_pair, 1))))


def build_tasks(dtypes, op_names, shape, repeats, rng, seed, budget_bytes):
    """Allocate buffers, validate, and return the runnable task list.

    `budget_bytes` is a TOTAL across every dtype, split evenly. Applying it per
    dtype is how v1's footprint got away from us: 4 GB each across 13 dtypes is
    52 GB, not 4 GB.
    """
    tasks, reports = [], []
    per_dtype_budget = max(1, budget_bytes // max(len(dtypes), 1))

    for dtype_name, dtype in dtypes.items():
        n_pairs = pairs_for_budget(shape, dtype, per_dtype_budget, repeats)
        pairs = make_pairs(shape, dtype, n_pairs, rng)
        a0, b0 = pairs[0]

        if "matmul" in op_names:
            ok, note = cross_check_matmul(a0, b0)
            reports.append({"dtype": dtype_name, "op": "matmul@cross-check",
                            "ok": ok, "note": note, "err": 0.0, "res_dtype": ""})

        for op in op_names:
            alloc_fn, out_fn, temp_kind = OPS[op]

            v = validate(op, alloc_fn, a0, b0)
            v["dtype"] = dtype_name
            reports.append(v)
            if not v["ok"]:
                continue

            probe = alloc_fn(a0, b0)
            res_dtype, res_shape = probe.dtype, probe.shape
            out_buf = np.empty(res_shape, dtype=res_dtype)

            if temp_kind == "like_out":
                tmp = np.empty(res_shape, dtype=res_dtype)
            elif temp_kind == "stack5":
                tmp = np.empty((5, shape[0]), dtype=res_dtype)
            else:
                tmp = None

            # out= form has to produce the same answer as the allocating form
            out_ok = True
            if out_fn is not None:
                try:
                    out_fn(a0, b0, out_buf, tmp)
                    target = tmp if op == "muladd_accum" else out_buf
                    out_ok = (mixed_error(as_f64(target), as_f64(probe))
                              < result_tolerance(res_dtype))
                    if not out_ok:
                        reports.append({"dtype": dtype_name, "op": f"{op}(out=)", "ok": False,
                                        "note": "out= result differs from allocating form",
                                        "err": 0.0, "res_dtype": ""})
                except Exception as exc:
                    out_ok = False
                    reports.append({"dtype": dtype_name, "op": f"{op}(out=)", "ok": False,
                                    "note": f"raised {type(exc).__name__}: {exc}",
                                    "err": 0.0, "res_dtype": ""})

            n_elem = int(np.prod(shape))
            for mode, fn in (("alloc", alloc_fn), ("out", out_fn if out_ok else None)):
                if fn is None:
                    continue
                for locality in ("hot", "cold"):
                    t = Task(op, dtype_name, mode, locality,
                             f"{op}__{dtype_name}__{mode}__{locality}")
                    t.fn = fn
                    t.pairs = pairs
                    t.out = out_buf
                    t.tmp = tmp
                    t.n_elem = n_elem
                    t.res_dtype = str(res_dtype)
                    t.res_itemsize = res_dtype.itemsize
                    t.op_itemsize = a0.dtype.itemsize
                    tasks.append(t)

    random.Random(seed).shuffle(tasks)
    return tasks, reports


def run(tasks, repeats, seed, target_us, max_call_ms=0.0):
    """Round-robin over every task, reshuffling the order each repeat."""
    order_shuffler = random.Random(seed + 1)

    # warm-up: touch every page and let numpy resolve its dispatch, untimed.
    # This is what stops the first timed pass over a dtype from being charged
    # for first-touch faults -- v1's `add` row was almost entirely this.
    print("  warming up and calibrating ...", file=sys.stderr)
    for t in tasks:
        for i in range(len(t.pairs)):
            t.call(i)
        t.calibrate(target_us)

    if max_call_ms > 0:
        slow = [t for t in tasks if t.est_us > max_call_ms * 1e3]
        if slow:
            tasks = [t for t in tasks if t.est_us <= max_call_ms * 1e3]
            cells = {}
            for t in slow:
                k = (t.op, t.dtype_name)
                cells[k] = max(cells.get(k, 0.0), t.est_us / 1e3)
            worst = sorted(cells.items(), key=lambda kv: -kv[1])
            print(f"  skipping {len(cells)} (op, dtype) cells over "
                  f"--max-call-ms={max_call_ms:g}:", file=sys.stderr)
            for (op, dt), ms in worst[:10]:
                print(f"    {op:18s} {dt:14s} {ms:9.2f} ms/call", file=sys.stderr)
            if len(worst) > 10:
                print(f"    ... and {len(worst) - 10} more", file=sys.stderr)

    est_s = sum(t.est_sample_us for t in tasks) * repeats / 1e6
    print(f"  {len(tasks)} tasks, estimated {est_s:.1f} s "
          f"({est_s / 60:.1f} min) of timed work", file=sys.stderr)

    times = {t.key: np.empty(repeats, dtype=np.float64) for t in tasks}
    order = list(range(len(tasks)))
    shuffler = order_shuffler

    for rep in range(repeats):
        shuffler.shuffle(order)
        for j in order:
            t = tasks[j]
            t0 = perf_counter()
            t.sample(rep * t.inner)
            times[t.key][rep] = (perf_counter() - t0) * 1e6 / t.inner
        if repeats >= 10 and (rep + 1) % max(1, repeats // 10) == 0:
            print(f"  ... {rep + 1}/{repeats} repeats", file=sys.stderr)
    return tasks, times


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def stat(x):
    return {"median": float(np.median(x)), "min": float(np.min(x)),
            "iqr": float(np.subtract(*np.percentile(x, [75, 25])))}


def report(tasks, times, dtypes, op_names, n_elem):
    by = {}
    for t in tasks:
        by[(t.op, t.dtype_name, t.mode, t.locality)] = stat(times[t.key])
    meta = {(t.op, t.dtype_name): (t.res_dtype, t.res_itemsize, t.op_itemsize)
            for t in tasks}

    def get(op, dt, mode, loc, field="median"):
        v = by.get((op, dt, mode, loc))
        return v[field] if v else float("nan")

    for op in op_names:
        print(f"\n{'=' * 104}\n{op}\n{'=' * 104}")
        print(f"{'dtype':14s} {'->result':12s} "
              f"{'hot out=':>10s} {'cold out=':>10s} {'hot alloc':>10s} {'cold alloc':>10s} "
              f"{'ns/elem':>9s} {'alloc cost':>11s} {'cold GB/s':>10s}")
        for dt in dtypes:
            if (op, dt) not in meta:
                continue
            res_dtype, res_size, op_size = meta[(op, dt)]
            ho, co = get(op, dt, "out", "hot"), get(op, dt, "out", "cold")
            ha, ca = get(op, dt, "alloc", "hot"), get(op, dt, "alloc", "cold")
            ns = ho * 1e3 / n_elem
            # unary ops read one operand, binary ops read two
            n_read = 1 if op in ("sqrt", "exp", "cos") else 2
            gbs = (n_read * op_size + res_size) * n_elem / (co * 1e-6) / 1e9
            promoted = " <-" if res_dtype != dt else "   "
            print(f"{dt:14s} {res_dtype:9s}{promoted} {ho:10.2f} {co:10.2f} "
                  f"{ha:10.2f} {ca:10.2f} {ns:9.3f} {ca - co:11.2f} {gbs:10.1f}")
        print("  '<-' marks a dtype numpy promoted before computing: that row is "
              "the promoted type's cost, not the input type's.")

    # ---- the SIMD question, isolated to the compute-bound ops --------------
    print(f"\n{'=' * 104}\nprecision scaling on cache-resident data (hot, out=), "
          f"median us\n{'=' * 104}")
    wide = [d for d in ("float32", "float64") if d in dtypes]
    soft = [d for d in dtypes if d.startswith("longdouble") or d.startswith("quad")]
    print(f"{'op':18s} " + " ".join(f"{d:>14s}" for d in wide + soft)
          + f" {'f64/f32':>9s}")
    for op in op_names:
        row = [get(op, d, "out", "hot") for d in wide + soft]
        if all(np.isnan(r) for r in row):
            continue
        ratio = (get(op, "float64", "out", "hot") / get(op, "float32", "out", "hot")
                 if "float32" in dtypes and "float64" in dtypes else float("nan"))
        print(f"{op:18s} " + " ".join(f"{r:14.2f}" for r in row) + f" {ratio:9.2f}")
    print("\n  f64/f32 near 2.0 => both paths vectorised, cost tracks lane count.")
    print("  f64/f32 well above 2.0 => the float64 loop is falling back to scalar libm.")


def env_metadata():
    md = {
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "NPY_DISABLE_CPU_FEATURES": os.environ.get("NPY_DISABLE_CPU_FEATURES", ""),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
    }
    for mod in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        try:
            m = __import__(mod, fromlist=["_"])
            md["cpu_baseline"] = getattr(m, "__cpu_baseline__", None)
            md["cpu_dispatch"] = getattr(m, "__cpu_dispatch__", None)
            feats = getattr(m, "__cpu_features__", {})
            md["cpu_features_enabled"] = sorted(k for k, v in feats.items() if v)
            break
        except Exception:
            continue
    try:
        md["blas"] = np.__config__.show(mode="dicts")  # numpy >= 2
    except Exception:
        md["blas"] = "unavailable"
    return md


# --------------------------------------------------------------------------


def run_one(n, args, dtypes, op_names, quiet=False):
    shape, n_elem = (n, 2, 2), n * 4
    budget = int(args.budget_gb * 1e9)

    rng = np.random.default_rng(args.seed)
    tasks, reports = build_tasks(dtypes, op_names, shape, args.repeats, rng,
                                 args.seed, budget)

    footprint = sum(2 * len(t.pairs) * n_elem * t.op_itemsize
                    for t in tasks if t.mode == "out" and t.locality == "cold") / 1e9
    failures = [r for r in reports if not r["ok"]]

    if not quiet:
        f64 = n_elem * 8 / 1024
        print(f"\n  shape {shape} = {n_elem} elements "
              f"({f64:.0f} KB per float64 operand), footprint {footprint:.2f} GB")
        print(f"\n{'=' * 104}\nvalidation: {len(reports) - len(failures)}/{len(reports)}"
              f" passed\n{'=' * 104}")
        for r in failures:
            print(f"  FAIL  {r['dtype']:14s} {r['op']:22s} {r['note']}")
        if not failures:
            print("  all ops produce results matching a float64 reference")

    print(f"  running {len(tasks)} tasks x {args.repeats} repeats (n={n}) ...",
          file=sys.stderr)
    tasks, times = run(tasks, args.repeats, args.seed, args.target_us,
                       args.max_call_ms)
    return tasks, times, reports, footprint


def sweep_report(results, dtypes, op_names):
    """ns/element vs working-set size: where the roofline knee sits."""
    ns = [n for n, _ in results]
    for locality in ("hot", "cold"):
        print(f"\n{'=' * 104}\nns per element, {locality}, out=  "
              f"(operand KB per float64 array in header)\n{'=' * 104}")
        hdr = "  ".join(f"{n * 4 * 8 / 1024:>9.0f}K" for n in ns)
        print(f"{'op':18s} {'dtype':12s} {hdr}")
        for op in op_names:
            for dt in dtypes:
                row = []
                for n, (tasks, times, _, _) in results:
                    key = f"{op}__{dt}__out__{locality}"
                    row.append(np.median(times[key]) * 1e3 / (n * 4)
                               if key in times else float("nan"))
                if all(np.isnan(v) for v in row):
                    continue
                print(f"{op:18s} {dt:12s} " + "  ".join(f"{v:10.3f}" for v in row))
    print("\n  Flat across sizes => ALU/latency bound. Rising with size => you have")
    print("  fallen off a cache level and are measuring the memory system instead.")


def save_results(path, results, args, md, sizes):
    """Written after every size, not just at the end, so a long sweep that gets
    killed at the largest n still leaves the completed sizes on disk."""
    payload, meta_by_n = {}, {}
    for n, (tasks, times, reports, footprint) in results:
        for t in tasks:
            payload[f"n{n}__{t.key}" if len(sizes) > 1 else t.key] = times[t.key]
        meta_by_n[str(n)] = {
            "n_elem": n * 4, "footprint_gb": footprint, "validation": reports,
            "inner_reps": {t.key: t.inner for t in tasks},
            "n_pairs": {t.key: len(t.pairs) for t in tasks},
            "result_dtypes": {f"{t.op}__{t.dtype_name}": t.res_dtype for t in tasks},
        }
    payload["__meta__"] = np.array(json.dumps({
        "argv": sys.argv, "sizes": sizes, "sizes_completed": [n for n, _ in results],
        "repeats": args.repeats, "budget_gb": args.budget_gb,
        "target_us": args.target_us, "max_call_ms": args.max_call_ms,
        "env": md, "per_size": meta_by_n,
    }, default=str))
    np.savez_compressed(path, **payload)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", type=int, default=20_000, help="leading dim; arrays are (n,2,2)")
    p.add_argument("-r", "--repeats", type=int, default=50)
    p.add_argument("--sweep", default="", help="comma-separated n values, e.g. "
                                               "500,4000,64000,2000000")
    p.add_argument("--budget-gb", type=float, default=4.0,
                   help="TOTAL cap on the distinct-operand set across all dtypes, "
                        "split evenly; cold mode only needs it larger than L3")
    p.add_argument("--max-call-ms", type=float, default=0.0,
                   help="skip any (op, dtype) whose single call exceeds this. "
                        "Useful at large n, where quad and longdouble cost "
                        "~1 s/call and dominate the whole run")
    p.add_argument("--dtypes", default="", help="comma-separated subset, e.g. "
                                                "float32,float64,int32,int64")
    p.add_argument("--target-us", type=float, default=200.0,
                   help="target duration of one timed sample; calls are repeated "
                        "internally to reach it, so small n stays meaningful")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="v2", help="suffix for the output .npz")
    p.add_argument("--ops", default=",".join(OPS), help="comma-separated subset")
    p.add_argument("--no-quad", action="store_true", help="skip numpy_quaddtype")
    args = p.parse_args()

    op_names = [o.strip() for o in args.ops.split(",") if o.strip()]
    unknown = [o for o in op_names if o not in OPS]
    if unknown:
        p.error(f"unknown ops: {unknown}. available: {list(OPS)}")

    dtypes = build_dtypes(not args.no_quad)
    if args.dtypes:
        want = [d.strip() for d in args.dtypes.split(",") if d.strip()]
        missing = [d for d in want if d not in dtypes]
        if missing:
            p.error(f"unknown dtypes: {missing}. available: {list(dtypes)}")
        dtypes = {d: dtypes[d] for d in want}
    md = env_metadata()

    print(f"numpy {md['numpy']} on {md['machine']} / {md['platform']}")
    if md.get("cpu_baseline"):
        print(f"  SIMD baseline: {md['cpu_baseline']}")
        print(f"  SIMD dispatch: {md['cpu_dispatch']}")
    if md["NPY_DISABLE_CPU_FEATURES"]:
        print(f"  !! NPY_DISABLE_CPU_FEATURES={md['NPY_DISABLE_CPU_FEATURES']}")
    print(f"  {len(dtypes)} dtypes, {len(op_names)} ops, {args.repeats} repeats")

    sizes = [int(s) for s in args.sweep.split(",") if s.strip()] or [args.n]
    out = f"bench_{args.tag}.npz"
    results = []
    for n in sizes:
        results.append((n, run_one(n, args, dtypes, op_names, quiet=len(sizes) > 1)))
        save_results(out, results, args, md, sizes)
        if len(sizes) > 1:
            print(f"  n={n} done, checkpointed to {out}", file=sys.stderr)

    if len(sizes) == 1:
        tasks, times, reports, _ = results[0][1]
        report(tasks, times, list(dtypes), op_names, sizes[0] * 4)
    else:
        sweep_report([(n, r) for n, r in results], list(dtypes), op_names)

    save_results(out, results, args, md, sizes)
    print(f"\nsaved run-by-run timings + metadata to {out}")
    print("keys are  " + ("n<N>__" if len(sizes) > 1 else "")
          + "<op>__<dtype>__<alloc|out>__<hot|cold>")


if __name__ == "__main__":
    main()
