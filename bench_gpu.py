#!/usr/bin/env python3
"""
CuPy counterpart to bench_v2.py, for A100-class hardware.

Same structure as the CPU script so the two sets of numbers are comparable:
hot/cold locality, alloc vs out=, an n sweep, per-op validation against a
float64 NumPy reference, and run-by-run timings saved to .npz.

Three things differ because it is a GPU:

1. Timing. Host-side perf_counter with cuda.Stream.null.synchronize() around
   each call is correct but charges every sample the kernel launch (~3-5 us)
   and the sync return. An elementwise add over 80k float32 elements is ~0.4 us
   of real GPU time on an A100, so host timing would be >90% overhead. CUDA
   events timestamp on the device instead. Both are implemented; --timer
   selects, and the report prints the two side by side so the gap is visible.

2. Size. An A100 has 1.5 TB/s of HBM and needs ~10-100M elements in flight to
   saturate. At n=20_000 (80k elements) you are measuring launch overhead, not
   arithmetic. The default sweep goes to n=25_000_000 for that reason.

3. Fusion. Stacked-2x2 matmul via `a @ b` dispatches to gemmStridedBatched,
   which is a poor fit for 2x2. The elementwise form is better but still
   round-trips every intermediate through HBM. A single fused kernel does the
   whole product in one pass and is the honest upper bound; all three are
   measured.

Usage:
    python bench_gpu.py                         # default n
    python bench_gpu.py --sweep 20000,250000,2500000,25000000 -r 30
    python bench_gpu.py --timer host            # your original timing method
    python bench_gpu.py --device 1              # pick a GPU
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from time import perf_counter
from typing import Any

import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    sys.exit("cupy is not installed. For CUDA 12.x:  pip install cupy-cuda12x")


# --------------------------------------------------------------------------
# dtypes
# --------------------------------------------------------------------------

FLOAT_DTYPES = {
    "GPU fp16": cp.float16,
    "GPU fp32": cp.float32,
    "GPU fp64": cp.float64,
}

INT_DTYPES = {
    "GPU int8": cp.int8, "GPU uint8": cp.uint8,
    "GPU int16": cp.int16, "GPU uint16": cp.uint16,
    "GPU int32": cp.int32, "GPU uint32": cp.uint32,
    "GPU int64": cp.int64, "GPU uint64": cp.uint64,
}


# --------------------------------------------------------------------------
# operations
# --------------------------------------------------------------------------


def matmul_2x2_explicit(a, b):
    """Stacked 2x2 product built from elementwise ufuncs, as in the CPU script.
    Every intermediate is a separate kernel and a separate HBM round trip."""
    c00 = a[:, 0, 0] * b[:, 0, 0] + a[:, 0, 1] * b[:, 1, 0]
    c01 = a[:, 0, 0] * b[:, 0, 1] + a[:, 0, 1] * b[:, 1, 1]
    c10 = a[:, 1, 0] * b[:, 0, 0] + a[:, 1, 1] * b[:, 1, 0]
    c11 = a[:, 1, 0] * b[:, 0, 1] + a[:, 1, 1] * b[:, 1, 1]
    return cp.stack([c00, c01, c10, c11], axis=-1).reshape(a.shape)


def matmul_2x2_explicit_out(a, b, o, t):
    """Same, writing into preallocated buffers. `t` is a (5, n) scratch."""
    a00, a01, a10, a11 = a[:, 0, 0], a[:, 0, 1], a[:, 1, 0], a[:, 1, 1]
    b00, b01, b10, b11 = b[:, 0, 0], b[:, 0, 1], b[:, 1, 0], b[:, 1, 1]
    c00, c01, c10, c11, s = t[0], t[1], t[2], t[3], t[4]

    cp.multiply(a00, b00, out=c00); cp.multiply(a01, b10, out=s); cp.add(c00, s, out=c00)
    cp.multiply(a00, b01, out=c01); cp.multiply(a01, b11, out=s); cp.add(c01, s, out=c01)
    cp.multiply(a10, b00, out=c10); cp.multiply(a11, b10, out=s); cp.add(c10, s, out=c10)
    cp.multiply(a10, b01, out=c11); cp.multiply(a11, b11, out=s); cp.add(c11, s, out=c11)

    o.reshape(-1, 4)[...] = t[:4].T
    return o


# One kernel, one read of a and b, one write of c. No intermediates in HBM.
_MM2X2_SRC = r"""
    const int base = i * 4;
    const T a00 = a[base], a01 = a[base+1], a10 = a[base+2], a11 = a[base+3];
    const T b00 = b[base], b01 = b[base+1], b10 = b[base+2], b11 = b[base+3];
    c[base]   = a00 * b00 + a01 * b10;
    c[base+1] = a00 * b01 + a01 * b11;
    c[base+2] = a10 * b00 + a11 * b10;
    c[base+3] = a10 * b01 + a11 * b11;
"""

try:
    _mm2x2_kernel = cp.ElementwiseKernel(
        "raw T a, raw T b", "raw T c", _MM2X2_SRC, "mm2x2_fused")
except Exception:  # pragma: no cover
    _mm2x2_kernel = None


def matmul_2x2_fused(a, b):
    o = cp.empty_like(a)
    _mm2x2_kernel(a, b, o, size=a.shape[0])
    return o


def matmul_2x2_fused_out(a, b, o, t):
    _mm2x2_kernel(a, b, o, size=a.shape[0])
    return o


try:
    _muladd_kernel = cp.ElementwiseKernel(
        "T a, T b", "T c", "c = a * b + a", "muladd_fused")
except Exception:  # pragma: no cover
    _muladd_kernel = None


OPS = {
    # name                    alloc                         out=                                                     temp
    "add":                    (lambda a, b: a + b,          lambda a, b, o, t: cp.add(a, b, out=o),                  None),
    "mul":                    (lambda a, b: a * b,          lambda a, b, o, t: cp.multiply(a, b, out=o),             None),
    "div":                    (lambda a, b: a / b,          lambda a, b, o, t: cp.divide(a, b, out=o),               None),
    "sqrt":                   (lambda a, b: cp.sqrt(a),     lambda a, b, o, t: cp.sqrt(a, out=o),                    None),
    "exp":                    (lambda a, b: cp.exp(a),      lambda a, b, o, t: cp.exp(a, out=o),                     None),
    "cos":                    (lambda a, b: cp.cos(a),      lambda a, b, o, t: cp.cos(a, out=o),                     None),
    # two kernels, one HBM round trip for the intermediate
    "muladd":                 (lambda a, b: a * b + a,
                               lambda a, b, o, t: cp.add(cp.multiply(a, b, out=t), a, out=o),                        "like_out"),
    # one kernel; the delta against `muladd` is the cost of that HBM round trip
    "muladd_fused":           (lambda a, b: _muladd_kernel(a, b),
                               lambda a, b, o, t: _muladd_kernel(a, b, o),                                           None),
    "matmul":                 (lambda a, b: a @ b,          lambda a, b, o, t: cp.matmul(a, b, out=o),               None),
    "matmul_explicit":        (matmul_2x2_explicit,         matmul_2x2_explicit_out,                                 "stack5"),
    "matmul_explicit_fused":  (matmul_2x2_fused,            matmul_2x2_fused_out,                                    None),
}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def make_pairs(shape, dtype, n_pairs, seed):
    """Operands generated on device; no host transfer is ever timed."""
    rng = cp.random.default_rng(seed)
    pairs = []
    for _ in range(n_pairs):
        a = (1 + cp.abs(rng.standard_normal(shape, dtype=cp.float64))).astype(dtype)
        b = (1 + cp.abs(rng.standard_normal(shape, dtype=cp.float64))).astype(dtype)
        pairs.append((a, b))
    return pairs


def pairs_for_budget(shape, dtype, budget_bytes, cap):
    per_pair = 2 * int(np.prod(shape)) * np.dtype(dtype).itemsize
    return int(max(2, min(cap, budget_bytes // max(per_pair, 1))))


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def result_tolerance(dt):
    try:
        eps = float(np.finfo(dt).eps)
    except (TypeError, ValueError):
        eps = 0.0
    return 32 * max(eps, float(np.finfo(np.float64).eps))


def mixed_error(got, ref):
    return float(np.max(np.abs(got - ref) / (np.abs(ref) + 1.0)))


def numpy_equivalent(op_name, a_h, b_h):
    """Reference computed on the host in float64."""
    if op_name in ("matmul_explicit", "matmul_explicit_fused", "matmul"):
        c00 = a_h[:, 0, 0] * b_h[:, 0, 0] + a_h[:, 0, 1] * b_h[:, 1, 0]
        c01 = a_h[:, 0, 0] * b_h[:, 0, 1] + a_h[:, 0, 1] * b_h[:, 1, 1]
        c10 = a_h[:, 1, 0] * b_h[:, 0, 0] + a_h[:, 1, 1] * b_h[:, 1, 0]
        c11 = a_h[:, 1, 0] * b_h[:, 0, 1] + a_h[:, 1, 1] * b_h[:, 1, 1]
        return np.stack([c00, c01, c10, c11], axis=-1).reshape(a_h.shape)
    return {
        "add": lambda: a_h + b_h,
        "mul": lambda: a_h * b_h,
        "div": lambda: a_h / b_h,
        "sqrt": lambda: np.sqrt(a_h),
        "exp": lambda: np.exp(a_h),
        "cos": lambda: np.cos(a_h),
        "muladd": lambda: a_h * b_h + a_h,
        "muladd_fused": lambda: a_h * b_h + a_h,
    }[op_name]()


def validate(op_name, alloc_fn, a, b):
    out = {"op": op_name, "ok": True, "note": "", "err": 0.0, "res_dtype": None}
    try:
        got_dev = alloc_fn(a, b)
        cp.cuda.Stream.null.synchronize()
    except Exception as exc:
        out["ok"] = False
        out["note"] = f"raised {type(exc).__name__}: {exc}"
        return out

    out["res_dtype"] = str(getattr(got_dev, "dtype", "?"))
    got = cp.asnumpy(got_dev).astype(np.float64)
    ref = numpy_equivalent(op_name, cp.asnumpy(a).astype(np.float64),
                           cp.asnumpy(b).astype(np.float64))

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
    tol = result_tolerance(got_dev.dtype)
    if err > tol:
        out["ok"] = False
        out["note"] = f"mismatch vs float64 reference, err {err:.3g} > tol {tol:.3g}"
    return out


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------


class Task:
    def __init__(self, op, dtype_name, mode, locality, key):
        self.op, self.dtype_name = op, dtype_name
        self.mode, self.locality, self.key = mode, locality, key
        self.fn: Any = None
        self.pairs: list = []
        self.out = None
        self.tmp = None
        self.res_dtype = ""
        self.res_itemsize = 0
        self.op_itemsize = 0
        self.inner = 1
        self.est_us = 0.0
        self.est_sample_us = 0.0

    def _launch(self, base, inner):
        fn, npairs = self.fn, len(self.pairs)
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

    def sample_event(self, base, ev_start, ev_end):
        """Device-side timing. Launches are enqueued back to back between two
        event records, so per-launch host overhead is amortised over `inner`."""
        ev_start.record()
        self._launch(base, self.inner)
        ev_end.record()
        ev_end.synchronize()
        return cp.cuda.get_elapsed_time(ev_start, ev_end) * 1e3 / self.inner  # us

    def sample_host(self, base, _s=None, _e=None):
        """Host-side timing, i.e. the sync/perf_counter/sync pattern. Includes
        launch and sync round trip in every sample."""
        cp.cuda.Stream.null.synchronize()
        t0 = perf_counter()
        self._launch(base, self.inner)
        cp.cuda.Stream.null.synchronize()
        return (perf_counter() - t0) * 1e6 / self.inner

    def calibrate(self, target_us, sampler, ev_start, ev_end):
        self.inner = 1
        for _ in range(4):
            used = self.inner
            el = sampler(self, 0, ev_start, ev_end) * used
            self.est_us = el / max(used, 1)
            if el > target_us * 0.5:
                break
            self.inner = min(20_000,
                             max(used + 1, int(used * target_us / max(el, 0.05))))
        self.est_sample_us = self.est_us * self.inner
        return self.inner


def build_tasks(dtypes, op_names, shape, repeats, seed, budget_bytes):
    """`budget_bytes` is a TOTAL across all dtypes, split evenly. Per-dtype it
    would be 8 GB x 3 = 24 GB of operands at n=25e6, which does not leave room
    on a 40 GB A100 for output and scratch buffers."""
    tasks, reports = [], []
    per_dtype_budget = max(1, budget_bytes // max(len(dtypes), 1))

    free_b, _ = cp.cuda.runtime.memGetInfo()
    want = sum(2 * pairs_for_budget(shape, d, per_dtype_budget, repeats)
               * int(np.prod(shape)) * np.dtype(d).itemsize for d in dtypes.values())
    if want > 0.8 * free_b:
        sys.exit(f"operands would need {want / 1e9:.1f} GB but only "
                 f"{free_b / 1e9:.1f} GB is free on the device.\n"
                 f"Lower --budget-gb, --repeats, or -n.")

    for dtype_name, dtype in dtypes.items():
        n_pairs = pairs_for_budget(shape, dtype, per_dtype_budget, repeats)
        pairs = make_pairs(shape, dtype, n_pairs, seed)
        a0, b0 = pairs[0]

        for op in op_names:
            alloc_fn, out_fn, temp_kind = OPS[op]
            if op == "matmul_explicit_fused" and _mm2x2_kernel is None:
                continue
            if op == "muladd_fused" and _muladd_kernel is None:
                continue

            v = validate(op, alloc_fn, a0, b0)
            v["dtype"] = dtype_name
            reports.append(v)
            if not v["ok"]:
                continue

            probe = alloc_fn(a0, b0)
            res_dtype, res_shape = probe.dtype, probe.shape
            out_buf = cp.empty(res_shape, dtype=res_dtype)

            if temp_kind == "like_out":
                tmp = cp.empty(res_shape, dtype=res_dtype)
            elif temp_kind == "stack5":
                tmp = cp.empty((5, shape[0]), dtype=res_dtype)
            else:
                tmp = None

            out_ok = True
            if out_fn is not None:
                try:
                    got = out_fn(a0, b0, out_buf, tmp)
                    cp.cuda.Stream.null.synchronize()
                    target = got if got is not None else out_buf
                    out_ok = (mixed_error(cp.asnumpy(target).astype(np.float64),
                                          cp.asnumpy(probe).astype(np.float64))
                              < result_tolerance(res_dtype))
                    if not out_ok:
                        reports.append({"dtype": dtype_name, "op": f"{op}(out=)",
                                        "ok": False, "err": 0.0, "res_dtype": "",
                                        "note": "out= differs from allocating form"})
                except Exception as exc:
                    out_ok = False
                    reports.append({"dtype": dtype_name, "op": f"{op}(out=)", "ok": False,
                                    "err": 0.0, "res_dtype": "",
                                    "note": f"raised {type(exc).__name__}: {exc}"})

            for mode, fn in (("alloc", alloc_fn), ("out", out_fn if out_ok else None)):
                if fn is None:
                    continue
                for locality in ("hot", "cold"):
                    t = Task(op, dtype_name, mode, locality,
                             f"{op}__{dtype_name}__{mode}__{locality}")
                    t.fn, t.pairs, t.out, t.tmp = fn, pairs, out_buf, tmp
                    t.res_dtype = str(res_dtype)
                    t.res_itemsize = res_dtype.itemsize
                    t.op_itemsize = a0.dtype.itemsize
                    tasks.append(t)

    random.Random(seed).shuffle(tasks)
    return tasks, reports


def run(tasks, repeats, seed, target_us, timer):
    sampler = Task.sample_event if timer == "event" else Task.sample_host
    ev_start, ev_end = cp.cuda.Event(), cp.cuda.Event()
    times = {t.key: np.empty(repeats, dtype=np.float64) for t in tasks}
    order = list(range(len(tasks)))
    shuffler = random.Random(seed + 1)

    print("  warming up and calibrating ...", file=sys.stderr)
    for t in tasks:                      # warm up: JIT, module load, memory pool
        t._launch(0, min(4, len(t.pairs)))
    cp.cuda.Stream.null.synchronize()
    for t in tasks:
        t.calibrate(target_us, sampler, ev_start, ev_end)

    est_s = sum(t.est_sample_us for t in tasks) * repeats / 1e6
    print(f"  {len(tasks)} tasks, estimated {est_s:.1f} s "
          f"({est_s / 60:.1f} min) of timed work", file=sys.stderr)

    for rep in range(repeats):
        shuffler.shuffle(order)
        for j in order:
            t = tasks[j]
            times[t.key][rep] = sampler(t, rep * t.inner, ev_start, ev_end)
        if repeats >= 10 and (rep + 1) % max(1, repeats // 10) == 0:
            print(f"  ... {rep + 1}/{repeats} repeats", file=sys.stderr)
    return times


def timer_comparison(tasks, seed, target_us):
    """Same tasks measured both ways, so the host-timing overhead is explicit."""
    ev_start, ev_end = cp.cuda.Event(), cp.cuda.Event()
    picks = [t for t in tasks if t.mode == "out" and t.locality == "hot"
             and t.op in ("add", "cos", "matmul_explicit_fused")]
    if not picks:
        return
    print(f"\n{'=' * 100}\nCUDA-event vs host-sync timing, same kernels "
          f"(median us per call)\n{'=' * 100}")
    print(f"{'op':24s} {'dtype':12s} {'inner':>6s} {'event':>10s} {'host':>10s} "
          f"{'host-event':>11s} {'overhead':>9s}")
    for t in picks:
        saved, t.inner = t.inner, 1        # inner=1 is the honest per-call cost
        ev = float(np.median([Task.sample_event(t, i, ev_start, ev_end)
                              for i in range(20)]))
        ho = float(np.median([Task.sample_host(t, i, ev_start, ev_end)
                              for i in range(20)]))
        t.inner = saved
        print(f"{t.op:24s} {t.dtype_name:12s} {1:6d} {ev:10.2f} {ho:10.2f} "
              f"{ho - ev:11.2f} {ho / max(ev, 1e-9):8.1f}x")
    print("  The gap is launch + sync round trip. It is roughly constant, so it")
    print("  swamps small kernels and disappears into large ones.")


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def stat(x):
    return {"median": float(np.median(x)), "min": float(np.min(x))}


def report(tasks, times, dtypes, op_names, n_elem, peak_gbs):
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
        print(f"\n{'=' * 100}\n{op}\n{'=' * 100}")
        print(f"{'dtype':12s} {'->result':10s} {'hot out=':>10s} {'cold out=':>10s} "
              f"{'hot alloc':>10s} {'cold alloc':>11s} {'GB/s hot':>9s} {'% peak':>8s}")
        for dt in rows:
            res_dtype, res_size, op_size = meta[(op, dt)]
            ho, co = get(op, dt, "out", "hot"), get(op, dt, "out", "cold")
            ha, ca = get(op, dt, "alloc", "hot"), get(op, dt, "alloc", "cold")
            n_read = 1 if op in ("sqrt", "exp", "cos") else 2
            gbs = (n_read * op_size + res_size) * n_elem / (ho * 1e-6) / 1e9
            print(f"{dt:12s} {res_dtype:10s} {ho:10.2f} {co:10.2f} {ha:10.2f} "
                  f"{ca:11.2f} {gbs:9.1f} {100 * gbs / peak_gbs:7.1f}%")

    print(f"\n{'=' * 100}\nprecision scaling, hot out=, median us\n{'=' * 100}")
    have = [d for d in ("GPU fp16", "GPU fp32", "GPU fp64") if d in dtypes]
    print(f"{'op':24s} " + " ".join(f"{d:>12s}" for d in have)
          + f" {'f64/f32':>9s} {'f32/f16':>9s}")
    for op in op_names:
        row = [get(op, d, "out", "hot") for d in have]
        if all(np.isnan(v) for v in row):
            continue
        r1 = (get(op, "GPU fp64", "out", "hot") / get(op, "GPU fp32", "out", "hot")
              if "GPU fp64" in dtypes and "GPU fp32" in dtypes else float("nan"))
        r2 = (get(op, "GPU fp32", "out", "hot") / get(op, "GPU fp16", "out", "hot")
              if "GPU fp32" in dtypes and "GPU fp16" in dtypes else float("nan"))
        print(f"{op:24s} " + " ".join(f"{v:12.2f}" for v in row)
              + f" {r1:9.2f} {r2:9.2f}")
    print("\n  Bandwidth-bound ops track bytes moved, so both ratios should sit")
    print("  near 2.0. A100 vector fp64 is half-rate against fp32 (9.7 vs 19.5")
    print("  TFLOPS), so compute-bound ops land near 2.0 as well -- unlike a")
    print("  consumer card, where fp64 is 1/32 rate and the ratio blows out.")
    print("  Note fp16 here is native silicon, not NumPy's float32 emulation.")


def sweep_report(results, dtypes, op_names):
    ns = [n for n, _ in results]
    for locality in ("hot", "cold"):
        print(f"\n{'=' * 100}\nns per element, {locality}, out=  "
              f"(elements per array in header)\n{'=' * 100}")
        print(f"{'op':24s} {'dtype':12s} "
              + "  ".join(f"{n * 4:>11,d}" for n in ns))
        for op in op_names:
            for dt in dtypes:
                row = []
                for n, (_, times, _, _) in results:
                    key = f"{op}__{dt}__out__{locality}"
                    row.append(np.median(times[key]) * 1e3 / (n * 4)
                               if key in times else float("nan"))
                if all(np.isnan(v) for v in row):
                    continue
                print(f"{op:24s} {dt:12s} " + "  ".join(f"{v:11.4f}" for v in row))
    print("\n  Falling steeply with size => launch overhead dominated at small n.")
    print("  Flattening => the GPU is saturated and you are reading real throughput.")


def device_metadata(dev):
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    attrs = dev.attributes
    clock_khz = attrs.get("MemoryClockRate", 0)
    bus_bits = attrs.get("GlobalMemoryBusWidth", 0)
    peak_gbs = 2 * clock_khz * 1e3 * bus_bits / 8 / 1e9
    if peak_gbs <= 0:  # attribute names vary across CuPy versions
        peak_gbs = float("nan")
    free_b, total_b = cp.cuda.runtime.memGetInfo()
    return {
        "name": props["name"].decode(),
        "cc": f"{props['major']}.{props['minor']}",
        "sms": attrs.get("MultiProcessorCount"),
        "peak_hbm_gbs": peak_gbs,
        "mem_total_gb": total_b / 1e9,
        "mem_free_gb": free_b / 1e9,
        "cupy": cp.__version__,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }, peak_gbs


# --------------------------------------------------------------------------


def run_one(n, args, dtypes, op_names, quiet=False):
    shape = (n, 2, 2)
    tasks, reports = build_tasks(dtypes, op_names, shape, args.repeats,
                                 args.seed, int(args.budget_gb * 1e9))
    footprint = sum(2 * len(t.pairs) * n * 4 * t.op_itemsize
                    for t in tasks if t.mode == "out" and t.locality == "cold") / 1e9
    failures = [r for r in reports if not r["ok"]]

    if not quiet:
        print(f"\n  shape {shape} = {n * 4:,} elements, device footprint "
              f"{footprint:.2f} GB")
        print(f"\n{'=' * 100}\nvalidation: {len(reports) - len(failures)}/"
              f"{len(reports)} passed\n{'=' * 100}")
        for r in failures:
            print(f"  FAIL  {r['dtype']:12s} {r['op']:24s} {r['note']}")
        if not failures:
            print("  all ops match a float64 NumPy reference")

    print(f"  running {len(tasks)} tasks x {args.repeats} repeats (n={n}) ...",
          file=sys.stderr)
    times = run(tasks, args.repeats, args.seed, args.target_us, args.timer)
    return tasks, times, reports, footprint


def save_results(path, results, args, md, sizes, peak_gbs):
    """Written after every size so a killed sweep keeps its completed sizes."""
    payload, meta_by_n = {}, {}
    for n, (tasks, times, reports, footprint) in results:
        for t in tasks:
            payload[f"n{n}__{t.key}" if len(sizes) > 1 else t.key] = times[t.key]
        meta_by_n[str(n)] = {
            "n_elem": n * 4, "footprint_gb": footprint, "validation": reports,
            "inner_reps": {t.key: t.inner for t in tasks},
            "n_pairs": {t.key: len(t.pairs) for t in tasks},
        }
    payload["__meta__"] = np.array(json.dumps({
        "argv": sys.argv, "sizes": sizes,
        "sizes_completed": [n for n, _ in results], "repeats": args.repeats,
        "timer": args.timer, "device": md, "peak_hbm_gbs": peak_gbs,
        "per_size": meta_by_n,
    }, default=str))
    np.savez_compressed(path, **payload)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", type=int, default=2_500_000, help="leading dim; (n,2,2)")
    p.add_argument("-r", "--repeats", type=int, default=50)
    p.add_argument("--sweep", default="", help="comma-separated n values")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--timer", choices=("event", "host"), default="event")
    p.add_argument("--budget-gb", type=float, default=8.0,
                   help="cap on the device-side operand set per dtype")
    p.add_argument("--target-us", type=float, default=200.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="gpu")
    p.add_argument("--ops", default=",".join(OPS))
    p.add_argument("--with-ints", action="store_true",
                   help="also run the integer dtypes, for parity with the CPU run")
    args = p.parse_args()

    op_names = [o.strip() for o in args.ops.split(",") if o.strip()]
    unknown = [o for o in op_names if o not in OPS]
    if unknown:
        p.error(f"unknown ops: {unknown}. available: {list(OPS)}")

    dev = cp.cuda.Device(args.device)
    dev.use()
    md, peak_gbs = device_metadata(dev)

    dtypes = dict(FLOAT_DTYPES)
    if args.with_ints:
        dtypes.update(INT_DTYPES)

    print(f"cupy {md['cupy']} on device {args.device}: {md['name']} (sm_{md['cc']}, "
          f"{md['sms']} SMs)")
    print(f"  HBM {md['mem_total_gb']:.1f} GB, {md['mem_free_gb']:.1f} GB free, "
          f"peak bandwidth {peak_gbs:.0f} GB/s")
    print(f"  timer: {args.timer}, {len(dtypes)} dtypes, {len(op_names)} ops, "
          f"{args.repeats} repeats")
    if _mm2x2_kernel is None:
        print("  note: fused 2x2 kernel failed to compile; that op is skipped")

    sizes = [int(s) for s in args.sweep.split(",") if s.strip()] or [args.n]
    out = f"bench_{args.tag}.npz"
    results = []
    for n in sizes:
        results.append((n, run_one(n, args, dtypes, op_names, quiet=len(sizes) > 1)))
        cp.get_default_memory_pool().free_all_blocks()
        save_results(out, results, args, md, sizes, peak_gbs)
        if len(sizes) > 1:
            print(f"  n={n} done, checkpointed to {out}", file=sys.stderr)

    if len(sizes) == 1:
        tasks, times, _, _ = results[0][1]
        report(tasks, times, list(dtypes), op_names, sizes[0] * 4, peak_gbs)
        timer_comparison(tasks, args.seed, args.target_us)
    else:
        sweep_report(results, list(dtypes), op_names)

    save_results(out, results, args, md, sizes, peak_gbs)
    print(f"\nsaved run-by-run timings + metadata to {out}")
    print("keys are  " + ("n<N>__" if len(sizes) > 1 else "")
          + "<op>__<dtype>__<alloc|out>__<hot|cold>")


if __name__ == "__main__":
    main()
