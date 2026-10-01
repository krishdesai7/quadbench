#!/usr/bin/env python3
"""
Pure-Python containers next to the native numpy dtypes.

Three implementations of the same stacked-2x2 workload that bench_v2.py times:

  py-list-float   flat list[float], 4n elements        (math.* for the unary ops)
  py-list-f32     flat list[np.float32], 4n elements   (np.* scalar ufuncs, so the
                  result stays np.float32; math.sqrt would hand back a Python
                  float and quietly stop measuring the type)
  py-obj-array    NDArray[object] of shape (n, 2, 2) holding Python floats. +,
                  *, / and @ dispatch to float's own methods per element. numpy
                  has no sqrt/exp/cos for object dtype (float has no .sqrt()),
                  so those go through np.frompyfunc(math.<fn>): a Python call
                  per element, which is what object arrays cost in practice.

The point is the gap, not the absolute numbers: expect these to be far slower
and far larger than anything in bench_v2.py. Timing machinery (Task, run:
shuffled round-robin, calibration, --max-call-ms) is bench_v2's, so the
numbers are taken the same way. The output archive uses the same key schema
(`n<N>__<op>__<impl>__alloc__<hot|cold>`), so clean_npz.py reads it unchanged.

Only the `alloc` destination exists: a list has no out= form, so `out` and
`out_pool` rows are simply absent (clean_npz emits them as nulls).

Memory is recorded in __meta__ because bench_v2's itemsize arithmetic means
nothing for Python objects. Per implementation and size, bytes per element two
ways: deep sizeof (container + every element object) and tracemalloc across
construction. They should roughly agree; if they do not, one of them is lying.

The matmul op is the explicit 2x2 product on the flat layout (stride 4); for a
list there is no separate "a @ b" to compare it against. matmul_explicit and
muladd_accum from bench_v2 are numpy-specific and are not run here.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import tracemalloc
from typing import Any, Callable

import numpy as np

import bench_v2
from bench_v2 import Task, env_metadata, mixed_error, result_tolerance, run

# --------------------------------------------------------------------------
# operations on flat lists
# --------------------------------------------------------------------------


def _add(a, b):
    return [x + y for x, y in zip(a, b)]


def _mul(a, b):
    return [x * y for x, y in zip(a, b)]


def _div(a, b):
    return [x / y for x, y in zip(a, b)]


def _muladd(a, b):
    return [x * y + x for x, y in zip(a, b)]


def _unary(fn):
    def op(a, _b):
        return [fn(x) for x in a]
    return op


def _matmul_flat(a, b):
    """Stacked 2x2 product on a flat list, row-major, stride 4."""
    out = []
    ext = out.extend
    for i in range(0, len(a), 4):
        a00, a01, a10, a11 = a[i:i + 4]
        b00, b01, b10, b11 = b[i:i + 4]
        ext((a00 * b00 + a01 * b10, a00 * b01 + a01 * b11,
             a10 * b00 + a11 * b10, a10 * b01 + a11 * b11))
    return out


def _list_ops(sqrt, exp, cos) -> dict[str, Callable]:
    return {
        "add": _add, "mul": _mul, "div": _div,
        "sqrt": _unary(sqrt), "exp": _unary(exp), "cos": _unary(cos),
        "muladd": _muladd, "matmul": _matmul_flat,
    }


def _obj_ops() -> dict[str, Callable]:
    sqrt, exp, cos = (np.frompyfunc(f, 1, 1) for f in (math.sqrt, math.exp, math.cos))
    return {
        "add": lambda a, b: a + b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,
        "sqrt": lambda a, _b: sqrt(a),
        "exp": lambda a, _b: exp(a),
        "cos": lambda a, _b: cos(a),
        "muladd": lambda a, b: a * b + a,
        "matmul": lambda a, b: a @ b,
    }


# --------------------------------------------------------------------------
# implementations: how to build an operand, how to size it, which ops apply
# --------------------------------------------------------------------------


class Impl:
    def __init__(self, name, build, ops, res_dtype, ref_dtype):
        self.name, self.build, self.ops = name, build, ops
        self.res_dtype = res_dtype      # label recorded in the archive
        self.ref_dtype = ref_dtype      # precision the result is held to


def _build_list_float(x64):
    return x64.reshape(-1).tolist()


def _build_list_f32(x64):
    return list(x64.astype(np.float32).reshape(-1))


def _build_obj_array(x64):
    return x64.astype(object)           # elements are Python floats


IMPLS: dict[str, Impl] = {
    "py-list-float": Impl("py-list-float", _build_list_float,
                          _list_ops(math.sqrt, math.exp, math.cos),
                          "float", np.float64),
    "py-list-f32": Impl("py-list-f32", _build_list_f32,
                        _list_ops(np.sqrt, np.exp, np.cos),
                        "float32", np.float32),
    "py-obj-array": Impl("py-obj-array", _build_obj_array, _obj_ops(),
                         "object", np.float64),
}
DEFAULT_OPS = ["add", "mul", "div", "sqrt", "exp", "cos", "muladd", "matmul"]


def operand_f64(shape, rng):
    """Same distribution as bench_v2.make_pairs: values in roughly [1, 5.5]."""
    return 1 + np.abs(rng.standard_normal(shape))


def source_f64(x):
    """The float64 value of the operand actually held (f32 rounding included)."""
    return np.asarray(x, dtype=np.float64).reshape(-1)


# --------------------------------------------------------------------------
# memory
# --------------------------------------------------------------------------


def deep_bytes(x) -> int:
    """Container plus every element object. No sharing is assumed, which holds
    here: every element comes from tolist()/astype/list(), so each is its own
    object."""
    if isinstance(x, np.ndarray):
        return x.nbytes + sum(sys.getsizeof(e) for e in x.reshape(-1))
    return sys.getsizeof(x) + sum(sys.getsizeof(e) for e in x)


def measure_memory(impl: Impl, shape, rng) -> dict:
    """Bytes per element for one operand, by deep sizeof and by tracemalloc."""
    n_elem = int(np.prod(shape))
    src = operand_f64(shape, rng)          # built before tracing: not counted
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        x = impl.build(src)
        traced = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    deep = deep_bytes(x)
    res = impl.ops["add"](x, x)
    return {
        "operand_sizeof_B_per_elem": deep / n_elem,
        "operand_tracemalloc_B_per_elem": traced / n_elem,
        "result_sizeof_B_per_elem": deep_bytes(res) / n_elem,
    }


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def validate(op_name, impl: Impl, a, b, ref_a, ref_b):
    """Run the op and compare with bench_v2's float64 numpy op on the same values."""
    out = {"op": op_name, "dtype": impl.name, "ok": True, "note": "",
           "err": 0.0, "res_dtype": impl.res_dtype, "mode": "alloc"}
    try:
        got = np.asarray(impl.ops[op_name](a, b), dtype=np.float64)
    except Exception as exc:
        out.update(ok=False, note=f"raised {type(exc).__name__}: {exc}")
        return out

    ref = np.asarray(bench_v2.OPS[op_name][0](ref_a, ref_b), dtype=np.float64)
    if got.size != ref.size:
        out.update(ok=False, note=f"size {got.size} != reference {ref.size}")
        return out
    got = got.reshape(ref.shape)
    if not np.all(np.isfinite(got)):
        out.update(ok=False, note="non-finite values in result")
        return out

    err = mixed_error(got, ref)
    tol = max(result_tolerance(np.dtype(impl.ref_dtype)), 1e-6)
    out["err"] = err
    if err > tol:
        out.update(ok=False,
                   note=f"mismatch vs float64 reference, max blended err "
                        f"{err:.3g} > {tol:.3g}")
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def build_tasks(impls, op_names, n, repeats, rng, seed, budget_bytes):
    shape, n_elem = (n, 2, 2), n * 4
    tasks, reports, memory, n_pairs_by_impl = [], [], {}, {}
    per_impl_budget = max(1, budget_bytes // max(len(impls), 1))

    for impl in impls:
        mem = measure_memory(impl, shape, np.random.default_rng(seed))
        memory[impl.name] = mem
        per_pair = 2 * n_elem * mem["operand_sizeof_B_per_elem"]
        n_pairs = int(max(2, min(repeats, per_impl_budget // max(per_pair, 1))))
        n_pairs_by_impl[impl.name] = n_pairs

        f64_pairs = [(operand_f64(shape, rng), operand_f64(shape, rng))
                     for _ in range(n_pairs)]
        pairs = [(impl.build(a), impl.build(b)) for a, b in f64_pairs]
        del f64_pairs
        a0, b0 = pairs[0]
        ref_a = source_f64(a0).reshape(shape)
        ref_b = source_f64(b0).reshape(shape)

        for op in op_names:
            v = validate(op, impl, a0, b0, ref_a, ref_b)
            reports.append(v)
            if not v["ok"]:
                continue
            for locality in ("hot", "cold"):
                t = Task(op, impl.name, "alloc", locality,
                         f"{op}__{impl.name}__alloc__{locality}")
                t.fn = impl.ops[op]
                t.pairs = pairs
                t.n_elem = n_elem
                t.res_dtype = impl.res_dtype
                tasks.append(t)

    random.Random(seed).shuffle(tasks)
    return tasks, reports, memory, n_pairs_by_impl


def run_one(n, args, impls, op_names, quiet=False):
    rng = np.random.default_rng(args.seed)
    tasks, reports, memory, n_pairs = build_tasks(
        impls, op_names, n, args.repeats, rng, args.seed,
        int(args.budget_gb * 1e9))
    failures = [r for r in reports if not r["ok"]]

    if not quiet:
        print(f"\n  shape ({n}, 2, 2) = {n * 4} elements")
        for name, m in memory.items():
            print(f"  {name:14s} {m['operand_sizeof_B_per_elem']:6.1f} B/elem "
                  f"(sizeof)  {m['operand_tracemalloc_B_per_elem']:6.1f} B/elem "
                  f"(tracemalloc)  {n_pairs[name]} pairs")
    print(f"  validation: {len(reports) - len(failures)}/{len(reports)} passed",
          file=sys.stderr)
    for r in failures:
        print(f"  FAIL  {r['dtype']:14s} {r['op']:8s} {r['note']}", file=sys.stderr)

    print(f"  running {len(tasks)} tasks x {args.repeats} repeats (n={n}) ...",
          file=sys.stderr)
    tasks, times = run(tasks, args.repeats, args.seed, args.target_us,
                       args.max_call_ms)
    footprint = sum(2 * n_pairs[name] * n * 4 * m["operand_sizeof_B_per_elem"]
                    for name, m in memory.items()) / 1e9
    return tasks, times, reports, footprint, memory, n_pairs


def report(results, impls, op_names):
    """Median ns/element, cold, plus bytes/element, one table per size."""
    for n, (_, times, _, _, memory, _) in results:
        print(f"\n{'=' * 88}\nn={n} ({n * 4} elements): median ns/element, "
              f"cold\n{'=' * 88}")
        print(f"{'op':8s} " + " ".join(f"{i.name:>15s}" for i in impls))
        for op in op_names:
            row = []
            for i in impls:
                key = f"{op}__{i.name}__alloc__cold"
                row.append(np.median(times[key]) * 1e3 / (n * 4)
                           if key in times else float("nan"))
            if all(np.isnan(v) for v in row):
                continue
            print(f"{op:8s} " + " ".join(f"{v:15.1f}" for v in row))
        print(f"{'B/elem':8s} " + " ".join(
            f"{memory[i.name]['operand_sizeof_B_per_elem']:15.1f}" for i in impls))


def save_results(path, results, args, md, sizes):
    payload, meta_by_n = {}, {}
    for n, (tasks, times, reports, footprint, memory, _) in results:
        for t in tasks:
            payload[f"n{n}__{t.key}" if len(sizes) > 1 else t.key] = times[t.key]
        meta_by_n[str(n)] = {
            "n_elem": n * 4, "footprint_gb": footprint, "validation": reports,
            "inner_reps": {t.key: t.inner for t in tasks},
            "n_pairs": {t.key: len(t.pairs) for t in tasks},
            "result_dtypes": {f"{t.op}__{t.dtype_name}": t.res_dtype for t in tasks},
            "memory": memory,
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
    p.add_argument("-n", type=int, default=20_000,
                   help="leading dim; arrays are (n,2,2), 4n elements")
    p.add_argument("-r", "--repeats", type=int, default=30)
    p.add_argument("--sweep", default="", help="comma-separated n values")
    p.add_argument("--budget-gb", type=float, default=4.0,
                   help="TOTAL cap on distinct operands across implementations, "
                        "by deep size (Python objects are ~8x a float64)")
    p.add_argument("--max-call-ms", type=float, default=0.0,
                   help="skip any (op, impl) whose single call exceeds this")
    p.add_argument("--impls", default=",".join(IMPLS))
    p.add_argument("--ops", default=",".join(DEFAULT_OPS))
    p.add_argument("--target-us", type=float, default=200.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="pure", help="suffix for the output .npz")
    args = p.parse_args()

    op_names = [o.strip() for o in args.ops.split(",") if o.strip()]
    unknown = [o for o in op_names if o not in DEFAULT_OPS]
    if unknown:
        p.error(f"unknown ops: {unknown}. available: {DEFAULT_OPS}")
    names = [i.strip() for i in args.impls.split(",") if i.strip()]
    missing = [i for i in names if i not in IMPLS]
    if missing:
        p.error(f"unknown impls: {missing}. available: {list(IMPLS)}")
    impls = [IMPLS[i] for i in names]

    md = env_metadata()
    print(f"numpy {md['numpy']}, python {md['python']} on {md['machine']} / "
          f"{md['platform']}")
    print(f"  {len(impls)} implementations, {len(op_names)} ops, "
          f"{args.repeats} repeats")

    sizes = [int(s) for s in args.sweep.split(",") if s.strip()] or [args.n]
    out = f"bench_{args.tag}.npz"
    results = []
    for n in sizes:
        results.append((n, run_one(n, args, impls, op_names, quiet=len(sizes) > 1)))
        save_results(out, results, args, md, sizes)
        if len(sizes) > 1:
            print(f"  n={n} done, checkpointed to {out}", file=sys.stderr)

    report(results, impls, op_names)
    print(f"\nsaved run-by-run timings + metadata to {out}")
    print("keys are  " + ("n<N>__" if len(sizes) > 1 else "")
          + "<op>__<impl>__alloc__<hot|cold>")


if __name__ == "__main__":
    main()
