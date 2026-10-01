from time import perf_counter
import numpy as np
from numpy_quaddtype import QuadPrecDType

N = 10_000
REPEATS = 10_000

dtypes = {
    "np.float16": np.float16,
    "np.float32": np.float32,
    "np.float64": np.float64,
    "np.float80": np.float128,
    "quad-sleef": QuadPrecDType(backend="sleef"),
    "np.int8": np.int8,
    "np.uint8": np.uint8,
    "np.int16": np.int16,
    "np.uint16": np.uint16,
    "np.int32": np.int32,
    "np.uint32": np.uint32,
    "np.int64": np.int64,
    "np.uint64": np.uint64,
}

# Pre-generate REPEATS distinct operand pairs per dtype
arrays = {}
rng = np.random.default_rng(seed=42)

for name, dtype in dtypes.items():
    pairs = []
    for _ in range(REPEATS):
        a_raw = 1 + np.abs(rng.standard_normal((N, 2, 2)))
        b_raw = 1 + np.abs(rng.standard_normal((N, 2, 2)))
        pairs.append((a_raw.astype(dtype), b_raw.astype(dtype)))
    arrays[name] = pairs


def bench_distinct(op, operand_pairs):
    # Warmup on the first batch
    _ = op(*operand_pairs[0])

    times = np.empty(len(operand_pairs), dtype=np.float64)
    for i, (a, b) in enumerate(operand_pairs):
        t0 = perf_counter()
        _ = op(a, b)
        times[i] = (perf_counter() - t0) * 1e6  # Microseconds
    return times


def matmul_2x2_explicit(a, b):
    c00 = a[:, 0, 0] * b[:, 0, 0] + a[:, 0, 1] * b[:, 1, 0]
    c01 = a[:, 0, 0] * b[:, 0, 1] + a[:, 0, 1] * b[:, 1, 1]
    c10 = a[:, 1, 0] * b[:, 0, 0] + a[:, 1, 1] * b[:, 1, 0]
    c11 = a[:, 1, 0] * b[:, 0, 1] + a[:, 1, 1] * b[:, 1, 1]
    return np.stack([c00, c01, c10, c11], axis=-1).reshape(a.shape)


operations = {
    "add": lambda a, b: a + b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "sqrt": lambda a, b: np.sqrt(a),
    "exp": lambda a, b: np.exp(a),
    "muladd_inplace": lambda a, b: np.add(a * b, a, out=a),
    "muladd": lambda a, b: a * b + a,
    "matmul": lambda a, b: a @ b,
    "cos": lambda a, b: np.cos(a),
    "matmul_explicit": matmul_2x2_explicit,
}

raw_benchmark_data = {}

for opname, op in operations.items():
    print(f"\n{opname}")

    results_avg = {}

    for name, operand_pairs in arrays.items():
        times = bench_distinct(op, operand_pairs)

        # Store individual run-by-run times for saving
        # Key format: 'add__np.float64'
        raw_benchmark_data[f"{opname}__{name}"] = times

        results_avg[name] = np.mean(times)

    baseline = results_avg["np.float64"]

    for name, t_avg in results_avg.items():
        print(
            f"{name:15s} "
            f"{t_avg:9.2f} µs avg  "
            f"{t_avg/baseline:7.2f}x vs np.float64"
        )

# Save all run-by-run timings into benchmark_results.npz
np.savez_compressed("benchmark_results.npz", **raw_benchmark_data)
print("\nSaved run-by-run timings to benchmark_results.npz")
