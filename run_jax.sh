#!/bin/bash
#SBATCH -A m3246_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 02:00:00
#SBATCH -J quadbench-jax
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.out

# jax.numpy side experiment: how jax lands next to the numpy and cupy numbers.
# Writes to NEW run directories, so nothing already in data/ is touched:
#
#   data/jax_sweep    full op grid at the four gpu_sweep80 sizes
#   data/jax_dense    size sweep, the ops where eager/jit fusion is the story
#   data/jax_latency  the same small sizes with --timer blocking
#   data/jax_cpu      CPU backend + NumPy rows, same node, for the host baseline
#
# Sizes match sbatch_gpu.sh so the parquet joins straight onto the cupy runs.
# Submit from the repo root:   sbatch run_jax.sh

cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

echo "=== $(date) | job ${SLURM_JOB_ID:-?} on $(hostname) ==="

# JAX ships its own CUDA in the jax-cuda13-plugin wheels, so unlike the cupy job
# this deliberately does NOT module load cudatoolkit -- a system CUDA on
# LD_LIBRARY_PATH ahead of the wheels is the usual way this breaks.
module list 2>&1

# Fail fast rather than discovering afterwards that every JAX run silently fell
# back to the CPU backend and produced numbers that mean nothing.
if ! uv run python -c "
import jax, sys
d = jax.devices()
print('devices:', d)
sys.exit(0 if d[0].platform == 'gpu' else 1)
"; then
    echo "FATAL: JAX did not find a GPU -- every run below would be CPU." >&2
    echo "       Check nvidia-smi and that jax-cuda13-plugin installed." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# run <run_dir> <tag> <bench args...>
#   benches, files the archive under data/<run_dir>/, converts it. Never aborts
#   the job -- a later run should still get its turn if an earlier one dies.
# ---------------------------------------------------------------------------
STATUS=()
run() {
    local dir="$1" tag="$2"; shift 2
    local npz="bench_${tag}.npz"
    echo
    echo "############################################################"
    echo "### $dir  ($(date +%H:%M:%S))"
    echo "###   uv run bench_jax.py $*"
    echo "############################################################"

    if [[ -e "data/$dir" ]]; then
        echo "SKIP: data/$dir already exists, refusing to overwrite it."
        STATUS+=("$dir: SKIPPED (directory exists)")
        return
    fi

    local note="ok"
    rm -f "$npz"
    if ! uv run bench_jax.py "$@" --tag "$tag"; then
        echo "FAILED: bench_jax.py exited nonzero for $dir"
        if [[ ! -e "$npz" ]]; then
            STATUS+=("$dir: FAILED, no archive written")
            return
        fi
        echo "  a partial checkpoint exists; keeping it"
        note="PARTIAL (bench exited nonzero)"
    fi

    if ! { mkdir -p "data/$dir" && mv "$npz" "data/$dir/"; }; then
        STATUS+=("$dir: FAILED to file the archive")
        return
    fi
    if ! uv run clean_npz.py "$dir"; then
        STATUS+=("$dir: archive kept, clean_npz failed")
        return
    fi

    # Ops that failed validation, plus any task that recompiled inside the timed
    # loop. Both only ever land in the JSON; this is the only place they surface.
    uv run python - "data/$dir/$dir.json" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
bad, recompiled = {}, 0
for size, block in meta["per_size"].items():
    for v in block.get("validation", []):
        if not v.get("ok"):
            bad.setdefault((v.get("op"), v.get("dtype"), v.get("mode"),
                            v.get("note", "")[:60]), []).append(size)
    recompiled += len(block.get("compile_check", {}).get("recompiled", []))
if bad:
    print("  DROPPED OPS:")
    for (op, dt, mode, note), sizes in sorted(bad.items()):
        print(f"    {op} / {dt} / {mode} at {len(sizes)} size(s): {note}")
else:
    print("  all ops validated")
print(f"  recompiles inside timed loops: {recompiled}"
      + ("  <-- those samples include compilation" if recompiled else ""))
PY
    STATUS+=("$dir: $note")
}

# ---------------------------------------------------------------------------
# 1. Full op grid at the gpu_sweep80 sizes, both eager and jit. This is the run
#    that lines up against data/gpu_sweep80 op for op.
# ---------------------------------------------------------------------------
run jax_sweep sweep \
    --sweep 20000,250000,2500000,25000000 -r 30

# ---------------------------------------------------------------------------
# 2. Dense size sweep on the ops where whole-expression jit is the story:
#    chain and matmul_explicit fuse, matmul goes down the batched-GEMM path.
#    250M elements needs ~14 GB of operands, which fits on an 80GB card.
# ---------------------------------------------------------------------------
run jax_dense dense \
    --sweep 250,500,1000,2500,5000,10000,25000,50000,100000,250000,500000,1000000,2500000,6250000,25000000,62500000 \
    --ops add,exp,chain,matmul,matmul_explicit -r 30 --budget-gb 8

# ---------------------------------------------------------------------------
# 3. Per-call latency at the small sizes: block_until_ready on every call, which
#    is the JAX analogue of the host-timer run in sbatch_gpu.sh. The delta
#    against jax_dense at the same n is the dispatch + sync round trip.
# ---------------------------------------------------------------------------
run jax_latency latency \
    --sweep 250,1000,5000,25000,100000,500000,2500000 \
    --ops add,exp --timer blocking -r 30

# ---------------------------------------------------------------------------
# 4. CPU backend on the same node, with the NumPy rows alongside. Gives a
#    jax-cpu vs numpy vs jax-gpu comparison that shares one host, which is the
#    only way that particular ratio means anything.
# ---------------------------------------------------------------------------
run jax_cpu cpu \
    --platform cpu --sweep 20000,250000,2500000 \
    --ops add,exp,chain,matmul,matmul_explicit -r 30 --with-numpy --budget-gb 4

echo
echo "=== summary ($(date)) ==="
printf '  %s\n' "${STATUS[@]}"
echo
echo "Parquet + JSON are in data/jax_sweep, data/jax_dense, data/jax_latency,"
echo "data/jax_cpu. The destination column holds eager/jit, not alloc/out."
