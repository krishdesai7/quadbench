#!/bin/bash
#SBATCH -A m3246_g
#SBATCH -C gpu&hbm80g
#SBATCH -q shared
#SBATCH -G 1
#SBATCH -c 32
#SBATCH -t 03:00:00
#SBATCH -J quadbench-gpu
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.out
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=you@example.com

# All outstanding GPU work in one job. Writes to NEW run directories, so nothing
# already in data/ is touched:
#
#   data/gpu_sweep80   full op grid at the original four sizes  (restores matmul)
#   data/gpu_dense80   16 sizes from 1k to 250M elements        (replaces gpu_dense)
#   data/gpu_host80    host timer at small sizes                (launch overhead)
#
# Each run is unpacked to Parquet on the spot, and any op the harness dropped is
# printed at the end -- those failures only ever land in the run's JSON, never on
# the console, which is how the cuBLAS ImportError went unnoticed last time.
#
# Submit from the repo root:   sbatch sbatch_gpu.sh

cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

echo "=== $(date) | job ${SLURM_JOB_ID:-?} on $(hostname) ==="

module load cudatoolkit/13.2 nccl/2.29.2-cu13
module list 2>&1

# Fail fast and loudly rather than discovering afterwards that every cuBLAS op
# was silently dropped from the archive.
if ! uv run python -c "import cupy as cp; a=cp.ones((4,2,2)); assert float((a@a).sum())==32.0"; then
    echo "FATAL: cuBLAS unavailable after module load -- matmul would be dropped." >&2
    echo "       Check that cudatoolkit/13.2 is really loaded, then resubmit." >&2
    exit 1
fi
uv run python - <<'PY'
import cupy as cp
d = cp.cuda.Device(0)
name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
free, total = d.mem_info
print(f"device: {name}   free/total HBM: {free/2**30:.1f}/{total/2**30:.1f} GiB")
PY

# ---------------------------------------------------------------------------
# run <run_dir> <tag> <bench args...>
#   benches, files the archive under data/<run_dir>/, converts it, and reports
#   any dropped op. Never aborts the job -- a later run should still get its
#   turn if an earlier one dies.
# ---------------------------------------------------------------------------
STATUS=()
run() {
    local dir="$1" tag="$2"; shift 2
    local npz="bench_${tag}.npz"
    echo
    echo "############################################################"
    echo "### $dir  ($(date +%H:%M:%S))"
    echo "###   uv run bench_gpu.py $*"
    echo "############################################################"

    if [[ -e "data/$dir" ]]; then
        echo "SKIP: data/$dir already exists, refusing to overwrite it."
        STATUS+=("$dir: SKIPPED (directory exists)")
        return
    fi

    local note="ok"
    rm -f "$npz"
    if ! uv run bench_gpu.py "$@" --tag "$tag"; then
        echo "FAILED: bench_gpu.py exited nonzero for $dir"
        # the harness checkpoints after every size, so a partial archive is
        # still worth keeping
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

    # Ops the harness validated as broken and dropped. Silent on the console
    # during the run; this is the only place they surface.
    uv run python - "data/$dir/$dir.json" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
bad = {}
for size, block in meta["per_size"].items():
    for v in block.get("validation", []):
        if not v.get("ok"):
            bad.setdefault((v.get("op"), v.get("dtype"), v.get("note", "")[:70]), []).append(size)
if bad:
    print("  DROPPED OPS:")
    for (op, dt, note), sizes in sorted(bad.items()):
        print(f"    {op} / {dt} at {len(sizes)} size(s): {note}")
else:
    print("  all ops validated")
PY
    STATUS+=("$dir: $note")
}

# ---------------------------------------------------------------------------
# 1. Full op grid, original sizes. Default --ops = all 11, so this is the run
#    that brings back matmul, matmul_explicit and matmul_explicit_fused. It is
#    the direct replacement for the 40GB gpu_sweep, so Part 3 can be one card.
# ---------------------------------------------------------------------------
run gpu_sweep80 sweep80 \
    --sweep 20000,250000,2500000,25000000 -r 30

# ---------------------------------------------------------------------------
# 2. Dense size sweep, now including matmul, and reaching the two top sizes the
#    previous attempt never delivered (the archive on my end stops at n=2.5M).
#    250M elements needs ~14 GB of operands, which fits on an 80GB card.
# ---------------------------------------------------------------------------
run gpu_dense80 dense80 \
    --sweep 250,500,1000,2500,5000,10000,25000,50000,100000,250000,500000,1000000,2500000,6250000,25000000,62500000 \
    --ops add,mul,exp,matmul -r 30 --budget-gb 8

# ---------------------------------------------------------------------------
# 3. Host timer at the small sizes. host - event isolates the ~8.4 us per-call
#    floor; re-run here only so every GPU number shares one card and one module
#    environment.
# ---------------------------------------------------------------------------
run gpu_host80 host80 \
    --sweep 250,1000,5000,25000,100000,500000,2500000 \
    --ops add,exp --timer host -r 30

echo
echo "=== summary ($(date)) ==="
printf '  %s\n' "${STATUS[@]}"
echo
echo "Parquet + JSON are in data/gpu_sweep80, data/gpu_dense80, data/gpu_host80."
