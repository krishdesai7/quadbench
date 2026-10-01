#!/bin/bash
#SBATCH -A m3246
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -t 02:00:00
#SBATCH -J quadbench-pure
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.out

# Pure-Python containers (list[float], list[np.float32], NDArray[object]) next
# to the native dtypes. CPU only, single-threaded. No -c: the shared QOS then
# hands out one core and its proportional share of memory (~4 GB), which is why
# --budget-gb is kept small below.
# Writes to a NEW run directory, so nothing already in data/ is touched:
#
#   data/cpu_pure   9 sizes, 400 to 4M elements (n = 100 .. 1M)
#
# The cap at n = 1M is deliberate: past it a single call is seconds, and the
# trend is already unambiguous. --max-call-ms drops anything slower than 2 s.
# Rough cost: 25-40 min, dominated by py-list-f32 at the top two sizes.
#
# Submit from the repo root, naming the account on the command line:
#   sbatch -A m3246 sbatch_pure.sh
# SBATCH_ACCOUNT in the environment (e.g. m3246_g) outranks the #SBATCH -A line
# above, and a GPU account with -C cpu is rejected as "does not match any
# supported policy".

cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

echo "=== $(date) | job ${SLURM_JOB_ID:-?} on $(hostname) ==="

DIR=cpu_pure
NPZ=bench_pure.npz

if [[ -e "data/$DIR" ]]; then
    echo "FATAL: data/$DIR already exists, refusing to overwrite it." >&2
    exit 1
fi

rm -f "$NPZ"
if ! uv run bench_pure.py \
    --sweep 100,320,1000,3200,10000,32000,100000,320000,1000000 \
    -r 30 --budget-gb 2 --max-call-ms 2000 --tag pure; then
    echo "FAILED: bench_pure.py exited nonzero"
    [[ -e "$NPZ" ]] || exit 1
    echo "  a partial checkpoint exists; keeping it"
fi

mkdir -p "data/$DIR" && mv "$NPZ" "data/$DIR/" || exit 1
uv run clean_npz.py "$DIR" || { echo "archive kept, clean_npz failed" >&2; exit 1; }

# Ops that failed validation only ever land in the JSON; surface them here.
uv run python - "data/$DIR/$DIR.json" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
bad = {}
for size, block in meta["per_size"].items():
    for v in block.get("validation", []):
        if not v.get("ok"):
            bad.setdefault((v["op"], v["dtype"], v["note"][:60]), []).append(size)
if bad:
    print("  DROPPED OPS:")
    for (op, impl, note), sizes in sorted(bad.items()):
        print(f"    {op} / {impl} at {len(sizes)} size(s): {note}")
else:
    print("  all ops validated")
print("\n  bytes per element (sizeof / tracemalloc) at the largest size:")
last = meta["per_size"][str(meta["sizes_completed"][-1])]["memory"]
for impl, m in last.items():
    print(f"    {impl:14s} {m['operand_sizeof_B_per_elem']:6.1f} / "
          f"{m['operand_tracemalloc_B_per_elem']:6.1f}")
PY
echo "=== done $(date) ==="
