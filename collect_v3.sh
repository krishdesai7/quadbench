#!/usr/bin/env bash
# Data collection for the v3 analysis. Run on the x86 box for 1-3, on the A100
# node for 4-5. Each block is independent; run them in any order, and skip any
# you do not want. Every command checkpoints after each size, so a run that is
# killed still leaves the completed sizes on disk.
#
# After each run, move the .npz into data/<run>/ and clean it:
#     mkdir -p data/cpu_dense && mv bench_dense.npz data/cpu_dense/
#     uv run clean_npz.py cpu_dense
#
# Rough costs are measured wall time from the existing runs scaled by task
# count; they assume nothing else is on the machine.

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Dense CPU size sweep                                        ~45-75 min
# ---------------------------------------------------------------------------
# 18 log-spaced sizes instead of 3, on 4 dtypes instead of 13. This is the run
# that turns the three-point slopegraphs into curves and locates the cache
# transitions instead of asserting them. n is the leading dim; elements = 4n,
# so this spans 400 to 8M elements (3 KB to 64 MB per float64 operand).
numactl --cpunodebind=0 --membind=0 \
uv run bench_v2.py \
  --sweep 100,180,320,560,1000,1800,3200,5600,10000,18000,32000,56000,100000,180000,320000,560000,1000000,2000000 \
  --dtypes float16,float32,float64,int64,quad-sleef \
  --ops add,mul,div,exp,sqrt,matmul \
  -r 30 --budget-gb 8 --max-call-ms 50 --tag dense

# ---------------------------------------------------------------------------
# 2. The allocation experiment                                   ~30-50 min
# ---------------------------------------------------------------------------
# --out-pool-gb enables the third destination mode (see the bench_v2.py
# docstring). It cycles the destination through a 2 GB pool of preallocated,
# already-faulted buffers, so the destination is cache-cold without being
# freshly allocated. That splits the alloc-vs-out difference into
# read-for-ownership and allocation, which is currently a hypothesis in the
# notebook rather than a measurement.
#
# Needs ~2 GB per distinct result buffer shape on top of the operand budget;
# with these dtypes that is about 6 GB of pools. Drop --out-pool-gb to 1.0 if
# the box is tight. Sizes are only the ones that bracket the sign flip.
numactl --cpunodebind=0 --membind=0 \
uv run bench_v2.py \
  --sweep 4000,16000,64000,256000,1000000,2000000 \
  --dtypes float32,float64,int32,int64 \
  --ops add,mul,exp \
  -r 30 --budget-gb 8 --out-pool-gb 2.0 --tag alloc

# ---------------------------------------------------------------------------
# 3. cpu_dram, more repeats                                      ~25 min
# ---------------------------------------------------------------------------
# Same configuration as the existing cpu_dram run, at 50 repeats instead of 20.
# It is the noisiest run in the set (p90 IQR 21%) and it carries the
# 12%-vs-46% allocation claim, which currently has no error bars behind it.
numactl --cpunodebind=0 --membind=0 \
uv run bench_v2.py -n 2000000 -r 50 \
  --dtypes float16,float32,float64,int32,int64 \
  --max-call-ms 50 --budget-gb 8 --tag dram50

# ---------------------------------------------------------------------------
# 3b. Is 27.7 GB/s a core limit or a socket limit?               ~5 min
# ---------------------------------------------------------------------------
# The notebook calls 27.7 GB/s a ceiling without anything to calibrate it
# against. Four concurrent single-threaded copies pinned to one node: if
# aggregate throughput scales, 27.7 is a per-core limit and the "one bandwidth
# ceiling" reading is about the core, not the memory system.
for i in 0 1 2 3; do
  numactl --physcpubind=$i --membind=0 \
  uv run bench_v2.py -n 2000000 -r 10 --dtypes float64 --ops add \
    --budget-gb 2 --tag "par$i" &
done
wait
# Also useful, and cheap:  lscpu > data/lscpu.txt

# ---------------------------------------------------------------------------
# 3c. Why float16 is slow on this CPU                            ~10 min
# ---------------------------------------------------------------------------
# In the existing cpu_sweep, float16 add costs 6.0 ns/element against float32's
# 0.25 -- 24x SLOWER for half the bytes -- and it costs the same 6.0 ns whether
# the op is an add or a cosine, and the same at 2 000 elements as at 256 000.
# A per-element cost that ignores both the operation and the memory system is
# a scalar loop, not arithmetic.
#
# The obvious explanation is that float16 has no native ALU and everything
# routes through float32. But this CPU reports F16C, which converts eight
# halves per instruction, so conversion should be nearly free. add_via_f32
# tests that directly: it widens to float32 by hand, operates, and narrows
# back, allocating three intermediates on the way. If it BEATS the native
# float16 ufunc, the cost is numpy's inner loop, not the format.
#
# The float32 rows are the control: promote_types makes the widening a no-op
# there, so they price the astype copies on their own. (A smoke run on an
# arm64 laptop already shows add 176 us vs add_via_f32 25 us at n=20 000;
# worth confirming on the real box.)
numactl --cpunodebind=0 --membind=0 \
uv run bench_v2.py \
  --sweep 4000,64000,1000000 \
  --dtypes float16,float32,float64 \
  --ops add,mul,exp,add_via_f32,mul_via_f32,exp_via_f32 \
  -r 30 --budget-gb 4 --tag f16

# ---------------------------------------------------------------------------
# 4. Dense GPU size sweep                                        ~20-30 min
# ---------------------------------------------------------------------------
# 16 sizes instead of 4, reaching down to 1000 elements -- below anything the
# CPU runs cover. This is what makes the saturation curve real (it currently
# has one point below 1M) and what lets the CPU/GPU comparison have an overlap
# region instead of two disjoint segments.
uv run bench_gpu.py \
  --sweep 250,500,1000,2500,5000,10000,25000,50000,100000,250000,500000,1000000,2500000,6250000,25000000,62500000 \
  --ops add,mul,exp,matmul \
  -r 30 --budget-gb 8 --tag gpudense

# ---------------------------------------------------------------------------
# 5. GPU launch overhead, measured rather than inferred          ~5 min
# ---------------------------------------------------------------------------
# Same small sizes, host timer instead of CUDA events. host - event is the
# launch and synchronisation overhead, which Part 3 currently attributes
# without measuring.
uv run bench_gpu.py \
  --sweep 250,1000,5000,25000,100000,500000,2500000 \
  --ops add,exp --timer host \
  -r 30 --tag gpuhost
