#!/usr/bin/env bash
# verify.sh — automated verifier for the LDS stride-padding task.
#
# Usage:
#   ./verify.sh <path-to-llama-build-dir> [baseline.json]
#
# What it does (exit 0 = pass):
#   1. Correctness: test-backend-ops MUL_MAT full sweep (all dtypes/shapes,
#      including shapes the candidate never sees — the held-out set).
#   2. Performance: paired benchmark of the primary shape (m=4096 n=512
#      k=14336, q4_0_rocmfp4_fast) plus the two held-out shapes via
#      `test-backend-ops perf -o MUL_MAT`, N=7 synchronized runs, first
#      discarded, median kept per shape.
#   3. Compares against the pinned baseline.json (primary shape must beat
#      it by >=12%; held-out shapes must not regress by >2%).
#   4. Reviewer checks (NOT automated by this script):
#      - shader checksum: SHMEM_STRIDE_PAD must be the ONLY diff vs the
#        starting commit (git diff --stat restricted to the allowed file);
#      - coopmat path proof: run the benchmark once with
#        RADV_DEBUG=shaderstats:nocache and confirm mul_mm coopmat kernels
#        appear in the stats.
set -uo pipefail

BUILD="${1:?usage: verify.sh <build-dir> [baseline.json]}"
BASELINE_JSON="${2:-baseline.json}"
N=7
TOL_PERF=0.12   # required speedup over baseline
TOL_REGRESS=0.02

BIN="$BUILD/bin"
[ -x "$BIN/test-backend-ops" ] || { echo "FAIL: no test-backend-ops in $BIN"; exit 1; }

echo "== 1. correctness (MUL_MAT full sweep) =="
"$BIN/test-backend-ops" -o MUL_MAT 2>&1 | tail -3
CORRECT=$("$BIN/test-backend-ops" -o MUL_MAT 2>&1 | grep -c 'result = ok' || true)
TOTAL=$("$BIN/test-backend-ops" -o MUL_MAT 2>&1 | grep -c 'result =' || true)
echo "correctness: $CORRECT/$TOTAL ok"
[ "$TOTAL" -gt 0 ] || { echo "FAIL: no MUL_MAT results parsed (parser or binary broken)"; exit 1; }
[ "$CORRECT" = "$TOTAL" ] || { echo "FAIL: correctness"; exit 1; }

echo "== 2. performance (paired, first run discarded, median of $((N-1))) =="
run_perf() { # runs perf sweep, prints TFLOPS lines for primary + held-out shapes
  "$BIN/test-backend-ops" perf -o MUL_MAT 2>&1 | grep -E 'q4_0_rocmfp4_fast.*(m=4096.*n=512.*k=14336|m=1024.*n=512.*k=14336|m=4096.*n=2048.*k=5120)'
}
# discard boost
run_perf > /dev/null
for i in $(seq 2 $N); do run_perf; done > perf_runs.txt
# median per shape via python
TOL_PERF="$TOL_PERF" TOL_REGRESS="$TOL_REGRESS" N_OBS="$((N-1))" python3 - "$BASELINE_JSON" <<'EOF'
import os, re, statistics, sys, json
runs = {}
for line in open('perf_runs.txt'):
    m = re.search(r'shape=\(([^)]+)\).*?([0-9.]+)\s*(GFLOPS|TFLOPS)', line)
    if m:
        val = float(m.group(2))
        if m.group(3) == 'GFLOPS':
            val /= 1000.0  # normalize to TFLOPS (baseline units)
        runs.setdefault(m.group(1).replace(' ', ''), []).append(val)
med = {k: statistics.median(v) for k, v in runs.items()}
base = json.load(open(sys.argv[1]))
tol_perf = float(os.environ.get('TOL_PERF', 0.12))
tol_regress = float(os.environ.get('TOL_REGRESS', 0.02))
n_obs = int(os.environ.get('N_OBS', '6'))
print("medians (TFLOPS):", med)
primary = base.get('_primary') or max(med, key=lambda k: med[k])  # pinned key from baseline.json; fallback fastest
held_out = base.get('_held_out', [])
ok = True
# every declared shape must actually be benchmarked, with enough observations
for shape in [primary] + list(held_out):
    if shape not in med:
        print(f"FAIL: declared shape {shape} not benchmarked"); ok = False
    elif len(runs[shape]) < n_obs:
        print(f"FAIL: shape {shape} has {len(runs[shape])} observations, need {n_obs}"); ok = False
for shape, v in med.items():
    if shape in base and not shape.startswith('_'):
        ratio = v / base[shape]
        if shape == primary and ratio < 1 + tol_perf:
            print(f"FAIL: primary {shape} speedup {ratio:.3f} < {1 + tol_perf:.2f}"); ok = False
        if shape != primary and ratio < 1 - tol_regress:
            print(f"FAIL: held-out {shape} regressed {ratio:.3f}"); ok = False
sys.exit(0 if ok else 1)
EOF
PERF_RC=$?

echo "== 3. coopmat path proof (reviewer check) =="
echo "NOTE: run the benchmark once with RADV_DEBUG=shaderstats:nocache and confirm mul_mm coopmat kernels appear in the stats (manual reviewer gate; not automated here)."

echo "== result =="
[ $PERF_RC -eq 0 ] && echo "PASS" || { echo "FAIL: performance"; exit 1; }
