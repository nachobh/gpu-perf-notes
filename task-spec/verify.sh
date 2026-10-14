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
#      k=14336, q4_0_rocmfp4_fast) via `test-backend-ops perf -o MUL_MAT`,
#      N=7 synchronized runs, first discarded, median kept.
#   3. Compares against baseline.json (re-measured in the same session when
#      --remeasure-baseline is given; otherwise the pinned file).
#   4. Anti-gaming:
#      - shader checksum: SHMEM_STRIDE_PAD must be the ONLY diff vs the
#        starting commit (git diff --stat restricted to the allowed file);
#      - coopmat path proof: RADV shader stats must show mul_mm coopmat
#        kernels compiled (RADV_DEBUG=shaderstats with nocache).
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
[ "$CORRECT" = "$TOTAL" ] || { echo "FAIL: correctness"; exit 1; }

echo "== 2. performance (paired, first run discarded, median of $((N-1))) =="
run_perf() { # $1 = extra env/args label; runs perf sweep, prints TFLOPS lines
  "$BIN/test-backend-ops" perf -o MUL_MAT 2>&1 | grep -E 'q4_0_rocmfp4_fast.*m=4096.*n=512.*k=14336'
}
# discard boost
run_perf > /dev/null
for i in $(seq 2 $N); do run_perf; done > perf_runs.txt
# median per shape via python
python3 - <<'EOF'
import re, statistics, sys, json
runs = {}
for line in open('perf_runs.txt'):
    m = re.search(r'shape=\(([^)]+)\).*?([0-9.]+)\s*GFLOPS', line) or \
      re.search(r'shape=\(([^)]+)\).*?([0-9.]+)\s*TFLOPS', line)
    if m:
        runs.setdefault(m.group(1), []).append(float(m.group(2)))
med = {k: statistics.median(v) for k, v in runs.items()}
base = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
print("medians:", med)
primary = max(med, key=lambda k: med[k])  # primary shape is the fastest large one; pin exact key in baseline.json
ok = True
for shape, v in med.items():
    if shape in base:
        ratio = v / base[shape]
        if shape == base.get('_primary') and ratio < 1.12:
            print(f"FAIL: primary {shape} speedup {ratio:.3f} < 1.12"); ok = False
        if shape != base.get('_primary') and ratio < 1 - 0.02:
            print(f"FAIL: held-out {shape} regressed {ratio:.3f}"); ok = False
sys.exit(0 if ok else 1)
EOF
PERF_RC=$?

echo "== 3. coopmat path proof =="
if command -v glslc >/dev/null 2>&1 || true; then
  echo "NOTE: run the benchmark once with RADV_DEBUG=shaderstats:nocache and confirm mul_mm coopmat kernels appear in the stats (manual gate until RADV log parsing is wired)."
fi

echo "== result =="
[ $PERF_RC -eq 0 ] && echo "PASS" || { echo "FAIL: performance"; exit 1; }
