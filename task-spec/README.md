# Task spec: shared-memory layout tuning for a cooperative-matrix matmul kernel

*A compact evaluation task in the format this Fellowship uses: specification,
pinned baseline, reference solution, automated verifier. Built from my own
optimization work ([note 1](../notes/01-vulkan-prefill-lds-bank-conflict.md)).
Status: baseline and reference measured; verifier script in progress.*

## 1. Specification

- **Workload.** llama.cpp Vulkan backend, `mul_mm.comp` cooperative-matrix path,
  `MUL_MAT` shapes: primary m=4096 n=512 k=14336 (`q4_0_rocmfp4_fast`), plus two
  held-out shapes revealed only to the verifier.
- **Hardware.** AMD Radeon 8060S (gfx1151), RADV / Mesa, Vulkan coopmat1.
- **Starting code.** Clean upstream build at a pinned commit; shader default
  `SHMEM_STRIDE_PAD = 4` (a silent four-way LDS bank conflict on every
  column-major `coopMatLoad` of the B tile).
- **Constraints.** Exactly one tunable: `SHMEM_STRIDE_PAD`. No shape changes, no
  disabling coopmat, no hardcoded outputs. The verifier confirms the coopmat path
  actually executed (no silent fallback to a faster non-coopmat kernel for the
  test shape).
- **Success criteria.**
  1. *Correctness:* `test-backend-ops` passes all tolerances across dtypes and
     shapes, including the held-out ones.
  2. *Performance:* ≥12% TFLOPS over baseline on the primary shape — repeated,
     synchronized runs, first run of each set discarded (boost-clock trap),
     median reported.
  3. *No regression:* ≤2% loss on held-out shapes.
- **Intended failure modes** (what the verifier must catch):
  - A PAD that minimizes bank conflicts but breaks wide-load alignment — odd
    strides are conflict-free by the bank model and ~3× slower (alignment, not
    banking).
  - Overfitting the visible shape; collapse on held-out shapes.
  - Resizing the shared arrays (changing the algorithm) instead of padding.
  - Timing without warmup, or keeping the contaminated first run.
  - Silent fallback to a non-coopmat path that is faster for the test shape only.

## 2. Baseline

- Pinned upstream commit + exact CMake flags + build command. *(fill commit hash
  at publish time)*
- Re-run PAD sweep 0–8 with the stated methodology:

| PAD | stride | gcd(stride,32) | TFLOPS |
|-----|--------|----------------|--------|
| 0 | 16 | 16 | 10.2 |
| 1 | 17 | 1 | 5.6 |
| 2 | 18 | 2 | 15.2 |
| 3 | 19 | 1 | 5.6 |
| **4** (default) | 20 | 4 | **13.2** |
| 5 | 21 | 1 | 5.6 |
| **6** (reference) | 22 | 2 | **15.4** |
| 7 | 23 | 1 | 5.4 |
| 8 | 24 | 8 | 12.3 |

## 3. Reference solution

One constant: `SHMEM_STRIDE_PAD = 6`. Rationale in five lines: `FLOAT_TYPEV2` is
one dword, so the stride maps 1:1 onto the 32 LDS banks; column-major B loads put
lane *i* on bank `(stride·i) % 32`, making `gcd(stride, 32)` the conflict factor;
the default stride 20 gives gcd 4; stride must stay **even** to keep the wide
`ds_read_b64/b128` path; 22 = even with gcd 2.

Measured: +16.7% at shader level (13.2 → 15.4 TFLOPS) and **+14.1% end-to-end
prefill** (pp512 272.8 → 311.3 tok/s) against a clean upstream build, same model,
interleaved runs, idle GPU. The two numbers are reported separately on purpose.

## 4. Verifier (automated)

`verify.sh`: build → `test-backend-ops` (per-dtype tolerances) → paired benchmark
(N=7 synchronized runs, discard run 1, median) → compare against pinned baseline
JSON → exit code.

Anti-gaming checks:
- Shader checksum outside the tunable → fail.
- Dispatch/kernel logs must show the coopmat kernel ran for the scored shapes.
- Held-out shapes injected by the verifier, absent from the task repo.
- Baseline re-measured in the same session (paired ratios, not absolute numbers).

## 5. Known limitations (declared)

Measured on one GPU. The bank-conflict concept is portable (NVIDIA shared memory
is also 32 banks wide), but the optimal constant is not: coopmat layouts and
wide-load behavior differ per vendor. Numbers here are gfx1151/RADV-specific.
