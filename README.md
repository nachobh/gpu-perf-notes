# GPU performance notes — AMD Strix Halo (Radeon 8060S, gfx1151)

Inference and kernel optimization on a single AMD Strix Halo machine (Radeon 8060S
iGPU, 103 GB unified VRAM, ~256 GB/s UMA ceiling), mostly on a
[llama.cpp](https://github.com/nachobh/llama.cpp) fork (Vulkan + ROCm/HIP).
Everything is measured locally against commit-pinned upstream baselines.
Dead ends are recorded alongside wins, because most of the value is in not
repeating them.

## The notes

1. **[Vulkan prefill: a one-constant +14% win from LDS bank conflicts](notes/01-vulkan-prefill-lds-bank-conflict.md)**
   `gcd(SHMEM_STRIDE, 32)` analysis of the coopmat matmul, PAD sweep table
   (5.4 → 15.4 TFLOPS), why odd strides are 3× slower despite being
   conflict-free, why upstream's default silently ships a 4-way conflict to
   every non-Intel device, adaptive draft-length for speculative decoding
   (+48% prose on a model-card-recommended setting), a measured roofline
   (decode at 80–83% of LPDDR5X bandwidth), six dead ends with evidence, and
   the measurement traps on this machine (first-run boost clock, lying
   `gpu_busy_percent`, …).

2. **[ROCmFPx FP4 port + the batched spec-decode fix](notes/02-rocmfpx-fp4-port-and-batched-spec-decode-fix.md)**
   Hand-port of six FP-family GGUF quant types with bit-identical validation
   against an independent oracle build (and two upstream bugs found on the
   way); FP4 vs a strong K-quant honestly measured (12% smaller, +7% batch-1,
   PPL gap "small but probably real" with overlapping error bars); and how
   FP4 batched speculative decoding went from a 25% deficit to a 10% lead on
   prose (and +24%/+48% on HTML/digit workloads) — per-weight scale lookups
   amortized, a branch-free dequant path, and a mixed-type `output.weight`
   tensor costing 22 ms/step found and fixed.

## Method (applies to every note)

- Baseline = clean upstream build, same CMake flags, commit hash stated.
- First run of any set discarded (boost clock); medians over repeated,
  paired, synchronized runs.
- Correctness before speed: numerical checks across dtypes/shapes
  (`test-backend-ops` full sweeps), PPL with error bars.
- Combined vs isolated gains always separated (quantization ≠ spec-decode ≠
  layout fix).
- Third-party reference numbers (e.g. community reports and model cards for
  ROCmFPx GGUFs on identical hardware) are labeled as such, never blended into
  my own tables.

## Task spec

[`task-spec/`](task-spec/README.md) — a compact evaluation task built from
note 1: specification, pinned baseline, reference solution, and an automated
verifier ([`verify.sh`](task-spec/verify.sh)) with held-out shapes and
anti-gaming checks. Baseline values are being re-measured with the stated
methodology before final publication.

## Verification: fork vs upstream (2026-10-14)

Cross-check of the merged fork branch against a clean upstream build
(`6868a52c`), same machine, interleaved arms, first round discarded
(boost clock), Qwen3.5-9B MTP Q4_K_M (a GDN-hybrid model — deliberately
*not* the model the wins were found on):

| arm | base (upstream) | fork |
|---|---|---|
| pp512 / pp2048 (llama-bench, t/s) | 1020 / 1008 | 1036 / 1011 |
| tg128 (t/s) | 31.7 | 31.5 |
| prose gen, no spec (t/s, mean of 4 runs) | 30.3 | 31.1 |
| prose gen, MTP spec n-max 4 (t/s, mean of 4 runs) | 34.6 | 37.3 |
| adaptive draft n∈[3,7] (t/s, median of 3) | — (feature absent upstream) | 35.2 |

Honest reading: **no regression from the merge** (pp/tg within noise); MTP gives
+14% on the upstream build (30.3 → 34.6) and +20% on the fork build
(31.1 → 37.3) — MTP runs are high-variance (base ranged 31.0–38.4 across
repeats), so treat the means as ±3; the adaptive controller
matches the tuned fixed setting without tuning (n=4 is already near-optimal here — its
documented +48% win is on a model whose recommended fixed setting is wrong,
see note 1 §2b). The +14% prefill win is matmul-bound-model-specific and does
not transfer to this GDN hybrid, as note 1 predicts for non-matmul-bound
workloads.

## Related repos

- **[llama.cpp fork, branch `perf-master-2026-10`](https://github.com/nachobh/llama.cpp/tree/perf-master-2026-10)** —
  the code behind these notes (ROCmFPx port, Vulkan LDS fix, batched
  spec-decode fixes, adaptive draft length, qwen4exp support), merged with
  upstream master (zero commits behind, all fixes preserved, builds + tests
  green).
- **[consejo-de-sabios](https://github.com/nachobh/consejo-de-sabios)** —
  multi-model debate pipeline with quality-vs-latency benchmarks.
- **[zimage-studio](https://github.com/nachobh/zimage-studio)** — local
  multimodal panel (diffusion + LoRA + TTS/STT + MCP).
- **[gpt2-from-scratch-llama2c](https://github.com/nachobh/gpt2-from-scratch-llama2c)**
  — GPT-2 trained from scratch with numerically verified GGUF export.
