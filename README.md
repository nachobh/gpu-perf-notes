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
   FP4 batched speculative decoding went from a 25% deficit to a 46% lead —
   per-weight scale lookups amortized, a branch-free dequant path, and a
   mixed-type `output.weight` tensor costing 22 ms/step found and fixed.

## Method (applies to every note)

- Baseline = clean upstream build, same CMake flags, commit hash stated.
- First run of any set discarded (boost clock); medians over repeated,
  paired, synchronized runs.
- Correctness before speed: numerical checks across dtypes/shapes
  (`test-backend-ops` full sweeps), PPL with error bars.
- Combined vs isolated gains always separated (quantization ≠ spec-decode ≠
  layout fix).
- Third-party reference numbers (e.g. kingjones777's published results on
  identical hardware) are labeled as such, never blended into my own tables.

## Task spec

[`task-spec/`](task-spec/README.md) — a compact evaluation task built from
note 1: specification, pinned baseline, reference solution, and an automated
verifier with held-out shapes and anti-gaming checks.

## Related repos

- **[llama.cpp fork](https://github.com/nachobh/llama.cpp)** — the code behind
  these notes (ROCmFPx port, Vulkan LDS fix, batched spec-decode fixes,
  adaptive draft length, qwen4exp support).
- **[consejo-de-sabios](https://github.com/nachobh/consejo-de-sabios)** —
  multi-model debate pipeline with quality-vs-latency benchmarks.
- **[zimage-studio](https://github.com/nachobh/zimage-studio)** — local
  multimodal panel (diffusion + LoRA + TTS/STT + MCP).
- **[gpt2-from-scratch-llama2c](https://github.com/nachobh/gpt2-from-scratch-llama2c)**
  — GPT-2 trained from scratch with numerically verified GGUF export.
