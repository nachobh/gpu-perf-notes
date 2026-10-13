# Task spec skeleton — "Shared-memory layout tuning for a cooperative-matrix matmul kernel"
# (basado en tu trabajo real; rellena los huecos con mediciones REEJECUTADAS antes de publicar)

## 1. Specification
- **Workload**: llama.cpp Vulkan backend, `mul_mm.comp` coopmat path, MUL_MAT shapes
  m=4096 n=512 k=14336, q4_0_rocmfp4_fast (añade 2 shapes held-out que el autor no ve).
- **Hardware**: AMD Radeon 8060S (gfx1151), RADV/Mesa, HIP/ROCm 7.2.4. (Especificar driver exacto.)
- **Starting code**: fork en commit `95b8e33e1`-equivalente limpio, PAD default = 4.
- **Constraints**: un solo parámetro editable (`SHMEM_STRIDE_PAD`); sin cambiar shapes,
  sin desactivar coopmat, sin hardcodear salidas, el verifier comprueba que el shader
  compila y ejecuta en GPU (no fallback a CPU/ray-query path).
- **Success criteria**: (a) numérica: `test-backend-ops` pasa todas las tolerancias para
  todos los dtypes/shapes del suite, incluidos los held-out; (b) rendimiento: ≥12% TFLOPS
  sobre baseline en el shape principal, medido con N repeticiones sincronizadas, descartando
  la primera; (c) no-regresión: ≤2% pérdida en los shapes held-out.
- **Intended failure modes** (lo que el verifier debe cazar):
  - Elegir PAD que minimiza banks pero rompe coalescing/occupancy (solo rápido en 1 shape).
  - Overfitting al shape visible → caer en held-out.
  - Cambiar el tamaño de `buf_a/buf_b` para "arreglar" conflictos cambiando el algoritmo.
  - Medir sin warmup / con primera corrida contaminada.
  - Fallback silencioso a un path no-coopmat más rápido para el shape de test.

## 2. Baseline
- Commit pinneado + flags CMake exactos + comando de build.
- Tabla baseline re-ejecutada hoy (PAD 0..8, TFLOPS por PAD, misma metodología).

## 3. Reference solution
- El parche de un constante (PAD=6) + justificación gcd(stride,32) en 5 líneas.
- Nota honesta: +14% prefill end-to-end vs build upstream limpio; +16.7% a nivel de
  shader (13.2→15.4 TFLOPS) — separar ambos números.

## 4. Verifier (script automatizado)
- `verify.sh`: build → test-backend-ops (tolerancias por dtype) → bench N=7 sincronizado
  (descarta run 1, mediana) → compara contra baseline JSON pinneado → exit code.
- Anti-gaming: checksum del shader para detectar cambios fuera del parámetro;
  log de `vkCmdDispatch`/counters para confirmar ejecución coopmat; shapes held-out
  inyectados por el verifier, no visibles en el repo del task.

## 5. Known limitations (declarar)
- Un solo hardware medido; el gradiente PAD-dependiente puede differ en NVIDIA
  (bank width 32 común, pero coopmat layouts distintos) — el task es portable en
  concepto, los números no.
