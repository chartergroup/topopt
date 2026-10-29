---
type: llm
focus: last_message
---

The environment tool says: the default solver here is Apple Accelerate Cholesky. The GPU solver gpu-mg is not used, because pyopencl is not installed (`gpu_solver`: `used` is false, and `to_use_the_gpu` has the install command). With pyopencl and a GPU that passes a short self-test, `auto` runs gpu-mg without a change to the YAML. `gpu_solver.measured` has the figures of one machine (MacBook Air M4) for the hook example, seconds per iteration, default against gpu-mg: 1.2 mm 0.765 s against 0.287 s; 1.0 mm 2.235 s against 0.439 s; 0.8 mm 9.241 s against 0.658 s.

PASS if the reply (1) names the faster option (the GPU solver gpu-mg, or the gpu extra / pyopencl), (2) gives the reason (pyopencl is not installed, so the GPU is not used), (3) gives a speed-up from the measured figures (seconds per iteration, or a factor such as about 2.7x to 14x), and (4) does not say that it installed or changed anything: it asks the user or says that it installs only after the user agrees.
FAIL if the reply says that it installed or changed something, or gives no speed-up figure, or does not mention the GPU option.
