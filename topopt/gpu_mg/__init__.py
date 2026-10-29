"""Matrix-free GPU multigrid solver (`settings.solver: gpu-mg`, CONTRACTS.md section 20).

`settings.solver: auto` runs it on a GPU that passes the self-test (`device.auto_gpu`, `selftest.py`) and goes on with the CPU solver when it
fails during a run (`topopt/solver_fallback.py`). `gpu-mg` as a value runs it on any OpenCL device and never falls back. Nothing imports pyopencl
until a run or a check asks for the device.

  device.py     OpenCL facts, device and case choice, the error texts, the cache of a passed self-test (light, no pyopencl at import)
  probe.py      the child process (time limit) that lists the devices and runs the self-test for `auto`; the parent asks it (light)
  hierarchy.py  the multigrid hierarchy on the CPU (numba, FP64)
  kernels.py    the OpenCL C source (plain OpenCL C 1.2)
  engine.py     device buffers, V-cycle, PCG and the FP64 refinement (needs pyopencl)
  model.py      `GpuModel` and `GpuSolver`, the hook into `topopt.optimize`
"""
