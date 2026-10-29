# Provenance of `mma_reference_steps.npz`

Recorded on 2026-10-26 from mmapy 0.3.1 (GPL-3.0-or-later), called once as a black box (import and call only; its source was not read) by a script that is not in this repository; recorder settings: 64 indices kept for each array with n > 64, real steps from runs with voxel size 4.0 mm; Linux x86_64, Python 3.12.13, numpy 2.5.3, scipy 1.18.1.

The file holds numbers only. It holds no code of the reference package. The package is not a dependency of topopt and no test imports it.

The test is `tests/test_mma_reference_steps.py`. Its docstring says what the file holds, the tolerances, and why they hold on other platforms.

## Content

- `meta`: a JSON text. It holds the format number, the date, the platform of the recording, the defaults of the reference package, and the table of the cases (n, m, steps, seed, settings).
- `<case>__x`, `__low`, `__upp`: the new design and the new asymptotes of each step of a random problem. For n up to 64 the arrays are whole. For a larger n they hold 64 fixed indices (`__idx`) and `__stat_x`, `__stat_low`, `__stat_upp` hold the sum, the minimum and the maximum of the whole array.
- `<case>__lam`, `__y`, `__z`: the multipliers, the elastic variables and the variable z of each step.
- `<case>__inp`: four numbers of the inputs of each step (f0 and the sums of df0, g and dg). The test uses them to check that numpy made the same random problem.
- `mma_class__*` and `continue__*`: the steps of the `Mma` class test and the state of the continue test.
- `real_<run>_<call>__in_*`: the arguments of one `mma_update` call, taken from the own code during a short real run (`short`: `examples/hook/hook_short.yaml`; `stress`: `examples/hook/hook_stress.yaml`; both with `voxel_size: 4.0`, 40 and 30 passes). `real_<run>_<call>__x`, `__low`, `__upp`, `__lam`, `__y`, `__z`: the recorded result for those arguments.

The random problems follow the own outputs: at each step the reference got the same arguments as the own code (the next x is the clipped own x, the next asymptotes are the own asymptotes).
