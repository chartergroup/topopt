# Third-party notices

This file lists the third-party packages that `pyproject.toml` declares. It also lists the
third-party files in this repository and the components whose licences matter.
It gives facts. It does not give legal advice.

The test `tests/test_third_party_notices.py` fails when a package of `pyproject.toml` has no
row in the table of its group, or when the pin cell of a row does not name every pin of that
package in `pyproject.toml`. A package can have more than one pin (one for each environment
marker). Its row then names all of them, separated by `;`, with the condition in brackets, for
example `2.5.3; 2.3.5 (macOS x86_64)`.

## Facts in short

- Every package in `dependencies` has a permissive licence (MIT, BSD, Apache-2.0, MIT-CMU).
  topopt makes the MMA design update with its own code (`topopt/mma_sub.py`): a new implementation
  written from the published note of K. Svanberg (2007).
- GPL, LGPL and proprietary packages are only in optional extras: gmsh (extras `tet` and
  `verify`, GPL-2.0-or-later), cholespy (extra `verify`, GPL-2.0-or-later in effect), the Intel
  packages that the extra `pardiso` pulls in.
- The extra `step` (OpenCASCADE, the wheel `cadquery-ocp-novtk`) has an Apache-2.0 wrapper. Its wheel bundles OCCT (LGPL-2.1 with
  an exception) and image libraries, some under GPL terms or GPL options. It is accepted under the rule for gmsh and cholespy
  (optional extra, installed from PyPI, never vendored, never imported by the core); see
  "Components whose licences matter".
- pyopencl (MIT) is a core dependency on an Apple silicon Mac and on Linux x86_64 (the marker in `pyproject.toml`). The extras
  `gpu` and `gpu-test` have it too. None of them has a GPL package. siphash24, which pytools pulls in, offers an
  LGPL option as one of two licences.
- The Linux x86_64 wheels of numpy and scipy contain two runtime libraries under GPL and LGPL
  terms. The macOS x86_64 wheels (numpy 2.3.5, scipy 1.18.1) contain those two and a third, the
  GCC runtime library libgcc_s, under the same GPL terms as libgfortran. Pillow lists two optional
  GPL and LGPL components. See "Libraries inside and beside binary wheels".
- macOS x86_64 (an Intel Mac) has other pins for four packages: numpy 2.3.5, numba 0.62.1,
  llvmlite 0.45.1 and cryptography 48.0.1. Their licences are permissive. See the rows below.
- setuptools (MIT), the build backend of topopt, is pinned in both lock files, and the setup installs it into the environment
  (see "Build backend"). Its wheel bundles autocommand under LGPL-3.0; the build of topopt does not import it (see "Libraries inside
  and beside binary wheels").
- No package of the tables is vendored. pip installs each one.
- This repository holds three third-party files, all MIT: three.js r160 for the dashboard, and three.js r128 with its
  `OrbitControls` for the refine page. Their full licence texts are in "Third-party files in this repository". The same section names two data files that hold facts
  from maker pages and data sheets, and the files that come from other repositories of the project.

## Components whose licences matter

- **MMA design update (`topopt/mma_sub.py`): the own code of topopt (MIT).** It is a new implementation
  written from the 2007 note of K. Svanberg, "MMA and GCMMA - two methods for nonlinear optimization"
  (the note describes the method of his paper of 1987). It holds no code of another MMA program, and
  it is not a translation of one. The test `tests/test_mma_reference_steps.py` compares it with recorded
  numbers; `tests/data/mma_reference_steps.md` says where they come from. No package is needed for it.
- **gmsh 4.15.2 (extra `tet`): GPL-2.0-or-later.** Platforms: macOS and Linux x86_64 (the
  marker in `pyproject.toml`). The core of topopt never imports gmsh. `topopt/tet/worker.py`
  runs it in a child process (`python -m topopt.tet.worker`). The test
  `test_gmsh_is_not_imported_by_the_core` in `tests/test_tet.py` checks that `import topopt`,
  the CLI, the MCP server and `topopt.tet` do not load it. The licence file of the gmsh wheel
  (`share/doc/gmsh/LICENSE.txt`) states the GNU GPL version 2 or later, with an exception that
  lets the user combine gmsh with code of Netgen, METIS, OpenCASCADE and ParaView under their own
  licences. The `CREDITS.txt` file next to it lists other code in gmsh under other licences. This
  check does not cover that list. The user installs gmsh. This repository does not hold a copy.
- **pypardiso 0.4.7 and Intel MKL (extra `pardiso`): Linux x86_64 only.** pypardiso is
  BSD-3-Clause. It pulls in Intel packages that are binary only and carry Intel terms (see
  "Packages that other packages pull in"). `topopt/solver.py` imports pypardiso through
  `import_pypardiso()` when the solver `pardiso` runs, and never on macOS. The user installs
  it. This repository does not hold a copy. `LICENSE_NOTE` in `topopt/solver_choice.py` gives
  the same facts to the user at install time.
- **cholespy 2.2.0 (extra `verify`): GPL-2.0-or-later in effect.** The wheel bundles CHOLMOD
  Supernodal (SuiteSparse 5.4.0), also Cholesky, Check and Partition (LGPL-2.1-or-later per the
  same upstream file). The package metadata says BSD-3-Clause. Optional extra; not vendored; the
  core never imports it.
- **cadquery-ocp-novtk 7.9.3.1.1 (extra `step`): Apache-2.0 wrapper; the wheel bundles LGPL and GPL components.** `topopt/step.py`
  imports `OCP` inside one function (`_import_ocp`), so `import topopt` does not load it (`tests/test_step.py` checks this). The
  user installs it from PyPI; this repository does not hold a copy. Facts from the Linux x86_64 wheel (installed) and from the
  macOS arm64 wheel (read as a zip file); the wheels hold no licence text for the bundled libraries, so the licences below come from
  the Ubuntu `copyright` file of each package version in the SBOM of the Linux wheel (`sboms/auditwheel.cdx.json`, no licence
  field), and from upstream for OCCT, libgomp and zstd. Not checked by a lawyer.
  - OCCT 7.9.3 (46 `libTK*` files on Linux, 50 on macOS): LGPL-2.1 with the Open CASCADE exception 1.0 (upstream).
  - FreeImage 3.18.0 (both platforms): `GPL-2.0 or GPL-3.0 or FIPL-1.0` (a choice; the FreeImage Public License is the non-GPL option).
  - jbigkit `libjbig` 2.1 (Linux only): `GPL-2+`, no other option.
  - LibRaw (Linux 0.19.5, macOS libraw.23): `LGPL-2.1 or CDDL-1.0` (a choice), with dcraw parts under their own terms.
  - libgomp (Linux): the GCC runtime, `GPL-3.0-or-later WITH GCC-exception-3.1`.
  - FreeType (`GPL-2+ or FTL`) and zstd 1.5.7 (`BSD-3-Clause or GPL-2.0`) are choices with a permissive option.
  - The other bundled libraries (libtiff, libpng, libjpeg-turbo, OpenJPEG, JPEG-XR, OpenEXR, libwebp, lcms2, fontconfig, libuuid, liblzma)
    have permissive licences. The macOS-only libraries are libsharpyuv, Imath, libLerc, libdeflate, zlib, libc++ and OpenJPH; their upstream
    licences are permissive (BSD-3-Clause, BSD-3-Clause, Apache-2.0, MIT, Zlib, Apache-2.0 with the LLVM exception, BSD-2-Clause); the wheel
    holds no licence text for them.
  - **Decision.** The package is accepted as the optional extra `step` under the rule for gmsh and
    cholespy: it is installed from PyPI on the user's machine, this repository never holds a copy, and the core never imports it. The metadata
    says Apache-2.0 and the bundled files do not, so `scripts/ci/license_policy.toml` has an `override` (category `copyleft`) and an `allow`
    entry that names the libraries: libjbig (GPL-2+) and libgomp (GPL-3.0 with the GCC exception) on Linux only, the GPL options of FreeImage,
    FreeType and zstd, and the LGPL libraries (OpenCASCADE, LibRaw). The legs `linux-x86_64` and `macos-arm64` install the extra, so the licence
    step of CI reads this package. Not legal advice.
- **siphash24 1.9 (pulled in by pytools, which pyopencl needs: core on an Apple silicon Mac and on Linux x86_64): `Apache-2.0 OR LGPL-2.1-or-later`.** The
  metadata offers both licences. It is a package that pip installs; this repository does not hold a
  copy.

## Core packages

Source codes: `meta` = the installed `METADATA` of the pinned version (read with
`importlib.metadata`; the field is in brackets). `wheel` = a licence file inside the installed
wheel (in the `.dist-info` folder or in its `licenses` sub-folder). `upstream` = the licence
file of the upstream project at the pinned tag.

| Package | Pin | Licence (SPDX) | Checked from |
|---|---|---|---|
| numpy | 2.5.3; 2.3.5 (macOS x86_64) | 2.5.3: BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0. 2.3.5: BSD-3-Clause (its metadata has no SPDX expression; its licence file also names MIT and Zlib parts) | 2.5.3: meta (License-Expression); wheel `LICENSE.txt`. 2.3.5 (wheel `numpy-2.3.5-cp312-cp312-macosx_10_13_x86_64.whl`): meta (License text is BSD 3-clause; classifier "BSD License"; no License-Expression); wheel `LICENSE.txt` (BSD 3-clause text; it names lapack-lite BSD-3-Clause, dragon4 MIT and libdivide Zlib) |
| scipy | 1.18.1 | BSD-3-Clause | meta (License text is BSD 3-clause; classifier "BSD License"); wheel `LICENSE.txt` |
| scikit-image | 0.26.0 | BSD-3-Clause | meta (License text; it lists some files under BSD-2-Clause or MIT) |
| trimesh | 5.1.0 | MIT | meta (License text; classifier MIT); wheel `LICENSE.md` |
| manifold3d | 3.5.4 | Apache-2.0 | meta (classifier "Apache Software License"; no License field); wheel `LICENSE` (Apache License 2.0 text); upstream `LICENSE` at tag `v3.5.4` (the same text) |
| pyamg | 5.3.0 | MIT | meta (License "MIT"; classifier MIT); wheel `LICENSE.txt` |
| PyYAML | 6.0.3 | MIT | meta (License "MIT"; classifier MIT) |
| numba | 0.67.0; 0.62.1 (macOS x86_64) | BSD-2-Clause | wheel `LICENSE` (2-clause BSD text, Anaconda, Inc.; the text in 0.62.1 is the same as in 0.67.0); meta says "BSD" in both (0.62.1 also has the classifier "BSD License") |
| rtree | 1.4.1 | MIT | meta (License-Expression); wheel `LICENSE.txt` |
| networkx | 3.7 | BSD-3-Clause | meta (License-Expression); wheel `LICENSE.txt` |
| Pillow | 12.3.0 | MIT-CMU | meta (License-Expression); wheel `LICENSE` |
| pyopencl | 2026.1.4 (Apple silicon Mac and Linux x86_64) | MIT | meta (License-Expression MIT); wheel `licenses/LICENSE` (MIT/X Consortium text). The Linux x86_64 wheel bundles the ocl-icd loader (`pyopencl/.libs/libOpenCL-*.so*`, listed in the wheel `RECORD`); the same `LICENSE` holds its licence text (2-clause BSD, BSD-2-Clause), and it names parts of pyopencl under Apache-2.0 (derived from Thrust, the scan code) and under BSD-3-Clause (Random123) |

numba needs llvmlite. The `meta` License-Expression of llvmlite 0.49.0 is
`BSD-2-Clause AND Apache-2.0 WITH LLVM-exception`. On macOS x86_64, numba 0.62.1 needs llvmlite
0.45.1. See "Packages that other packages pull in".

`requirements.lock` pins 55 packages: the core packages, `pytest`, `pytest-xdist`, `mcp`, their
transitive packages, and setuptools (see "Build backend"). The metadata of all 55 shows a permissive licence (MIT, MIT-0, BSD,
Apache-2.0, PSF-2.0, MIT-CMU), and siphash24 offers Apache-2.0 as one of two licences. Every pin of both lock files has a row in a table of
this file, and the test `tests/test_third_party_notices.py` fails when one has none. The rows of
the transitive packages are in "Packages that other packages pull in". Their licences were read
from the PyPI JSON of each pinned version on 2026-10-25.

`requirements-macos-x86_64.lock` pins 51 packages: those of `requirements.lock` without pyopencl and the three packages that only
it needs (pytools, platformdirs and siphash24; an Intel Mac gets no pyopencl). It moves four pins back: numpy 2.3.5,
numba 0.62.1, llvmlite 0.45.1 and cryptography 48.0.1. The metadata of these four was read in the
macOS x86_64 wheels (see "How the licences were checked"). It shows a permissive licence for all
four.

## Extra: dev

| Package | Pin | Licence (SPDX) | Checked from |
|---|---|---|---|
| pytest | none (9.1.1 checked) | MIT | meta (License-Expression); wheel `LICENSE` |
| pytest-xdist | none (3.8.0 checked) | MIT | meta (License-Expression); wheel `licenses/LICENSE` |

## Extra: mcp

| Package | Pin | Licence (SPDX) | Checked from |
|---|---|---|---|
| mcp | 2.3.0 | MIT | meta (License "MIT"; classifier MIT); wheel `LICENSE` |
| cryptography | 48.0.1 (macOS x86_64 only) | Apache-2.0 OR BSD-3-Clause | meta (License-Expression); wheel `licenses/LICENSE` (the user may take either licence: `LICENSE.APACHE` or `LICENSE.BSD`, both in the wheel) |

cryptography is in this extra with a pin for macOS x86_64 only (the marker in `pyproject.toml`).
Version 50.0.2 (the pin of `requirements.lock`) has no macOS x86_64 wheel (`pip download` found
none on 2026-10-20; the last version with one is 48.0.1). mcp 2.3.0 needs
`pyjwt[crypto]`, and pyjwt 2.15.1 asks for `cryptography>=3.4.0` in its extra `crypto`, so pip
installs cryptography on every platform. 50.0.2 has the same License-Expression as 48.0.1.

The other packages that this extra installs (pydantic, starlette, uvicorn, httpx2 and others)
carry MIT, MIT-0, BSD, Apache-2.0 or PSF-2.0 licences in their `meta` fields.

## Extra: pardiso

Linux x86_64 only (the marker in `pyproject.toml`).

| Package | Pin | Licence (SPDX) | Checked from |
|---|---|---|---|
| pypardiso | 0.4.7 | BSD-3-Clause | meta (License-Expression); wheel `LICENSE.txt` |

## Extra: tet

macOS and Linux x86_64 (the marker in `pyproject.toml`).

| Package | Pin | Licence (SPDX) | Checked from |
|---|---|---|---|
| gmsh | 4.15.2 | GPL-2.0-or-later | meta (License "GPLv2+"; classifier "GNU General Public License v2 or later (GPLv2+)"); wheel `share/doc/gmsh/LICENSE.txt` (GPL version 2 or later, with the Gmsh exception) |

## Extra: gpu

The same pin as the core row, without the marker. The extra stays for an old command and for a platform that the marker leaves out.

| Package | Pin | Licence (SPDX) | Checked from |
|---|---|---|---|
| pyopencl | 2026.1.4 | MIT | meta (PyPI JSON, License-Expression MIT); wheel `LICENSE` (MIT/X Consortium text) |

## Extra: gpu-test

Linux x86_64 for pocl-binary-distribution (the marker in `pyproject.toml`).

| Package | Pin | Licence (SPDX) | Checked from |
|---|---|---|---|
| pyopencl | 2026.1.4 | MIT | meta (PyPI JSON, License-Expression MIT); wheel `LICENSE` (MIT/X Consortium text) |
| pocl-binary-distribution | 3.0 | MIT | wheel file `pocl_binary_distribution/.libs/share/pocl/licenses/POCL.COPYING` (MIT text; the wheel metadata has no licence field) |

## Extra: verify

gmsh has the same marker as in the extra `tet`.

| Package | Pin | Licence (SPDX) | Checked from |
|---|---|---|---|
| gmsh | 4.15.2 | GPL-2.0-or-later | same as extra `tet` |
| scikit-fem | 12.0.2 | BSD-3-Clause | wheel `licenses/LICENSE` (three numbered clauses); classifier "BSD License" |
| cholespy | 2.2.0 | BSD-3-Clause (own code); GPL-2.0-or-later in effect | the wheel `licenses/LICENSE` is the BSD-3-Clause text of its own code and the classifier says "BSD License"; the compiled module `cholespy/_cholespy_core.abi3.so` holds the symbols `cholmod_super_symbolic`, `cholmod_super_numeric` and `SuiteSparse/CHOLMOD` path strings (CHOLMOD Supernodal); upstream `CHOLMOD/Doc/License.txt` at SuiteSparse tag `v5.4.0` gives the Supernodal module as "GNU General Public License ... version 2 ... or any later version" |

## Extra: step

Same wheel set as in "Components whose licences matter": macOS (arm64 and x86_64) and Linux (x86_64 and arm64, glibc 2.31 or newer). The macOS arm64 wheel
and the Linux x86_64 wheel were checked.

| Package | Pin | Licence (SPDX) | Checked from |
|---|---|---|---|
| cadquery-ocp-novtk | 7.9.3.1.1 | Apache-2.0 (the wrapper). The wheel bundles OCCT 7.9.3 under LGPL-2.1 with the Open CASCADE exception, and libraries under GPL options (see "Components whose licences matter") | meta (License "Apache-2.0"; classifier "Apache Software License"); the wheel holds no licence file. The bundled libraries: SBOM of the Linux wheel and the Ubuntu copyright files, see "How the licences were checked" |

## Build backend

`pyproject.toml` names setuptools in `[build-system]` (`setuptools>=77`), not as a dependency. Both lock files pin it. Step 2 of the
setup installs it into the environment, and step 3 links the topopt code with it (`--no-build-isolation`), so the link after a plugin
update needs no network. topopt does not import it. Its file `distutils-precedence.pth` makes each start of the Python of the
environment import its module `_distutils_hack` (MIT, part of setuptools).

| Package | Pin | Licence (SPDX) | Checked from |
|---|---|---|---|
| setuptools | 84.2.1 | MIT | meta (License-Expression); wheel `licenses/LICENSE`. It bundles 12 packages, see "Libraries inside and beside binary wheels" |

## Packages that other packages pull in

These packages are not named in `pyproject.toml`. pip installs them for the core packages (scikit-image, numba and pyopencl),
for the extras `dev`, `mcp`, `pardiso`, `step` and `gpu`, and llvmlite for numba. `mkl` is not pinned: pypardiso asks for
`mkl != 2024.0`. The versions below are the versions that were checked. The rows from `annotated-types` to `uvicorn` are the
pins of `requirements.lock` that no other table names. "Pulled in by" lists the hard requirements in the installed
metadata (environment markers are not evaluated). Where the installed copy has another version than the pin, the row says so.

| Package | Pulled in by | Licence (as the package states it) | Checked from |
|---|---|---|---|
| mkl 2026.2.0 | pypardiso | Intel Simplified Software License (October 2022), non-OSI, binary only | meta (License; classifier "Other/Proprietary License"); wheel `LICENSE.txt` |
| tbb 2023.2.0 | mkl | Intel Simplified Software License (October 2022), non-OSI, binary only | meta (License); wheel `LICENSE.txt` |
| tcmlib 1.6.0 | tbb, umf | Intel Simplified Software License (October 2022), non-OSI, binary only | meta (License); wheel `LICENSE.txt` |
| intel-openmp 2026.2.0 | mkl | Intel End User License Agreement for Developer Tools (August 2024), non-OSI | meta (License); wheel `LICENSE.txt` |
| intel-cmplr-lib-ur 2026.2.0 | intel-openmp | Intel End User License Agreement for Developer Tools (August 2024), non-OSI | meta (License); wheel `LICENSE.txt` |
| umf 1.2.0 | intel-cmplr-lib-ur | Apache-2.0 (the metadata says "Apache-2.0 with LLVM exceptions" with the classifier "Other/Proprietary License") | wheel `LICENSE.txt` (Apache License 2.0 text) |
| cadquery-ocp-proxy 7.9.3.1.1 | cadquery-ocp-novtk | Apache-2.0 | meta (License "Apache-2.0"; classifier "Apache Software License"); it holds Python files only (a version tracker) |
| execnet 2.1.2 | pytest-xdist | MIT | meta (License-Expression; classifier MIT); wheel `licenses/LICENSE` |
| pytools 2026.1.1 | pyopencl | MIT | meta (PyPI JSON, License-Expression MIT); wheel `LICENSE` (MIT/X Consortium text) |
| platformdirs 4.13.5 | pyopencl, pytools | MIT | meta (PyPI JSON, License-Expression MIT); wheel `LICENSE` |
| typing_extensions 4.16.0 | pyopencl, pytools | PSF-2.0 | meta (PyPI JSON, License-Expression PSF-2.0); wheel `LICENSE` |
| siphash24 1.9 | pytools | Apache-2.0 OR LGPL-2.1-or-later | meta (PyPI JSON, License-Expression); the wheel has no licence file |
| llvmlite 0.45.1 | numba 0.62.1 (macOS x86_64 only; the pin is in `requirements-macos-x86_64.lock`) | BSD (as the metadata states it; the wheel `LICENSE` is the 2-clause BSD text of Continuum Analytics, Inc.) | meta (License "BSD"; no License-Expression; no licence classifier); wheel `licenses/LICENSE`. The wheel holds `llvmlite/binding/libllvmlite.dylib` and no LLVM licence file. llvmlite 0.49.0 (the other platforms) differs: its License-Expression adds `Apache-2.0 WITH LLVM-exception`, and its wheel holds `LICENSE.thirdparty` (the LLVM licence text) |
| annotated-types 0.8.0 | pydantic | MIT | meta (PyPI JSON of the pin, License-Expression); wheel `licenses/LICENSE` |
| anyio 4.15.1 | httpx2, mcp, sse-starlette, starlette | MIT | meta (PyPI JSON of the pin, License-Expression); wheel `licenses/LICENSE` |
| attrs 26.1.0 | jsonschema, referencing | MIT | meta (PyPI JSON of the pin, License-Expression); wheel `licenses/LICENSE` |
| cffi 2.1.1 | cryptography | MIT-0 | meta (PyPI JSON of the pin, License-Expression); wheel `licenses/LICENSE` |
| click 8.5.0 | uvicorn | BSD-3-Clause | meta (PyPI JSON of the pin, License-Expression); wheel `licenses/LICENSE.txt` |
| h11 0.16.0 | httpcore2, uvicorn | MIT | meta (PyPI JSON of the pin, License "MIT"; classifier MIT License); wheel `licenses/LICENSE.txt` |
| httpcore2 2.13.1 | httpx2 | BSD-3-Clause | meta (PyPI JSON of the pin, License-Expression); the installed 2.16.0 holds `licenses/LICENSE.md` |
| httpx2 2.13.1 | mcp | BSD-3-Clause | meta (PyPI JSON of the pin, License-Expression); the installed 2.16.0 holds `licenses/LICENSE.md` |
| idna 3.20 | anyio, httpx2 | BSD-3-Clause | meta (PyPI JSON of the pin, License-Expression); wheel `licenses/LICENSE.md` |
| imageio 2.38.1 | scikit-image | BSD-2-Clause | meta (PyPI JSON of the pin, License-Expression); wheel `licenses/LICENSE` |
| iniconfig 2.3.1 | pytest | MIT | meta (PyPI JSON of the pin, License-Expression); wheel `licenses/LICENSE` |
| jsonschema 4.26.0 | mcp | MIT | meta (PyPI JSON of the pin, License-Expression); wheel `licenses/COPYING` |
| jsonschema-specifications 2025.9.1 | jsonschema | MIT | meta (PyPI JSON of the pin, License-Expression); wheel `licenses/COPYING` |
| lazy-loader 0.6 | scikit-image | BSD-3-Clause | meta (PyPI JSON of the pin, License-Expression); wheel `licenses/LICENSE.md` |
| mcp-types 2.3.0 | mcp | MIT | meta (PyPI JSON of the pin, License "MIT"; classifier MIT License); wheel `licenses/LICENSE` |
| opentelemetry-api 1.45.1 | mcp | Apache-2.0 | meta (PyPI JSON of the pin, License-Expression); wheel `licenses/LICENSE` |
| packaging 26.3 | lazy-loader, pytest, scikit-image | Apache-2.0 OR BSD-2-Clause | meta (PyPI JSON of the pin, License-Expression); wheel `licenses/LICENSE` |
| pluggy 1.6.0 | pytest | MIT | meta (PyPI JSON of the pin, License "MIT"; classifier MIT License); the installed 1.7.0 holds `licenses/LICENSE` |
| pycparser 3.1 | cffi | BSD-3-Clause | meta (PyPI JSON of the pin, License-Expression); the installed 3.12 holds `licenses/LICENSE` |
| pydantic 2.14.0 | mcp, mcp-types | MIT | meta (PyPI JSON of the pin, License-Expression); the installed 2.14.1 holds `licenses/LICENSE` |
| pydantic-core 2.50.0 | pydantic | MIT | meta (PyPI JSON of the pin, License-Expression); the installed 2.50.1 holds `licenses/LICENSE` |
| pygments 2.21.0 | pytest | BSD-2-Clause | meta (PyPI JSON of the pin, License-Expression); wheel `licenses/LICENSE` |
| pyjwt 2.15.1 | mcp | MIT | meta (PyPI JSON of the pin, License-Expression); wheel `licenses/LICENSE` |
| python-multipart 0.0.32 | mcp | Apache-2.0 | meta (PyPI JSON of the pin, License-Expression); wheel `licenses/LICENSE.txt` |
| referencing 0.37.0 | jsonschema, jsonschema-specifications | MIT | meta (PyPI JSON of the pin, License-Expression); wheel `licenses/COPYING` |
| rpds-py 2026.9.1 | jsonschema, referencing | MIT | meta (PyPI JSON of the pin, License-Expression); wheel `licenses/LICENSE` |
| sse-starlette 3.5.0 | mcp | BSD-3-Clause | meta (PyPI JSON of the pin, License-Expression); wheel `licenses/LICENSE` |
| starlette 1.7.0 | mcp, sse-starlette | BSD-3-Clause | meta (PyPI JSON of the pin, License-Expression); the installed 1.9.0 holds `licenses/LICENSE.md` |
| tifffile 2026.9.20 | scikit-image | BSD-3-Clause | meta (PyPI JSON of the pin, License "BSD-3-Clause"); the installed 2026.10.15 holds `licenses/LICENSE` |
| truststore 0.10.4 | httpcore2, httpx2 | MIT | meta (PyPI JSON of the pin, License-Expression); the installed 0.10.5 holds `licenses/LICENSE` |
| typing-inspection 0.4.4 | mcp, pydantic | MIT | meta (PyPI JSON of the pin, License-Expression); wheel `licenses/LICENSE` |
| uvicorn 0.54.0 | mcp | BSD-3-Clause | meta (PyPI JSON of the pin, License-Expression); wheel `licenses/LICENSE.md` |

The Intel EULA for Developer Tools has more limits than the Simplified Software License.
`LICENSE_NOTE` in `topopt/solver_choice.py` names use in a hosted service as an example. The
text of each licence is in the `LICENSE.txt` file of the installed package.

## Libraries inside and beside binary wheels

These items are in the licence files of the Linux x86_64 wheels that were installed for the
check, and in the macOS x86_64 wheels that were read as zip files (see "How the licences were
checked"). The macOS arm64 wheels were not inspected.

- **numpy 2.5.3 and scipy 1.18.1 (core), Linux x86_64 wheels.** Each wheel holds libgfortran
  (`GPL-3.0-or-later WITH GCC-exception-3.1`), libquadmath (`LGPL-2.1-or-later`), and
  OpenBLAS (`BSD-3-Clause`) with LAPACK (`BSD-3-Clause-Open-MPI`). Source: the licence file in
  the wheel (`LICENSE.txt` in both).
- **numpy 2.3.5 and scipy 1.18.1 (core), macOS x86_64 wheels.** The libraries are the same, in
  the `.dylibs` folder of each wheel, with one more: libgcc_s (`libgcc_s.1.1.dylib`). The licence
  file names libgfortran and libgcc together as "GCC runtime library"
  (`GPL-3.0-or-later WITH GCC-exception-3.1`). The Linux wheels of numpy 2.5.3 and scipy 1.18.1 have no libgcc file. The wheel
  of numpy holds `libscipy_openblas64_.dylib`, `libgfortran.5.dylib`, `libquadmath.0.dylib` and
  `libgcc_s.1.1.dylib`; the wheel of scipy holds `libscipy_openblas.dylib` and the same three
  runtime libraries. Source: `LICENSE.txt` in the `.dist-info` folder of each wheel
  (`numpy-2.3.5-cp312-cp312-macosx_10_13_x86_64.whl`,
  `scipy-1.18.1-cp312-cp312-macosx_10_15_x86_64.whl`).
- **cryptography 48.0.1 (extra `mcp`, macOS x86_64).** The wheel
  (`cryptography-48.0.1-cp311-abi3-macosx_10_9_universal2.whl`) has no separate library file. It
  holds a compiled module (`cryptography/hazmat/bindings/_rust.abi3.so`). Its software bill of
  materials (`sboms/sbom.json`) lists openssl 4.0.1 with no licence field. The second list
  (`sboms/cryptography-rust.cyclonedx.json`, 39 Rust packages) gives permissive licences or
  choices of licences. One entry, self_cell 1.2.2, has `Apache-2.0 OR GPL-2.0-only`.
- **Pillow 12.3.0 (core).** The software bill of materials in the wheel
  (`sboms/pillow-12.3.0.cdx.json`) lists a FriBiDi loader shim (`LGPL-2.1-or-later`) and marks
  FriBiDi (`LGPL-2.1-or-later`) as optional. The shim loads `libfribidi` at run time (the
  words of the list). The wheel does not hold `libfribidi`. The same list marks libimagequant
  (`GPL-3.0-or-later`) as optional. The wheel holds no libimagequant file (`pillow.libs` has
  none, and no `.so` file of the wheel holds the libimagequant function `liq_attr_create`).
  The list gives permissive licences to the other bundled libraries (FreeType under FTL,
  libjpeg, libtiff, libwebp, HarfBuzz and more).
- **pocl-binary-distribution 3.0 (extra `gpu-test`, Linux x86_64).** The wheel holds pocl (MIT,
  `POCL.COPYING`), hwloc (BSD, 3-clause text, `HWLOC.COPYING`) and LLVM
  (`Apache-2.0 WITH LLVM-exception`, `LLVM_LICENSE.txt`), all under
  `.libs/share/pocl/licenses/`. A macOS wheel was not checked (PyPI has a Linux x86_64 wheel only).
- **cadquery-ocp-novtk 7.9.3.1.1 (extra `step`).** The Linux x86_64 wheel holds 68 files in `cadquery_ocp_novtk.libs`: 46 OCCT `libTK*`,
  FreeImage (`libfreeimage-3.18.0.so`), `libjbig`, LibRaw, libtiff, libpng16, libjpeg, libopenjp2, libjpegxr and libjxrglue, OpenEXR (Half, Iex,
  IlmImf, IlmThread), libwebp and libwebpmux, liblcms2, libfreetype, libfontconfig, libgomp, libzstd, liblzma and libuuid. The macOS arm64 wheel
  holds 75 files in `.dylibs`: 50 OCCT `libTK*`, FreeImage, LibRaw (`libraw.23`), libtiff, libfreetype, libzstd, libpng, libjpeg, libopenjp2,
  libjpegxr, liblcms2, libwebp and libsharpyuv, OpenEXR 3 and Imath, libopenjph, libLerc, libdeflate, libz, liblzma and libc++; it has no libjbig. The licences are in
  "Components whose licences matter". Neither wheel holds a licence text for them.
- **setuptools 84.2.1 (the build backend, both lock files).** The wheel (`py3-none-any`) holds 12 packages in `setuptools/_vendor/`,
  each with its `.dist-info`. autocommand 2.2.2 is under LGPL-3.0 (License "LGPLv3", the classifier LGPLv3, the LGPL version 3 text in
  its `LICENSE`). The other 11 are permissive: backports.tarfile 1.2.0 and jaraco.text 4.0.0 (MIT, by their classifiers),
  jaraco.context 6.1.0, jaraco.functools 4.4.0, more-itertools 10.8.0, platformdirs 4.4.0, tomli 2.4.0, wheel 0.46.3 and zipp 3.23.0 (MIT),
  importlib-metadata 8.7.1 (Apache-2.0), packaging 26.3 (`Apache-2.0 OR BSD-2-Clause`). In the wheel, only two helper scripts of
  jaraco.text (`show-newlines.py`, `strip-prefix.py`) import autocommand. The build of topopt (the hook `build_editable` of
  setuptools, run with the Python of the environment on 2026-10-27) loaded no autocommand module.
- **rtree 1.4.1 (core).** The wheel holds `libspatialindex`. The wheel has no separate
  licence file for it. Its licence text is not in this check.

## Third-party files in this repository

- `topopt/dashboard/vendor/three.module.min.js`: three.js r160 (the constant `REVISION` in the file is
  `"160"`), MIT, "Copyright 2010-2023 Three.js Authors". The file is the module build of the npm
  package `three` 0.160.0 (`build/three.module.min.js`; sha256
  `3e690ac7d180b0aadf0891bea39eec643e29e2d3e75c99b18689518665f69ba6`, checked on 2026-10-25 against the
  package from `registry.npmjs.org`). The `@license` header of the file holds the copyright line and
  `SPDX-License-Identifier: MIT` but not the permission text. The text of the npm package (`LICENSE`) is
  below. The dashboard loads the file from the package. The `.mcpb` bundle and the wheel hold it too.

  ```
  The MIT License

  Copyright © 2010-2023 three.js authors

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
  ```

- `topopt/refine/vendor/three.min.js` and `topopt/refine/vendor/OrbitControls.js`: three.js r128 (the constant `REVISION` in
  `three.min.js` is `"128"`) and its `OrbitControls`, MIT, "Copyright 2010-2021 Three.js Authors". The files are
  `build/three.min.js` and `examples/js/controls/OrbitControls.js` of the npm package `three` 0.128.0 (sha256
  `9274bbcec8d96168626c732b5d31c775aa8cfb7eaa0599bec0c175908a2c1ce2` and
  `02bb4ade710f3e607329e37a21f098bc3ac70eb6e33daf8a65e79f4db785e7b2`, checked on 2026-10-27 against the package from
  `registry.npmjs.org`). The `@license` header of `three.min.js` holds the copyright line and `SPDX-License-Identifier: MIT`;
  `OrbitControls.js` has no header. The text of the npm package (`LICENSE`) is below. The refine step puts both files into
  each `result/refine.html`, so the page needs no network. The `.mcpb` bundle and the wheel hold them too.

  ```
  The MIT License

  Copyright © 2010-2021 three.js authors

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
  ```

- Two data files hold facts that come from pages and data sheets of other makers. This repository holds
  no copy of those pages or sheets.
  - `topopt/data/printers.yaml`: the printer catalog (`catalog_version` 2026-10-24; 15 models). The header
    of the file says that it is our own list of facts, written by hand from maker pages, and that no slicer
    profile is copied into it. Each row names its source (`source_url`, `field_sources`) and the date on
    which the page was opened.
  - `topopt_mcp/materials/reference.md`: printed PETG and PLA values from data sheets of filament makers.
    The file says that all its data comes from the sheets named in its section "Sheets", opened on
    2026-10-22. The server serves it as a read-only resource. Topopt does not read the numbers.
- Some files copy, adapt or port code from earlier projects of the same author. The author contributed that code
  under the MIT licence of this repository, with the organisation named in `LICENSE` as the holder. This file does not list
  that code as third-party. The code comments name the source:
  - from an earlier project of the same author, a bike part: `topopt/filter.py` (the
    functions `heaviside_project`, `heaviside_deriv`, `simp_E` and `simp_dEdrho`) and
    `topopt/refine/build.py` (adapted from its markup tool; `topopt/refine/README.md` says the same).
  - from an earlier topology optimization pipeline of the same author (each comment names the file of that
    pipeline; some name the revision d43bcf3): `topopt/verify/p2.py`, `topopt/verify/stats.py`, `topopt/verify/factor.py`,
    `topopt/tet/mesher.py`, `topopt/tet/surface.py`, `topopt/interfaces.py`, `topopt/smooth_sdf.py` and
    `topopt/print_report.py`.
  - from the research code of the GPU solver: `topopt/gpu_mg/` (the comments say "G1/G1b research code").
  Those earlier projects are not in this repository. Their code had no licence header.

## System component

- Apple Accelerate (macOS only). It is part of macOS. `topopt/native/accel_chol.c` is a small
  shim that the user builds on the Mac and that links to it. This repository does not hold
  any Accelerate code.

## How the licences were checked

- Date: 2026-10-20. Platform: Linux x86_64, Python 3.12.13. Environment: the pinned packages
  of `pyproject.toml` with the extras `dev`, `mcp`, `pardiso` and `tet`.
- pyopencl, pytools, platformdirs, typing_extensions and siphash24 were read in the installed copies (Linux x86_64, Python 3.12) on
  2026-10-26, when pyopencl became a core dependency (#176): `meta` is the installed `METADATA`, `wheel` is the file in the
  `.dist-info/licenses` folder. siphash24 has no licence file.
- setuptools 84.2.1 was read in the installed copy (Linux x86_64, Python 3.12) on 2026-10-27: `meta` is the installed `METADATA`,
  `wheel` is the file in the `.dist-info/licenses` folder. The packages that it bundles were read in the `.dist-info` folders of
  `setuptools/_vendor/` (License-Expression, else License and the classifiers).
- The extras `gpu-test` and `verify` were not installed for the check. For them, `meta`
  is the PyPI JSON of the pinned version, and `wheel` is a licence file that was read inside the
  PyPI wheel (opened as a zip file) of the pinned version: the Linux x86_64 wheel for
  pocl-binary-distribution and cholespy, and the `any` wheel for scikit-fem. The rows of pyopencl,
  pytools, platformdirs, typing_extensions and siphash24 were read earlier from the PyPI JSON and the wheels of the pinned
  versions; the installed copies (first item above) gave the same licences.
- The macOS x86_64 pins (numpy 2.3.5, numba 0.62.1, llvmlite 0.45.1, cryptography 48.0.1) and the
  scipy 1.18.1 wheel for that platform were not installed. The wheels came from PyPI with
  `pip download --no-deps --only-binary=:all: --platform macosx_10_15_x86_64 --python-version 3.12`
  (older macOS x86_64 tags and `macosx_10_9_universal2` were allowed too) on 2026-10-20. They were
  read as zip files: `meta` is the `METADATA` file of the wheel, and `wheel` is a licence file in
  it. The wheels: `numpy-2.3.5-cp312-cp312-macosx_10_13_x86_64`,
  `numba-0.62.1-cp312-cp312-macosx_10_15_x86_64`,
  `llvmlite-0.45.1-cp312-cp312-macosx_10_15_x86_64`,
  `scipy-1.18.1-cp312-cp312-macosx_10_15_x86_64` and
  `cryptography-48.0.1-cp311-abi3-macosx_10_9_universal2`. The dependency from mcp 2.3.0 to
  `pyjwt[crypto]` and from pyjwt 2.15.1 to cryptography is in the `METADATA` of those two (pure
  Python) wheels.
- The extra `step` was checked on 2026-10-23: the Linux x86_64 wheel was installed (Python 3.12.13; METADATA, the file list of
  `cadquery_ocp_novtk.libs` and the SBOM were read). The macOS arm64 wheel
  (`cadquery_ocp_novtk-7.9.3.1.1-cp312-cp312-macosx_11_0_arm64.whl`, sha256 `50931ba2...dbf4bd`, equal to the PyPI digest) was downloaded and read as a
  zip file. The licences of the bundled libraries come from the Ubuntu `copyright` files (changelogs.ubuntu.com) of the package versions in
  the SBOM; for OCCT, libgomp and zstd, from the upstream projects.
- REL-1 check of 2026-10-25 (Linux x86_64, Python 3.12.13): one venv with every extra (`dev`, `mcp`,
  `pardiso`, `tet`, `gpu`, `gpu-test`, `verify`, `step`) was read with `importlib.metadata`: 68
  distributions. The rows of the tables above agree with the installed licence fields. The pins of
  `requirements.lock` and `requirements-macos-x86_64.lock` were read from the PyPI JSON of each pinned
  version (`https://pypi.org/pypi/NAME/VERSION/json`). The vendored three.js file was compared with the
  npm package `three` 0.160.0.
- To read the licence fields of one package, run:
  `python -c "import importlib.metadata as m; d = m.metadata('NAME'); print(d['License-Expression'], d['License'], d.get_all('Classifier'))"`
- A metadata field is what the package maker wrote. Where a field and a licence file differ,
  the row says so.
- A new version can change a licence. When you change a pin in `pyproject.toml`, read the
  licence fields of the new version and update the row. The test checks that the pin in the
  row is the pin in `pyproject.toml`; it does not check the licence.
