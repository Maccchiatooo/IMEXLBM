# Contributing to IMEXLBM

Thanks for your interest. This is a small research code, so the process is
short — but a few of the rules below exist for reasons that are not obvious
from reading the tree, and skipping them tends to cost a supercomputer queue
slot to discover.

## The one structural rule

There is **one copy of the sources**, in `src/`. Everything else builds against
it:

```
src/                  portable sources — the only copy
machines/aurora/      ALCF Aurora   — SYCL, 6x PVC Max 1550
machines/polaris/     ALCF Polaris  — CUDA, 4x A100
machines/frontier/    OLCF Frontier — HIP,  4x MI250X (8 GCDs)
tools/                shared helper scripts
tests/                regression test
CMakeLists.txt        CPU build (Kokkos Serial/OpenMP) for development and CI
```

The backend is chosen entirely by **which Kokkos you link**, never by editing
code:

```cpp
using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;
```

**Machine-specific compiler flags belong in `machines/<name>/Makefile.<name>`
and nowhere else.** In particular, do not add GPU flags to `CMakeLists.txt` —
that file exists to give contributors and CI a CPU build, and mixing the two
concerns is how the per-machine tuning in those Makefiles gets quietly lost.
The Makefiles carry long comments explaining *why* each flag is there (SYCL AOT
vs JIT, `LaunchBounds` meaning three different things across CUDA/HIP/SYCL,
`-ffp-model=precise` matching how the Kokkos module was built). Read the
comment before changing a flag.

## Building for development

You need a C++17 compiler, MPI, and Kokkos with a CPU backend.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If Kokkos is not on the default search path:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKokkos_ROOT=/path/to/kokkos
```

On macOS with Homebrew, `-DOpenMP_ROOT=$(brew --prefix libomp)` is usually also
needed.

`IMEXLBM_SCALING=ON` and `IMEXLBM_PROFILE=ON` mirror `SCALING=1` and
`PROFILE=1` in the machine Makefiles.

## Running the tests

```bash
./tests/regression.sh build/imexlbm
```

It runs a 32³ Taylor-Green case on 1, 2 and 4 ranks and asserts:

- **mass conservation** — `sum_i f_i` is an exact invariant of the discrete
  dynamics, so drift beyond rounding is a bug in streaming, in the halo
  exchange, or a race;
- **momentum conservation** — likewise exact, and zero for this initial field;
- **decomposition independence** — the same case on 1, 2 and 4 ranks must agree
  to near machine epsilon, because collision is per-cell and streaming is a
  permutation.

The header of the script explains each check and, just as importantly, why
kinetic-energy monotonicity is *not* asserted.

If you add a test, **make it fail on purpose once** before you commit it. A
test that has never been red is not evidence of anything.

## What CI does and does not cover

CI builds with Kokkos on a CPU backend and runs the regression test. It cannot
build for CUDA, HIP or SYCL — GitHub runners have no GPUs.

So: **if your change touches `src/`, say in the pull request which machine you
built and ran it on.** A green CI means the code is still correct and still
compiles; it says nothing about whether the Aurora build still links.

## Pull requests

- Branch off `main`, one topic per branch.
- Commit subject in the imperative, ≤ 50 characters, no trailing period
  ("Add …", not "Added …"). Use the body to say *why*, not *what* — the diff
  already says what.
- Pull requests are squash-merged, so `main` keeps one commit per change.
- CI must be green before merge.

## Reporting a problem

Open an issue. For anything that fails on a machine, the bug report template
asks for the machine, backend, Kokkos version, compiler and rank count — those
five fields are usually the whole diagnosis, and without them a report is not
actionable.

## Citation

If IMEXLBM contributed to published work, please cite the paper listed in the
README.
