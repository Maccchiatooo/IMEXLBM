# IMEXLBM

[![CI](https://github.com/Maccchiatooo/IMEXLBM/actions/workflows/ci.yml/badge.svg)](https://github.com/Maccchiatooo/IMEXLBM/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21866050.svg)](https://doi.org/10.5281/zenodo.21866050)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

A portable lattice-Boltzmann solver for heterogeneous platforms. One source set
builds for NVIDIA, AMD and Intel GPUs through the
[Kokkos](https://github.com/kokkos/kokkos) performance-portability library —
the backend is chosen entirely by which Kokkos you link, never by editing code:

```cpp
using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;
```

That resolves to `CudaSpace`, `HIPSpace` or `SYCLDeviceUSMSpace` per machine.

## What it solves

A three-dimensional Taylor-Green vortex, the standard transition-to-turbulence
benchmark, on a fully periodic domain.

- **D3Q27** lattice, single-relaxation-time collision. The `tau0 + 0.5` in the
  relaxation rate is the transformed implicit treatment of the collision term —
  the IMEX in the name.
- **Pressure-based incompressible formulation**: `sum_i f_i = 3p` and
  `sum_i f_i e_i = u`, so both moments are recovered exactly by the equilibrium
  and are conserved by collision.
- **3D Cartesian MPI decomposition**, factorised automatically from the rank
  count into whichever `nx x ny x nz` grid gives the most cube-like subdomain.
  Any rank count works.
- **26-neighbour halo exchange** with three ghost layers, passing device
  pointers straight into `MPI_Isend`/`MPI_Irecv`.

## Repository layout

```
src/                 portable sources — the only copy
machines/
  aurora/            ALCF Aurora   — SYCL, 6x PVC Max 1550
  polaris/           ALCF Polaris  — CUDA, 4x A100
  frontier/          OLCF Frontier — HIP,  4x MI250X (8 GCDs)
tools/               parse_scaling.sh, shared by all three machines
tests/               regression test
CMakeLists.txt       CPU build (Kokkos Serial/OpenMP) for development and CI
input.in             example input deck
```

Each `machines/<name>/` directory is self-contained — Makefile, environment
script, job script, and a README with that machine's specifics and caveats.
**Start with the README for your machine**:
[Aurora](machines/aurora/README.md) ·
[Polaris](machines/polaris/README.md) ·
[Frontier](machines/frontier/README.md).

Machine-specific compiler flags live in those Makefiles and nowhere else; the
CMake build exists to give contributors and CI a CPU build. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Quick start

### On a laptop or cluster (CPU)

Needs a C++17 compiler, MPI, and Kokkos with a CPU backend.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

cd build
cp ../input.in .          # then shrink the grid — see below
mpirun -n 2 ./imexlbm
```

If Kokkos is not on the default search path, add
`-DKokkos_ROOT=/path/to/kokkos`. On macOS with Homebrew you will usually also
need `-DOpenMP_ROOT=$(brew --prefix libomp)`.

> The shipped `input.in` is 256³, which needs roughly 7.8 GB for the two `q=27`
> distribution arrays. Drop it to `32 32 32` (about 14 MB) for a quick local run.

`-DIMEXLBM_SCALING=ON` and `-DIMEXLBM_PROFILE=ON` mirror `SCALING=1` and
`PROFILE=1` in the machine Makefiles.

### On a supercomputer (GPU)

```bash
cd machines/aurora                # or polaris, or frontier
source env_aurora.sh              # prints a PASS/FAIL self-check
make -f Makefile.aurora
cp ../../input.in .               # the solver reads input.in from the CWD
qsub submit_aurora.sh             # sbatch on Frontier
```

Frontier additionally needs a one-time `./build_kokkos_frontier.sh` on a login
node; Polaris and Aurora use a site-provided Kokkos module. Polaris's
environment script is `env.sh`. Read that machine's README before submitting —
each has caveats about queues, project names and GPU pinning.

## Input deck

`input.in` is nine whitespace-separated numbers, read in this order. The solver
reads it from the current working directory.

```
rho0   R    Re          density, (unused), Reynolds number
u0     Time inter       peak velocity, number of steps, reporting interval
nx     ny   nz          global grid
```

Anything after the ninth number is ignored, which is why the shipped deck can
carry a legend below the values. `R` is parsed into `System::R` and never read
again anywhere in the code; it is kept only so existing decks stay valid.

Derived quantities, with the Taylor-Green convention `L = nx / 2π`:

```
cs² = 1/3          Ma  = u0 / cs
nu  = rho0 u0 nx / (2π Re)          tau = u0 nx / (Re cs² 2π)
```

## Tests

```bash
./tests/regression.sh build/imexlbm
```

Runs a 32³ case on 1, 2 and 4 ranks and asserts three things:

| Check | Why it is meaningful |
|---|---|
| Mass conservation | `sum_i f_i` is an *exact* invariant of the discrete dynamics, so drift beyond rounding is a bug in streaming, in the halo exchange, or a race — never "numerical error" |
| Momentum conservation | Likewise exact, and zero for this initial field |
| Decomposition independence | Collision is per-cell and streaming is a permutation, so 1, 2 and 4 ranks must agree to near machine epsilon. This is the check that exercises the halo exchange |

The script's header explains each check, and why kinetic-energy monotonicity is
deliberately *not* asserted.

CI runs this on every pull request. It cannot build for CUDA, HIP or SYCL —
GitHub runners have no GPUs — so a green badge means the physics and the CPU
build are intact, not that the Aurora build still links.

## At a glance

| | [Polaris](machines/polaris/README.md) | [Frontier](machines/frontier/README.md) | [Aurora](machines/aurora/README.md) |
|---|---|---|---|
| GPU | 4x A100 `cc80` | 4x MI250X = **8 GCDs** `gfx90a` | 6x PVC = **12 tiles** |
| Ranks/node | 4 | 8 | 12 |
| Backend | CUDA | HIP | SYCL (AOT) |
| Compiler | `CC` (nvc++) | `CC` (amdclang++) | `mpic++ -cxx=icpx` |
| Kokkos | module, `KOKKOS_HOME` | **build it yourself** | module, `KOKKOS_ROOT` |
| Scheduler | PBS (`qsub`) | Slurm (`sbatch`) | PBS (`qsub`) |
| Local rank | `$PMI_LOCAL_RANK` | `$SLURM_LOCALID` | `$PALS_LOCAL_RANKID` |
| GPU pinning | `CUDA_VISIBLE_DEVICES` | `ROCR_VISIBLE_DEVICES` | `ZE_AFFINITY_MASK` |
| GPU-aware MPI | `MPICH_GPU_SUPPORT_ENABLED=1` | `MPICH_GPU_SUPPORT_ENABLED=1` | `MPIR_CVAR_ENABLE_GPU=1` |

**All three require GPU-aware MPI.** `exchange_f()` passes device pointers
straight into `MPI_Isend`/`MPI_Irecv`; without it, the first exchange faults.

## Requirements

- C++17 compiler (Kokkos 5.x itself requires C++20)
- MPI, GPU-aware for the GPU builds
- Kokkos — built against 4.6.02 (Polaris), 4.7.02 (Frontier) and 5.1.1 (CI)
- CMake ≥ 3.23, for the CPU build only; the machine builds use plain `make`

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports should go through the issue
template, which asks for machine, backend, Kokkos version, compiler and rank
count — with three vendor toolchains in play, those five fields are usually
most of the diagnosis.

## License

BSD 3-Clause. See [LICENSE](LICENSE).

## Citation

If IMEXLBM contributed to published work, please cite the paper:

> Zhao, C., Patel, S., Balakrishnan, R. and Lee, T., 2025. IMEXLBM: A portable
> lattice-Boltzmann solver for heterogeneous platforms. In *Fluids Engineering
> Division Summer Meeting* (Vol. 88995, p. V001T03A016). American Society of
> Mechanical Engineers. [doi:10.1115/FEDSM2025-158589](https://doi.org/10.1115/FEDSM2025-158589)

To cite the software itself, use the DOI of the exact version you ran — a
reproducer needs that snapshot, not whatever `main` looks like later. The badge
above is the concept DOI, which always resolves to the newest release.
[CITATION.cff](CITATION.cff) carries both, and GitHub's **Cite this repository**
button renders them as BibTeX or APA.
