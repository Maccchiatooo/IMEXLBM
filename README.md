# IMEXLBM — Polaris (ALCF)

## Quick start

```bash
source env.sh                   # prints a PASS/FAIL self-check
make -f Makefile.polaris
```

## Node / machine summary

| | Polaris |
|---|---|
| GPU | 4x NVIDIA A100 40GB, `cc80` |
| CPU | 1x AMD EPYC 7543P (Milan), 32 cores / 64 threads |
| Ranks per node | 4 (one per A100), `--depth=8` |
| PrgEnv | `PrgEnv-nvidia` (nvc++ under the `CC` wrapper) |
| Kokkos | prebuilt `/soft` module, `KOKKOS_HOME` |
| Kokkos backend | CUDA |
| Scheduler | PBS Pro (`qsub`) |
| GPU pinning | `CUDA_VISIBLE_DEVICES`, rank from `$PMI_LOCAL_RANK` |
| GPU-aware MPI | `MPICH_GPU_SUPPORT_ENABLED=1` + `craype-accel-nvidia80` (GTL) |

## Files

| File | Notes |
|---|---|
| `env.sh` | modules + CUDA + Kokkos paths + runtime knobs, with self-check |
| `Makefile.polaris` | `CC` (nvc++), `-cuda -gpu=cc80,nordc` |
| `set_affinity_gpu_polaris.sh` | ALCF's standard script; reversed GPU mapping for NIC affinity |
| `submit_script.sh` | **stale — see below** |

### `submit_script.sh` will not run as written

It predates Polaris' move from Cobalt to PBS Pro, and has several problems:

```bash
export OMPI_CXX=/grand/IMEXLBM/czhao/Kokkos/kokkos/bin/nvcc_wrapper
NODES='cat $COBALT_NODEFILE|wc-|'
PROCS=$((NODES*8))
make -j KOKKOS_DEVICES=Cuda
mpirun -f $COBALT_NODEFILE -n $PROCS LBM.exe
```

* `$COBALT_NODEFILE` no longer exists — Polaris is PBS Pro, so it is
  `$PBS_NODEFILE`, and the launcher is `mpiexec` (Cray PALS), not `mpirun -f`.
* `NODES=` uses single quotes, not backticks, so it assigns the literal string;
  and `wc-|` is a typo for `wc -l`. `PROCS` therefore evaluates to 0.
* `*8` assumes 8 ranks per node. Polaris has **4** A100s — this should be `*4`.
* `LBM.exe` is not the binary `Makefile.polaris` builds (`LBM3D_polaris`).
* `OMPI_CXX` is an **OpenMPI** variable pointing at a stale personal path;
  this build uses Cray MPICH through the `CC` wrapper and ignores it.
* `make -j KOKKOS_DEVICES=Cuda` is the old inline-Kokkos build style and does
  not match `Makefile.polaris`, which needs `-f Makefile.polaris`.

I left the file untouched since you didn't ask for it to be ported. If you want
a working PBS version (the `Frontier/` and `Aurora/` folders both have one), say
so and I'll add `submit_polaris.sh`. The shape would be:

```bash
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -q debug
...
mpiexec -n $((NNODES*4)) --ppn 4 --depth=8 --cpu-bind depth \
        ./set_affinity_gpu_polaris.sh ./LBM3D_polaris
```

## Strong scaling study, 1 -> 256 nodes

```bash
make -f Makefile.polaris SCALING=1     # build the scaling binary FIRST
qsub submit_strong_scaling.sh   # edit -A first
./parse_scaling.sh scaling_<jobid>
```

Sweep: **1, 2, 4, 8, 16, 32, 64, 128, 256 nodes** = 4 -> 1024 ranks.

### One job, not nine

One qsub per node count fights the queue policy: `debug` is 1-2 nodes and
`debug-scaling` is 1-10, so most of the sweep cannot even be submitted there.
Instead this asks for 256 nodes once (`prod`, which is 10-496) and runs every
size inside that allocation, giving mpiexec a `--hostfile` that is the first N
lines of `$PBS_NODEFILE`.

Strong scaling means the work per run shrinks as N grows, so the sweep is
dominated by the 1-node run. Bonus: every point runs on the *same physical
nodes*, so node-to-node variation cannot masquerade as a scaling effect.

### Why 512^3

Polaris' A100s have 40 GB each, and the 1-node baseline is the binding
constraint:

| grid | per GPU (4/node) | fits 40 GB? |
|---|---|---|
| 512^3 | ~23 GB | yes |
| 768^3 | ~79 GB | **no** |

512 = 2^9 divides cleanly across every rank count (4 x 2^k), and at 256 nodes
each rank still holds 131k cells.

**Cross-machine comparison:** Frontier and Aurora both default to 768^3, so
Polaris numbers are *not* directly comparable as shipped. To compare, run
768^3 here starting from 4 nodes (~20 GB/GPU):

```bash
qsub -v NX=768,NY=768,NZ=768,NODE_LIST="4 8 16 32 64 128 256" submit_strong_scaling.sh
```

The 1- and 2-node points simply do not exist at that size on Polaris.

### The build must be `SCALING=1`

`main.cpp` times `Collision()` every step with a fence pair and a `printf`.
Great for kernel tuning, wrong for scaling runs: it costs tens of microseconds
per step and would penalise exactly the high-node-count points, manufacturing a
scaling cliff that is not real. `SCALING=1` compiles it out into a **separate**
binary with separate objects, so your normal build is untouched.

`PROFILE=1` additionally builds the per-phase profiler (collision / pack /
MPI / unpack / stream, with GFLOP/s and GB/s). It adds two fences to make
phases attributable, so it is **not** a throughput build -- read structure from
it, read MLUPS from the SCALING build.

### Reading the output

`parse_scaling.sh` prints mean MLUPS, speedup and parallel efficiency per node
count (`--csv` for a spreadsheet). It discards the first reporting interval of
every run -- that interval absorbs kernel load, first-touch allocation and MPI
setup, and averaging it in would hurt the short high-node runs far more than the
1-node run.

Expect efficiency to fall off at the high end: with `ghost=3` the halo/interior
ratio grows as the subdomain shrinks, so communication comes to dominate. That
is a real property of the decomposition, not a measurement artefact.

## Source changes

`lbm.hpp` and `lbm.cpp` are now **portable** — byte-identical across all three
trees, and they still build on Polaris exactly as before. Two changes:

### 1. No vendor memory space in the source

All 15 uses of `Kokkos::CudaSpace` now go through one alias:

```cpp
using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;
```

On Polaris this resolves right back to `Kokkos::CudaSpace`, so **nothing changes
for this build**. It resolves to `HIPSpace` on Frontier and
`SYCLDeviceUSMSpace` on Aurora. A `static_assert` rejects a host-only Kokkos,
which would otherwise put every buffer in `HostSpace` and hand host pointers to
the GPU-aware MPI calls in `exchange_f()`.

### 2. Backend-conditional `LaunchBounds`

Your tuned values are preserved verbatim under `KOKKOS_ENABLE_CUDA`:

```cpp
#if defined(KOKKOS_ENABLE_CUDA)
using launch_bounds_3 = Kokkos::LaunchBounds<256, 4>;    // as before
using launch_bounds_4 = Kokkos::LaunchBounds<128, 10>;   // as before
#else
...
#endif
```

The guard exists because the second parameter means `minBlocksPerSM` on CUDA
but `minWavesPerEU` on HIP, where `4` would cap `Collision` at ~64 VGPRs and
force scratch spilling on MI250X. Kokkos' SYCL backend ignores `LaunchBounds`
entirely. **Polaris behaviour is unchanged.**

## Pre-existing note worth revisiting

The comment block above `Collision()` in `lbm.cpp` says:

> tile `{1,1,64}` = 64 threads/block must match `LaunchBounds<64,...>`

but the code actually uses tile `{1,1,128}` with `LaunchBounds<256, 4>`. Not a
bug — just a stale comment from an earlier tuning pass. Left as-is.

## Keeping in sync with the other trees

The `.cpp` / `.hpp` files here are **copies**. The canonical set lives in the
parent directory; `Frontier/` and `Aurora/` hold copies too. All four are
identical and portable, so edits must be mirrored or the trees drift:

```bash
for f in *.cpp *.hpp; do
  diff -q "$f" "../$f"; diff -q "$f" "../Frontier/$f"; diff -q "$f" "../Aurora/$f"
done
```

## References

* [Running Jobs on Polaris](https://docs.alcf.anl.gov/polaris/running-jobs/)
* [Example Job Scripts](https://docs.alcf.anl.gov/running-jobs/example-job-scripts/)
* [Kokkos on Polaris](https://docs.alcf.anl.gov/polaris/programming-models/kokkos-polaris/)
