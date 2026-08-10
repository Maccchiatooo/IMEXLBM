# IMEXLBM — Frontier (OLCF) port

## Quick start

```bash
./build_kokkos_frontier.sh          # login node, ~10 min, ONE TIME ONLY
source env_frontier.sh              # prints a PASS/FAIL self-check
make -f Makefile.frontier
sbatch submit_frontier.sh
```

## Before you submit

1. **`submit_frontier.sh` line 13** — set `#SBATCH -A CHANGE_ME_PROJECT_ID` to
   your OLCF project ID. The job is rejected at submit time otherwise.
2. `build_kokkos_frontier.sh` installs to
   `$HOME/opt/kokkos-4.6.02-hip-gfx90a`. Override with `KOKKOS_PREFIX=...` and
   set the matching `KOKKOS_HOME` before sourcing `env_frontier.sh`.

## Polaris -> Frontier差异

| | Polaris (ALCF) | Frontier (OLCF) |
|---|---|---|
| GPU | 4x A100, `cc80`, CUDA | 4x MI250X = **8 GCDs**, `gfx90a`, HIP |
| Ranks per node | 4 | **8** (one per GCD) |
| PrgEnv | `PrgEnv-nvidia` (nvc++) | `PrgEnv-amd` (amdclang++) |
| Accel module | `craype-accel-nvidia80` | `craype-accel-amd-gfx90a` |
| Kokkos | pre-built `/soft` module | **no module — build it yourself** |
| Scheduler | PBS (`qsub`) | Slurm (`sbatch` / `srun`) |
| GPU pinning | `CUDA_VISIBLE_DEVICES` | `ROCR_VISIBLE_DEVICES` |
| Memory space | `Kokkos::CudaSpace` | `Kokkos::HIPSpace` |

## File map

| Frontier file | Polaris original |
|---|---|
| `env_frontier.sh` | `../env.sh` |
| `Makefile.frontier` | `../Makefile.polaris` |
| `set_affinity_gpu_frontier.sh` | `../set_affinity_gpu_polaris.sh` |
| `submit_frontier.sh` | `../submit_script.sh` |
| `build_kokkos_frontier.sh` | *(new — no Kokkos module on Frontier)* |

## Strong scaling study, 1 -> 256 nodes

```bash
make -f Makefile.frontier SCALING=1     # build the scaling binary FIRST
sbatch submit_strong_scaling.sh   # edit -A first
./parse_scaling.sh scaling_<jobid>
```

Sweep: **1, 2, 4, 8, 16, 32, 64, 128, 256 nodes** = 8 -> 2048 ranks.

### One job, not nine

Nine separate sbatch jobs would queue independently, land on different nodes,
and take days of wall-clock to all schedule. This asks for 256 nodes once and
runs every size inside that allocation. Slurm makes the subsetting easy --
`srun -N <n>` just uses the first n nodes, so no hostfile juggling is needed.

Strong scaling means the work per run shrinks as N grows, so the sweep is
dominated by the 1-node run. Bonus: every point runs on the *same physical
nodes*, so node-to-node variation cannot masquerade as a scaling effect.

### Why 768^3

768^3 x ~696 B/cell = 315 GB over 8 GCDs = **~39 GB/GCD** against 64 GB of
HBM. 768 = 2^8 x 3 divides cleanly across every rank count (8 x 2^k), and at
256 nodes each rank still holds 221k cells.

This is the **same grid Aurora uses**, so those two machines are directly
comparable. Polaris uses 512^3 instead -- its 40 GB A100s cannot hold 768^3 on
a single node.

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

The `.cpp` / `.hpp` files here are **portable** — they build unmodified on both
machines. Two changes vs. the Polaris originals:

### 1. No vendor memory space in the source (`lbm.hpp`, `lbm.cpp`)

All 15 uses of `Kokkos::CudaSpace` now go through one alias:

```cpp
using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;
```

It resolves to `CudaSpace` or `HIPSpace` depending on which Kokkos you link, so
the backend is a build-time choice rather than a source edit. A `static_assert`
rejects a host-only (Serial/OpenMP) Kokkos, which would otherwise silently put
every buffer in `HostSpace` and hand **host** pointers to the GPU-aware MPI
calls in `exchange_f()`.

### 2. Backend-conditional `LaunchBounds` (`lbm.hpp`)

The second `LaunchBounds` template parameter means different things per backend:

* CUDA — `__launch_bounds__(maxT, minBlocksPerSM)`; `4` is a mild hint.
* HIP — `__launch_bounds__(maxT, minWavesPerEU)`; `4` means 4 waves/EU x 4 EUs
  = 16 waves/CU, capping the kernel at **~64 VGPRs/thread**.

`Collision`'s `double fl[27]` register array alone wants ~54 VGPRs, so that cap
would force scratch (local-memory) spilling on MI250X and destroy performance.

The tuned Polaris values are therefore applied only under `KOKKOS_ENABLE_CUDA`;
every other backend leaves the occupancy floor unset and lets the compiler
choose. (The Kokkos SYCL backend on Aurora ignores `LaunchBounds` entirely, so
the same guard covers that tree too.)

## GPU pinning is manual, deliberately

Slurm's `--gpu-bind=closest` is documented to **hang GPU-aware MPI on Frontier**,
and this code cannot run without GPU-aware MPI (`exchange_f()` passes `HIPSpace`
device pointers straight into `MPI_Isend`/`MPI_Irecv`). So the job uses
`--gpu-bind=none` and `set_affinity_gpu_frontier.sh` pins each rank by hand.

The map is neither identity nor reversed — it comes from the node's L3/NUMA to
GCD table:

```
L3 region (cores)   0-7  8-15 16-23 24-31 32-39 40-47 48-55 56-63
closest GCD          4     5     2     3     6     7     0     1
```

This is why `srun -c 7` matters: low-noise mode reserves the first core of each
of the 8 L3 regions, leaving 56 allocatable cores, and 56/8 = 7 puts local rank
`i` on L3 region `i` — exactly what the table above assumes.

## Not yet verified

None of this has been compiled or run — it was ported without Frontier access.
The shell scripts pass `bash -n`; that is the extent of local verification.

* Expect the first `make` to need small flag adjustments. Likeliest source of
  warnings: the Cray `CC` wrapper duplicating `-x hip` / `--offload-arch=gfx90a`
  when `craype-accel-amd-gfx90a` is loaded (harmless in most CPE versions).
* **Check for register spilling** before trusting any timing. Build with
  `-Rpass-analysis=kernel-resource-usage` and confirm `ScratchSize: 0` on the
  `collision` kernel.
* The `{1,1,128}` MDRange tile sizes were tuned for A100's 32-wide warp.
  MI250X wavefronts are 64, so these are worth re-sweeping on the machine.

## Keeping in sync with the parent directory

The `.cpp` / `.hpp` files here are **copies**. The canonical set lives in the
parent directory; `../Polaris/` and `../Aurora/` hold copies too. All four are
identical and portable, so edits must be mirrored or the trees drift:

```bash
for f in *.cpp *.hpp; do
  diff -q "$f" "../$f"; diff -q "$f" "../Polaris/$f"; diff -q "$f" "../Aurora/$f"
done
```

## References

* [Frontier User Guide](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html)
* [Crusher Quick-Start Guide](https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html)
