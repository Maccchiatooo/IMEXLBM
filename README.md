# IMEXLBM — Aurora (ALCF) port

Self-contained build tree for Aurora. Ported from the Polaris (ALCF) version in
the parent directory; a Frontier (OLCF) port lives in `../Frontier/`.

## Quick start

```bash
source env_aurora.sh            # prints a PASS/FAIL self-check
make -f Makefile.aurora
qsub submit_aurora.sh
```

No Kokkos build step — Aurora ships a prebuilt SYCL Kokkos module.
(`build_kokkos_aurora.sh` is included but **optional**; you almost certainly do
not need it.)

## Before you submit

1. **`submit_aurora.sh` line 18** — set `#PBS -A CHANGE_ME_PROJECT_NAME` to your
   ALCF project name.
2. **Check the queue.** The script defaults to `-q debug`, which is 1–2 nodes and
   max 1 hour. For larger runs: `debug-scaling` (2–256), `capacity` (1–16, up to
   7 days), or `prod` (the routing queue, **≥ 256 nodes**).
3. **Submit from your project directory** (`/lus/flare/<project>/...`), not from
   `$HOME` and not from `/soft/modulefiles` — ALCF documents that jobs submitted
   from those locations can end abruptly.

## Per-phase profiling

`main.cpp` only ever timed `Collision`. `PROFILE=1` builds a full breakdown of
every phase of the step:

```bash
make -f Makefile.aurora PROFILE=1     # -> LBM3D_aurora_profile
# run it exactly like the normal binary; it prints a table at exit
```

| Phase | What it covers |
|---|---|
| `collision (kernel)` | fused collision, fenced |
| `pack: enqueue faces/edges/corners` | host cost of issuing 6 / 12 / 8 strided copies |
| `pack: fence (GPU copies)` | GPU execution of all 26 pack copies |
| `mpi: post 26 isend/irecv` | posting the 26 channels |
| `mpi: waitall` | blocking wait — the actual communication cost |
| `unpack: enqueue faces/edges/corners` | host cost of issuing 26 more copies |
| `stream: enqueue kernel` | issuing `stream1` |
| `stream: fence` | GPU exec of unpack copies **and** `stream1` |
| `computeMacroscopic` | only when output is enabled |

Reported as mean / min / max **across ranks** of each rank's per-step mean, plus
% of step. Step 1 is discarded (SYCL kernel load, first-touch, MPI setup). An
`(unattributed)` row shows whatever the phases don't account for, so you can
tell at a glance whether the breakdown is complete.

### Why enqueue and fence are separate

Kokkos `deep_copy`/`parallel_for` on an execution-space instance are **async** —
timing the enqueue tells you nothing about GPU cost. Each async region is
therefore split into host-side *enqueue* and the blocking *fence* where the GPU
work is actually charged.

This deliberately does **not** insert fences between the face/edge/corner
groups. `pack_f` enqueues 26 copies and fences once — that is its OPT-T3
optimisation — and fencing between groups to "measure them separately" would
destroy the thing being measured. So those rows show enqueue cost only, and all
26 copies' GPU time lands in `pack: fence`.

Same for `unpack_f`, which has no fence at all (it is stream-ordered ahead of
`stream1`). Its GPU time is charged to `stream: fence`, and that row is labelled
to say so.

The one fence added is after `Collision`, which is otherwise unattributable —
which is exactly why a PROFILE build is a **separate binary**. Read the phase
breakdown from it; read MLUPS from the SCALING build.

### What to look for

From your 1-node smoke test, collision was only ~12% of the step and the other
88% was `Streaming()`. This breakdown splits that 88% into pack enqueue, pack
GPU, MPI wait, unpack enqueue, and stream GPU — which tells you which fix pays:

* **`mpi: waitall` dominant** → halo *volume* is the problem. Every channel
  ships all 27 directions when only 9 cross a face (3 per edge, 1 per corner);
  pruning is a ~3× reduction. The unused `d_dirset` member in `lbm.hpp` is a
  half-finished attempt at exactly this.
* **enqueue rows dominant** → 52 kernel launches/step is the problem, and the
  fix is fusing the copies into one packing kernel.
* **`pack: fence` dominant** → the strided copies themselves are inefficient.

## Strong scaling study, 1 → 256 nodes

```bash
make -f Makefile.aurora SCALING=1        # separate binary, see below
qsub submit_strong_scaling.sh            # edit -A first
./parse_scaling.sh scaling_<jobid>       # after it finishes
```

Sweep: **1, 2, 4, 8, 16, 32, 64, 128, 256 nodes** = 12 → 3072 ranks.

### One job, not nine

The natural approach — one `qsub` per node count — fights the queue policy:
`debug` and `debug-scaling` each allow only **one job running/accruing/queued
per user**, so nine submissions would have to be babysat serially.

Instead the script requests 256 nodes **once** and runs every size inside that
allocation, giving `mpiexec` a `--hostfile` that is the first N lines of
`$PBS_NODEFILE`. Strong scaling means work per run shrinks as N grows, so the
sweep is dominated by the 1-node run and fits inside the 1-hour
`debug-scaling` limit. Bonus: every point runs on the *same physical nodes*, so
node-to-node variation can't masquerade as a scaling effect.

### Why 768³

Strong scaling fixes the global problem size, so it has to satisfy all sizes at
once. 768³ works because:

* **Fits on 1 node.** 768³ × ~696 B/cell = 315 GB over 12 tiles ≈ **26 GB/tile**
  against 64 GB HBM. (1024³ would be 62 GB/tile — no room for ghosts.)
* **Divides cleanly.** 768 = 2⁸·3, and every rank count is 12·2ᵏ, so no point is
  penalised by load imbalance.
* **Still non-degenerate at 256 nodes:** ~48×48×64 = 147k cells per rank.

Override with `NX=... NY=... NZ=... qsub -v ...` or by editing the block at the
top of the script.

Note `System.cpp` derives `tau` from `sx`, so tau at 768³ differs from your 256³
runs. Irrelevant for throughput — this measures performance, not physics.

### The build must be `SCALING=1`

`main.cpp` times `Collision()` every step with a `Kokkos::fence()` pair and a
`printf`. That is great for kernel tuning and **wrong for scaling runs**: it
costs tens of microseconds per step, and at 256 nodes a step is only ~100 µs of
compute, so the instrumentation would dominate what it is measuring — and it
would penalise exactly the high-node-count points, manufacturing a scaling
cliff that isn't real.

`SCALING=1` compiles that block out via `-DLBM_NO_PERSTEP_TIMING` and builds a
**separate** binary (`LBM3D_aurora_scaling`) with separate objects, so your
normal build is untouched and both can coexist. The change in `main.cpp` is
guarded, so default behaviour is byte-identical to before.

### Reading the output

`parse_scaling.sh` prints mean MLUPS, speedup, and parallel efficiency per node
count (`--csv` for a spreadsheet). It **discards the first reporting interval**
of every run — that interval absorbs SYCL kernel load, first-touch allocation
and MPI connection setup, and averaging it in would hurt the short high-node
runs far more than the 1-node run. min/max are shown so you can see spread
rather than trusting a bare mean.

Expect efficiency to fall off at the high end. With `ghost=3`, the
halo-to-interior ratio grows from ~6% at 1 node to ~38% at 256 as the subdomain
shrinks, so communication comes to dominate. That is a real property of the
decomposition, not a measurement artefact — and it is arguably the most useful
thing the study will tell you.

## Polaris → Aurora差异

| | Polaris (ALCF) | Aurora (ALCF) |
|---|---|---|
| GPU | 4x A100, `cc80`, CUDA | 6x PVC Max 1550 = **12 tiles**, SYCL |
| Ranks per node | 4 | **12** (one per tile) |
| CPU | — | 2x52 cores; **cores 0 and 52 reserved** |
| Compiler | `CC` (nvc++) | `mpic++ -cxx=icpx` |
| Kokkos | `/soft` module, `KOKKOS_HOME` | `module load kokkos`, **`KOKKOS_ROOT`** |
| Kokkos backend | CUDA | SYCL, **ahead-of-time (AOT)** |
| Scheduler | PBS (`qsub`) | PBS (`qsub`) — same family |
| Nodefile | `$COBALT_NODEFILE` | `$PBS_NODEFILE` |
| GPU pinning | `CUDA_VISIBLE_DEVICES` | `ZE_AFFINITY_MASK` |
| Local rank var | `$PMI_LOCAL_RANK` | `$PALS_LOCAL_RANKID` |
| GPU-aware MPI | `MPICH_GPU_SUPPORT_ENABLED=1` | `MPIR_CVAR_ENABLE_GPU=1` (default) |
| Memory space | `Kokkos::CudaSpace` | `Kokkos::SYCLDeviceUSMSpace` |

## File map

| Aurora file | Polaris original |
|---|---|
| `env_aurora.sh` | `../env.sh` |
| `Makefile.aurora` | `../Makefile.polaris` |
| `set_affinity_gpu_aurora.sh` | `../set_affinity_gpu_polaris.sh` |
| `submit_aurora.sh` | `../submit_script.sh` |
| `build_kokkos_aurora.sh` | *(optional — the module normally suffices)* |
| `submit_strong_scaling.sh` | *(new)* 1→256 node sweep in a single allocation |
| `parse_scaling.sh` | *(new)* turns the sweep logs into a speedup/efficiency table |

## Source changes

The `.cpp` / `.hpp` files here are **portable** — byte-identical to the Polaris
and Frontier trees, and they build unmodified on all three machines.

### 1. No vendor memory space in the source (`lbm.hpp`, `lbm.cpp`)

All 15 uses of `Kokkos::CudaSpace` now go through one alias:

```cpp
using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;
```

It resolves to `CudaSpace` / `HIPSpace` / `SYCLDeviceUSMSpace` depending on
which Kokkos you link. This matters more on Aurora than on Frontier: the module
carries **three** backends (Serial, OpenMP, SYCL), so a misconfigured build can
silently pick a host backend. A `static_assert` rejects that case — it would
otherwise put every buffer in `HostSpace` and hand **host** pointers to the
GPU-aware MPI calls in `exchange_f()`.

### 2. Backend-conditional `LaunchBounds` (`lbm.hpp`)

The tuned Polaris values are now applied only under `KOKKOS_ENABLE_CUDA`. The
second `LaunchBounds` parameter is `minBlocksPerSM` on CUDA but `minWavesPerEU`
on HIP, and the Kokkos SYCL backend **ignores `LaunchBounds` entirely** — on
Aurora it is a no-op either way, so the guard costs nothing here and prevents
the value from silently hurting Frontier.

## Two things to verify on the first run

### Register/private-memory pressure in `Collision`

The fused `Collision` kernel holds `double fl[27]` = 216 B of private memory per
work-item. On PVC that may exceed the default GRF budget and spill. Check it:

```
-Xsycl-target-backend "-device pvc -options -ze-opt-print-reg-usage"
```

If it spills, the PVC knob is large-GRF mode — add
`-options -ze-opt-large-register-file` to `SYCL_AOT_LDFLAGS` in
`Makefile.aurora`. (This is the Aurora analogue of the `LaunchBounds` occupancy
problem on Frontier.)

### Kernel argument size — the most likely thing to bite

Every kernel uses `KOKKOS_CLASS_LAMBDA`, which captures `*this` **by value** —
and `LBM` holds roughly 170 `View` members (the 52 `f_face_send`/`f_edge_send`/
`f_corner_send` entries are captured too, even though `lbm.hpp` marks them
"currently unused"). That is on the order of 8 KB of closure per launch.

Kokkos handles oversized functors by staging them through device memory rather
than kernel arguments, so this should work — but SYCL is stricter about kernel
argument size and device-copyability than CUDA is, and the staging costs a copy
per launch. If you hit a kernel-argument limit at compile time, or see
suspicious per-launch overhead in the `collision` timings that `main.cpp`
prints, the fix is to hoist the handful of Views each kernel actually uses into
locals and switch that kernel from `KOKKOS_CLASS_LAMBDA` to `KOKKOS_LAMBDA`:

```cpp
auto f_ = f; auto e_ = e; auto t_ = t;      // capture 3 Views, not 170
Kokkos::parallel_for("collision", policy, KOKKOS_LAMBDA(...) { ... });
```

Deleting the unused `f_face_send` / `f_edge_send` / `f_corner_send` arrays would
also cut the capture by about a third on its own.

## Not yet verified

None of this has been compiled or run — it was ported without Aurora access.
The shell scripts pass `bash -n`; that is the extent of local verification.

* The AOT flags follow ALCF's documented Kokkos makefile, but AOT builds are
  slow to link — expect the first `make` to take a while, and do it on a login
  node.
* The `{1,1,128}` MDRange tile sizes were tuned for A100's 32-wide warp. PVC's
  natural SIMD width is 16 (or 32), so these are worth re-sweeping.
* `sycl-ls` in the self-check only reports meaningfully on a compute node;
  on a login node it will show no GPUs, which is expected, not a failure.

## Keeping in sync with the other trees

The `.cpp` / `.hpp` files here are **copies**. The canonical set lives in the
parent directory; `../Polaris/` and `../Frontier/` hold copies too. All four are
identical and portable, so edits must be mirrored or the trees drift:

```bash
for f in *.cpp *.hpp; do
  diff -q "$f" "../$f"; diff -q "$f" "../Polaris/$f"; diff -q "$f" "../Frontier/$f"
done
```

## References

* [Running Jobs on Aurora](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/)
* [Kokkos on Aurora](https://docs.alcf.anl.gov/aurora/programming-models/kokkos-aurora/)
* [Aurora Example Program Makefile](https://docs.alcf.anl.gov/aurora/compiling-and-linking/aurora-example-program-makefile/)
