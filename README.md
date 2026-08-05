# IMEXLBM: A Portable Lattice-Boltzmann Solver for Heterogeneous Platforms

One portable source set, three machine-specific build trees.

```
.                       canonical sources (edit here)
├── Polaris/            ALCF  — 4x A100,        CUDA
├── Frontier/           OLCF  — 4x MI250X,      HIP
└── Aurora/             ALCF  — 6x PVC Max 1550, SYCL
```

Each folder is **self-contained** — sources, build files, job script, and a
README with the machine's specifics and caveats. Start with the README in the
folder for your machine.

## The sources are shared and identical

`main.cpp`, `lbm.cpp`, `lbm.hpp`, `mpi_view_transfer.cpp`, `System.cpp`,
`System.hpp` are byte-identical in all four locations. The backend is chosen
entirely by **which Kokkos you link**, not by editing code:

```cpp
using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;
```

That resolves to `CudaSpace` / `HIPSpace` / `SYCLDeviceUSMSpace` per machine.

Because they are copies rather than symlinks, edits must be mirrored. To check:

```bash
for f in *.cpp *.hpp; do
  for d in Polaris Frontier Aurora; do diff -q "$f" "$d/$f"; done
done
```

## At a glance

| | Polaris | Frontier | Aurora |
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

All three require GPU-aware MPI — `exchange_f()` passes device pointers straight
into `MPI_Isend`/`MPI_Irecv`. Without it, the first exchange faults.

## 📚 Citation

If you use **IMEXLBM** in your research, please cite the following publications:

### Key Publications
> [1] **Zhao, C., Patel, S., Balakrishnan, R. and Lee, T., 2025.** IMEXLBM: A portable lattice-Boltzmann solver for heterogeneous platforms. In *Fluids Engineering Division Summer Meeting* (Vol. 88995, p. V001T03A016). American Society of Mechanical Engineers.

