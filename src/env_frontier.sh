#!/bin/bash
# =============================================================================
# IMEXLBM / LBM3D_frontier  environment for Frontier (OLCF)
# Usage:  source env_frontier.sh
# Goal:   one source sets EVERYTHING (compiler env + ROCm + Kokkos libs + PATH)
#         so binaries run on any compute node without per-node library fixups.
#
# Frontier vs Polaris, in one table:
#   GPU              A100 (cc80, CUDA)        MI250X (gfx90a, HIP)
#   GPUs per node    4                        4 cards = 8 GCDs (8 MPI ranks)
#   PrgEnv           PrgEnv-nvidia (nvc++)    PrgEnv-amd (amdclang++)
#   accel module     craype-accel-nvidia80    craype-accel-amd-gfx90a
#   Kokkos           /soft module             build it yourself (no OLCF module)
#   scheduler        PBS (qsub)               Slurm (sbatch/srun)
#   GPU pinning      CUDA_VISIBLE_DEVICES     ROCR_VISIBLE_DEVICES
# =============================================================================

# ---- 1. Programming environment ---------------------------------------------
# PrgEnv-amd gives amdclang++ under the `CC` wrapper, which is what
# Makefile.frontier's `-x hip --offload-arch=gfx90a` flags expect.
# (PrgEnv-cray also works if you keep CCE and ROCm at matching Clang majors,
# but PrgEnv-amd is the low-friction choice for a HIP+Kokkos code.)
module reset
module load PrgEnv-amd
module load rocm

# The Cray PE only exists on Frontier LOGIN nodes. OLCF also has home*/dtn*
# hosts for file management, and ssh can land you there -- on those, all three
# module loads below fail and every path check downstream reports a confusing
# [FAIL]. Catch the real cause here instead.
if ! module is-avail PrgEnv-amd 2>/dev/null && ! command -v CC >/dev/null 2>&1; then
    echo "  [FAIL] Cray programming environment not found on $(hostname)."
    echo "         This looks like an OLCF home/dtn node, not a Frontier login"
    echo "         node. Reconnect with:  ssh $USER@frontier.olcf.ornl.gov"
    echo "         (hostname there is login##, not home##)"
fi

# craype-accel-amd-gfx90a makes the CC wrapper link the GTL (GPU Transport
# Layer) MPI library. REQUIRED because exchange_f() passes GPU (HIPSpace)
# buffers straight to MPI_Isend/Irecv; without GTL + GPU-aware MPI, MPI treats
# the device pointer as host memory and segfaults on the first exchange.
module load craype-accel-amd-gfx90a

# ---- 2. ROCm ----------------------------------------------------------------
# ROCM_PATH is set by `module load rocm`. Makefile.frontier links
# $(ROCM_PATH)/lib -lamdhip64, so it must be set.
export ROCM_PATH=${ROCM_PATH:?ROCM_PATH unset - did 'module load rocm' fail?}

# ---- 3. Kokkos (HIP build for gfx90a) ---------------------------------------
# There is NO Kokkos module on Frontier. Build it once with
# ./build_kokkos_frontier.sh, then point KOKKOS_HOME at the install prefix.
# It MUST be the HIP/gfx90a build compiled against the SAME ROCm you load here,
# or you get undefined hip symbols at link time / device-code mismatches.
# Makefile.frontier uses $(KOKKOS_HOME) for both -I include and -L lib, so the
# variable must be named KOKKOS_HOME (not KOKKOS_ROOT).
export KOKKOS_HOME=${KOKKOS_HOME:-$HOME/opt/kokkos-4.6.02-hip-gfx90a}
if [ -d "$KOKKOS_HOME/lib64" ]; then
    export KOKKOS_LIB=$KOKKOS_HOME/lib64
else
    export KOKKOS_LIB=$KOKKOS_HOME/lib
fi
export KOKKOS_INC=$KOKKOS_HOME/include

# ---- 4. PATH / LD_LIBRARY_PATH ----------------------------------------------
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$KOKKOS_LIB:$ROCM_PATH/lib:$LD_LIBRARY_PATH

# ---- 5. Runtime / build knobs -----------------------------------------------
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=1          # 1 rank per GCD; the GPU does the work

# GPU-aware MPI is REQUIRED for this code: exchange_f() hands GPU buffers to
# MPI_Isend/Irecv. Must be 1, and the binary must be linked with GTL (see the
# craype-accel-amd-gfx90a load above).
export MPICH_GPU_SUPPORT_ENABLED=1

# 26 neighbour channels per rank per step (6 faces + 12 edges + 8 corners) is a
# lot of concurrent unexpected messages. 'hybrid' lets libfabric fall back to
# software matching instead of dying with an "RX match mode" / dropped-message
# error at scale. Cheap insurance; costs nothing at small node counts.
export FI_CXI_RX_MATCH_MODE=hybrid
# Round-robin ranks over the 4 NICs by NUMA locality (8 ranks, 4 NICs).
export MPICH_OFI_NIC_POLICY=NUMA

# ---- 6. Self-check -----------------------------------------------------------
echo "---- env_frontier.sh self-check ----"
_ok=1
if [ -d "$ROCM_PATH" ]; then
    echo "  [ok]   ROCM_PATH  = $ROCM_PATH"
else
    echo "  [FAIL] ROCM_PATH  = $ROCM_PATH  (does not exist)"; _ok=0
fi
if ls "$KOKKOS_LIB"/libkokkoscore.so* >/dev/null 2>&1; then
    echo "  [ok]   KOKKOS_LIB = $KOKKOS_LIB"
else
    echo "  [FAIL] KOKKOS_LIB = $KOKKOS_LIB  (libkokkoscore.so* not found)"
    echo "         -> run ./build_kokkos_frontier.sh first"; _ok=0
fi
if [ -e "$KOKKOS_INC/Kokkos_Core.hpp" ]; then
    echo "  [ok]   KOKKOS_HOME= $KOKKOS_HOME"
else
    echo "  [FAIL] KOKKOS_HOME= $KOKKOS_HOME  (include/Kokkos_Core.hpp not found)"; _ok=0
fi
# Confirm the Kokkos we found is actually a HIP build. A Serial/OpenMP-only
# Kokkos compiles (lbm.hpp's static_assert catches it) but wastes a build cycle.
if [ -e "$KOKKOS_INC/KokkosCore_config.h" ]; then
    if grep -q "KOKKOS_ENABLE_HIP" "$KOKKOS_INC/KokkosCore_config.h"; then
        echo "  [ok]   Kokkos built with HIP backend"
    else
        echo "  [FAIL] Kokkos at KOKKOS_HOME is NOT a HIP build"; _ok=0
    fi
fi
if command -v CC >/dev/null 2>&1; then
    echo "  [ok]   CC         = $(command -v CC)  ($(CC --version 2>/dev/null | head -1))"
else
    echo "  [FAIL] CC not found in PATH"; _ok=0
fi
# Warn (don't fail) if an existing binary isn't linked with GTL — it will
# segfault at exchange_f under GPU-aware MPI.
if [ -e ./LBM3D_frontier ]; then
    if ldd ./LBM3D_frontier 2>/dev/null | grep -qi "gtl"; then
        echo "  [ok]   LBM3D_frontier linked with GTL (mpi_gtl_hsa)"
    else
        echo "  [warn] LBM3D_frontier is NOT linked with GTL -> will segfault in"
        echo "         exchange_f under GPU-aware MPI. Rebuild after loading"
        echo "         craype-accel-amd-gfx90a:"
        echo "         make -f Makefile.frontier clean && make -f Makefile.frontier"
    fi
fi
if [ "$_ok" -eq 1 ]; then
    echo "  --> environment looks good."
else
    echo "  --> FIX the [FAIL] item(s) above before building/running."
fi
unset _ok
echo "------------------------------------"
