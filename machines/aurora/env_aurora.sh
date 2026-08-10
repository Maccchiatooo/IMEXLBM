#!/bin/bash
# =============================================================================
# IMEXLBM / LBM3D_aurora  environment for Aurora (ALCF)
# Usage:  source env_aurora.sh
# Goal:   one source sets EVERYTHING (oneAPI + Kokkos + runtime knobs) so
#         binaries run on any compute node without per-node library fixups.
#
# Aurora vs Polaris, in one table:
#   GPU              4x A100 (cc80, CUDA)     6x PVC Max 1550 = 12 TILES (SYCL)
#   Ranks per node   4                        12 (one per tile)
#   PrgEnv           PrgEnv-nvidia (nvc++)    oneAPI (mpic++ -cxx=icpx)
#   Kokkos           /soft module, KOKKOS_HOME  module load kokkos, KOKKOS_ROOT
#   Kokkos backend   CUDA                     SYCL, ahead-of-time (AOT)
#   Scheduler        PBS (qsub)               PBS (qsub) -- same family
#   Launcher         mpiexec + COBALT/PBS     mpiexec (Cray PALS)
#   GPU pinning      CUDA_VISIBLE_DEVICES     ZE_AFFINITY_MASK
#   GPU-aware MPI    MPICH_GPU_SUPPORT_ENABLED  MPIR_CVAR_ENABLE_GPU (default 1)
# =============================================================================

# ---- 1. Programming environment ---------------------------------------------
# The default oneAPI module is the one the central Kokkos was built against.
# Do NOT swap oneAPI versions without also rebuilding Kokkos: the AOT device
# images and the SYCL runtime ABI must match. `module help kokkos` prints the
# oneAPI version it was built with.
module load cmake

# ---- 2. Kokkos (prebuilt, SYCL AOT for PVC) ---------------------------------
# Unlike Frontier (no module at all), Aurora ships a prebuilt Kokkos with
# Serial + OpenMP + SYCL backends. It sets KOKKOS_ROOT and prepends CPATH,
# LIBRARY_PATH and LD_LIBRARY_PATH, so no manual path surgery is needed.
# Makefile.aurora reads $(KOKKOS_ROOT) for both -I include and -L lib64.
module load kokkos

# ---- 3. Runtime / build knobs -----------------------------------------------
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=1          # 1 rank per tile; the GPU does the work

# GPU-aware MPI. Aurora MPICH defaults this to 1, but set it explicitly: this
# code REQUIRES it. exchange_f() passes SYCL device-USM pointers straight into
# MPI_Isend/Irecv; with MPIR_CVAR_ENABLE_GPU=0 the MPI library treats them as
# host pointers and the first exchange faults.
export MPIR_CVAR_ENABLE_GPU=1

# COMPOSITE hierarchy makes ZE_AFFINITY_MASK accept the "<gpu>.<tile>" form
# that set_affinity_gpu_aurora.sh writes. Under FLAT the 12 tiles are flat
# device ids 0-11 instead and that mask syntax is invalid, so pin the mode
# rather than inheriting whatever the system default happens to be.
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE

# ---- 4. Self-check -----------------------------------------------------------
echo "---- env_aurora.sh self-check ----"
_ok=1
if [ -n "${KOKKOS_ROOT:-}" ] && [ -d "$KOKKOS_ROOT" ]; then
    echo "  [ok]   KOKKOS_ROOT= $KOKKOS_ROOT"
else
    echo "  [FAIL] KOKKOS_ROOT unset or missing (did 'module load kokkos' fail?)"; _ok=0
fi
if [ -e "$KOKKOS_ROOT/include/Kokkos_Core.hpp" ]; then
    echo "  [ok]   Kokkos headers found"
else
    echo "  [FAIL] $KOKKOS_ROOT/include/Kokkos_Core.hpp not found"; _ok=0
fi
if ls "$KOKKOS_ROOT"/lib64/libkokkoscore.* >/dev/null 2>&1; then
    echo "  [ok]   Kokkos libs found in $KOKKOS_ROOT/lib64"
else
    echo "  [FAIL] libkokkoscore.* not found in $KOKKOS_ROOT/lib64"; _ok=0
fi
# Confirm the module is the SYCL build. A Serial/OpenMP-only Kokkos compiles
# far enough to trip the static_assert in lbm.hpp, but catching it here is
# faster than waiting for the build.
if [ -e "$KOKKOS_ROOT/include/KokkosCore_config.h" ]; then
    if grep -q "KOKKOS_ENABLE_SYCL" "$KOKKOS_ROOT/include/KokkosCore_config.h"; then
        echo "  [ok]   Kokkos built with SYCL backend"
    else
        echo "  [FAIL] Kokkos at KOKKOS_ROOT is NOT a SYCL build"; _ok=0
    fi
fi
if command -v icpx >/dev/null 2>&1; then
    echo "  [ok]   icpx       = $(command -v icpx)  ($(icpx --version 2>/dev/null | head -1))"
else
    echo "  [FAIL] icpx not found in PATH"; _ok=0
fi
if command -v mpic++ >/dev/null 2>&1; then
    echo "  [ok]   mpic++     = $(command -v mpic++)"
else
    echo "  [FAIL] mpic++ not found in PATH"; _ok=0
fi
# Compute nodes only: confirm the runtime actually sees 6 GPUs / 12 tiles.
if command -v sycl-ls >/dev/null 2>&1; then
    _ngpu=$(sycl-ls 2>/dev/null | grep -ci "level_zero.*gpu" || true)
    echo "  [info] sycl-ls reports $_ngpu Level-Zero GPU device(s)"
    echo "         (expect 6 on a compute node with COMPOSITE, 1 under a pinned mask)"
    unset _ngpu
fi
if [ "$_ok" -eq 1 ]; then
    echo "  --> environment looks good."
else
    echo "  --> FIX the [FAIL] item(s) above before building/running."
fi
unset _ok
echo "----------------------------------"
