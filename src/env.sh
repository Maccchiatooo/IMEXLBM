#!/bin/bash
# =============================================================================
# IMEXLBM / LBM3D_polaris  environment for Polaris (ALCF)
# Usage:  source env.sh
# Goal:   one source sets EVERYTHING (compiler env + CUDA + Kokkos libs + PATH)
#         so binaries run on any compute node without per-node library fixups.
# =============================================================================

# ---- 1. Programming environment ---------------------------------------------
# Makefile.polaris uses `CC` -> nvc++ with `-cuda -gpu=cc80` flags, which are
# NVIDIA-compiler-specific. So we MUST be in PrgEnv-nvidia, NOT PrgEnv-gnu.
module use /soft/modulefiles
module load PrgEnv-nvidia
# We need the GTL (GPU Transport Layer) MPI library on the link line. REQUIRED
# because exchange_f() passes GPU device buffers straight to MPI_Irecv; without
# GTL + GPU-aware MPI, MPI treats the device pointer as host memory and
# segfaults in _cray_mpi_memcpy_rome.
#
# The usual way is `module load craype-accel-nvidia80`, but on Polaris that
# often refuses to load: it requires a module literally NAMED cudatoolkit or
# cuda (Lmod's atleast("cudatoolkit","11.0")), while ALCF ships
# "cudatoolkit-standalone" — a different name, so the test fails no matter
# which version you load.
#
# So: try the modules, but do not depend on them. We resolve the GTL flags
# ourselves below and put them on the link line explicitly, which is all
# craype-accel-nvidia80 would have done. This is the same approach
# Makefile.frontier uses for mpi_gtl_hsa.
module load cudatoolkit-standalone 2>/dev/null
module load craype-accel-nvidia80 2>/dev/null

# Resolve GTL flags. If craype-accel-nvidia80 DID load, it exports these; use
# them. Otherwise fall back to the cray-mpich install tree.
# Makefile.polaris puts $(GTL_FLAGS) on LDLIBS.
if [ -n "${PE_MPICH_GTL_DIR_nvidia80:-}" ]; then
    export GTL_FLAGS="${PE_MPICH_GTL_DIR_nvidia80} ${PE_MPICH_GTL_LIBS_nvidia80}"
elif [ -n "${CRAY_MPICH_ROOTDIR:-}" ] && [ -d "${CRAY_MPICH_ROOTDIR}/gtl/lib" ]; then
    export GTL_FLAGS="-L${CRAY_MPICH_ROOTDIR}/gtl/lib -lmpi_gtl_cuda"
else
    export GTL_FLAGS=""
fi

# ---- 2. CUDA toolkit ---------------------------------------------------------
# IMPORTANT: point CUDA_HOME at the SAME cuda that the Kokkos build below was
# compiled against, or cudart versions can mismatch.
# Verify the real path with:  ls -l <KOKKOS_DIR>/../  and match the cudatoolkit.
export CUDA_HOME=/soft/compilers/cudatoolkit/cuda-12.9.0   # <-- confirm this exists

# ---- 3. Kokkos (nvidia build, matching the -I used in Makefile.polaris) ------
# Your compile line used the PrgEnv-nvidia Kokkos include tree, so link/runtime
# must use the SAME tree's lib64. Confirm libkokkoscore.so.4.6 lives here:
#   ls $KOKKOS_LIB/libkokkoscore.so.4.6
# NOTE: Makefile.polaris references $(KOKKOS_HOME) for both -I include and
# -L lib64, so the variable MUST be named KOKKOS_HOME (not KOKKOS_ROOT), or the
# build expands to "-I/include" and fails to find Kokkos_Core.hpp.
export KOKKOS_HOME=/soft/libraries/kokkos/kokkos-4.6.02/shared/PrgEnv-nvidia/8.6.0/nvidia/default/cuda/cudatoolkit/default
export KOKKOS_LIB=$KOKKOS_HOME/lib64
export KOKKOS_INC=$KOKKOS_HOME/include

# ---- 4. PATH / LD_LIBRARY_PATH ----------------------------------------------
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$KOKKOS_LIB:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ---- 5. Runtime / build knobs -----------------------------------------------
export NVCC_WRAPPER_DEFAULT_COMPILER=CC
export CRAY_ACCEL_TARGET=nvidia80
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
# WITHOUT this, Kokkos starts 64 OpenMP threads PER RANK ("Detected: 64 cores,
# Requested: 64 threads per process"). At 4 ranks/node that is 256 threads on
# 32 physical cores. OpenMP threads spin-wait, and Cray MPICH needs CPU to make
# progress on transfers -- starving it drops GPU-to-GPU MPI from ~100 GB/s
# (measured with mpi_gpu_bw) to ~6 GB/s in the solver.
# All compute is on the GPU here, so one host thread per rank is correct.
export OMP_NUM_THREADS=1

# GPU-aware MPI is REQUIRED for this code: exchange_f() hands GPU buffers to
# MPI_Irecv. Must be 1, and the binary must be linked with GTL (see the
# craype-accel-nvidia80 load above). Setting this to 0 causes a SIGSEGV inside
# Cray MPI's local-copy path when it memcpy's a device pointer as host memory.
export MPICH_GPU_SUPPORT_ENABLED=1

# ---- 6. Self-check -----------------------------------------------------------
# Prints a clear PASS/FAIL so you catch a wrong path immediately, instead of
# discovering it as a "cannot open shared object file" mid-profiling.
echo "---- env.sh self-check ----"
_ok=1
if [ -d "$CUDA_HOME" ]; then
    echo "  [ok]   CUDA_HOME  = $CUDA_HOME"
else
    echo "  [FAIL] CUDA_HOME  = $CUDA_HOME  (does not exist)"; _ok=0
fi
if [ -e "$KOKKOS_LIB/libkokkoscore.so.4.6" ]; then
    echo "  [ok]   KOKKOS_LIB = $KOKKOS_LIB"
else
    echo "  [FAIL] KOKKOS_LIB = $KOKKOS_LIB  (libkokkoscore.so.4.6 not found)"; _ok=0
fi
if [ -e "$KOKKOS_INC/Kokkos_Core.hpp" ]; then
    echo "  [ok]   KOKKOS_HOME= $KOKKOS_HOME"
else
    echo "  [FAIL] KOKKOS_HOME= $KOKKOS_HOME  (include/Kokkos_Core.hpp not found)"; _ok=0
fi
if command -v CC >/dev/null 2>&1; then
    echo "  [ok]   CC         = $(command -v CC)  ($(CC --version 2>&1 | head -1))"
else
    echo "  [FAIL] CC not found in PATH"; _ok=0
fi
# THE check this file previously lacked. Every path above can be correct while
# the GTL is still missing from the link line, and then the build succeeds and
# dies in exchange_f at runtime. Check for the actual library, not the module:
# the module is one way to get it, not the thing we need.
if [ -n "${GTL_FLAGS:-}" ]; then
    _gtldir=${GTL_FLAGS#-L}; _gtldir=${_gtldir%% *}
    if ls "$_gtldir"/libmpi_gtl_cuda.* >/dev/null 2>&1; then
        echo "  [ok]   GTL        = $GTL_FLAGS"
    else
        echo "  [FAIL] GTL flags set but libmpi_gtl_cuda.* not found in $_gtldir"
        _ok=0
    fi
    unset _gtldir
else
    echo "  [FAIL] cannot locate the CUDA GTL library -> exchange_f will SIGSEGV"
    echo "         under GPU-aware MPI. Find it with:"
    echo "           ls \$CRAY_MPICH_ROOTDIR/gtl/lib"
    echo "           find /opt/cray -name 'libmpi_gtl_cuda*' 2>/dev/null"
    echo "         then set GTL_FLAGS=\"-L<dir> -lmpi_gtl_cuda\" in env.sh."
    _ok=0
fi
# Warn (don't fail) if an existing binary isn't linked with GTL — it will
# segfault at exchange_f under GPU-aware MPI. Rebuild after loading
# craype-accel-nvidia80 to fix.
if [ -e ./LBM3D_polaris ]; then
    if ldd ./LBM3D_polaris 2>/dev/null | grep -qi gtl; then
        echo "  [ok]   LBM3D_polaris linked with GTL"
    else
        echo "  [warn] LBM3D_polaris is NOT linked with GTL -> will segfault in"
        echo "         exchange_f under GPU-aware MPI. Rebuild: make -f Makefile.polaris clean && make -f Makefile.polaris"
    fi
fi
if [ "$_ok" -eq 1 ]; then
    echo "  --> environment looks good."
else
    echo "  --> FIX the [FAIL] path(s) above before building/running."
fi
unset _ok
echo "---------------------------"
