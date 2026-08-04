#!/bin/bash
# =============================================================================
# Build Kokkos 4.6.02 with the HIP backend for Frontier (MI250X / gfx90a).
#
# Polaris had a pre-built Kokkos module (/soft/libraries/kokkos/...); OLCF does
# NOT ship a Kokkos module on Frontier, so you build it once yourself.
#
#   Run from a LOGIN node (compiles fine there, ~10 min):
#       ./build_kokkos_frontier.sh
#   Then:
#       source env_frontier.sh
#       make -f Makefile.frontier
#
# Installs to $KOKKOS_PREFIX (default below). Put it somewhere on a filesystem
# the compute nodes can read — home ($HOME) and project space both work; do NOT
# use /tmp.
# =============================================================================
set -euo pipefail

KOKKOS_VERSION=${KOKKOS_VERSION:-4.6.02}
KOKKOS_PREFIX=${KOKKOS_PREFIX:-$HOME/opt/kokkos-${KOKKOS_VERSION}-hip-gfx90a}
BUILD_ROOT=${BUILD_ROOT:-$HOME/src}

# ---- modules (must match what env_frontier.sh loads at build/run time) -------
module reset
module load PrgEnv-amd
module load rocm
module load craype-accel-amd-gfx90a
module load cmake

echo "==> ROCM_PATH = ${ROCM_PATH:-<unset>}"
echo "==> installing Kokkos ${KOKKOS_VERSION} to ${KOKKOS_PREFIX}"

mkdir -p "$BUILD_ROOT"
cd "$BUILD_ROOT"

if [ ! -d "kokkos-${KOKKOS_VERSION}" ]; then
    curl -L -o "kokkos-${KOKKOS_VERSION}.tar.gz" \
        "https://github.com/kokkos/kokkos/archive/refs/tags/${KOKKOS_VERSION}.tar.gz"
    tar xf "kokkos-${KOKKOS_VERSION}.tar.gz"
fi

rm -rf "kokkos-${KOKKOS_VERSION}/build-hip"
cmake -S "kokkos-${KOKKOS_VERSION}" -B "kokkos-${KOKKOS_VERSION}/build-hip" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$KOKKOS_PREFIX" \
    -DCMAKE_CXX_COMPILER=hipcc \
    -DCMAKE_CXX_STANDARD=17 \
    -DBUILD_SHARED_LIBS=ON \
    -DKokkos_ENABLE_HIP=ON \
    -DKokkos_ARCH_VEGA90A=ON \
    -DKokkos_ARCH_ZEN3=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=OFF

cmake --build "kokkos-${KOKKOS_VERSION}/build-hip" -j 16
cmake --install "kokkos-${KOKKOS_VERSION}/build-hip"

echo
echo "==> done. Add this to env_frontier.sh (or export before sourcing it):"
echo "    export KOKKOS_HOME=${KOKKOS_PREFIX}"
