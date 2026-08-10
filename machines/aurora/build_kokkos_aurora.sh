#!/bin/bash
# =============================================================================
# OPTIONAL — you almost certainly do NOT need this.
#
# Aurora ships a prebuilt Kokkos (`module load kokkos`, Serial + OpenMP + SYCL
# AOT), and env_aurora.sh uses it. Build your own only if you need a Kokkos
# version the module does not provide, or non-default backend options.
#
# This follows ALCF's "Configuring Your Own Kokkos Build on Aurora" recipe.
# Match the oneAPI version the central module was built with — `module help
# kokkos` prints it — or the AOT device images and SYCL runtime ABI diverge.
#
#   ./build_kokkos_aurora.sh
#   export KOKKOS_ROOT=<the install prefix printed at the end>
#   source env_aurora.sh        # will pick up your KOKKOS_ROOT if exported
#   make -f Makefile.aurora
# =============================================================================
set -euo pipefail

KOKKOS_VERSION=${KOKKOS_VERSION:-4.5.01}
KOKKOS_PREFIX=${KOKKOS_PREFIX:-$HOME/opt/kokkos-${KOKKOS_VERSION}-sycl-pvc}
BUILD_ROOT=${BUILD_ROOT:-$HOME/src}

module load cmake

echo "==> installing Kokkos ${KOKKOS_VERSION} to ${KOKKOS_PREFIX}"

mkdir -p "$BUILD_ROOT"
cd "$BUILD_ROOT"

if [ ! -d "kokkos-${KOKKOS_VERSION}" ]; then
    curl -L -o "kokkos-${KOKKOS_VERSION}.tar.gz" \
        "https://github.com/kokkos/kokkos/archive/refs/tags/${KOKKOS_VERSION}.tar.gz"
    tar xf "kokkos-${KOKKOS_VERSION}.tar.gz"
fi

rm -rf "kokkos-${KOKKOS_VERSION}/build-sycl"
cmake -S "kokkos-${KOKKOS_VERSION}" -B "kokkos-${KOKKOS_VERSION}/build-sycl" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$KOKKOS_PREFIX" \
    -DCMAKE_CXX_COMPILER=icpx \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS="-fp-model=precise" \
    -DBUILD_SHARED_LIBS=ON \
    -DKokkos_ENABLE_SYCL=ON \
    -DKokkos_ARCH_INTEL_PVC=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_SERIAL=ON

cmake --build "kokkos-${KOKKOS_VERSION}/build-sycl" -j 16
cmake --install "kokkos-${KOKKOS_VERSION}/build-sycl"

echo
echo "==> done. Export this BEFORE sourcing env_aurora.sh:"
echo "    export KOKKOS_ROOT=${KOKKOS_PREFIX}"
echo "    (and comment out the 'module load kokkos' line in env_aurora.sh,"
echo "     or it will overwrite KOKKOS_ROOT with the central install)"
