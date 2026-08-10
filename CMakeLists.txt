cmake_minimum_required(VERSION 3.23)
project(IMEXLBM VERSION 1.0.2 LANGUAGES CXX
        DESCRIPTION "A portable lattice-Boltzmann solver for heterogeneous platforms")

# ---------------------------------------------------------------------------
# SCOPE: this build targets CPU backends (Kokkos Serial / OpenMP) for local
# development, testing and CI. The GPU builds for Aurora, Polaris and Frontier
# are owned by the hand-written Makefiles in machines/<name>/ -- they encode
# machine-specific flags (SYCL AOT, -gpu=cc80,nordc, -munsafe-fp-atomics) that
# are deliberately NOT duplicated here. Do not add GPU flags to this file.
# ---------------------------------------------------------------------------

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

# Mirrors $(wildcard *.cpp) in the machine Makefiles: every .cpp under src/ is
# part of the solver. CONFIGURE_DEPENDS re-globs when the directory changes, so
# a newly added source is never silently left out of the build.
file(GLOB IMEXLBM_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(imexlbm ${IMEXLBM_SOURCES})
target_include_directories(imexlbm PRIVATE src)
target_link_libraries(imexlbm PRIVATE Kokkos::kokkos MPI::MPI_CXX)

# A floor, not a ceiling -- Kokkos raises this if it was built against a newer
# standard (Kokkos 5.x requires C++20).
target_compile_features(imexlbm PRIVATE cxx_std_17)

# The machine Makefiles keep the GPU-backend assert in lbm.hpp armed; this
# CPU-only build deliberately disarms it. See the comment at that assert.
target_compile_definitions(imexlbm PRIVATE LBM_ALLOW_HOST_BUILD)

# Mirrors SCALING=1 / PROFILE=1 in the machine Makefiles.
option(IMEXLBM_SCALING "Compile out the per-step collision timing" OFF)
option(IMEXLBM_PROFILE "Enable the per-phase profiler (implies SCALING)" OFF)
if(IMEXLBM_PROFILE)
  target_compile_definitions(imexlbm PRIVATE LBM_PROFILE LBM_NO_PERSTEP_TIMING)
elseif(IMEXLBM_SCALING)
  target_compile_definitions(imexlbm PRIVATE LBM_NO_PERSTEP_TIMING)
endif()
