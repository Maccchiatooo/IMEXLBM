#ifndef _LBM_H_
#define _LBM_H_

#include <cmath>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Timer.hpp>
#include <mpi.h>
#include "profile.hpp"

#define q 27
#define dim 3
#define ghost 3

typedef Kokkos::TeamPolicy<> team_policy;
typedef Kokkos::TeamPolicy<>::member_type member_type;
typedef Kokkos::RangePolicy<> range_policy;
typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrange_policy2;
// LaunchBounds<maxThreadsPerBlock, minOccupancy>: the SECOND parameter means
// different things per backend, so it is only safe to keep the tuned Polaris
// values on CUDA.
//   CUDA (Polaris) : __launch_bounds__(maxT, minBlocksPerSM) -- a mild hint,
//                    these are the measured-good A100 values. Keep them.
//   HIP (Frontier) : __launch_bounds__(maxT, minWavesPerEU) -- 4 waves/EU x 4
//                    EUs = 16 waves/CU, capping Collision at ~64 VGPRs/thread.
//                    Its `double fl[27]` register array alone wants ~54 VGPRs,
//                    so that cap forces scratch (local-memory) spilling on
//                    MI250X and destroys performance.
//   SYCL (Aurora)  : ignored by the Kokkos SYCL backend; the PVC equivalent
//                    knob is -ze-opt-large-register-file at link time.
// Off CUDA, leave the occupancy floor unset and let the compiler choose.
// Before trusting any timing on a new machine, confirm the Collision kernel is
// not spilling:
//   Frontier: -Rpass-analysis=kernel-resource-usage, check "ScratchSize: 0"
//   Aurora:   -Xsycl-target-backend "-device pvc -options -ze-opt-print-reg-usage"
// CUDA occupancy floor (the minBlocksPerSM argument). THIS IS A REAL TUNING
// KNOB -- the original code hardcoded 4, which is very likely costing you.
//
// On A100 an SM has 65536 32-bit registers. LaunchBounds<256, N> asks for at
// least N blocks of 256 threads resident, i.e. 256*N threads/SM, so the
// compiler is capped at 65536/(256*N) registers per thread:
//     N=4  -> 1024 threads/SM ->  64 regs/thread
//     N=2  ->  512 threads/SM -> 128 regs/thread
//     N=0  -> compiler chooses freely
// Collision holds `double fl[27]` = 54 registers for that array ALONE (a
// double is two 32-bit registers), before addressing, loop state and the
// accumulators. At N=4 it cannot fit and spills to local memory, which is
// consistent with collision measuring only ~44% of peak bandwidth while the
// streaming kernel reaches ~60%+ on the same data volume.
//
// Default is now 0 (let the compiler decide). To A/B against the original:
//     make -f Makefile.polaris PROFILE=1 USER_CPPFLAGS=-DLBM_CUDA_MINBLOCKS=4
// To see actual register counts and spill bytes:
//     make -f Makefile.polaris PROFILE=1 USER_CPPFLAGS=-gpu=ptxinfo
#ifndef LBM_CUDA_MINBLOCKS
#define LBM_CUDA_MINBLOCKS 0
#endif

#if defined(KOKKOS_ENABLE_CUDA)
using launch_bounds_3 = Kokkos::LaunchBounds<256, LBM_CUDA_MINBLOCKS>;
using launch_bounds_4 = Kokkos::LaunchBounds<128, LBM_CUDA_MINBLOCKS>;
#else
using launch_bounds_3 = Kokkos::LaunchBounds<256>;
using launch_bounds_4 = Kokkos::LaunchBounds<128>;
#endif

using mdrange_policy4 = Kokkos::MDRangePolicy<Kokkos::Rank<4, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
using mdrange_policy4_lb = Kokkos::MDRangePolicy<Kokkos::Rank<4, Kokkos::Iterate::Right, Kokkos::Iterate::Right>, launch_bounds_4>;
using mdrange_policy3 = Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
using mdrange_policy3_lb = Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Right>, launch_bounds_3>;
// Portable device memory space.
//   Polaris  (NVIDIA A100)  -> Kokkos::CudaSpace
//   Frontier (AMD MI250X)   -> Kokkos::HIPSpace
// Resolving it from the default execution space means the source never names a
// vendor space, so the SAME sources build on both machines; the backend is
// chosen entirely by which Kokkos build you link against.
// The static_assert guards against accidentally linking a Serial/OpenMP-only
// Kokkos, which would silently put every buffer in HostSpace and hand host
// pointers to the GPU-aware MPI calls in exchange_f().
//
// The machine Makefiles leave it armed, so a forgotten `module load` fails the
// build instead of producing a binary that runs but is two orders of magnitude
// slower than it should be. The CMake build (CPU-only, for local development
// and CI) defines LBM_ALLOW_HOST_BUILD to disarm it: a host build is correct --
// host pointers into a plain MPI is a valid path -- it is just never what you
// want on a GPU machine.
using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;
#ifndef LBM_ALLOW_HOST_BUILD
static_assert(!std::is_same<DeviceSpace, Kokkos::HostSpace>::value,
              "Kokkos must be built with a GPU backend (CUDA on Polaris, HIP on "
              "Frontier, SYCL on Aurora); a host-only build breaks the GPU-aware "
              "MPI path. Define LBM_ALLOW_HOST_BUILD to build for CPU anyway.");
#endif

// lbm.hpp — View 别名加 LayoutRight(最右维 k 变 stride-1)
using buffer_f      = Kokkos::View<double ****, Kokkos::LayoutRight, DeviceSpace>;
using buffer_u      = Kokkos::View<double ***,  Kokkos::LayoutRight, DeviceSpace>;
using buffer_div    = Kokkos::View<double ***,  Kokkos::LayoutRight, DeviceSpace>;
using buffer_pack_f = Kokkos::View<double ****, Kokkos::LayoutRight, DeviceSpace>;
using buffer_pack_u = Kokkos::View<double ***,  Kokkos::LayoutRight, DeviceSpace>;

struct LBM
{

    MPI_Comm comm;
    int nranks;

    int rx, ry, rz;
    // rank
    int me;
    // axis for each rank
    int px, py, pz;

    int glx, gly, glz;
    // include ghost nodes
    int lx, ly, lz;
    // local start, local end, local length
    int l_s[3], l_e[3], l_l[3];

    // local axis
    int x_lo = 0, x_hi = 0, y_lo = 0, y_hi = 0, z_lo = 0, z_hi = 0;
    double rho0, mu, cs2, tau0, u0;
    int face_recv[6], face_send[6];
    // 12 edges
    int edge_recv[12], edge_send[12];
    // 8 points
    int point_recv[8], point_send[8];

    // NOTE: the array-of-buffers members below are reserved for the future
    // fully loop-driven pack/unpack refactor; currently unused.
    buffer_pack_f f_face_send[6], f_face_recv[6];
    buffer_pack_f f_edge_send[12], f_edge_recv[12];
    buffer_pack_f f_corner_send[8], f_corner_recv[8];

    buffer_pack_f m_left, m_right, m_down, m_up, m_front, m_back;
    buffer_pack_f m_leftout, m_rightout, m_downout, m_upout, m_frontout, m_backout;
    buffer_pack_f m_leftup, m_rightup, m_leftdown, m_rightdown, m_frontup, m_backup, m_frontdown, m_backdown, m_frontleft, m_backleft, m_frontright, m_backright;
    buffer_pack_f m_leftupout, m_rightupout, m_leftdownout, m_rightdownout, m_frontupout, m_backupout, m_frontdownout, m_backdownout, m_frontleftout, m_backleftout, m_frontrightout, m_backrightout;
    // 8 points
    buffer_pack_f m_frontleftup, m_frontrightup, m_frontleftdown, m_frontrightdown, m_backleftup, m_backleftdown, m_backrightup, m_backrightdown;
    buffer_pack_f m_frontleftupout, m_frontrightupout, m_frontleftdownout, m_frontrightdownout, m_backleftupout, m_backleftdownout, m_backrightupout, m_backrightdownout;
    Kokkos::View<int **, DeviceSpace> d_dirset; // OPT-T5 pruned-direction table

    buffer_pack_u u_left, u_right, u_down, u_up, u_front, u_back;
    buffer_pack_u u_leftout, u_rightout, u_downout, u_upout, u_frontout, u_backout;

    buffer_pack_u u_leftup, u_rightup, u_leftdown, u_rightdown, u_backleft, u_backright, u_frontleft, u_frontright, u_backdown, u_backup, u_frontdown, u_frontup;
    buffer_pack_u u_leftupout, u_rightupout, u_leftdownout, u_rightdownout, u_backleftout, u_backrightout, u_frontleftout, u_frontrightout, u_backdownout, u_backupout, u_frontdownout, u_frontupout;

    buffer_pack_u u_frontleftdown, u_frontrightdown, u_frontleftup, u_frontrightup, u_backleftdown, u_backrightdown, u_backleftup, u_backrightup;
    buffer_pack_u u_frontleftdownout, u_frontrightdownout, u_frontleftupout, u_frontrightupout, u_backleftdownout, u_backrightdownout, u_backleftupout, u_backrightupout;

    // particle distribution eqution
    buffer_f f, ft, fb;
    // macro scopic equation
    buffer_u ua, va, wa, rho, p;
    // usr define
    Kokkos::View<int ***, DeviceSpace> usr, ran;
    // bounce back notation
    Kokkos::View<int *, DeviceSpace> bb;
    // weight function
    Kokkos::View<double *, DeviceSpace> t;
    // discrete velocity
    Kokkos::View<int **, DeviceSpace> e;

    // OPT-H: cs2 was declared but never assigned anywhere — Initialize()
    // used it uninitialized in the Taylor-Green pressure field.
    LBM(MPI_Comm comm_, int sx, int sy, int sz, double &tau, double &rho0, double &u0)
        : comm(comm_), glx(sx), gly(sy), glz(sz), tau0(tau), rho0(rho0), u0(u0)
    {
        cs2 = 1.0 / 3.0;
    };

    void setup_Cartesian();
    void setup_Local();
    void setup_MPI();
    void setup_f();

    void Initialize();
    void Collision();
    void Streaming();
    void Boundary();
    void ComputeMacroscopic();
    void Update();
    void Update1();
    void MPIoutput(int n);
    void Output(int n);

    // Globally reduced conserved quantities; see the definition in lbm.cpp for
    // why mass and momentum are exact invariants of the discrete dynamics.
    void Conserved(double &mass, double &mom_x, double &mom_y, double &mom_z,
                   double &ke);

    void pack_f(buffer_f ff);
    void unpack_f(buffer_f ff);

    void exchange_f();

    void passf(buffer_f ff);
};
#endif
