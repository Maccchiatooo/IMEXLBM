#ifndef _LBM_PROFILE_H_
#define _LBM_PROFILE_H_

// =============================================================================
// Per-phase timing instrumentation. Compiled out entirely unless -DLBM_PROFILE
// (Makefile.aurora PROFILE=1), so the normal and scaling builds pay nothing.
//
// WHAT IT MEASURES, AND WHY THE SPLIT LOOKS ODD
// ---------------------------------------------
// Kokkos deep_copy/parallel_for on an execution-space instance are ASYNC: the
// host enqueues work and returns immediately. Timing the enqueue tells you
// almost nothing about GPU cost.
//
// So each async region is split into two numbers:
//   * "enqueue"  — host-side cost of issuing the operations. Small, but NOT
//                  negligible here: pack_f + unpack_f issue 52 strided copies
//                  per step, and per-launch overhead is real.
//   * "fence"    — the blocking wait, which is where the GPU work is actually
//                  charged.
//
// Crucially this does NOT insert new fences into the async pipeline. pack_f
// deliberately enqueues 26 copies and fences ONCE (its OPT-T3 optimisation);
// fencing between the face/edge/corner groups to "measure them separately"
// would destroy that optimisation and report a structure that no longer
// matches the code being measured. The groups therefore show enqueue cost
// only, and all 26 copies' GPU time lands in "pack: fence".
//
// unpack_f is the one place this rule is knowingly bent. It normally has NO
// fence (stream1 is stream-ordered behind it), which would hide unpack's GPU
// time inside "stream: fence" together with stream1. Since telling those two
// apart is the whole point, a PROFILE-ONLY fence is added at the end of
// unpack_f so each gets its own row. It costs one extra synchronisation per
// step and is absent from every non-profile build.
//
// The other added fence is after Collision, otherwise unattributable.
// Those two fences are why PROFILE builds are a separate binary from SCALING
// builds — do not take throughput numbers from a profile build.
// =============================================================================

#include <mpi.h>

#ifdef LBM_PROFILE
#include <Kokkos_Core.hpp>

namespace lbmprof
{
enum Phase
{
    P_STEP = 0,      // whole time step (the denominator for %)
    P_COLLISION,     // fused collision kernel, fenced
    P_PACK_FACE,     // enqueue 6 face copies
    P_PACK_EDGE,     // enqueue 12 edge copies
    P_PACK_CORNER,   // enqueue 8 corner copies
    P_PACK_FENCE,    // GPU execution of all 26 pack copies
    P_MPI_POST,      // 26 x (Isend + Irecv)
    P_MPI_WAIT,      // Waitall(recv) + Waitall(send)
    P_UNPACK_FACE,   // enqueue 6 face copies
    P_UNPACK_EDGE,   // enqueue 12 edge copies
    P_UNPACK_CORNER, // enqueue 8 corner copies
    P_UNPACK_FENCE,  // GPU exec of the 26 unpack copies (PROFILE-ONLY fence)
    P_STREAM_ENQ,    // enqueue stream1 kernel
    P_STREAM_FENCE,  // GPU exec of stream1
    P_MACRO,         // ComputeMacroscopic (only when output is enabled)
    NPHASE
};

// Peak device memory bandwidth PER RANK (one GPU tile / GCD), GB/s, used for
// the "% peak BW" column. Conveniently ~1.6 TB/s on all three targets:
// PVC Max 1550 tile 1600, MI250X GCD 1638, A100-40 1555. Override at compile
// time with -DLBM_PROF_PEAK_GBPS=... if you want to be exact.
#ifndef LBM_PROF_PEAK_GBPS
#define LBM_PROF_PEAK_GBPS 1600.0
#endif

extern const char *const names[NPHASE];
extern double acc[NPHASE];   // sum of per-step times
extern double smin[NPHASE];  // fastest step seen
extern double smax[NPHASE];  // slowest step seen
extern double mark[NPHASE];
extern long calls[NPHASE];

// Workload registration, so the report can turn times into GFLOP/s and GB/s.
// set_cells: local INTERIOR cells owned by this rank, and q (=27).
// set_halo_bytes: total bytes across the 26 send / 26 recv halo buffers.
void set_cells(long cells, int q);
void set_halo_bytes(double send_bytes, double recv_bytes);

inline void begin(int p) { mark[p] = MPI_Wtime(); }
inline void end(int p)
{
    const double dt = MPI_Wtime() - mark[p];
    acc[p] += dt;
    // Step-to-step spread is tracked as a RANGE, not a variance. The textbook
    // E[X^2]-E[X]^2 shortcut cancels catastrophically here: per-step times are
    // ~2e-3 s, so E[X^2] and E[X]^2 agree to ~6 significant figures and the
    // difference is noise. It silently reported 0.0% jitter for every phase on
    // real data. min/max cannot fail that way.
    if (calls[p] == 0 || dt < smin[p]) smin[p] = dt;
    if (calls[p] == 0 || dt > smax[p]) smax[p] = dt;
    ++calls[p];
}

void reset();
void report(MPI_Comm comm, int nsteps);
} // namespace lbmprof

#define LBM_PROF_BEGIN(p) ::lbmprof::begin(::lbmprof::p)
#define LBM_PROF_END(p) ::lbmprof::end(::lbmprof::p)
#define LBM_PROF_RESET() ::lbmprof::reset()
#define LBM_PROF_REPORT(c, n) ::lbmprof::report((c), (n))
#define LBM_PROF_FENCE() Kokkos::fence()
#define LBM_PROF_SET_CELLS(c, qq) ::lbmprof::set_cells((c), (qq))
#define LBM_PROF_SET_HALO(s, r) ::lbmprof::set_halo_bytes((s), (r))

#else

#define LBM_PROF_BEGIN(p) ((void)0)
#define LBM_PROF_END(p) ((void)0)
#define LBM_PROF_RESET() ((void)0)
#define LBM_PROF_REPORT(c, n) ((void)0)
#define LBM_PROF_FENCE() ((void)0)
#define LBM_PROF_SET_CELLS(c, qq) ((void)0)
#define LBM_PROF_SET_HALO(s, r) ((void)0)

#endif // LBM_PROFILE
#endif // _LBM_PROFILE_H_
