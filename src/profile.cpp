#include "profile.hpp"

#ifdef LBM_PROFILE
#include <cstdio>

namespace lbmprof
{
const char *const names[NPHASE] = {
    "STEP TOTAL",
    "  collision (kernel)",
    "  pack: enqueue faces",
    "  pack: enqueue edges",
    "  pack: enqueue corners",
    "  pack: fence (GPU copies)",
    "  mpi: post 26 isend/irecv",
    "  mpi: waitall",
    "  unpack: enqueue faces",
    "  unpack: enqueue edges",
    "  unpack: enqueue corners",
    "  unpack: fence (GPU copies)",
    "  stream: enqueue kernel",
    "  stream: fence (stream1 GPU)",
    "  computeMacroscopic",
};

double acc[NPHASE] = {0.0};
double smin[NPHASE] = {0.0};
double smax[NPHASE] = {0.0};
double mark[NPHASE] = {0.0};
long calls[NPHASE] = {0};

// ---- registered workload (per rank) -----------------------------------------
static long g_cells = 0;
static int g_q = 27;
static double g_halo_send = 0.0;
static double g_halo_recv = 0.0;

static const double FLOP_COLLISION = 594.0;
static const double FLOP_MOMENTS = 189.0;

void set_cells(long cells, int q) { g_cells = cells; g_q = q; }
void set_halo_bytes(double s, double r) { g_halo_send = s; g_halo_recv = r; }

void reset()
{
    for (int i = 0; i < NPHASE; ++i)
    {
        acc[i] = 0.0;
        smin[i] = 0.0;
        smax[i] = 0.0;
        calls[i] = 0;
    }
}

// ---- groups -----------------------------------------------------------------
// Bandwidth is reported PER GROUP, never per sub-row. Kokkos deep_copy on an
// execution-space instance is async, so the work can land in "enqueue" or in
// "fence" depending on buffer size, backend and driver mood -- observed to flip
// between runs of this very code. Dividing a group's bytes by one half of that
// split produces garbage (it once read 914% of peak). The group total is stable
// because it does not care where inside the group the time was spent.
enum { G_COLLISION = 0, G_PACK, G_MPI, G_UNPACK, G_STREAM, G_MACRO, NGROUP };

static const char *const gnames[NGROUP] = {
    "collision", "pack (enq+fence)", "mpi (post+wait)",
    "unpack (enq+fence)", "stream (enq+fence)", "computeMacroscopic",
};
static const int gmembers[NGROUP][5] = {
    {P_COLLISION, -1, -1, -1, -1},
    {P_PACK_FACE, P_PACK_EDGE, P_PACK_CORNER, P_PACK_FENCE, -1},
    {P_MPI_POST, P_MPI_WAIT, -1, -1, -1},
    {P_UNPACK_FACE, P_UNPACK_EDGE, P_UNPACK_CORNER, P_UNPACK_FENCE, -1},
    {P_STREAM_ENQ, P_STREAM_FENCE, -1, -1, -1},
    {P_MACRO, -1, -1, -1, -1},
};

static double group_bytes(int g)
{
    const double fb = (double)g_cells * g_q * sizeof(double);
    switch (g)
    {
    case G_COLLISION: return 2.0 * fb;                 // read f, write f
    case G_PACK:      return 2.0 * g_halo_send;        // read f, write bufs
    case G_MPI:       return g_halo_send + g_halo_recv;
    case G_UNPACK:    return 2.0 * g_halo_recv;        // read bufs, write f
    case G_STREAM:    return 2.0 * fb;                 // read f, write ft
    case G_MACRO:     return fb + 4.0 * g_cells * sizeof(double);
    default:          return 0.0;
    }
}

static double group_flops(int g)
{
    if (g == G_COLLISION) return FLOP_COLLISION * (double)g_cells;
    if (g == G_MACRO)     return FLOP_MOMENTS * (double)g_cells;
    return 0.0;
}

void report(MPI_Comm comm, int nsteps)
{
    int me = 0, nranks = 1;
    MPI_Comm_rank(comm, &me);
    MPI_Comm_size(comm, &nranks);

    if (nsteps <= 0)
    {
        if (me == 0) printf("\n[profile] nothing to report (nsteps <= 0)\n");
        return;
    }

    // Per-rank mean ms/step, plus the fastest and slowest single step seen.
    double local[NPHASE], lo[NPHASE], hi[NPHASE];
    for (int i = 0; i < NPHASE; ++i)
    {
        const double n = (calls[i] > 0) ? (double)calls[i] : 1.0;
        local[i] = (acc[i] / n) * 1.0e3;
        lo[i] = smin[i] * 1.0e3;
        hi[i] = smax[i] * 1.0e3;
    }

    double mn[NPHASE], mx[NPHASE], sm[NPHASE], glo[NPHASE], ghi[NPHASE];
    MPI_Reduce(local, mn, NPHASE, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(local, mx, NPHASE, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(local, sm, NPHASE, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(lo, glo, NPHASE, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(hi, ghi, NPHASE, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (me != 0) return;

    const double step = sm[P_STEP] / nranks;

    printf("\n=== per-phase profile: %d ranks, %d timed steps ===\n", nranks, nsteps);
    printf("mean ms = across-rank mean of each rank's per-step mean\n");
    printf("spread%% = (slowest step - fastest step) / mean, over all steps and\n");
    printf("          ranks. >20%% => the mean is not a stable number.\n");
    printf("imbal   = max/min ACROSS RANKS. >>1 means load imbalance.\n\n");

    printf("%-36s %9s %8s %8s %7s %6s %7s\n", "phase", "mean ms",
           "min ms", "max ms", "spread%", "imbal", "% step");
    printf("%-36s %9s %8s %8s %7s %6s %7s\n", "------------------------------------",
           "---------", "--------", "--------", "-------", "------", "-------");

    double attributed = 0.0;
    for (int i = 0; i < NPHASE; ++i)
    {
        const double mean = sm[i] / nranks;
        if (i != P_STEP)
        {
            if (calls[i] == 0) continue;
            attributed += mean;
        }
        printf("%-36s %9.4f %8.4f %8.4f %7.1f %6.2f %7.1f\n", names[i], mean,
               glo[i], ghi[i],
               (mean > 0.0) ? 100.0 * (ghi[i] - glo[i]) / mean : 0.0,
               (mn[i] > 0.0) ? mx[i] / mn[i] : 0.0,
               (step > 0.0) ? 100.0 * mean / step : 0.0);
    }
    printf("%-36s %9.4f %8s %8s %7s %6s %7.1f\n", "  (unattributed)",
           step - attributed, "", "", "", "",
           (step > 0.0) ? 100.0 * (step - attributed) / step : 0.0);

    // ---- group summary: the only place rates are reported --------------------
    printf("\n%-22s %9s %7s %10s %10s %7s\n",
           "group", "mean ms", "% step", "GFLOP/s", "GB/s", "% BW");
    printf("%-22s %9s %7s %10s %10s %7s\n", "----------------------",
           "---------", "-------", "----------", "----------", "-------");

    const double peak = LBM_PROF_PEAK_GBPS * nranks;
    for (int g = 0; g < NGROUP; ++g)
    {
        double ms = 0.0;
        bool any = false;
        for (int m = 0; m < 5 && gmembers[g][m] >= 0; ++m)
        {
            const int p = gmembers[g][m];
            if (calls[p] == 0) continue;
            ms += sm[p] / nranks;
            any = true;
        }
        if (!any) continue;

        const double secs = ms * 1.0e-3;
        const double by = group_bytes(g), fl = group_flops(g);
        char gf[16], gb[16], bw[16];
        if (fl > 0.0 && secs > 0.0) snprintf(gf, sizeof gf, "%10.1f", fl * nranks / secs / 1e9);
        else                        snprintf(gf, sizeof gf, "%10s", "-");
        if (by > 0.0 && secs > 0.0)
        {
            const double g_bps = by * nranks / secs / 1e9;
            snprintf(gb, sizeof gb, "%10.1f", g_bps);
            snprintf(bw, sizeof bw, "%7.1f", 100.0 * g_bps / peak);
        }
        else { snprintf(gb, sizeof gb, "%10s", "-"); snprintf(bw, sizeof bw, "%7s", "-"); }

        printf("%-22s %9.4f %7.1f %s %s %s\n", gnames[g], ms,
               (step > 0.0) ? 100.0 * ms / step : 0.0, gf, gb, bw);
    }

    printf("\nworkload/rank: %ld interior cells, halo %.2f MB send + %.2f MB recv\n",
           g_cells, g_halo_send / 1048576.0, g_halo_recv / 1048576.0);
    printf("mpi GB/s is the halo volume over the wall time of post+wait: an\n");
    printf("  effective rate including synchronisation, not link bandwidth.\n");
    printf("PROFILE build (2 extra fences) -- take throughput from SCALING.\n\n");
}
} // namespace lbmprof
#endif // LBM_PROFILE
