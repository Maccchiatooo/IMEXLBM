#include "lbm.hpp"
#include <utility> // std::swap
#define pi 3.1415926

namespace
{
constexpr int Q27 = 27;

KOKKOS_INLINE_FUNCTION double LB_W(const int a)
{constexpr double w[Q27] = {
    8.0 / 27.0,
    2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0,
    1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0,
    1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0,
    1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0,
    1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0,
    1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0};
  return w[a];
}
KOKKOS_INLINE_FUNCTION int LB_E(const int a, const int d) 
{constexpr int e[Q27][3] = {
    {0, 0, 0},
    {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
    {1, 1, 0}, {-1, -1, 0}, {1, -1, 0}, {-1, 1, 0},
    {1, 0, 1}, {-1, 0, -1}, {1, 0, -1}, {-1, 0, 1},
    {0, 1, 1}, {0, -1, -1}, {0, 1, -1}, {0, -1, 1},
    {1, 1, 1}, {-1, -1, -1}, {1, -1, 1}, {-1, 1, -1},
    {1, 1, -1}, {-1, -1, 1}, {1, -1, -1}, {-1, 1, 1}};
   return e[a][d];
}

constexpr int LB_OPP[Q27] = {
    0, 2, 1, 4, 3, 6, 5,
    8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17,
    20, 19, 22, 21, 24, 23, 26, 25};
}

void LBM::Initialize()
{

    setup_MPI();

    f = buffer_f("f", q, lx, ly, lz);
    ft = buffer_f("ft", q, lx, ly, lz);
    fb = buffer_f("fb", q, lx, ly, lz);

    ua = buffer_u("u", lx, ly, lz);
    va = buffer_u("v", lx, ly, lz);
    wa = buffer_u("w", lx, ly, lz); 
    rho = buffer_u("rho", lx, ly, lz);
    p = buffer_u("p", lx, ly, lz);

    e = Kokkos::View<int **, DeviceSpace>("e", q, dim);
    t = Kokkos::View<double *, DeviceSpace>("t", q);
    usr = Kokkos::View<int ***, DeviceSpace>("usr", lx, ly, lz);
    ran = Kokkos::View<int ***, DeviceSpace>("ran", lx, ly, lz);
    bb = Kokkos::View<int *, DeviceSpace>("b", q);

    auto bb_mirror = Kokkos::create_mirror_view(Kokkos::HostSpace(), bb);
    auto t_mirror = Kokkos::create_mirror_view(Kokkos::HostSpace(), t);
    auto e_mirror = Kokkos::create_mirror_view(Kokkos::HostSpace(), e);
    for (int a = 0; a < q; ++a)
    {
        t_mirror(a) = LB_W(a);
        bb_mirror(a) = LB_OPP[a];
        for (int d = 0; d < dim; ++d)
            e_mirror(a, d) = LB_E(a, d);
    }
    Kokkos::deep_copy(t, t_mirror);
    Kokkos::deep_copy(e, e_mirror);
    Kokkos::deep_copy(bb, bb_mirror);

    // Register the per-rank workload so the profiler can turn phase times into
    // GFLOP/s and GB/s. Interior cells only — ghosts are not updated.
    LBM_PROF_SET_CELLS((long)l_l[0] * l_l[1] * l_l[2], q);
    // macroscopic value initialization

    Kokkos::parallel_for(
        "init_macro", mdrange_policy3({0, 0, 0}, {lx, ly, lz}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
            const double xx = (double)(i - ghost + x_lo) / (double)glx * 2.0 * pi;
            const double yy = (double)(j - ghost + y_lo) / (double)gly * 2.0 * pi;
            const double zz = (double)(k - ghost + z_lo) / (double)glz * 2.0 * pi;
            const double sx = sin(xx), cx = cos(xx);
            const double sy = sin(yy), cy = cos(yy);
            const double cz = cos(zz);

            ua(i, j, k) = u0 * sx * cy * cz;
            va(i, j, k) = -u0 * cx * sy * cz;
            wa(i, j, k) = 0.0;
            p(i, j, k) = rho0 * cs2 +
                         rho0 * u0 * u0 / 16.0 * (cos(2.0 * xx) + cos(2.0 * yy)) *
                             (cos(2.0 * zz) + 2.0);
            rho(i, j, k) = rho0;
        });

    // distribution function initialization
    Kokkos::parallel_for(
        "initf", mdrange_policy4({0, 0, 0, 0}, {q, lx, ly, lz}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j, const int k) {

            const double u = ua(i, j, k), v = va(i, j, k), w = wa(i, j, k);
            const double edu = e(ii,0) * u + e(ii,1) * v + e(ii,2) * w;
            const double udu = u * u + v * v + w * w;

            f(ii, i, j, k) =
                t(ii) * (3.0 * p(i, j, k) + 3.0 * edu + 4.5 * edu * edu - 1.5 * udu);
            ft(ii, i, j, k) = 0.0;
        });

    Kokkos::fence();
};
// =============================================================================
// FUSED COLLISION (moments + collision in one 3D kernel)
//
// Folds the old Update() (moment reduction) INTO Collision:
//   - one 3D thread per cell, inner loop over 27 directions
//   - load 27 f into register array fl[] ONCE
//   - reduce (p,u,v,w) from fl[]  (no global read of ua/va/wa/p)
//   - collide using fl[]          (no second global read of f)
//   - single store of f
// => f: read1/write1 per step; (u,v,w,p) never touch global memory.
//
// MUST VERIFY:
//   * correctness: output u/v/w/p min-max must match the ORIGINAL build.
//   * register pressure: fl[27] = 216 B/thread may spill. profile with
//     ncu --set full and check Registers Per Thread / Achieved Occupancy /
//     local-memory spill. If it spills, tune LaunchBounds<64,N> or drop fl[].
//   * tile {1,1,64} = 64 threads/block must match LaunchBounds<64,...> in the
//     mdrange_policy3_lb typedef in lbm.hpp.
// =============================================================================
void LBM::Collision()
{
    const double inv_tau = 1.0 / (tau0 + 0.5);

    Kokkos::parallel_for(
        "collision",
        mdrange_policy3_lb({l_s[0], l_s[1], l_s[2]},
                           {l_e[0], l_e[1], l_e[2]},
                           {1, 1, 128}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
            // 1. load 27 directions once; reduce to p,u,v,w
            double fl[Q27];
            double pl = 0.0, ul = 0.0, vl = 0.0, wl = 0.0;
#pragma unroll
            for (int ii = 0; ii < Q27; ++ii) {
                const double fv = f(ii, i, j, k);
                fl[ii] = fv;
                pl += fv;
                ul += fv * e(ii, 0);
                vl += fv * e(ii, 1);
                wl += fv * e(ii, 2);
            }
            const double u = ul, v = vl, w = wl;
            const double pp = pl * (1.0 / 3.0);
            const double udu = u * u + v * v + w * w;

            // 2. collide using register copy; single store
#pragma unroll
            for (int ii = 0; ii < Q27; ++ii) {
                const double edu = e(ii, 0) * u + e(ii, 1) * v + e(ii, 2) * w;
                const double feq =
                    t(ii) * (3.0 * pp + 3.0 * edu + 4.5 * edu * edu - 1.5 * udu);
                f(ii, i, j, k) = fl[ii] - (fl[ii] - feq) * inv_tau;
            }
        });
};

void LBM::Streaming()
{
    passf(f);

    // RESTRUCTURED to mirror Collision: 3D policy over (i,j,k) with the 27
    // directions unrolled INSIDE, instead of a 4D policy with one element per
    // work-item. Identical semantics, same bytes moved. Two reasons it is
    // faster:
    //   * memory-level parallelism — 27 independent loads in flight per
    //     work-item instead of 1. The old version was latency-bound: each
    //     work-item's single load could not issue until three e() global reads
    //     returned, and there was no second load to overlap with it.
    //   * 27x fewer workgroups to schedule (one per cell, not one per
    //     cell-direction), each doing 27x the work.
    // LB_E() instead of the e View: with ii a compile-time constant under
    // #pragma unroll, the constexpr table folds to literals, so the offsets
    // cost no memory traffic at all.
    LBM_PROF_BEGIN(P_STREAM_ENQ);
    Kokkos::parallel_for(
        "stream1",
        mdrange_policy3({ghost, ghost, ghost},
                        {lx - ghost, ly - ghost, lz - ghost},
                        {1, 1, 128}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
#pragma unroll
            for (int ii = 0; ii < Q27; ++ii) {
                ft(ii, i, j, k) =
                    f(ii, i - LB_E(ii, 0), j - LB_E(ii, 1), k - LB_E(ii, 2));
            }
        });
    LBM_PROF_END(P_STREAM_ENQ);

    // In a normal build this fence also covers unpack_f's 26 copies, which are
    // stream-ordered ahead of stream1. In a PROFILE build unpack_f fences
    // itself, so this row is stream1 alone.
    LBM_PROF_BEGIN(P_STREAM_FENCE);
    Kokkos::fence();
    LBM_PROF_END(P_STREAM_FENCE);
    std::swap(f, ft);
};

// RENAMED from Update(): now only called before OUTPUT (not every step).
// Identical math to the original Update(); explicit (u,v,w,p) materialization
// used only when writing output. Also the correctness reference for the fused
// Collision above.
void LBM::ComputeMacroscopic()
{


    LBM_PROF_BEGIN(P_MACRO);
    Kokkos::parallel_for(
        "moments",
        mdrange_policy3({ghost, ghost, ghost}, {lx - ghost, ly - ghost, lz - ghost},{1,1,128}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
            double pl = 0.0, ul = 0.0, vl = 0.0, wl = 0.0;
#pragma unroll
            for (int ii = 0; ii < Q27; ++ii) // D3Q27: q is 27 by construction
            {
                const double fv = f(ii, i, j, k);
                pl += fv;
                ul += fv * e(ii,0);
                vl += fv * e(ii,1);
                wl += fv * e(ii,2);
            }
            p(i, j, k) = pl * (1.0 / 3.0);
            ua(i, j, k) = ul;
            va(i, j, k) = vl;
            wa(i, j, k) = wl;
        });

    Kokkos::fence();
    LBM_PROF_END(P_MACRO);
};

// ---------------------------------------------------------------------------
// Globally reduced diagnostics.
//
// In this pressure-based formulation sum_i f_i = 3p and sum_i f_i e_i = u, and
// the equilibrium reproduces both moments exactly. With sum_i t_i = 1,
// sum_i t_i e_i = 0 and sum_i t_i e_ia e_ib = delta_ab / 3:
//
//     sum_i feq_i     = 3p + 4.5*(u^2/3) - 1.5*u^2 = 3p = sum_i f_i
//     sum_i feq_i e_i = 3 u_b delta_ab / 3         = u_a = sum_i f_i e_i
//
// so BGK collision leaves both untouched, and streaming on a fully periodic
// domain is a permutation of f. Mass and momentum are therefore invariants of
// the discrete dynamics up to floating-point rounding. A relative drift much
// larger than 1e-12 is a bug -- in streaming, in the halo exchange, or a race --
// not "numerical error".
//
// ke is NOT conserved: dissipating it is what the Taylor-Green vortex is for.
// It is reported so a test can check that it decays, and that its trajectory
// does not depend on how the domain was split across ranks.
//
// Cost is one 5-double MPI_Allreduce per call. The caller already issues an
// MPI_Barrier at the same cadence, so the marginal cost is in the noise even
// at full machine scale.
// ---------------------------------------------------------------------------
void LBM::Conserved(double &mass, double &mom_x, double &mom_y, double &mom_z,
                    double &ke)
{
    double lmass = 0.0, lmx = 0.0, lmy = 0.0, lmz = 0.0, lke = 0.0;

    Kokkos::parallel_reduce(
        "conserved",
        mdrange_policy3({l_s[0], l_s[1], l_s[2]}, {l_e[0], l_e[1], l_e[2]}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k,
                            double &m, double &mx, double &my, double &mz,
                            double &e_kin) {
            double pl = 0.0, ul = 0.0, vl = 0.0, wl = 0.0;
            for (int ii = 0; ii < Q27; ++ii)
            {
                const double fv = f(ii, i, j, k);
                pl += fv;
                ul += fv * e(ii, 0);
                vl += fv * e(ii, 1);
                wl += fv * e(ii, 2);
            }
            m += pl;
            mx += ul;
            my += vl;
            mz += wl;
            e_kin += 0.5 * (ul * ul + vl * vl + wl * wl);
        },
        Kokkos::Sum<double>(lmass), Kokkos::Sum<double>(lmx),
        Kokkos::Sum<double>(lmy), Kokkos::Sum<double>(lmz),
        Kokkos::Sum<double>(lke));
    Kokkos::fence();

    double local[5] = {lmass, lmx, lmy, lmz, lke};
    double total[5];
    MPI_Allreduce(local, total, 5, MPI_DOUBLE, MPI_SUM, comm);

    mass = total[0];
    mom_x = total[1];
    mom_y = total[2];
    mom_z = total[3];
    ke = total[4];
};

void LBM::MPIoutput(int n)
{
    // MPI_IO
    MPI_File fh;
    MPI_Status status;
    MPI_Offset offset = 0;

    MPI_Datatype FILETYPE, DATATYPE;
    // buffer
    int tp;
    float ttp;
    double fp;
    // min max
    double umin, umax, wmin, wmax, vmin, vmax, pmin, pmax;
    double uumin, uumax, wwmin, wwmax, vvmin, vvmax, ppmin, ppmax;
    // transfer
    double *uu, *vv, *ww, *pp, *xx, *yy, *zz;
    int start[3];
    uu = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));
    vv = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));
    ww = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));
    pp = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));
    xx = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));
    yy = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));
    zz = (double *)malloc(l_l[0] * l_l[1] * l_l[2] * sizeof(double));

    for (int k = 0; k < l_l[2]; k++)
    {
        for (int j = 0; j < l_l[1]; j++)
        {
            for (int i = 0; i < l_l[0]; i++)
            {

                uu[i + j * l_l[0] + k * l_l[1] * l_l[0]] = ua(i + ghost, j + ghost, k + ghost);
                vv[i + j * l_l[0] + k * l_l[1] * l_l[0]] = va(i + ghost, j + ghost, k + ghost);
                ww[i + j * l_l[0] + k * l_l[1] * l_l[0]] = wa(i + ghost, j + ghost, k + ghost);
                pp[i + j * l_l[0] + k * l_l[1] * l_l[0]] = p(i + ghost, j + ghost, k + ghost);
                xx[i + j * l_l[0] + k * l_l[1] * l_l[0]] = (double)(x_lo + i) / (glx - 1);
                yy[i + j * l_l[0] + k * l_l[1] * l_l[0]] = (double)(y_lo + j) / (gly - 1);
                zz[i + j * l_l[0] + k * l_l[1] * l_l[0]] = (double)(z_lo + k) / (glz - 1);
            }
        }
    }
    Kokkos::parallel_reduce(
        "minmax_uvwp",
        mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k,
                            double &umx, double &vmx, double &wmx, double &pmx,
                            double &umn, double &vmn, double &wmn, double &pmn) {
            const double u = ua(i, j, k);
            const double v = va(i, j, k);
            const double w = wa(i, j, k);
            const double pv = p(i, j, k);
            if (u > umx) umx = u;
            if (v > vmx) vmx = v;
            if (w > wmx) wmx = w;
            if (pv > pmx) pmx = pv;
            if (u < umn) umn = u;
            if (v < vmn) vmn = v;
            if (w < wmn) wmn = w;
            if (pv < pmn) pmn = pv;
        },
        Kokkos::Max<double>(umax), Kokkos::Max<double>(vmax),
        Kokkos::Max<double>(wmax), Kokkos::Max<double>(pmax),
        Kokkos::Min<double>(umin), Kokkos::Min<double>(vmin),
        Kokkos::Min<double>(wmin), Kokkos::Min<double>(pmin));
    Kokkos::fence();

    std::string str1 = "output" + std::to_string(n) + ".plt";
    const char *na = str1.c_str();
    std::string str2 = "#!TDV112";
    const char *version = str2.c_str();
    MPI_File_open(MPI_COMM_WORLD, na, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    {
        double mins[4] = {umin, vmin, wmin, pmin};
        double maxs[4] = {umax, vmax, wmax, pmax};
        double gmins[4], gmaxs[4];
        MPI_Reduce(mins, gmins, 4, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(maxs, gmaxs, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        uumin = gmins[0]; vvmin = gmins[1]; wwmin = gmins[2]; ppmin = gmins[3];
        uumax = gmaxs[0]; vvmax = gmaxs[1]; wwmax = gmaxs[2]; ppmax = gmaxs[3];
    }

    if (me == 0)
    {

        MPI_File_seek(fh, offset, MPI_SEEK_SET);
        // header !version number
        MPI_File_write(fh, version, 8, MPI_CHAR, &status);
        // INTEGER 1
        tp = 1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // 3*4+8=20
        // variable name
        tp = 7;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 120;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 121;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 122;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 117;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 118;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 119;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 112;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // 20+15*4=80
        // Zone Marker
        ttp = 299.0;
        MPI_File_write(fh, &ttp, 1, MPI_REAL, &status);
        // Zone Name
        tp = 90;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 79;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 78;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 69;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 32;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 48;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 48;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 49;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // 80 + 10 * 4 = 120

        // Strand id
        tp = -1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // SOLUTION TIME
        double nn = (double)n;
        fp = nn;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ZONE COLOR
        tp = -1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ZONE TYPE
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // SPECIFY VAR LOCATION
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ARE RAW LOCAL
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // NUMBER OF MISCELLANEOUS
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ORDERED ZONE
        tp = glx;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = gly;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        tp = glz;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // AUXILIARY
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // 120 + 13 * 4 = 172
        // EOHMARKER
        ttp = 357.0;
        MPI_File_write(fh, &ttp, 1, MPI_REAL, &status);
        // DATA SECTION
        ttp = 299.0;
        MPI_File_write(fh, &ttp, 1, MPI_REAL, &status);
        // VARIABLE DATA FORMAT
        tp = 2;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        MPI_File_write(fh, &tp, 1, MPI_INT, &status);

        // PASSIVE VARIABLE
        tp = 0;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // SHARING VARIABLE
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // ZONE NUMBER
        tp = -1;
        MPI_File_write(fh, &tp, 1, MPI_INT, &status);
        // 172 + 12 * 4 = 220
        // MIN AND MAX VALUE FLOAT 64
        fp = 0.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 1.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 0.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 1.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 0.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = 1.0;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = uumin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = uumax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = vvmin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = vvmax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = wwmin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = wwmax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = ppmin;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);
        fp = ppmax;
        MPI_File_write(fh, &fp, 1, MPI_DOUBLE, &status);

        // 220 + 14 * 8 = 332
    }

    offset = 332;

    int glolen[3] = {glx, gly, glz};
    int iniarr[3] = {0, 0, 0};
    int localstart[3] = {x_lo, y_lo, z_lo};
    MPI_Type_create_subarray(dim, glolen, l_l, localstart, MPI_ORDER_FORTRAN, MPI_DOUBLE, &DATATYPE);

    MPI_Type_commit(&DATATYPE);

    MPI_Type_contiguous(7, DATATYPE, &FILETYPE);

    MPI_Type_commit(&FILETYPE);

    MPI_File_set_view(fh, offset, MPI_DOUBLE, FILETYPE, "native", MPI_INFO_NULL);

    MPI_File_write_all(fh, xx, l_l[0] * l_l[1] * l_l[2], MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, yy, l_l[0] * l_l[1] * l_l[2], MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, zz, l_l[0] * l_l[1] * l_l[2], MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, uu, l_l[0] * l_l[1] * l_l[2], MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, vv, l_l[0] * l_l[1] * l_l[2], MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, ww, l_l[0] * l_l[1] * l_l[2], MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_write_all(fh, pp, l_l[0] * l_l[1] * l_l[2], MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_close(&fh);

    free(uu);
    free(vv);
    free(ww);
    free(pp);
    free(xx);
    free(yy);
    free(zz);

    MPI_Barrier(MPI_COMM_WORLD);
};

void LBM::Output(int n)
{
    std::ofstream outfile;
    std::string str = "output" + std::to_string(n) + std::to_string(me);
    outfile << std::setiosflags(std::ios::fixed);
    outfile.open(str + ".dat", std::ios::out);

    outfile << "variables=x,y,z,f" << std::endl;
    outfile << "zone I=" << lx - 6 << ",J=" << ly - 6 << ",K=" << lz - 6 << std::endl;

    for (int k = 3; k < lz - 3; k++)
    {
        for (int j = 3; j < ly - 3; j++)
        {
            for (int i = 3; i < lx - 3; i++)
            {

                outfile << std::setprecision(8) << std::setiosflags(std::ios::left) << x_lo + i - 3 << " " << y_lo + j - 3 << " " << z_lo + k - 3 << " " << f(0, i, j, k) << std::endl;
            }
        }
    }

    outfile.close();
    if (me == 0)
    {
        printf("\n");
        printf("The result %d is writen\n", n);
        printf("\n");
        printf("============================\n");
    }
};
