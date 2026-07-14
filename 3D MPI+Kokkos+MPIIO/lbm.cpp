#include "lbm.hpp"

namespace
{
constexpr int Q27 = 27;

constexpr double LB_W[Q27] = {
    8.0 / 27.0,
    2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0, 2.0 / 27.0,
    1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0,
    1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0,
    1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0, 1.0 / 54.0,
    1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0,
    1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0, 1.0 / 216.0};

constexpr int LB_E[Q27][3] = {
    {0, 0, 0},
    {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
    {1, 1, 0}, {-1, -1, 0}, {1, -1, 0}, {-1, 1, 0},
    {1, 0, 1}, {-1, 0, -1}, {1, 0, -1}, {-1, 0, 1},
    {0, 1, 1}, {0, -1, -1}, {0, 1, -1}, {0, -1, 1},
    {1, 1, 1}, {-1, -1, -1}, {1, -1, 1}, {-1, 1, -1},
    {1, 1, -1}, {-1, -1, 1}, {1, -1, -1}, {-1, 1, 1}};

constexpr int LB_OPP[Q27] = {
    0, 2, 1, 4, 3, 6, 5,
    8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17,
    20, 19, 22, 21, 24, 23, 26, 25};
}

void LBM::Initialize()
{

    f = decltype(f)("f", q, lx, ly, lz);
    ft = decltype(ft)("ft", q, lx, ly, lz);
    fb = decltype(fb)("fb", q, lx, ly, lz);
 
    ua = decltype(ua)("u", lx, ly, lz);
    va = decltype(va)("v", lx, ly, lz);
    wa = decltype(wa)("w", lx, ly, lz);
    rho = decltype(rho)("rho", lx, ly, lz);
    p = decltype(p)("p", lx, ly, lz);
 
    e = decltype(e)("e", q, dim);
    t = decltype(t)("t", q);
    usr = decltype(usr)("usr", lx, ly, lz);
    ran = decltype(ran)("ran", lx, ly, lz);
    bb = decltype(bb)("b", q);

    for (int a = 0; a < q; ++a)
    {
        t(a) = LB_W[a];
        bb(a) = LB_OPP[a];
        for (int d = 0; d < dim; ++d)
            e(a, d) = LB_E[a][d];
    }

    // macroscopic value initialization

    // macroscopic value initialization
    Kokkos::parallel_for(
        "initialize", mdrange_policy3({0, 0, 0}, {lx, ly, lz}), KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
            ua(i, j, k) = 0;
            va(i, j, k) = 0;
            wa(i, j, k) = 0;
            p(i, j, k) = 0;
            rho(i, j, k) = rho0;
        });

    // distribution function initialization
    Kokkos::parallel_for(
        "initf", mdrange_policy4({0, 0, 0, 0}, {q, lx, ly, lz}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j, const int k) {

            const double u = ua(i, j, k);
            const double v = va(i, j, k);
            const double w = wa(i, j, k);
            const double edu = LB_E[ii][0] * u + LB_E[ii][1] * v + LB_E[ii][2] * w;
            const double udu = u * u + v * v + w * w;

            f(ii, i, j, k) =
                LB_W[ii] * (3.0 * p(i, j, k) + 3.0 * edu + 4.5 * edu * edu - 1.5 * udu);

            ft(ii, i, j, k) = 0;
        });

    Kokkos::fence();
};
void LBM::Collision()
{
    // collision

    Kokkos::parallel_for(
        "collision", mdrange_policy4({0, l_s[0], l_s[1], l_s[2]}, {q, l_e[0], l_e[1], l_e[2]}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j, const int k) {
            const double u = ua(i, j, k);
            const double v = va(i, j, k);
            const double w = wa(i, j, k);
            const double edu = LB_E[ii][0] * u + LB_E[ii][1] * v + LB_E[ii][2] * w;
            const double udu = u * u + v * v + w * w;

            const double feq =
                LB_W[ii] * (3.0 * p(i, j, k) + 3.0 * edu + 4.5 * edu * edu - 1.5 * udu);

            f(ii, i, j, k) -= (f(ii, i, j, k) - feq) * inv_tau;
        });
};

void LBM::Streaming()
{
    if (x_lo == 0)
    {
        Kokkos::parallel_for(
            "bcl", mdrange_policy3({0, ghost - 1, ghost - 1}, {q, ly - ghost + 1, lz - ghost + 1}), KOKKOS_CLASS_LAMBDA(const int ii, const int j, const int k) {
                if (LB_E[ii][0] > 0)
                {
                    f(ii, l_s[0] - 1, j, k) =
                        f(LB_OPP[ii], l_s[0] + 1, j + 2 * LB_E[ii][1], k + 2 * LB_E[ii][2]);
                }
            });
    }

    if (x_hi == glx - 1)
    {
        Kokkos::parallel_for(
            "bcr", mdrange_policy3({0, ghost - 1, ghost - 1}, {q, ly - ghost + 1, lz - ghost + 1}), KOKKOS_CLASS_LAMBDA(const int ii, const int j, const int k) {
                {
                    f(ii, l_e[0], j, k) =
                        f(LB_OPP[ii], l_e[0] - 2, j + 2 * LB_E[ii][1], k + 2 * LB_E[ii][2]);
                }
            });
    }
    // front boundary bounce back
    if (y_lo == 0)
    {
        Kokkos::parallel_for(
            "bcf", mdrange_policy3({0, ghost - 1, ghost - 1}, {q, lx - ghost + 1, lz - ghost + 1}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int k) {
                if (LB_E[ii][1] > 0)
                {
                    f(ii, i, l_s[1] - 1, k) =
                        f(LB_OPP[ii], i + 2 * LB_E[ii][0], l_s[1] + 1, k + 2 * LB_E[ii][2]);
                }
            });
    }
    // back boundary bounce back
    Kokkos::fence();
    if (y_hi == gly - 1)
    {
        Kokkos::parallel_for(
            "bcb", mdrange_policy3({0, ghost - 1, ghost - 1}, {q, lx - ghost + 1, lz - ghost + 1}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int k) {
                if (LB_E[ii][1] < 0)
                {
                    f(ii, i, l_e[1], k) =
                        f(LB_OPP[ii], i + 2 * LB_E[ii][0], l_e[1] - 2, k + 2 * LB_E[ii][2]);
                }
            });
    }
    // bottom boundary bounce back
    Kokkos::fence();
    if (z_lo == 0)
    {
        Kokkos::parallel_for(
            "bcd", mdrange_policy3({0, ghost - 1, ghost - 1}, {q, lx - ghost + 1, ly - ghost + 1}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
                if (LB_E[ii][2] > 0)
                {
                    f(ii, i, j, l_s[2] - 1) =
                        f(LB_OPP[ii], i + 2 * LB_E[ii][0], j + 2 * LB_E[ii][1], l_s[2] + 1);
                }
            });
    }
    // top boundary bounce back
    Kokkos::fence();
    if (z_hi == glz - 1)
    {
        Kokkos::parallel_for(
            "bcu", mdrange_policy3({0, ghost - 1, ghost - 1}, {q, lx - ghost + 1, ly - ghost + 1}), KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j) {
                if (LB_E[ii][2] < 0)
                {
                    f(ii, i, j, l_e[2]) =
                        f(LB_OPP[ii], i + 2 * LB_E[ii][0], j + 2 * LB_E[ii][1], l_e[2] - 2);
                }
            });
    }

    // streaming process
    Kokkos::parallel_for(
        "stream1",
        mdrange_policy4({0, ghost, ghost, ghost}, {q, lx - ghost, ly - ghost, lz - ghost}),
        KOKKOS_CLASS_LAMBDA(const int ii, const int i, const int j, const int k) {
            ft(ii, i, j, k) = f(ii, i - LB_E[ii][0], j - LB_E[ii][1], k - LB_E[ii][2]);
        });

    Kokkos::fence();
    std::swap(f, ft);
};

void LBM::Update()
{


    Kokkos::parallel_for(
        "moments",
        mdrange_policy3({ghost, ghost, ghost}, {lx - ghost, ly - ghost, lz - ghost}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k) {
            double pl = 0.0, ul = 0.0, vl = 0.0, wl = 0.0;
#pragma unroll
            for (int ii = 0; ii < Q27; ++ii) // D3Q27: q is 27 by construction
            {
                const double fv = f(ii, i, j, k);
                pl += fv;
                ul += fv * LB_E[ii][0];
                vl += fv * LB_E[ii][1];
                wl += fv * LB_E[ii][2];
            }
            p(i, j, k) = pl * (1.0 / 3.0);
            ua(i, j, k) = ul;
            va(i, j, k) = vl;
            wa(i, j, k) = wl;
        });

    
    if (z_hi == glz - 1)
    {
        const int kk = l_e[2] - 1;
        Kokkos::parallel_for(
            "lid_bc",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ghost, ghost}, {lx - ghost, ly - ghost}),
            KOKKOS_CLASS_LAMBDA(const int i, const int j) {
                ua(i, j, kk) = 0.1;
            });
    }
    Kokkos::fence();
};

void LBM::MPIoutput(int n)
{
    // MPI_IO
    MPI_File fh;
    MPIO_Request request;
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

    parallel_reduce(
        " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = ua(i,j,k);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(umax));
    Kokkos::fence();
    parallel_reduce(
        " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = va(i,j,k);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(vmax));
    Kokkos::fence();
    parallel_reduce(
        " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = wa(i,j,k);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(wmax));
    Kokkos::fence();
    parallel_reduce(
        " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = p(i,j,k);
         if(my_value > valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Max<double>(pmax));
    Kokkos::fence();
    parallel_reduce(
        " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = ua(i,j,k);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(umin));
    Kokkos::fence();
    parallel_reduce(
        " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = va(i,j,k);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(vmin));
    Kokkos::fence();
    parallel_reduce(
        " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = wa(i,j,k);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(wmin));
    Kokkos::fence();
    parallel_reduce(
        " Label", mdrange_policy3({ghost, ghost, ghost}, {l_e[0], l_e[1], l_e[2]}),
        KOKKOS_CLASS_LAMBDA(const int i, const int j, const int k, double &valueToUpdate) {
         double my_value = p(i,j,k);
         if(my_value < valueToUpdate ) valueToUpdate = my_value; }, Kokkos ::Min<double>(pmin));
    Kokkos::fence();
    std::string str1 = "output" + std::to_string(n) + ".plt";
    const char *na = str1.c_str();
    std::string str2 = "#!TDV112";
    const char *version = str2.c_str();
    MPI_File_open(MPI_COMM_WORLD, na, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    MPI_Reduce(&umin, &uumin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&umax, &uumax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&vmin, &vvmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&vmax, &vvmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&wmin, &wwmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&wmax, &wwmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&pmin, &ppmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&pmax, &ppmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (comm.me == 0)
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
    std::string str = "output" + std::to_string(n) + std::to_string(comm.me);
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

                outfile << std::setprecision(8) << setiosflags(std::ios::left) << x_lo + i - 3 << " " << y_lo + j - 3 << " " << z_lo + k - 3 << " " << f(0, i, j, k) << std::endl;
            }
        }
    }

    outfile.close();
    if (comm.me == 0)
    {
        printf("\n");
        printf("The result %d is writen\n", n);
        printf("\n");
        printf("============================\n");
    }
};
