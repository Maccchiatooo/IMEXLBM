#include "mpi.h"
#include "lbm.hpp"
#include "System.hpp"
#include <Kokkos_Core.hpp>

int main(int argc, char *argv[])
{

    double start, end;
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        System s1;
        if (rank == 0)
        {
            s1.Monitor();
        }
        MPI_Barrier(MPI_COMM_WORLD);
        LBM l1(MPI_COMM_WORLD, s1.sx, s1.sy, s1.sz, s1.tau, s1.rho0, s1.u0);

        l1.Initialize();

        Kokkos::fence();
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        double t_last = start;
        
        for (int it = 1; it <= s1.Time; it++)
        {
// Per-step collision timing. Great for kernel tuning, WRONG for scaling runs:
// the fence pair plus a per-step printf costs tens of microseconds, which is a
// large fraction of a step once the per-rank subdomain gets small. At 256 nodes
// it would dominate what it is trying to measure.
// Build the scaling binary with -DLBM_NO_PERSTEP_TIMING to compile it out
// (Makefile.aurora SCALING=1 does this). Default behaviour is unchanged.
            LBM_PROF_BEGIN(P_STEP);
#ifndef LBM_NO_PERSTEP_TIMING
		Kokkos::fence();
const double t0 = MPI_Wtime();
l1.Collision();
Kokkos::fence();
const double t1 = MPI_Wtime();
const double collision_time = t1 - t0;

if (l1.me == 0) {
	    const double flop_per_cell = 594.0;
	        const double total_flop = flop_per_cell * s1.sx * s1.sy * s1.sz;
		    const double gflops = total_flop / collision_time / 1.0e9;
		        printf("collision | %8.4f ms | %10.2f GFLOPS\n", collision_time*1e3, gflops);
}
#else
            LBM_PROF_BEGIN(P_COLLISION);
            l1.Collision();
            LBM_PROF_FENCE();          // only fence this build adds
            LBM_PROF_END(P_COLLISION);
#endif

            l1.Streaming();
            LBM_PROF_END(P_STEP);
            //l1.Update();
            end = MPI_Wtime();

            // Discard step 1: SYCL kernel load, first-touch allocation and MPI
            // connection setup all land there and would skew every phase.
            if (it == 1) LBM_PROF_RESET();
            if (it % s1.inter == 0)
            {
                Kokkos::fence();
                MPI_Barrier(MPI_COMM_WORLD);
                const double now = MPI_Wtime();
                if (l1.me == 0)
                {
                    const double dt_int = now - t_last;
                    const double mlups = (double)s1.sx * s1.sy * s1.sz *
                                         s1.inter / dt_int / 1.0e6;
                    printf("step %6d | interval %8.4f s | total %8.4f s | %10.2f MLUPS\n",
                           it, dt_int, now - start, mlups);
                }
                t_last = now;
                // l1.MPIoutput(it / s1.inter);
            }
        }

        // nsteps-1 because step 1 was discarded as warm-up above.
        LBM_PROF_REPORT(MPI_COMM_WORLD, s1.Time - 1);
    }
    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
