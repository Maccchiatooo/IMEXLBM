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
            l1.Collision();
            l1.Streaming();
            l1.Update();
            end = MPI_Wtime();
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
    }
    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
