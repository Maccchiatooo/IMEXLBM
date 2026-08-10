// =============================================================================
// Minimal GPU-aware MPI bandwidth probe. Independent of the LBM code entirely
// -- it exists to answer one question: can MPI on this machine move device
// buffers between ranks faster than the ~6 GB/s/rank the solver is seeing?
//
//   make -f Makefile.polaris mpi_gpu_bw
//   mpiexec -n 2 --ppn 2 --depth=8 --cpu-bind depth \
//           ./set_affinity_gpu_polaris.sh ./mpi_gpu_bw
//
// Two modes are reported:
//   PINGPONG  one large message at a time. Best case; measures the raw path.
//   BURST     26 concurrent Isend/Irecv pairs, mimicking exchange_f()'s
//             26-channel pattern. If BURST is much worse than PINGPONG, the
//             problem is the message pattern, not the link.
//
// Interpreting it:
//   PINGPONG ~= 6 GB/s   -> the machine/MPI is the limit; the solver is not at
//                           fault, and the only lever is sending fewer bytes.
//   PINGPONG >> 6 GB/s   -> MPI can go fast; something about how exchange_f
//                           uses it is the problem. Compare BURST to localise.
// =============================================================================
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <cstdio>
#include <vector>
#include <cstdlib>   // getenv, atoi

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    {
        int me, np;
        MPI_Comm_rank(MPI_COMM_WORLD, &me);
        MPI_Comm_size(MPI_COMM_WORLD, &np);
        if (np < 2)
        {
            if (me == 0) printf("need at least 2 ranks\n");
            Kokkos::finalize(); MPI_Finalize(); return 1;
        }
        // Which ranks are paired. With 4 ranks this selects WHICH GPU PAIRS get
        // measured, and that turns out to matter: an all-to-all NVLink topology
        // gives the same answer for every choice, anything else does not.
        //   PEER_XOR=1  0<->1, 2<->3   (the default, and all we had measured)
        //   PEER_XOR=2  0<->2, 1<->3
        //   PEER_XOR=3  0<->3, 1<->2
        // exchange_f needs ALL of these to be fast; it talks to every rank.
        const char *pxe = getenv("PEER_XOR");
        const int pxor = pxe ? atoi(pxe) : 1;
        const int peer = me ^ pxor;
        if (me == 0) printf("PEER_XOR=%d (rank0 <-> rank%d)\n", pxor, peer);

        using Buf = Kokkos::View<double *, Kokkos::DefaultExecutionSpace::memory_space>;
        const int ITERS = 50, NCHAN = 26;

        if (me == 0)
        {
            printf("\n%-10s %12s %14s %14s\n", "MB/msg", "mode", "GB/s (pair)", "us/msg");
            printf("%-10s %12s %14s %14s\n", "----------", "------------",
                   "--------------", "--------------");
        }

        // Warm up: the first MPI exchange pays one-time IPC handle setup, which
        // otherwise contaminates the smallest size measured.
        {
            Buf w("w", 1 << 20);
            for (int it = 0; it < 5; ++it)
            {
                MPI_Request rq[2];
                MPI_Irecv(w.data(), 1 << 20, MPI_DOUBLE, peer, 99, MPI_COMM_WORLD, &rq[0]);
                MPI_Isend(w.data(), 1 << 20, MPI_DOUBLE, peer, 99, MPI_COMM_WORLD, &rq[1]);
                MPI_Waitall(2, rq, MPI_STATUSES_IGNORE);
            }
        }

        for (double mb : {0.25, 1.0, 4.0, 16.0})
        {
            const size_t n = (size_t)(mb * 1024 * 1024 / sizeof(double));

            // ---- PINGPONG: one message in flight ----------------------------
            {
                Buf s("s", n), r("r", n);
                Kokkos::fence();
                MPI_Barrier(MPI_COMM_WORLD);
                const double t0 = MPI_Wtime();
                for (int it = 0; it < ITERS; ++it)
                {
                    MPI_Request rq[2];
                    MPI_Irecv(r.data(), n, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, &rq[0]);
                    MPI_Isend(s.data(), n, MPI_DOUBLE, peer, 0, MPI_COMM_WORLD, &rq[1]);
                    MPI_Waitall(2, rq, MPI_STATUSES_IGNORE);
                }
                const double dt = MPI_Wtime() - t0;
                if (me == 0)
                {
                    const double bytes = 2.0 * n * sizeof(double) * ITERS;
                    printf("%-10.2f %12s %14.2f %14.2f\n", mb, "PINGPONG",
                           bytes / dt / 1e9, dt * 1e6 / ITERS);
                }
            }

            // ---- BURST: 26 concurrent, like exchange_f -----------------------
            {
                std::vector<Buf> s, r;
                for (int c = 0; c < NCHAN; ++c)
                {
                    s.emplace_back("s", n);
                    r.emplace_back("r", n);
                }
                Kokkos::fence();
                MPI_Barrier(MPI_COMM_WORLD);
                const double t0 = MPI_Wtime();
                for (int it = 0; it < ITERS; ++it)
                {
                    MPI_Request rq[2 * NCHAN];
                    for (int c = 0; c < NCHAN; ++c)
                    {
                        MPI_Irecv(r[c].data(), n, MPI_DOUBLE, peer, c, MPI_COMM_WORLD, &rq[c]);
                        MPI_Isend(s[c].data(), n, MPI_DOUBLE, peer, c, MPI_COMM_WORLD, &rq[NCHAN + c]);
                    }
                    MPI_Waitall(2 * NCHAN, rq, MPI_STATUSES_IGNORE);
                }
                const double dt = MPI_Wtime() - t0;
                if (me == 0)
                {
                    const double bytes = 2.0 * n * sizeof(double) * NCHAN * ITERS;
                    printf("%-10.2f %12s %14.2f %14.2f\n", mb, "BURST x26",
                           bytes / dt / 1e9, dt * 1e6 / (ITERS * NCHAN));
                }

                // ---- BURST + TOUCH: the one structural difference left ------
                // exchange_f's send buffers are written by pack_f's GPU copies
                // immediately before MPI reads them; this benchmark's are not.
                // Rewriting them with a kernel + fence each iteration is the
                // only thing that still separates the two. If this collapses to
                // the solver's ~6 GB/s, that is the mechanism.
                Kokkos::fence();
                MPI_Barrier(MPI_COMM_WORLD);
                const double t1 = MPI_Wtime();
                for (int it = 0; it < ITERS; ++it)
                {
                    for (int c = 0; c < NCHAN; ++c)
                    {
                        Buf b = s[c];
                        Kokkos::parallel_for("touch", n,
                            KOKKOS_LAMBDA(const size_t i) { b(i) = (double)i; });
                    }
                    Kokkos::fence();          // mirrors pack_f's ex.fence()

                    MPI_Request rq[2 * NCHAN];
                    for (int c = 0; c < NCHAN; ++c)
                    {
                        MPI_Irecv(r[c].data(), n, MPI_DOUBLE, peer, c, MPI_COMM_WORLD, &rq[c]);
                        MPI_Isend(s[c].data(), n, MPI_DOUBLE, peer, c, MPI_COMM_WORLD, &rq[NCHAN + c]);
                    }
                    MPI_Waitall(2 * NCHAN, rq, MPI_STATUSES_IGNORE);
                }
                const double dt1 = MPI_Wtime() - t1;
                if (me == 0)
                {
                    const double bytes = 2.0 * n * sizeof(double) * NCHAN * ITERS;
                    printf("%-10.2f %12s %14.2f %14.2f\n", mb, "BURST+TOUCH",
                           bytes / dt1 / 1e9, dt1 * 1e6 / (ITERS * NCHAN));
                }
            }
        }
        // ---- MIXED: exchange_f's actual message mix and peer spread ----------
        // Everything above sends 26 EQUAL messages to ONE peer, and every such
        // configuration measured 90-148 GB/s. The solver, at 256^3 on 4 ranks
        // with rx=1,ry=2,rz=2, instead sends per step:
        //     2 x 3.70 MB  x-faces, to ITSELF (rx=1 wraps onto the same rank)
        //     2 x 7.23 MB  z-faces, both to the SAME peer (rz=2 wraps)
        //     2 x 7.23 MB  y-faces, both to the same other peer
        //    12 x ~30 KB   edges
        //     8 x 216 B    corners
        // That is the last structural difference left. If this mode reproduces
        // ~6 GB/s, the message mix is the mechanism; if it stays fast, the
        // microbenchmark cannot reach the cause and the answer is to cut bytes.
        if (np >= 4)
        {
            const int selfp = me, p1 = me ^ 1, p2 = me ^ 2;
            struct Msg { size_t bytes; int peer; };

            auto run_mix = [&](const char *name, const std::vector<Msg> &mix) {
                if (mix.empty()) return;
                std::vector<Buf> sb, rb;
                double total = 0.0;
                for (auto &m : mix)
                {
                    const size_t n = m.bytes / sizeof(double) + 1;
                    sb.emplace_back("s", n);
                    rb.emplace_back("r", n);
                    total += 2.0 * n * sizeof(double);
                }
                Kokkos::fence();
                MPI_Barrier(MPI_COMM_WORLD);
                const double t0 = MPI_Wtime();
                for (int it = 0; it < ITERS; ++it)
                {
                    std::vector<MPI_Request> rq(2 * mix.size());
                    for (size_t c = 0; c < mix.size(); ++c)
                    {
                        const size_t n = sb[c].extent(0);
                        MPI_Irecv(rb[c].data(), n, MPI_DOUBLE, mix[c].peer, (int)c,
                                  MPI_COMM_WORLD, &rq[c]);
                        MPI_Isend(sb[c].data(), n, MPI_DOUBLE, mix[c].peer, (int)c,
                                  MPI_COMM_WORLD, &rq[mix.size() + c]);
                    }
                    MPI_Waitall((int)rq.size(), rq.data(), MPI_STATUSES_IGNORE);
                }
                const double dt = MPI_Wtime() - t0;
                if (me == 0)
                    printf("%-14s %14.2f %14.2f   (%zu msgs, %.1f MB)\n", name,
                           total * ITERS / dt / 1e9, dt * 1e6 / (ITERS * mix.size()),
                           mix.size(), total / 1048576.0);
            };

            const size_t FACE = (size_t)(7.23 * 1048576), SELFF = (size_t)(3.70 * 1048576);
            std::vector<Msg> selfs, faces, edges, corners;
            for (int i = 0; i < 2; ++i)  selfs.push_back({SELFF, selfp});
            for (int i = 0; i < 2; ++i)  faces.push_back({FACE, p1});
            for (int i = 0; i < 2; ++i)  faces.push_back({FACE, p2});
            for (int i = 0; i < 12; ++i) edges.push_back({30u * 1024, (i % 2) ? p1 : p2});
            for (int i = 0; i < 8; ++i)  corners.push_back({216u, (i % 2) ? p1 : p2});

            auto cat = [](std::vector<std::vector<Msg>> parts) {
                std::vector<Msg> o;
                for (auto &p : parts) o.insert(o.end(), p.begin(), p.end());
                return o;
            };

            if (me == 0)
                printf("\n%-14s %14s %14s\n", "variant", "GB/s (pair)", "us/msg");

            // Bisect the mix: remove one ingredient at a time. Whichever removal
            // recovers BURST-like bandwidth is the culprit.
            run_mix("FULL",        cat({selfs, faces, edges, corners}));
            run_mix("no corners",  cat({selfs, faces, edges}));
            run_mix("no small",    cat({selfs, faces}));
            run_mix("no self",     cat({faces, edges, corners}));
            run_mix("faces only",  faces);
            run_mix("corners only", corners);
            run_mix("edges only",  edges);
        }

        if (me == 0) printf("\n");
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
