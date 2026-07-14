#include "lbm.hpp"


void LBM::setup_MPI()
{
    MPI_Comm_size(comm, &nranks);
    MPI_Comm_rank(comm, &me);

    setup_Cartesian();
    setup_Local();
    setup_f();

    MPI_Barrier(MPI_COMM_WORLD);
}

void LBM::setup_Cartesian()
{
    int x_tem, y_tem, z_tem;

    double min_e = 1e16;
    double u2 = 0;
    for (int i = 1; i < nranks + 1; i++)
    {
        if (nranks % i == 0)
        {
            x_tem = i;

            for (int j = 1; j < nranks / i + 1; j++)
            {
                if ((nranks / i) % j == 0)
                {
                    y_tem = j;
                    z_tem = nranks / i / j;

                    double loc_x = glx / x_tem;
                    double loc_y = gly / y_tem;
                    double loc_z = glz / z_tem;

                    double average = (loc_x + loc_y + loc_z) / 3.0;

                    u2 = (loc_x - average) * (loc_x - average) +
                         (loc_y - average) * (loc_y - average) +
                         (loc_z - average) * (loc_z - average);
                    if (u2 < min_e)
                    {
                        min_e = u2;
                        rx = x_tem;
                        ry = y_tem;
                        rz = z_tem;
                    }
                }
            }
        }
    }

    if (me == 0)
        printf("rx=%d,ry=%d,rz=%d\n", rx, ry, rz);

    px = me % rx;
    pz = (me / rx) % rz;
    py = (me / rx / rz);

    // 6 faces (periodic wraparound)
    face_send[0] = px == 0 ? me + rx - 1 : me - 1;                     // left
    face_send[1] = px == rx - 1 ? me - rx + 1 : me + 1;                // right
    face_send[2] = pz == 0 ? me + rx * (rz - 1) : me - rx;             // down
    face_send[3] = pz == rz - 1 ? me - rx * (rz - 1) : me + rx;        // up
    face_send[4] = py == 0 ? me + rx * rz * (ry - 1) : me - rx * rz;   // front
    face_send[5] = py == ry - 1 ? me - rx * rz * (ry - 1) : me + rx * rz; // back

    // 12 edges
    edge_send[0] = (pz == rz - 1) ? face_send[0] - rx * (rz - 1) : face_send[0] + rx;  // leftup
    edge_send[1] = (pz == rz - 1) ? face_send[1] - rx * (rz - 1) : face_send[1] + rx;  // rightup
    edge_send[2] = (pz == 0) ? face_send[0] + rx * (rz - 1) : face_send[0] - rx;       // leftdown
    edge_send[3] = (pz == 0) ? face_send[1] + rx * (rz - 1) : face_send[1] - rx;       // rightdown
    edge_send[4] = (pz == rz - 1) ? face_send[4] - rx * (rz - 1) : face_send[4] + rx;  // frontup
    edge_send[5] = (pz == 0) ? face_send[4] + rx * (rz - 1) : face_send[4] - rx;       // frontdown
    edge_send[6] = (pz == rz - 1) ? face_send[5] - rx * (rz - 1) : face_send[5] + rx;  // backup
    edge_send[7] = (pz == 0) ? face_send[5] + rx * (rz - 1) : face_send[5] - rx;       // backdown
    edge_send[8] = (px == rx - 1) ? face_send[4] - rx + 1 : face_send[4] + 1;          // frontright
    edge_send[9] = (px == 0) ? face_send[4] + rx - 1 : face_send[4] - 1;               // frontleft
    edge_send[10] = (px == rx - 1) ? face_send[5] - rx + 1 : face_send[5] + 1;         // backright
    edge_send[11] = (px == 0) ? face_send[5] + rx - 1 : face_send[5] - 1;              // backleft

    // 8 points
    point_send[0] = (px == rx - 1) ? edge_send[7] - rx + 1 : edge_send[7] + 1; // backrightdown
    point_send[1] = (px == rx - 1) ? edge_send[5] - rx + 1 : edge_send[5] + 1; // frontrightdown
    point_send[2] = (px == rx - 1) ? edge_send[4] - rx + 1 : edge_send[4] + 1; // frontrightup
    point_send[3] = (px == rx - 1) ? edge_send[6] - rx + 1 : edge_send[6] + 1; // backrightup
    point_send[4] = (px == 0) ? edge_send[4] + rx - 1 : edge_send[4] - 1;      // frontleftup
    point_send[5] = (px == 0) ? edge_send[7] + rx - 1 : edge_send[7] - 1;      // backleftdown
    point_send[6] = (px == 0) ? edge_send[5] + rx - 1 : edge_send[5] - 1;      // frontleftdown
    point_send[7] = (px == 0) ? edge_send[6] + rx - 1 : edge_send[6] - 1;      // backleftup

    // recv = opposite direction's sender
    face_recv[0] = face_send[1];
    face_recv[1] = face_send[0];
    face_recv[2] = face_send[3];
    face_recv[3] = face_send[2];
    face_recv[4] = face_send[5];
    face_recv[5] = face_send[4];

    edge_recv[0] = edge_send[3];
    edge_recv[1] = edge_send[2];
    edge_recv[2] = edge_send[1];
    edge_recv[3] = edge_send[0];
    edge_recv[4] = edge_send[7];
    edge_recv[5] = edge_send[6];
    edge_recv[6] = edge_send[5];
    edge_recv[7] = edge_send[4];
    edge_recv[8] = edge_send[11];
    edge_recv[9] = edge_send[10];
    edge_recv[10] = edge_send[9];
    edge_recv[11] = edge_send[8];

    point_recv[0] = point_send[4];
    point_recv[1] = point_send[7];
    point_recv[2] = point_send[5];
    point_recv[3] = point_send[6];
    point_recv[4] = point_send[0];
    point_recv[5] = point_send[2];
    point_recv[6] = point_send[3];
    point_recv[7] = point_send[1];
}


void LBM::setup_Local()
{
    l_l[0] = (px - glx % rx >= 0) ? glx / rx : glx / rx + 1;
    l_l[1] = (py - gly % ry >= 0) ? gly / ry : gly / ry + 1;
    l_l[2] = (pz - glz % rz >= 0) ? glz / rz : glz / rz + 1;
    // local length
    lx = l_l[0] + 2 * ghost;
    ly = l_l[1] + 2 * ghost;
    lz = l_l[2] + 2 * ghost;
    // local start
    l_s[0] = ghost;
    l_s[1] = ghost;
    l_s[2] = ghost;
    // local end
    l_e[0] = l_s[0] + l_l[0];
    l_e[1] = l_s[1] + l_l[1];
    l_e[2] = l_s[2] + l_l[2];

    auto block_lo = [](int gl, int r, int pc) {
        const int base = gl / r, rem = gl % r;
        return pc * base + (pc < rem ? pc : rem);
    };
    x_lo = block_lo(glx, rx, px);
    x_hi = x_lo + l_l[0] - 1;
    y_lo = block_lo(gly, ry, py);
    y_hi = y_lo + l_l[1] - 1;
    z_lo = block_lo(glz, rz, pz);
    z_hi = z_lo + l_l[2] - 1;
}

void LBM::setup_f()
{

    // 6 faces
    m_left = buffer_pack_f("m_left", q, 1, ly, lz);
    m_right = buffer_pack_f("m_right", q, 1, ly, lz);
    m_down = buffer_pack_f("m_down", q, lx, ly, 1);
    m_up = buffer_pack_f("m_up", q, lx, ly, 1);
    m_front = buffer_pack_f("m_front", q, lx, 1, lz);
    m_back = buffer_pack_f("m_back", q, lx, 1, lz);
    // 12 edges
    m_leftup = buffer_pack_f("m_leftup", q, 1, l_l[1], 1);
    m_rightup = buffer_pack_f("m_rightup", q, 1, l_l[1], 1);
    m_leftdown = buffer_pack_f("m_leftdown", q, 1, l_l[1], 1);
    m_rightdown = buffer_pack_f("m_rightdown", q, 1, l_l[1], 1);
    m_backleft = buffer_pack_f("m_backleft", q, 1, 1, l_l[2]);
    m_backright = buffer_pack_f("m_backright", q, 1, 1, l_l[2]);
    m_frontleft = buffer_pack_f("m_frontleft", q, 1, 1, l_l[2]);
    m_frontright = buffer_pack_f("m_frontright", q, 1, 1, l_l[2]); // OPT-T4
    m_backdown = buffer_pack_f("m_backdown", q, l_l[0], 1, 1);
    m_backup = buffer_pack_f("m_backup", q, l_l[0], 1, 1);
    m_frontdown = buffer_pack_f("m_frontdown", q, l_l[0], 1, 1);
    m_frontup = buffer_pack_f("m_frontup", q, l_l[0], 1, 1);
    // 8 corners
    m_frontleftdown = buffer_pack_f("m_fld", q, 1, 1, 1);
    m_frontrightdown = buffer_pack_f("m_frd", q, 1, 1, 1);
    m_frontleftup = buffer_pack_f("m_flu", q, 1, 1, 1);
    m_frontrightup = buffer_pack_f("m_fru", q, 1, 1, 1);
    m_backleftdown = buffer_pack_f("m_bld", q, 1, 1, 1);
    m_backrightdown = buffer_pack_f("m_brd", q, 1, 1, 1);
    m_backleftup = buffer_pack_f("m_blu", q, 1, 1, 1);
    m_backrightup = buffer_pack_f("m_bru", q, 1, 1, 1);

    // out direction
    m_leftout = buffer_pack_f("m_leftout", q, 1, ly, lz);
    m_rightout = buffer_pack_f("m_rightout", q, 1, ly, lz);
    m_downout = buffer_pack_f("m_downout", q, lx, ly, 1);
    m_upout = buffer_pack_f("m_upout", q, lx, ly, 1);
    m_frontout = buffer_pack_f("m_frontout", q, lx, 1, lz); // OPT-T4
    m_backout = buffer_pack_f("m_backout", q, lx, 1, lz);
    m_leftupout = buffer_pack_f("m_leftupout", q, 1, l_l[1], 1);
    m_rightupout = buffer_pack_f("m_rightupout", q, 1, l_l[1], 1);
    m_leftdownout = buffer_pack_f("m_leftdownout", q, 1, l_l[1], 1);
    m_rightdownout = buffer_pack_f("m_rightdownout", q, 1, l_l[1], 1);
    m_backleftout = buffer_pack_f("m_backleftout", q, 1, 1, l_l[2]);
    m_backrightout = buffer_pack_f("m_backrightout", q, 1, 1, l_l[2]);
    m_frontleftout = buffer_pack_f("m_frontleftout", q, 1, 1, l_l[2]);
    m_frontrightout = buffer_pack_f("m_frontrightout", q, 1, 1, l_l[2]); // OPT-T4
    m_backdownout = buffer_pack_f("m_backdownout", q, l_l[0], 1, 1);
    m_backupout = buffer_pack_f("m_backupout", q, l_l[0], 1, 1);
    m_frontdownout = buffer_pack_f("m_frontdownout", q, l_l[0], 1, 1);
    m_frontupout = buffer_pack_f("m_frontupout", q, l_l[0], 1, 1);
    m_frontleftdownout = buffer_pack_f("m_fldout", q, 1, 1, 1);
    m_frontrightdownout = buffer_pack_f("m_frdout", q, 1, 1, 1);
    m_frontleftupout = buffer_pack_f("m_fluout", q, 1, 1, 1);
    m_frontrightupout = buffer_pack_f("m_fruout", q, 1, 1, 1);
    m_backleftdownout = buffer_pack_f("m_bldout", q, 1, 1, 1);
    m_backrightdownout = buffer_pack_f("m_brdout", q, 1, 1, 1);
    m_backleftupout = buffer_pack_f("m_bluout", q, 1, 1, 1);
    m_backrightupout = buffer_pack_f("m_bruout", q, 1, 1, 1);
}

void LBM::pack_f(buffer_f ff)
{

    Kokkos::DefaultExecutionSpace ex;

    // 6 faces
    Kokkos::deep_copy(ex, m_leftout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(ghost, 1 + ghost), Kokkos::ALL, Kokkos::ALL));
    Kokkos::deep_copy(ex, m_rightout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(lx - ghost - 1, lx - ghost), Kokkos::ALL, Kokkos::ALL));
    Kokkos::deep_copy(ex, m_downout, Kokkos::subview(ff, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, std::make_pair(ghost, 1 + ghost)));
    Kokkos::deep_copy(ex, m_upout, Kokkos::subview(ff, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, std::make_pair(lz - ghost - 1, lz - ghost)));
    Kokkos::deep_copy(ex, m_frontout, Kokkos::subview(ff, Kokkos::ALL, Kokkos::ALL, std::make_pair(ghost, 1 + ghost), Kokkos::ALL));
    Kokkos::deep_copy(ex, m_backout, Kokkos::subview(ff, Kokkos::ALL, Kokkos::ALL, std::make_pair(ly - ghost - 1, ly - ghost), Kokkos::ALL));
    // 12 edges
    Kokkos::deep_copy(ex, m_leftupout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(ghost, 1 + ghost), std::make_pair(l_s[1], l_e[1]), std::make_pair(lz - ghost - 1, lz - ghost)));
    Kokkos::deep_copy(ex, m_rightupout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(lx - ghost - 1, lx - ghost), std::make_pair(l_s[1], l_e[1]), std::make_pair(lz - ghost - 1, lz - ghost)));
    Kokkos::deep_copy(ex, m_leftdownout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(ghost, 1 + ghost), std::make_pair(l_s[1], l_e[1]), std::make_pair(ghost, 1 + ghost)));
    Kokkos::deep_copy(ex, m_rightdownout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(lx - ghost - 1, lx - ghost), std::make_pair(l_s[1], l_e[1]), std::make_pair(ghost, 1 + ghost)));
    Kokkos::deep_copy(ex, m_frontleftout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(ghost, 1 + ghost), std::make_pair(ghost, 1 + ghost), std::make_pair(l_s[2], l_e[2])));
    Kokkos::deep_copy(ex, m_frontrightout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(lx - ghost - 1, lx - ghost), std::make_pair(ghost, 1 + ghost), std::make_pair(l_s[2], l_e[2])));
    Kokkos::deep_copy(ex, m_backleftout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(ghost, 1 + ghost), std::make_pair(ly - ghost - 1, ly - ghost), std::make_pair(l_s[2], l_e[2])));
    Kokkos::deep_copy(ex, m_backrightout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(lx - ghost - 1, lx - ghost), std::make_pair(ly - ghost - 1, ly - ghost), std::make_pair(l_s[2], l_e[2])));
    Kokkos::deep_copy(ex, m_frontupout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), std::make_pair(ghost, 1 + ghost), std::make_pair(lz - ghost - 1, lz - ghost)));
    Kokkos::deep_copy(ex, m_frontdownout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), std::make_pair(ghost, 1 + ghost), std::make_pair(ghost, 1 + ghost)));
    Kokkos::deep_copy(ex, m_backupout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), std::make_pair(ly - ghost - 1, ly - ghost), std::make_pair(lz - ghost - 1, lz - ghost)));
    Kokkos::deep_copy(ex, m_backdownout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), std::make_pair(ly - ghost - 1, ly - ghost), std::make_pair(ghost, 1 + ghost)));
    // 8 corners
    Kokkos::deep_copy(ex, m_frontleftdownout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(ghost, 1 + ghost), std::make_pair(ghost, 1 + ghost), std::make_pair(ghost, 1 + ghost)));
    Kokkos::deep_copy(ex, m_frontrightdownout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(lx - ghost - 1, lx - ghost), std::make_pair(ghost, 1 + ghost), std::make_pair(ghost, 1 + ghost)));
    Kokkos::deep_copy(ex, m_backleftdownout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(ghost, 1 + ghost), std::make_pair(ly - ghost - 1, ly - ghost), std::make_pair(ghost, 1 + ghost)));
    Kokkos::deep_copy(ex, m_backrightdownout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(lx - ghost - 1, lx - ghost), std::make_pair(ly - ghost - 1, ly - ghost), std::make_pair(ghost, 1 + ghost)));
    Kokkos::deep_copy(ex, m_frontleftupout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(ghost, 1 + ghost), std::make_pair(ghost, 1 + ghost), std::make_pair(lz - ghost - 1, lz - ghost)));
    Kokkos::deep_copy(ex, m_frontrightupout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(lx - ghost - 1, lx - ghost), std::make_pair(ghost, 1 + ghost), std::make_pair(lz - ghost - 1, lz - ghost)));
    Kokkos::deep_copy(ex, m_backleftupout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(ghost, 1 + ghost), std::make_pair(ly - ghost - 1, ly - ghost), std::make_pair(lz - ghost - 1, lz - ghost)));
    Kokkos::deep_copy(ex, m_backrightupout, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(lx - ghost - 1, lx - ghost), std::make_pair(ly - ghost - 1, ly - ghost), std::make_pair(lz - ghost - 1, lz - ghost)));

    ex.fence(); // MPI reads these device buffers next
}

void LBM::exchange_f()
{

    MPI_Request rs[26], rr[26];

    const buffer_pack_f fout[6] = {m_leftout, m_rightout, m_downout, m_upout, m_frontout, m_backout};
    const buffer_pack_f fin[6] = {m_left, m_right, m_down, m_up, m_front, m_back};
    const int fopp[6] = {1, 0, 3, 2, 5, 4};

    const buffer_pack_f eout[12] = {m_leftupout, m_rightupout, m_leftdownout, m_rightdownout,
                                    m_frontupout, m_frontdownout, m_backupout, m_backdownout,
                                    m_frontrightout, m_frontleftout, m_backrightout, m_backleftout};
    const buffer_pack_f ein[12] = {m_leftup, m_rightup, m_leftdown, m_rightdown,
                                   m_frontup, m_frontdown, m_backup, m_backdown,
                                   m_frontright, m_frontleft, m_backright, m_backleft};
    const int eopp[12] = {3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8};

    const buffer_pack_f pout[8] = {m_backrightdownout, m_frontrightdownout, m_frontrightupout, m_backrightupout,
                                   m_frontleftupout, m_backleftdownout, m_frontleftdownout, m_backleftupout};
    const buffer_pack_f pin[8] = {m_backrightdown, m_frontrightdown, m_frontrightup, m_backrightup,
                                  m_frontleftup, m_backleftdown, m_frontleftdown, m_backleftup};
    const int popp[8] = {4, 7, 5, 6, 0, 2, 3, 1};

    int ch = 0;
    for (int d = 0; d < 6; ++d, ++ch)
    {
        MPI_Isend(fout[d].data(), fout[d].size(), MPI_DOUBLE, face_send[d], ch, comm, &rs[ch]);
        MPI_Irecv(fin[fopp[d]].data(), fin[fopp[d]].size(), MPI_DOUBLE, face_recv[d], ch, comm, &rr[ch]);
    }
    for (int d = 0; d < 12; ++d, ++ch)
    {
        MPI_Isend(eout[d].data(), eout[d].size(), MPI_DOUBLE, edge_send[d], ch, comm, &rs[ch]);
        MPI_Irecv(ein[eopp[d]].data(), ein[eopp[d]].size(), MPI_DOUBLE, edge_recv[d], ch, comm, &rr[ch]);
    }
    for (int d = 0; d < 8; ++d, ++ch)
    {
        MPI_Isend(pout[d].data(), pout[d].size(), MPI_DOUBLE, point_send[d], ch, comm, &rs[ch]);
        MPI_Irecv(pin[popp[d]].data(), pin[popp[d]].size(), MPI_DOUBLE, point_recv[d], ch, comm, &rr[ch]);
    }

    MPI_Waitall(26, rr, MPI_STATUSES_IGNORE);
    MPI_Waitall(26, rs, MPI_STATUSES_IGNORE);
}

void LBM::unpack_f(buffer_f ff)
{
    Kokkos::DefaultExecutionSpace ex;

    // 6 faces
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(ghost - 1, ghost), Kokkos::ALL, Kokkos::ALL), m_left);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(lx - ghost, lx - ghost + 1), Kokkos::ALL, Kokkos::ALL), m_right);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, std::make_pair(ghost - 1, ghost)), m_down);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, std::make_pair(lz - ghost, lz - ghost + 1)), m_up);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, Kokkos::ALL, std::make_pair(ghost - 1, ghost), Kokkos::ALL), m_front);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, Kokkos::ALL, std::make_pair(ly - ghost, ly - ghost + 1), Kokkos::ALL), m_back);
    // 12 edges
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(ghost - 1, ghost), std::make_pair(l_s[1], l_e[1]), std::make_pair(lz - ghost, lz - ghost + 1)), m_leftup);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(lx - ghost, lx - ghost + 1), std::make_pair(l_s[1], l_e[1]), std::make_pair(lz - ghost, lz - ghost + 1)), m_rightup);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(ghost - 1, ghost), std::make_pair(l_s[1], l_e[1]), std::make_pair(ghost - 1, ghost)), m_leftdown);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(lx - ghost, lx - ghost + 1), std::make_pair(l_s[1], l_e[1]), std::make_pair(ghost - 1, ghost)), m_rightdown);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(ghost - 1, ghost), std::make_pair(ghost - 1, ghost), std::make_pair(l_s[2], l_e[2])), m_frontleft);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(lx - ghost, lx - ghost + 1), std::make_pair(ghost - 1, ghost), std::make_pair(l_s[2], l_e[2])), m_frontright);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(ghost - 1, ghost), std::make_pair(ly - ghost, ly - ghost + 1), std::make_pair(l_s[2], l_e[2])), m_backleft);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(lx - ghost, lx - ghost + 1), std::make_pair(ly - ghost, ly - ghost + 1), std::make_pair(l_s[2], l_e[2])), m_backright);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), std::make_pair(ghost - 1, ghost), std::make_pair(lz - ghost, lz - ghost + 1)), m_frontup);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), std::make_pair(ghost - 1, ghost), std::make_pair(ghost - 1, ghost)), m_frontdown);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), std::make_pair(ly - ghost, ly - ghost + 1), std::make_pair(lz - ghost, lz - ghost + 1)), m_backup);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(l_s[0], l_e[0]), std::make_pair(ly - ghost, ly - ghost + 1), std::make_pair(ghost - 1, ghost)), m_backdown);
    // 8 corners
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(ghost - 1, ghost), std::make_pair(ghost - 1, ghost), std::make_pair(ghost - 1, ghost)), m_frontleftdown);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(lx - ghost, lx - ghost + 1), std::make_pair(ghost - 1, ghost), std::make_pair(ghost - 1, ghost)), m_frontrightdown);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(ghost - 1, ghost), std::make_pair(ly - ghost, ly - ghost + 1), std::make_pair(ghost - 1, ghost)), m_backleftdown);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(lx - ghost, lx - ghost + 1), std::make_pair(ly - ghost, ly - ghost + 1), std::make_pair(ghost - 1, ghost)), m_backrightdown);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(ghost - 1, ghost), std::make_pair(ghost - 1, ghost), std::make_pair(lz - ghost, lz - ghost + 1)), m_frontleftup);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(lx - ghost, lx - ghost + 1), std::make_pair(ghost - 1, ghost), std::make_pair(lz - ghost, lz - ghost + 1)), m_frontrightup);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(ghost - 1, ghost), std::make_pair(ly - ghost, ly - ghost + 1), std::make_pair(lz - ghost, lz - ghost + 1)), m_backleftup);
    Kokkos::deep_copy(ex, Kokkos::subview(ff, Kokkos::ALL, std::make_pair(lx - ghost, lx - ghost + 1), std::make_pair(ly - ghost, ly - ghost + 1), std::make_pair(lz - ghost, lz - ghost + 1)), m_backrightup);

}

void LBM::passf(buffer_f ff)
{
    pack_f(ff);
    exchange_f();
    unpack_f(ff);
}
