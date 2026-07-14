#ifndef _LBM_H_
#define _LBM_H_

#include <cmath>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Timer.hpp>
#include <mpi.h>

#define q 27
#define dim 3
#define ghost 3

typedef Kokkos::TeamPolicy<> team_policy;
typedef Kokkos::TeamPolicy<>::member_type member_type;
typedef Kokkos::RangePolicy<> range_policy;
typedef Kokkos::MDRangePolicy<Kokkos::Rank<2>> mdrange_policy2;
typedef Kokkos::MDRangePolicy<Kokkos::Rank<3>> mdrange_policy3;
typedef Kokkos::MDRangePolicy<Kokkos::Rank<4>> mdrange_policy4;

using buffer_f = Kokkos::View<double ****, Kokkos::CudaSpace>;
using buffer_u = Kokkos::View<double ***, Kokkos::CudaSpace>;
using buffer_div = Kokkos::View<double ***, Kokkos::CudaSpace>;
using buffer_pack_f = Kokkos::View<double ****, Kokkos::CudaSpace>;
using buffer_pack_u = Kokkos::View<double ***, Kokkos::CudaSpace>;


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
    Kokkos::View<int ***, Kokkos::CudaSpace> usr, ran;
    // bounce back notation
    Kokkos::View<int *, Kokkos::CudaSpace> bb;
    // weight function
    Kokkos::View<double *, Kokkos::CudaSpace> t;
    // discrete velocity
    Kokkos::View<int **, Kokkos::CudaSpace> e;

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
    void Update();
    void Update1();
    void MPIoutput(int n);
    void Output(int n);

    void pack_f(buffer_f ff);
    void unpack_f(buffer_f ff);

    void exchange_f();

    void passf(buffer_f ff);
};
#endif
