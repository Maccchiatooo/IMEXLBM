#include "System.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

// OPT-S1: the constructor was fully commented out, leaving sx/sy/sz, Time,
// inter, tau, rho0, u0 as UNINITIALIZED garbage — the program could not run
// meaningfully. Restored the input.in read and the derived quantities.
// Taylor-Green convention: L = sx / (2*pi), nu = u0*L/Re, tau = nu/cs2.

System::System()
{
    cs2 = 1.0 / 3.0;
    cs = std::sqrt(cs2);

    std::ifstream input("input.in");
    if (!input)
    {
        std::cerr << "FATAL: cannot open input.in" << std::endl;
        std::abort();
    }
    input >> rho0 >> R >> Re;
    input >> u0 >> Time >> inter;
    input >> sx >> sy >> sz;

    Ma = u0 / cs;
    miu = rho0 * u0 * sx / 2.0 / 3.1415926 / Re;
    tau = u0 * sx / Re / cs2 / 2.0 / 3.1415926;
}

void System::Monitor()
{
    std::cout << "============================" << std::endl
              << "3D Taylor-Green Vortex" << std::endl
              << "Re    = " << Re << std::endl
              << "Ma    = " << Ma << std::endl
              << "rho   = " << rho0 << std::endl
              << "miu   = " << miu << std::endl
              << "tau   = " << tau << std::endl
              << "Time  = " << Time << std::endl
              << "inter = " << inter << std::endl
              << "nx,ny,nz = " << sx << " " << sy << " " << sz << std::endl
              << "============================" << std::endl;
};
