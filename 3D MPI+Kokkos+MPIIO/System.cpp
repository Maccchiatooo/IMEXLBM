#include "System.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

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
