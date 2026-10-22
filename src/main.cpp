#include <cstdio>

#include <omp.h>

#include "diagnostics.hpp"
#include "particles.hpp"

// Milestone 0 placeholder: proves the build, OpenMP linkage, and core library work.
int main() {
    std::printf("planet_sim 0.1.0\n");
    std::printf("OpenMP max threads: %d\n", omp_get_max_threads());

    Particles p;
    p.resize(2);
    p.mass[0] = 1.0;
    p.vx[0] = 1.0;
    p.mass[1] = 1.0;
    p.vx[1] = -1.0;

    Vec3 P = total_momentum(p);
    std::printf("particles: %zu  total mass: %g  momentum: (%g, %g, %g)  KE: %g\n", p.size(),
                total_mass(p), P.x, P.y, P.z, kinetic_energy(p));
    return 0;
}
