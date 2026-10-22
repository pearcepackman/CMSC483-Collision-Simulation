#include "gravity.hpp"

#include <cmath>

// OpenMP all-pairs gravity: the serial reference's loop with the outer loop split across
// threads. Each thread owns a block of particles i and is the only one writing their
// ax/ay/az[i]; positions and masses are only read. Every pair is computed twice (once per
// side), so there are no shared writes and no atomics (#31 compares the alternative).
// Each i is summed by one thread over j in the same order as gravity_serial.cpp, so the
// results match the reference.
void compute_gravity_omp(Particles &p, const SimParams &params) {
    const std::size_t n = p.size();
    const double eps2 = params.softening * params.softening;
    const double *x = p.x.data(), *y = p.y.data(), *z = p.z.data(), *m = p.mass.data();
    double *ax = p.ax.data(), *ay = p.ay.data(), *az = p.az.data();

    // static: every particle does the same amount of work, so equal blocks balance.
#pragma omp parallel for schedule(static) default(none) shared(n, eps2, x, y, z, m, ax, ay, az)
    for (std::size_t i = 0; i < n; ++i) {
        double axi = 0.0, ayi = 0.0, azi = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            double dx = x[j] - x[i];
            double dy = y[j] - y[i];
            double dz = z[j] - z[i];
            double r2 = dx * dx + dy * dy + dz * dz + eps2;
            double inv_r3 = 1.0 / (r2 * std::sqrt(r2)); // 1 / (|r|^2 + eps^2)^(3/2)
            axi += m[j] * dx * inv_r3;
            ayi += m[j] * dy * inv_r3;
            azi += m[j] * dz * inv_r3;
        }
        ax[i] = axi;
        ay[i] = ayi;
        az[i] = azi;
    }
}
