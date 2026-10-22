#include "gravity.hpp"

#include <cmath>

// Reference implementation: every other backend is tested against this file.
// Keep it simple and never "optimize" it.

void compute_gravity_serial(Particles &p, const SimParams &params) {
    const std::size_t n = p.size();
    const double eps2 = params.softening * params.softening;

    // Every pair is computed twice (once from each side): each particle sums the
    // pull of all the others, in the same order the parallel backends will use.
    for (std::size_t i = 0; i < n; ++i) {
        double axi = 0.0, ayi = 0.0, azi = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            double dx = p.x[j] - p.x[i];
            double dy = p.y[j] - p.y[i];
            double dz = p.z[j] - p.z[i];
            double r2 = dx * dx + dy * dy + dz * dz + eps2;
            double inv_r3 = 1.0 / (r2 * std::sqrt(r2)); // 1 / (|r|^2 + eps^2)^(3/2)
            axi += p.mass[j] * dx * inv_r3;
            ayi += p.mass[j] * dy * inv_r3;
            azi += p.mass[j] * dz * inv_r3;
        }
        p.ax[i] = axi;
        p.ay[i] = ayi;
        p.az[i] = azi;
    }
}

double potential_energy(const Particles &p, const SimParams &params) {
    const std::size_t n = p.size();
    const double eps2 = params.softening * params.softening;

    double u = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            double dx = p.x[j] - p.x[i];
            double dy = p.y[j] - p.y[i];
            double dz = p.z[j] - p.z[i];
            u -= p.mass[i] * p.mass[j] / std::sqrt(dx * dx + dy * dy + dz * dz + eps2);
        }
    }
    return u;
}
