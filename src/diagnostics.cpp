#include "diagnostics.hpp"

double total_mass(const Particles &p) {
    double m = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        m += p.mass[i];
    }
    return m;
}

Vec3 total_momentum(const Particles &p) {
    Vec3 P{0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < p.size(); ++i) {
        P.x += p.mass[i] * p.vx[i];
        P.y += p.mass[i] * p.vy[i];
        P.z += p.mass[i] * p.vz[i];
    }
    return P;
}

Vec3 total_angular_momentum(const Particles &p) {
    Vec3 L{0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < p.size(); ++i) {
        L.x += p.mass[i] * (p.y[i] * p.vz[i] - p.z[i] * p.vy[i]);
        L.y += p.mass[i] * (p.z[i] * p.vx[i] - p.x[i] * p.vz[i]);
        L.z += p.mass[i] * (p.x[i] * p.vy[i] - p.y[i] * p.vx[i]);
    }
    return L;
}

double kinetic_energy(const Particles &p) {
    double ke = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        double v2 = p.vx[i] * p.vx[i] + p.vy[i] * p.vy[i] + p.vz[i] * p.vz[i];
        ke += 0.5 * p.mass[i] * v2;
    }
    return ke;
}

double total_heat(const Particles &p) {
    double q = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        q += p.heat[i];
    }
    return q;
}
