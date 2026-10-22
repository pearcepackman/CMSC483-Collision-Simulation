#include <gtest/gtest.h>

#include <cmath>

#include "diagnostics.hpp"
#include "forces.hpp"
#include "gravity.hpp"
#include "integrator.hpp"
#include "params.hpp"
#include "particles.hpp"

namespace {

const double kPi = 3.14159265358979323846;
const int kStepsPerOrbit = 1000;
const int kOrbits = 1000;
const int kCheckEvery = 10; // check energy/separation once every 10 steps

// Two masses of 0.5, separation 1, each moving at 0.5 around the center of mass in
// opposite directions. With G = 1 that's an exactly circular orbit:
// centripetal v^2/r = 0.25/0.5 = 0.5 = m/d^2, and period 2*pi.
Particles circular_pair() {
    Particles p;
    p.resize(2);
    p.mass[0] = 0.5;
    p.mass[1] = 0.5;
    p.x[0] = -0.5;
    p.x[1] = 0.5;
    p.vy[0] = -0.5;
    p.vy[1] = 0.5;
    return p;
}

struct OrbitStats {
    double max_energy_err = 0.0; // max |E - E0| / |E0| over the whole run
    double max_err_first = 0.0;  // ... over the first 100 orbits
    double max_err_last = 0.0;   // ... over the last 100 orbits
    double max_sep_err = 0.0;    // max |separation - 1|
    double max_momentum = 0.0;   // max |total momentum|
};

double total_energy(const Particles &p, const SimParams &params) {
    return kinetic_energy(p) + potential_energy(p, params);
}

// Runs the orbit once with leapfrog and records how far it strays.
OrbitStats run_orbit() {
    Particles p = circular_pair();
    SimParams params;
    params.dt = 2.0 * kPi / kStepsPerOrbit;
    params.softening = 0.0; // the particles never get close
    params.contact_enabled = false;

    compute_accelerations(p, params, BACKEND_SERIAL); // leapfrog needs a(t=0)
    const double e0 = total_energy(p, params);

    OrbitStats s;
    const int total = kStepsPerOrbit * kOrbits;
    for (int step = 1; step <= total; ++step) {
        leapfrog_step(p, params, BACKEND_SERIAL);
        if (step % kCheckEvery != 0) {
            continue;
        }
        double err = std::fabs(total_energy(p, params) - e0) / std::fabs(e0);
        s.max_energy_err = std::fmax(s.max_energy_err, err);
        int orbit = step / kStepsPerOrbit;
        if (orbit < 100) {
            s.max_err_first = std::fmax(s.max_err_first, err);
        }
        if (orbit >= kOrbits - 100) {
            s.max_err_last = std::fmax(s.max_err_last, err);
        }
        double dx = p.x[1] - p.x[0], dy = p.y[1] - p.y[0], dz = p.z[1] - p.z[0];
        s.max_sep_err =
            std::fmax(s.max_sep_err, std::fabs(std::sqrt(dx * dx + dy * dy + dz * dz) - 1.0));
        Vec3 mom = total_momentum(p);
        s.max_momentum =
            std::fmax(s.max_momentum, std::sqrt(mom.x * mom.x + mom.y * mom.y + mom.z * mom.z));
    }
    return s;
}

// The run takes 10^6 steps, so do it once and share the stats across the tests
// (only matters when several tests run in one process, e.g. the sim_tests binary).
const OrbitStats &orbit_stats() {
    static const OrbitStats s = run_orbit();
    return s;
}

} // namespace

TEST(TwoBodyOrbit, EnergyErrorStaysBelow1e5Over1000Orbits) {
    const OrbitStats &s = orbit_stats();
    EXPECT_LT(s.max_energy_err, 1e-5);
}

TEST(TwoBodyOrbit, EnergyErrorDoesNotDrift) {
    // Leapfrog's energy error wobbles but stays bounded; a method that leaks or gains
    // energy each orbit would have a much bigger error at the end than at the start.
    const OrbitStats &s = orbit_stats();
    EXPECT_GT(s.max_err_first, 0.0); // sanity: the check measured something
    EXPECT_LT(s.max_err_last, 2.0 * s.max_err_first)
        << "first 100 orbits: " << s.max_err_first << ", last 100: " << s.max_err_last;
}

TEST(TwoBodyOrbit, OrbitStaysCircular) {
    const OrbitStats &s = orbit_stats();
    EXPECT_LT(s.max_sep_err, 1e-3);
}

TEST(TwoBodyOrbit, MomentumStaysZero) {
    // Starts at exactly zero. The two particles are exact mirror images (equal masses
    // of 0.5, a power of two), so their momenta cancel bit-for-bit; this catches bugs
    // that treat the particles differently. Bound kept as <= per CLAUDE.md.
    // Scale: the size of each particle's momentum (0.5 * 0.5).
    const OrbitStats &s = orbit_stats();
    EXPECT_LE(s.max_momentum, 1e-12 * 0.25);
}
