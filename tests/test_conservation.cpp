#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <omp.h>

#include "contact.hpp"
#include "diagnostics.hpp"
#include "forces.hpp"
#include "gravity.hpp"
#include "init.hpp"
#include "integrator.hpp"
#include "params.hpp"
#include "particles.hpp"
#include "sim.hpp"

// Conservation and determinism (#16), and the contact spring energy that makes the
// energy log complete (#59). Physics bugs that look fine on screen show up here: a
// force that isn't equal and opposite moves the momentum, a non-central force spins the
// system, and a missing energy term makes the total climb.

namespace {

SimParams contact_params(double k, double c) {
    SimParams params;
    params.contact_enabled = true;
    params.contact_k = k;
    params.contact_c = c;
    return params;
}

// Everything energy.csv's total counts.
double logged_total(const Particles &p, const SimParams &params) {
    return kinetic_energy(p) + potential_energy(p, params) + contact_energy(p, params) +
           total_heat(p);
}

// Two particles on the x axis.
Particles pair_on_x(double x0, double x1, double r, double m) {
    Particles p;
    p.resize(2);
    p.x[0] = x0;
    p.x[1] = x1;
    p.radius[0] = p.radius[1] = r;
    p.mass[0] = p.mass[1] = m;
    return p;
}

// Two unrelaxed 150-particle planets that touch after about 200 steps and are pressed
// together for the rest of the run: smoke.cfg's settings (contact k = 20, c = 0.05,
// dt = 0.002), smaller. `offset` moves planet 1 sideways for an off-center hit.
void two_planets(Particles &p, SimParams &params, double offset) {
    params = contact_params(20.0, 0.05);
    params.dt = 0.002;
    params.softening = 0.05;
    const double r = 0.5 * std::cbrt(150.0 / 250.0); // same density as smoke.cfg
    PlanetSpec a;
    a.n = 150;
    a.radius = r;
    a.particle_radius = 0.06;
    a.mass = 0.15;
    a.center = Vec3{-r - 0.05, 0.0, 0.0};
    a.velocity = Vec3{0.25, 0.0, 0.0};
    a.seed = 1;
    a.body = 0;
    PlanetSpec b = a;
    b.center = Vec3{r + 0.05, offset, 0.0};
    b.velocity = Vec3{-0.25, 0.0, 0.0};
    b.seed = 2;
    b.body = 1;
    p = Particles();
    make_planet(p, a);
    make_planet(p, b);
}

const int kCollisionSteps = 300;

// True if some particle of planet 0 overlaps some particle of planet 1. Contact energy
// or heat alone can't show the planets collided: neighbors inside one planet already
// touch.
bool planets_touch(const Particles &p) {
    for (std::size_t i = 0; i < p.size(); ++i) {
        for (std::size_t j = 0; j < p.size(); ++j) {
            if (p.body[i] != 0 || p.body[j] != 1) {
                continue;
            }
            double dx = p.x[i] - p.x[j], dy = p.y[i] - p.y[j], dz = p.z[i] - p.z[j];
            double reach = p.radius[i] + p.radius[j];
            if (dx * dx + dy * dy + dz * dz < reach * reach) {
                return true;
            }
        }
    }
    return false;
}

} // namespace

// ---- Contact spring energy (#59) ----

TEST(ContactEnergy, HandComputed) {
    const SimParams params = contact_params(20.0, 0.5);
    // Radii 0.5, centers 0.8 apart: overlap 0.2, so 0.5 * 20 * 0.04 = 0.4.
    Particles p = pair_on_x(0.0, 0.8, 0.5, 1.0);
    EXPECT_NEAR(contact_energy(p, params), 0.4, 1e-15);

    Particles apart = pair_on_x(0.0, 1.5, 0.5, 1.0);
    EXPECT_EQ(contact_energy(apart, params), 0.0);
    Particles touching = pair_on_x(0.0, 1.0, 0.5, 1.0); // exactly touching: no force either
    EXPECT_EQ(contact_energy(touching, params), 0.0);
    Particles same = pair_on_x(0.3, 0.3, 0.5, 1.0); // no direction to push: no force either
    EXPECT_EQ(contact_energy(same, params), 0.0);
    EXPECT_EQ(contact_energy(p, SimParams()), 0.0); // contact off

    Particles none, one;
    one.resize(1);
    one.radius[0] = 1.0;
    one.mass[0] = 1.0;
    EXPECT_EQ(contact_energy(none, params), 0.0);
    EXPECT_EQ(contact_energy(one, params), 0.0);

    // Three particles, only the first two overlap: that pair counts once, not twice.
    Particles three = pair_on_x(0.0, 0.8, 0.5, 1.0);
    three.resize(3);
    three.x[2] = 10.0;
    three.radius[2] = 0.5;
    three.mass[2] = 1.0;
    EXPECT_NEAR(contact_energy(three, params), 0.4, 1e-15);
}

namespace {

// The head-on bounce from test_contact.cpp, with the logged total checked at every
// step, including while the spring is compressed. Returns the largest rise above the
// start and the largest drop below it, both relative to the starting kinetic energy.
void bounce_total_extremes(double c, double *max_rise, double *max_drop) {
    Particles p = pair_on_x(-0.6, 0.6, 0.5, 1.0);
    p.vx[0] = 1.0;
    p.vx[1] = -1.0;
    SimParams params = contact_params(1e4, c);
    params.dt = 1e-4;
    params.softening = 0.0;
    compute_accelerations(p, params, BACKEND_SERIAL);
    const double ke0 = kinetic_energy(p);
    const double e0 = logged_total(p, params);
    *max_rise = 0.0;
    *max_drop = 0.0;
    for (int step = 0; step < 3000; ++step) {
        leapfrog_step(p, params, BACKEND_SERIAL);
        double d = (logged_total(p, params) - e0) / ke0;
        *max_rise = std::fmax(*max_rise, d);
        *max_drop = std::fmax(*max_drop, -d);
    }
}

} // namespace

TEST(ContactEnergy, UndampedBounceTotalStaysPut) {
    // Spring energy fills the gap while the pair is pressed together, so the total holds
    // at every step, not only after the bounce. Measured: up 1.3e-5, down 4.6e-5 (of the
    // starting kinetic energy). Without the spring term the total would drop by up to
    // the whole kinetic energy mid-bounce.
    double rise = 0.0, drop = 0.0;
    bounce_total_extremes(0.0, &rise, &drop);
    EXPECT_LT(rise, 1e-4);
    EXPECT_LT(drop, 1e-4);
}

TEST(ContactEnergy, DampedBounceTotalNeverClimbs) {
    // Restitution 0.5. The total jumps by up to 7e-3 of the starting kinetic energy for
    // a step as the damper switches on at full strength at first contact (its heat is
    // booked from that step's velocities), then settles back to ~1e-5. The bound has
    // ~3x margin; counting each pair twice or with the wrong sign moves the total by
    // about the whole kinetic energy.
    const double kPi = 3.14159265358979323846;
    const double zeta = -std::log(0.5) / std::sqrt(kPi * kPi + std::log(0.5) * std::log(0.5));
    double rise = 0.0, drop = 0.0;
    bounce_total_extremes(2.0 * zeta * std::sqrt(0.5 * 1e4), &rise, &drop);
    EXPECT_LT(rise, 2e-2);
    EXPECT_LT(drop, 2e-2);
}

// ---- Conservation through a collision (#16) ----

TEST(Conservation, MomentumThroughCollision) {
    Particles p;
    SimParams params;
    two_planets(p, params, 0.0);
    ASSERT_EQ(sim_init(p, params, BACKEND_SERIAL), p.size());
    const Vec3 p0 = total_momentum(p);
    double scale = 0.0; // sum of |m v|: what rounding errors are measured against
    for (std::size_t i = 0; i < p.size(); ++i) {
        scale += p.mass[i] * std::sqrt(p.vx[i] * p.vx[i] + p.vy[i] * p.vy[i] + p.vz[i] * p.vz[i]);
    }
    for (int s = 0; s < kCollisionSteps; ++s) {
        ASSERT_EQ(sim_step(p, params, BACKEND_SERIAL), p.size()) << "step " << s;
    }
    ASSERT_TRUE(planets_touch(p)); // the planets really are pressed together
    // Contact forces cancel exactly; gravity's pairs are summed separately on each side,
    // so they cancel only to rounding. Measured 1e-15 of the scale.
    const Vec3 p1 = total_momentum(p);
    EXPECT_LE(std::fabs(p1.x - p0.x), 1e-13 * scale);
    EXPECT_LE(std::fabs(p1.y - p0.y), 1e-13 * scale);
    EXPECT_LE(std::fabs(p1.z - p0.z), 1e-13 * scale);
}

TEST(Conservation, AngularMomentumThroughOffCenterCollision) {
    Particles p;
    SimParams params;
    two_planets(p, params, 0.3); // off-center: the merged pair starts to spin
    ASSERT_EQ(sim_init(p, params, BACKEND_SERIAL), p.size());
    const Vec3 l0 = total_angular_momentum(p);
    double scale = 0.0; // sum of |m (r x v)|
    for (std::size_t i = 0; i < p.size(); ++i) {
        double cx = p.y[i] * p.vz[i] - p.z[i] * p.vy[i];
        double cy = p.z[i] * p.vx[i] - p.x[i] * p.vz[i];
        double cz = p.x[i] * p.vy[i] - p.y[i] * p.vx[i];
        scale += p.mass[i] * std::sqrt(cx * cx + cy * cy + cz * cz);
    }
    ASSERT_GT(std::fabs(l0.z), 0.01 * scale); // there is real spin to keep
    for (int s = 0; s < kCollisionSteps; ++s) {
        ASSERT_EQ(sim_step(p, params, BACKEND_SERIAL), p.size()) << "step " << s;
    }
    ASSERT_TRUE(planets_touch(p));
    // Gravity and contact both push along the line between two particles, so they can't
    // change the spin. Measured 9e-16 of the scale.
    const Vec3 l1 = total_angular_momentum(p);
    EXPECT_LE(std::fabs(l1.x - l0.x), 1e-12 * scale);
    EXPECT_LE(std::fabs(l1.y - l0.y), 1e-12 * scale);
    EXPECT_LE(std::fabs(l1.z - l0.z), 1e-12 * scale);
}

TEST(Conservation, EnergyHoldsThroughCollision) {
    Particles p;
    SimParams params;
    two_planets(p, params, 0.3);
    ASSERT_EQ(sim_init(p, params, BACKEND_SERIAL), p.size());
    const double e0 = logged_total(p, params);
    double max_rise = 0.0, max_drop = 0.0;
    for (int s = 0; s < kCollisionSteps; ++s) {
        ASSERT_EQ(sim_step(p, params, BACKEND_SERIAL), p.size()) << "step " << s;
        double d = logged_total(p, params) - e0;
        max_rise = std::fmax(max_rise, d);
        max_drop = std::fmax(max_drop, -d);
    }
    ASSERT_TRUE(planets_touch(p));
    // Kinetic + potential + springs + heat should hold, both ways. Measured: up to 7.3e-5
    // above the start and 7.6e-5 below it, 0.1% of |E0|: an error that appears as
    // contacts begin and shrinks roughly in proportion to dt (docs/notes.md). A spring
    // term counted twice pushes the total up and a missing one (or heat that isn't
    // booked) pulls it down, by much more than these bounds.
    EXPECT_LT(max_rise, 1.5e-4) << "E0 = " << e0;
    EXPECT_LT(max_drop, 1.5e-4) << "E0 = " << e0;
}

// ---- Determinism (#16) ----

namespace {

Particles run_collision(Backend backend, int threads) {
    Particles p;
    SimParams params;
    two_planets(p, params, 0.3);
    int saved = omp_get_max_threads();
    omp_set_num_threads(threads);
    sim_init(p, params, backend);
    for (int s = 0; s < 250; ++s) {
        sim_step(p, params, backend);
    }
    omp_set_num_threads(saved);
    return p;
}

void expect_same_bits(const Particles &a, const Particles &b) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) { // same inputs and order: identical bits
        ASSERT_EQ(a.x[i], b.x[i]) << "particle " << i;
        ASSERT_EQ(a.y[i], b.y[i]) << "particle " << i;
        ASSERT_EQ(a.z[i], b.z[i]) << "particle " << i;
        ASSERT_EQ(a.vx[i], b.vx[i]) << "particle " << i;
        ASSERT_EQ(a.heat[i], b.heat[i]) << "particle " << i;
    }
}

} // namespace

// `==` is right here: these compare a run with itself (same backend, same inputs), not
// two backends, which CLAUDE.md says to compare with tolerances.
TEST(Determinism, SameSeedSameBits) {
    expect_same_bits(run_collision(BACKEND_SERIAL, 1), run_collision(BACKEND_SERIAL, 1));
}

TEST(Determinism, OpenMpThreadCountDoesNotChangeBits) {
    // Each particle's gravity is summed by one thread in a fixed order, so the thread
    // count can't change the result.
    expect_same_bits(run_collision(BACKEND_OMP, 1), run_collision(BACKEND_OMP, 4));
}
