#include <gtest/gtest.h>

#include <cmath>

#include "contact.hpp"
#include "diagnostics.hpp"
#include "forces.hpp"
#include "gravity.hpp"
#include "integrator.hpp"
#include "params.hpp"
#include "particles.hpp"

namespace {

const double kPi = 3.14159265358979323846;

SimParams contact_params(double k, double c) {
    SimParams params;
    params.contact_enabled = true;
    params.contact_k = k;
    params.contact_c = c;
    return params;
}

// Two particles on the x axis: particle 0 at x0, particle 1 at x1.
Particles pair_on_x(double x0, double x1, double r0, double r1, double m0, double m1) {
    Particles p;
    p.resize(2);
    p.x[0] = x0;
    p.x[1] = x1;
    p.radius[0] = r0;
    p.radius[1] = r1;
    p.mass[0] = m0;
    p.mass[1] = m1;
    return p;
}

double total_energy(const Particles &p, const SimParams &params) {
    return kinetic_energy(p) + potential_energy(p, params);
}

} // namespace

TEST(Contact, SeparatedPairGetsNoForce) {
    // Gap of 0.1 between the surfaces. Starting accelerations are nonzero so the test
    // also checks contact adds to them instead of overwriting.
    Particles p = pair_on_x(0.0, 1.1, 0.5, 0.5, 1.0, 1.0);
    p.vx[0] = 1.0; // approaching, but not touching yet: the damper must not act either
    p.ax[0] = 0.25;
    p.ay[1] = -0.5;
    compute_contact_bruteforce(p, contact_params(100.0, 3.0));
    EXPECT_EQ(p.ax[0], 0.25);
    EXPECT_EQ(p.ay[1], -0.5);
    EXPECT_EQ(p.ax[1], 0.0);
    EXPECT_EQ(p.az[0], 0.0);
}

TEST(Contact, ExactlyTouchingPairGetsNoForce) {
    Particles p = pair_on_x(0.0, 1.0, 0.5, 0.5, 1.0, 1.0); // d == r0 + r1
    // Approaching: at zero overlap the spring is 0 anyway, so only the damper would
    // show it if touching counted as contact.
    p.vx[0] = 1.0;
    p.vx[1] = -1.0;
    compute_contact_bruteforce(p, contact_params(100.0, 3.0));
    EXPECT_EQ(p.ax[0], 0.0);
    EXPECT_EQ(p.ax[1], 0.0);
}

TEST(Contact, OverlapPushesApartEqualAndOpposite) {
    // Different radii and masses, along u = (1,2,3)/sqrt(14) so every component differs
    // (a swapped x/y/z would show). Radii 0.3 + 0.7 = 1.0. Two overlaps, so the test
    // pins the spring as linear, not just its value at one point.
    const double k = 50.0;
    const double m0 = 2.0, m1 = 0.5;
    const double s14 = std::sqrt(14.0);
    const double ux = 1.0 / s14, uy = 2.0 / s14, uz = 3.0 / s14;
    const double overlaps[] = {0.1, 0.05};
    for (double overlap : overlaps) {
        SCOPED_TRACE(overlap);
        const double d = 1.0 - overlap;
        Particles p;
        p.resize(2);
        p.x[1] = d * ux;
        p.y[1] = d * uy;
        p.z[1] = d * uz;
        p.radius[0] = 0.3;
        p.radius[1] = 0.7;
        p.mass[0] = m0;
        p.mass[1] = m1;
        p.ax[0] = 1.0; // pre-existing (e.g. gravity): contact must add to it
        compute_contact_bruteforce(p, contact_params(k, 0.0));

        const double f = k * overlap;
        // Particle 0 is pushed along -u, particle 1 along +u.
        EXPECT_NEAR(p.ax[0], 1.0 - f * ux / m0, 1e-12);
        EXPECT_NEAR(p.ay[0], -f * uy / m0, 1e-12);
        EXPECT_NEAR(p.az[0], -f * uz / m0, 1e-12);
        EXPECT_NEAR(p.ax[1], f * ux / m1, 1e-12);
        EXPECT_NEAR(p.ay[1], f * uy / m1, 1e-12);
        EXPECT_NEAR(p.az[1], f * uz / m1, 1e-12);
        // Equal and opposite forces: m0 a0 = -m1 a1 (subtracting the pre-existing 1.0).
        EXPECT_NEAR(m0 * (p.ax[0] - 1.0), -m1 * p.ax[1], 1e-12);
        EXPECT_NEAR(m0 * p.ay[0], -m1 * p.ay[1], 1e-12);
        EXPECT_NEAR(m0 * p.az[0], -m1 * p.az[1], 1e-12);
    }
}

TEST(Contact, DamperUsesAllVelocityComponents) {
    // Same direction u = (1,2,3)/sqrt(14), centers 0.9 apart with radii 0.5 + 0.5, so
    // overlap 0.1. Relative velocity v0 - v1 = (0.5, -1.1, 1.6) has a part along u of
    // (0.5 - 2.2 + 4.8)/sqrt(14) = 3.1/sqrt(14): particle 0 is moving toward 1.
    // Force on 0 along -u: k * 0.1 + c * 3.1/sqrt(14) = 1 + 0.775/sqrt(14).
    const double k = 10.0, c = 0.25;
    const double s14 = std::sqrt(14.0);
    const double ux = 1.0 / s14, uy = 2.0 / s14, uz = 3.0 / s14;
    Particles p;
    p.resize(2);
    p.x[1] = 0.9 * ux;
    p.y[1] = 0.9 * uy;
    p.z[1] = 0.9 * uz;
    p.radius[0] = p.radius[1] = 0.5;
    p.mass[0] = p.mass[1] = 1.0;
    p.vx[0] = 0.3;
    p.vy[0] = -0.7;
    p.vz[0] = 1.1;
    p.vx[1] = -0.2;
    p.vy[1] = 0.4;
    p.vz[1] = -0.5;
    compute_contact_bruteforce(p, contact_params(k, c));
    const double f = 1.0 + 0.775 / s14;
    EXPECT_NEAR(p.ax[0], -f * ux, 1e-12);
    EXPECT_NEAR(p.ay[0], -f * uy, 1e-12);
    EXPECT_NEAR(p.az[0], -f * uz, 1e-12);
    EXPECT_NEAR(p.ax[1], f * ux, 1e-12);
    EXPECT_NEAR(p.ay[1], f * uy, 1e-12);
    EXPECT_NEAR(p.az[1], f * uz, 1e-12);
}

TEST(Contact, DamperResistsApproachAndSeparation) {
    // Overlap 0.1, k = 10 -> spring 1.0. Relative normal speed 2 with c = 0.25 -> damper 0.5.
    const double k = 10.0, c = 0.25;
    Particles p = pair_on_x(0.0, 0.9, 0.5, 0.5, 1.0, 1.0);
    p.vx[0] = 1.0; // approaching
    p.vx[1] = -1.0;
    compute_contact_bruteforce(p, contact_params(k, c));
    EXPECT_NEAR(p.ax[0], -1.5, 1e-12); // spring and damper both push apart
    EXPECT_NEAR(p.ax[1], 1.5, 1e-12);

    p.ax[0] = p.ax[1] = 0.0;
    p.vx[0] = -1.0; // separating: the damper now holds them back
    p.vx[1] = 1.0;
    compute_contact_bruteforce(p, contact_params(k, c));
    EXPECT_NEAR(p.ax[0], -0.5, 1e-12);
    EXPECT_NEAR(p.ax[1], 0.5, 1e-12);

    // Sideways motion has no normal component, so only the spring acts.
    p.ax[0] = p.ax[1] = 0.0;
    p.vx[0] = p.vx[1] = 0.0;
    p.vy[0] = 3.0;
    compute_contact_bruteforce(p, contact_params(k, c));
    EXPECT_NEAR(p.ax[0], -1.0, 1e-12);
    EXPECT_EQ(p.ay[0], 0.0);
}

TEST(Contact, SamePositionGivesNoForceAndNoNaN) {
    Particles p = pair_on_x(0.3, 0.3, 0.5, 0.5, 1.0, 1.0);
    compute_contact_bruteforce(p, contact_params(100.0, 3.0));
    EXPECT_EQ(p.ax[0], 0.0);
    EXPECT_EQ(p.ax[1], 0.0);
}

TEST(Contact, ClusterForcesCancelExactly) {
    // 27 particles on a squeezed 3x3x3 grid with varied masses and velocities: every
    // neighbor pair overlaps. Pair forces cancel bit for bit, so the total force
    // (sum of m * a) is zero up to the rounding of the sum itself.
    Particles p;
    p.resize(27);
    std::size_t idx = 0;
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            for (int cc = 0; cc < 3; ++cc) {
                p.x[idx] = 0.9 * a + 0.01 * b;
                p.y[idx] = 0.9 * b - 0.02 * cc;
                p.z[idx] = 0.9 * cc + 0.03 * a;
                p.vx[idx] = 0.1 * (a - b);
                p.vy[idx] = 0.2 * (cc - 1);
                p.vz[idx] = 0.05 * (a + b + cc);
                p.radius[idx] = 0.5;
                p.mass[idx] = 1.0 + 0.1 * static_cast<double>(idx);
                ++idx;
            }
        }
    }
    compute_contact_bruteforce(p, contact_params(100.0, 2.0));
    double fx = 0.0, fy = 0.0, fz = 0.0, fmax = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        fx += p.mass[i] * p.ax[i];
        fy += p.mass[i] * p.ay[i];
        fz += p.mass[i] * p.az[i];
        fmax = std::fmax(fmax, std::fabs(p.mass[i] * p.ax[i]));
    }
    ASSERT_GT(fmax, 1.0); // the cluster really is in contact
    EXPECT_LT(std::fabs(fx), 1e-12 * fmax);
    EXPECT_LT(std::fabs(fy), 1e-12 * fmax);
    EXPECT_LT(std::fabs(fz), 1e-12 * fmax);
}

TEST(Contact, ComputeAccelerationsIsGravityPlusContact) {
    Particles p = pair_on_x(0.0, 0.9, 0.5, 0.5, 1.0, 1.0);
    SimParams params = contact_params(10.0, 0.0);
    params.softening = 0.01;
    compute_accelerations(p, params, BACKEND_SERIAL);
    const double gravity = 0.9 / std::pow(0.81 + 1e-4, 1.5); // m d / (d^2 + eps^2)^(3/2)
    const double spring = 10.0 * 0.1;                        // apart
    EXPECT_NEAR(p.ax[0], gravity - spring, 1e-12);
    EXPECT_NEAR(p.ax[1], -(gravity - spring), 1e-12);
}

// ---- Heat from the damper (#46) ----

TEST(Contact, HeatRateIsHalfTheDamperPowerEach) {
    // Unequal masses 2 and 0.5 (m_eff = 0.4), approaching at relative normal speed 2,
    // overlap 0.1. Damper power c * vn^2 = 2 * 4 = 8, times the leapfrog factor
    // 1 - c dt / (2 m_eff) = 1 - 2 * 0.1 / 0.8 = 0.75, gives 6, so 3 for each particle.
    Particles p = pair_on_x(0.0, 0.9, 0.5, 0.5, 2.0, 0.5);
    p.vx[0] = 1.0;
    p.vx[1] = -1.0;
    p.heat_rate[0] = p.heat_rate[1] = 7.0; // stale values must be overwritten, not added to
    SimParams params = contact_params(10.0, 2.0);
    params.dt = 0.1;
    compute_contact_bruteforce(p, params);
    EXPECT_NEAR(p.heat_rate[0], 3.0, 1e-12);
    EXPECT_NEAR(p.heat_rate[1], 3.0, 1e-12);
    EXPECT_EQ(p.heat[0], 0.0); // contact sets the rate only; leapfrog_step adds heat

    // Separating heats just the same (the damper resists both ways).
    p.vx[0] = -1.0;
    p.vx[1] = 1.0;
    compute_contact_bruteforce(p, params);
    EXPECT_NEAR(p.heat_rate[0], 3.0, 1e-12);

    // Sideways motion: no normal speed, no heat. Apart: no contact, no heat.
    p.vx[0] = p.vx[1] = 0.0;
    p.vy[0] = 3.0;
    compute_contact_bruteforce(p, params);
    EXPECT_EQ(p.heat_rate[0], 0.0);
    EXPECT_EQ(p.heat_rate[1], 0.0);
    Particles apart = pair_on_x(0.0, 1.1, 0.5, 0.5, 2.0, 0.5);
    apart.vx[0] = 1.0;
    apart.heat_rate[0] = 7.0;
    compute_contact_bruteforce(apart, params);
    EXPECT_EQ(apart.heat_rate[0], 0.0);
}

TEST(Contact, NoContactMeansNoHeatRate) {
    Particles p = pair_on_x(0.0, 0.9, 0.5, 0.5, 1.0, 1.0); // overlapping, approaching
    p.vx[0] = 1.0;
    p.heat_rate[0] = p.heat_rate[1] = 7.0;
    SimParams params = contact_params(10.0, 2.0);
    params.contact_enabled = false;
    compute_accelerations(p, params, BACKEND_SERIAL);
    EXPECT_EQ(p.heat_rate[0], 0.0);
    EXPECT_EQ(p.heat_rate[1], 0.0);
}

TEST(Contact, ClusterHeatRateSumsEveryPairOnce) {
    // The 27-particle squeezed grid from ClusterForcesCancelExactly. The particles'
    // heating rates must add up to each pair's damper power counted once.
    Particles p;
    p.resize(27);
    std::size_t idx = 0;
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            for (int cc = 0; cc < 3; ++cc) {
                p.x[idx] = 0.9 * a + 0.01 * b;
                p.y[idx] = 0.9 * b - 0.02 * cc;
                p.z[idx] = 0.9 * cc + 0.03 * a;
                p.vx[idx] = 0.1 * (a - b);
                p.vy[idx] = 0.2 * (cc - 1);
                p.vz[idx] = 0.05 * (a + b + cc);
                p.radius[idx] = 0.5;
                p.mass[idx] = 1.0 + 0.1 * static_cast<double>(idx);
                ++idx;
            }
        }
    }
    SimParams params = contact_params(100.0, 2.0);
    params.dt = 0.01;
    double expected = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        for (std::size_t j = i + 1; j < p.size(); ++j) {
            double dx = p.x[i] - p.x[j], dy = p.y[i] - p.y[j], dz = p.z[i] - p.z[j];
            double d = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (d >= 1.0) {
                continue;
            }
            double vn =
                ((p.vx[i] - p.vx[j]) * dx + (p.vy[i] - p.vy[j]) * dy + (p.vz[i] - p.vz[j]) * dz) /
                d;
            double m_eff = p.mass[i] * p.mass[j] / (p.mass[i] + p.mass[j]);
            expected += 2.0 * vn * vn * (1.0 - 0.5 * 2.0 * 0.01 / m_eff);
        }
    }
    ASSERT_GT(expected, 0.1); // the cluster really is moving in contact
    Particles q = p;
    compute_contact_bruteforce(p, params);
    double sum = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        sum += p.heat_rate[i];
    }
    EXPECT_NEAR(sum, expected, 1e-12 * expected);

    // Through compute_accelerations with the OpenMP backend: contact is still the same
    // serial loop on every backend, so this only checks that OpenMP gravity leaves
    // heat_rate alone. It becomes a real check when contact gets an OpenMP version (#23).
    compute_accelerations(q, params, BACKEND_OMP);
    for (std::size_t i = 0; i < p.size(); ++i) {
        EXPECT_NEAR(q.heat_rate[i], p.heat_rate[i], 1e-12 * expected) << "particle " << i;
    }
}

TEST(Contact, OnlyLeapfrogStepsAddHeat) {
    Particles p = pair_on_x(-0.45, 0.45, 0.5, 0.5, 1.0, 1.0); // overlapping, approaching
    p.vx[0] = 1.0;
    p.vx[1] = -1.0;
    SimParams params = contact_params(100.0, 3.0);
    params.dt = 1e-3;
    compute_accelerations(p, params, BACKEND_SERIAL);
    compute_accelerations(p, params, BACKEND_SERIAL); // extra evaluations add nothing
    EXPECT_EQ(total_heat(p), 0.0);
    leapfrog_step(p, params, BACKEND_SERIAL);
    // One step adds exactly the rate from the step's own force evaluation times dt.
    ASSERT_GT(p.heat_rate[0], 0.0);
    EXPECT_EQ(p.heat[0], p.heat_rate[0] * params.dt);
    EXPECT_EQ(p.heat[1], p.heat_rate[1] * params.dt);
}

namespace {

struct BounceResult {
    double energy_change; // (E_after - E_before) / KE_before, both while separated
    double with_heat;     // the same, with the heat added to E_after
    double heat0, heat1;  // each particle's heat at the end
    double vx0_after;     // particle 0's velocity after the bounce
    double momentum_after;
    double separation_after;
};

// Two equal particles fly at each other along x with gravity and contact on, bounce,
// and fly apart. Energy (kinetic + gravitational) is compared before and after, when
// the spring is not compressed, so spring energy is zero at both points.
BounceResult head_on_bounce(double c) {
    const double m = 1.0, r = 0.5, k = 1e4;
    Particles p = pair_on_x(-0.6, 0.6, r, r, m, m);
    p.vx[0] = 1.0;
    p.vx[1] = -1.0;
    SimParams params = contact_params(k, c);
    // Spring period 2*pi*sqrt(m_eff / k) with m_eff = m / 2 is ~0.044; dt gives ~440
    // steps per period, well under PLAN's dt <= T/10.
    params.dt = 1e-4;
    params.softening = 0.0; // centers stay ~1 apart

    compute_accelerations(p, params, BACKEND_SERIAL);
    const double ke0 = kinetic_energy(p);
    const double e0 = total_energy(p, params);
    // ~0.1 to reach contact, ~0.022 in contact, then flying apart until the end of the
    // window. After the damped bounce the pair is still bound by gravity (E < 0): its
    // separation peaks near 1.41 and they would collide again, so a much longer run
    // would no longer end separated.
    for (int step = 0; step < 3000; ++step) {
        leapfrog_step(p, params, BACKEND_SERIAL);
    }
    BounceResult res;
    res.energy_change = (total_energy(p, params) - e0) / ke0;
    res.with_heat = (total_energy(p, params) + total_heat(p) - e0) / ke0;
    res.heat0 = p.heat[0];
    res.heat1 = p.heat[1];
    res.vx0_after = p.vx[0];
    Vec3 mom = total_momentum(p);
    res.momentum_after = std::fabs(mom.x) + std::fabs(mom.y) + std::fabs(mom.z);
    res.separation_after = p.x[1] - p.x[0];
    return res;
}

} // namespace

TEST(Contact, HeadOnBounceKeepsEnergyWithoutDamping) {
    BounceResult res = head_on_bounce(0.0);
    EXPECT_GT(res.separation_after, 1.0); // apart again
    EXPECT_LT(res.vx0_after, 0.0);        // bounced back
    EXPECT_LT(res.momentum_after, 1e-12);
    // Leapfrog with a conservative spring: once the pair has separated, the energy
    // error is only from the timestep. Measured 1.2e-5 at dt = 1e-4 (1.3e-6 at
    // dt = 5e-5), so this bound has ~8x margin; real energy loss would be ~0.1 or more.
    EXPECT_LT(std::fabs(res.energy_change), 1e-4);
    EXPECT_EQ(res.heat0 + res.heat1, 0.0); // no damper, no heat
}

TEST(Contact, HeadOnBounceLosesEnergyWithDamping) {
    // Damping for restitution e = 0.5 (docs/kickstart.md section 4):
    // zeta = -ln(e) / sqrt(pi^2 + ln^2(e)), c = 2 zeta sqrt(m_eff k).
    const double e = 0.5;
    const double zeta = -std::log(e) / std::sqrt(kPi * kPi + std::log(e) * std::log(e));
    const double c = 2.0 * zeta * std::sqrt(0.5 * 1e4);
    BounceResult res = head_on_bounce(c);
    EXPECT_GT(res.separation_after, 1.0);
    EXPECT_LT(res.vx0_after, 0.0);
    EXPECT_LT(res.momentum_after, 1e-12);
    // Restitution e keeps e^2 of the kinetic energy at impact and loses the rest.
    // Gravity speeds the pair up on the way in: centers 1.2 apart at the start, 1.0
    // at first touch, so KE at impact = KE0 + m^2 (1/1.0 - 1/1.2). Measured -0.882
    // against the expected -0.875; the formula ignores gravity during the contact.
    const double ke0 = 1.0; // two unit masses at speed 1
    const double ke_impact = ke0 + (1.0 / 1.0 - 1.0 / 1.2);
    const double expected = -(1.0 - e * e) * ke_impact / ke0;
    EXPECT_NEAR(res.energy_change, expected, 0.03);
}

TEST(Contact, HeatAccountsForTheEnergyTheDamperRemoves) {
    // Kinetic + potential + heat after the bounce must match the start. Measured at
    // dt = 1e-4: 1.0e-5 (e = 0.5), -2.5e-5 (e = 0.9) and 1.1e-5 (e = 0.1) of the
    // starting kinetic energy, the same size as the undamped bounce's 1.2e-5, so the
    // bound is the undamped test's. Without the leapfrog factor 1 - c dt / (2 m_eff)
    // the heat was 2.7e-3 too high at e = 0.5 and 9.8e-3 at e = 0.1.
    const double restitutions[] = {0.5, 0.9, 0.1};
    for (double e : restitutions) {
        SCOPED_TRACE(e);
        const double zeta = -std::log(e) / std::sqrt(kPi * kPi + std::log(e) * std::log(e));
        BounceResult res = head_on_bounce(2.0 * zeta * std::sqrt(0.5 * 1e4));
        EXPECT_LT(res.energy_change, -0.2); // a real loss to account for
        EXPECT_LT(std::fabs(res.with_heat), 1e-4);
        EXPECT_EQ(res.heat0, res.heat1); // equal masses: the two halves match exactly
    }
}
