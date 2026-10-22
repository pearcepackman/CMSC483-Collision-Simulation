#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "forces.hpp"
#include "gravity.hpp"
#include "params.hpp"
#include "particles.hpp"

namespace {

// |a - b| <= tol * max(|a|, |b|): relative comparison, per CLAUDE.md never `==`.
void expect_rel_near(double a, double b, double tol) {
    double scale = std::fmax(std::fabs(a), std::fabs(b));
    EXPECT_LE(std::fabs(a - b), tol * scale) << a << " vs " << b;
}

// Mass 2 at the origin, mass 3 at (2, 3, 6): separation 7. All three components
// differ, so mixing up x/y/z can't go unnoticed.
Particles hand_pair() {
    Particles p;
    p.resize(2);
    p.mass[0] = 2.0;
    p.mass[1] = 3.0;
    p.x[1] = 2.0;
    p.y[1] = 3.0;
    p.z[1] = 6.0;
    return p;
}

SimParams with_softening(double eps) {
    SimParams params;
    params.softening = eps;
    return params;
}

} // namespace

TEST(GravitySerial, TwoParticlesMatchHandComputedForce) {
    // eps = 0: |r|^3 = 343, so a0 = m1 * (2,3,6) / 343 and a1 = -m0 * (2,3,6) / 343.
    Particles p = hand_pair();
    compute_gravity_serial(p, with_softening(0.0));
    expect_rel_near(p.ax[0], 3.0 * 2.0 / 343.0, 1e-14);
    expect_rel_near(p.ay[0], 3.0 * 3.0 / 343.0, 1e-14);
    expect_rel_near(p.az[0], 3.0 * 6.0 / 343.0, 1e-14);
    expect_rel_near(p.ax[1], -2.0 * 2.0 / 343.0, 1e-14);
    expect_rel_near(p.ay[1], -2.0 * 3.0 / 343.0, 1e-14);
    expect_rel_near(p.az[1], -2.0 * 6.0 / 343.0, 1e-14);

    // eps = 24: (|r|^2 + eps^2)^(3/2) = (49 + 576)^(3/2) = 25^3 = 15625.
    compute_gravity_serial(p, with_softening(24.0));
    expect_rel_near(p.ax[0], 3.0 * 2.0 / 15625.0, 1e-14);
    expect_rel_near(p.ay[0], 3.0 * 3.0 / 15625.0, 1e-14);
    expect_rel_near(p.az[0], 3.0 * 6.0 / 15625.0, 1e-14);
}

TEST(GravitySerial, ThreeParticlesSumAllPulls) {
    // Masses 2, 3, 1 at 0, (2,3,6) and (4,6,12): pair distances 7, 7 and 14 (eps = 0).
    //   a0 = 3*(2,3,6)/343 + 1*(4,6,12)/2744    = (2,3,6) * 13/1372
    //   a1 = -2*(2,3,6)/343 + 1*(2,3,6)/343     = (2,3,6) * -1/343
    //   a2 = -2*(4,6,12)/2744 - 3*(2,3,6)/343   = (2,3,6) * -1/98
    Particles p;
    p.resize(3);
    p.mass[0] = 2.0;
    p.mass[1] = 3.0;
    p.mass[2] = 1.0;
    p.x[1] = 2.0;
    p.y[1] = 3.0;
    p.z[1] = 6.0;
    p.x[2] = 4.0;
    p.y[2] = 6.0;
    p.z[2] = 12.0;
    compute_gravity_serial(p, with_softening(0.0));
    const double scale[3] = {13.0 / 1372.0, -1.0 / 343.0, -1.0 / 98.0};
    for (std::size_t i = 0; i < 3; ++i) {
        expect_rel_near(p.ax[i], 2.0 * scale[i], 1e-14);
        expect_rel_near(p.ay[i], 3.0 * scale[i], 1e-14);
        expect_rel_near(p.az[i], 6.0 * scale[i], 1e-14);
    }
    // U = -(2*3/7 + 3*1/7 + 2*1/14) = -10/7
    expect_rel_near(potential_energy(p, with_softening(0.0)), -10.0 / 7.0, 1e-14);
}

TEST(GravitySerial, MatchesHigherPrecisionSumOnRandomSystem) {
    // Distances that aren't nice integers, checked against the same sum done in long
    // double. Pins the reference to full double precision, so a backend that quietly
    // drops to float (~1e-7 error) can't pass by matching a sloppy reference.
    const std::size_t n = 20;
    Particles p;
    p.resize(n);
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> pos(-1.0, 1.0), mass(0.5, 2.0);
    for (std::size_t i = 0; i < n; ++i) {
        p.x[i] = pos(rng);
        p.y[i] = pos(rng);
        p.z[i] = pos(rng);
        p.mass[i] = mass(rng);
    }
    const double eps = 0.05;
    compute_gravity_serial(p, with_softening(eps));

    for (std::size_t i = 0; i < n; ++i) {
        long double ref[3] = {0.0L, 0.0L, 0.0L};
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            long double d[3] = {(long double)p.x[j] - p.x[i], (long double)p.y[j] - p.y[i],
                                (long double)p.z[j] - p.z[i]};
            long double r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + (long double)eps * eps;
            long double inv_r3 = 1.0L / (r2 * std::sqrt(r2));
            for (int c = 0; c < 3; ++c) {
                ref[c] += p.mass[j] * d[c] * inv_r3;
            }
        }
        const double got[3] = {p.ax[i], p.ay[i], p.az[i]};
        double norm = (double)std::sqrt(ref[0] * ref[0] + ref[1] * ref[1] + ref[2] * ref[2]);
        for (int c = 0; c < 3; ++c) {
            EXPECT_LE(std::fabs(got[c] - (double)ref[c]), 1e-13 * norm)
                << "particle " << i << " coord " << c;
        }
    }
}

TEST(GravitySerial, ForceIsMinusGradientOfPotential) {
    // The force must be exactly the slope of the potential, or energy won't be
    // conserved (#8). Check every coordinate of a small random system with central
    // differences: -(U(x+h) - U(x-h)) / 2h == m_k * a_k.
    const std::size_t n = 5;
    Particles p;
    p.resize(n);
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> pos(-1.0, 1.0), mass(0.5, 2.0);
    for (std::size_t i = 0; i < n; ++i) {
        p.x[i] = pos(rng);
        p.y[i] = pos(rng);
        p.z[i] = pos(rng);
        p.mass[i] = mass(rng);
    }
    const SimParams params = with_softening(0.3);
    compute_gravity_serial(p, params);

    const double h = 1e-6;
    double scale = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        scale = std::fmax(scale, p.mass[k] * std::fabs(p.ax[k]));
        scale = std::fmax(scale, p.mass[k] * std::fabs(p.ay[k]));
        scale = std::fmax(scale, p.mass[k] * std::fabs(p.az[k]));
    }
    for (std::size_t k = 0; k < n; ++k) {
        std::vector<double> *coords[3] = {&p.x, &p.y, &p.z};
        const double force[3] = {p.mass[k] * p.ax[k], p.mass[k] * p.ay[k], p.mass[k] * p.az[k]};
        for (int c = 0; c < 3; ++c) {
            double saved = (*coords[c])[k];
            (*coords[c])[k] = saved + h;
            double u_plus = potential_energy(p, params);
            (*coords[c])[k] = saved - h;
            double u_minus = potential_energy(p, params);
            (*coords[c])[k] = saved;
            double numeric = -(u_plus - u_minus) / (2.0 * h);
            EXPECT_LE(std::fabs(numeric - force[c]), 1e-6 * scale)
                << "particle " << k << " coord " << c;
        }
    }
}

TEST(GravitySerial, ForcesAreEqualAndOpposite) {
    Particles p = hand_pair();
    compute_gravity_serial(p, with_softening(0.0));
    expect_rel_near(p.mass[0] * p.ax[0], -p.mass[1] * p.ax[1], 1e-14);
    expect_rel_near(p.mass[0] * p.ay[0], -p.mass[1] * p.ay[1], 1e-14);
    expect_rel_near(p.mass[0] * p.az[0], -p.mass[1] * p.az[1], 1e-14);
}

TEST(GravitySerial, TotalForceOnRandomSystemIsZero) {
    // Every pair pulls equally and oppositely, so sum(m * a) cancels. Compare against
    // the size of the terms being summed, since the exact answer is 0.
    const std::size_t n = 100;
    Particles p;
    p.resize(n);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> pos(-1.0, 1.0), mass(0.5, 2.0);
    for (std::size_t i = 0; i < n; ++i) {
        p.x[i] = pos(rng);
        p.y[i] = pos(rng);
        p.z[i] = pos(rng);
        p.mass[i] = mass(rng);
    }
    compute_gravity_serial(p, with_softening(0.01));

    double fx = 0.0, fy = 0.0, fz = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        fx += p.mass[i] * p.ax[i];
        fy += p.mass[i] * p.ay[i];
        fz += p.mass[i] * p.az[i];
        scale += p.mass[i] * std::sqrt(p.ax[i] * p.ax[i] + p.ay[i] * p.ay[i] + p.az[i] * p.az[i]);
    }
    EXPECT_LE(std::fabs(fx), 1e-12 * scale);
    EXPECT_LE(std::fabs(fy), 1e-12 * scale);
    EXPECT_LE(std::fabs(fz), 1e-12 * scale);
}

TEST(GravitySerial, SofteningKeepsZeroSeparationFinite) {
    Particles p;
    p.resize(2);
    p.mass[0] = 1.0;
    p.mass[1] = 1.0; // both at the origin
    compute_gravity_serial(p, with_softening(0.1));
    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(p.ax[i], 0.0); // zero displacement -> exactly zero pull
        EXPECT_EQ(p.ay[i], 0.0);
        EXPECT_EQ(p.az[i], 0.0);
    }
    double u = potential_energy(p, with_softening(0.1));
    EXPECT_TRUE(std::isfinite(u));
    expect_rel_near(u, -1.0 / 0.1, 1e-14); // -m0 m1 / eps
}

TEST(GravitySerial, PotentialEnergyMatchesHandComputed) {
    Particles p = hand_pair();
    expect_rel_near(potential_energy(p, with_softening(24.0)), -2.0 * 3.0 / 25.0, 1e-14);
    expect_rel_near(potential_energy(p, with_softening(0.0)), -2.0 * 3.0 / 7.0, 1e-14);
}

TEST(GravitySerial, EmptyAndSingleParticle) {
    Particles p;
    compute_gravity_serial(p, with_softening(0.1)); // N = 0: nothing to do, no crash
    EXPECT_EQ(potential_energy(p, with_softening(0.1)), 0.0);

    p.resize(1);
    p.mass[0] = 5.0;
    p.ax[0] = p.ay[0] = p.az[0] = 123.0; // junk from a previous step
    compute_gravity_serial(p, with_softening(0.1));
    EXPECT_EQ(p.ax[0], 0.0); // overwritten, no self-force
    EXPECT_EQ(p.ay[0], 0.0);
    EXPECT_EQ(p.az[0], 0.0);
    EXPECT_EQ(potential_energy(p, with_softening(0.1)), 0.0);
}

TEST(GravitySerial, OverwritesPreviousAccelerations) {
    Particles p = hand_pair();
    Particles fresh = hand_pair();
    for (std::size_t i = 0; i < 2; ++i) {
        p.ax[i] = p.ay[i] = p.az[i] = 1e6; // stale values must not leak into the result
    }
    compute_gravity_serial(p, with_softening(0.5));
    compute_gravity_serial(fresh, with_softening(0.5));
    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(p.ax[i], fresh.ax[i]);
        EXPECT_EQ(p.ay[i], fresh.ay[i]);
        EXPECT_EQ(p.az[i], fresh.az[i]);
    }
}

TEST(Forces, ContactOffMatchesSerialGravity) {
    Particles p = hand_pair();
    Particles expected = hand_pair();
    SimParams params = with_softening(0.5);
    params.contact_enabled = false;
    compute_accelerations(p, params, BACKEND_SERIAL);
    compute_gravity_serial(expected, params);
    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(p.ax[i], expected.ax[i]); // same code path, so identical bits
        EXPECT_EQ(p.ay[i], expected.ay[i]);
        EXPECT_EQ(p.az[i], expected.az[i]);
    }
}
