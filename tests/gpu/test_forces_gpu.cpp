#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "contact.hpp"
#include "diagnostics.hpp"
#include "forces.hpp"
#include "gravity.hpp"
#include "init.hpp"
#include "integrator.hpp"
#include "params.hpp"
#include "particles.hpp"
#include "sim.hpp"

// The GPU's float kernel and its contact forces (#26, #32, #66), against the serial
// gravity and the all-pairs CPU contact. Labelled `gpu`; each test skips without a GPU.
// Float can't match serial to 1e-12 like double, so every float bound here comes from a
// measurement (about 10x what was measured, written next to it).

namespace {

#define REQUIRE_GPU()                                                                              \
    do {                                                                                           \
        std::string info;                                                                          \
        if (!hip_init(&info)) {                                                                    \
            GTEST_SKIP() << info;                                                                  \
        }                                                                                          \
    } while (0)

// Sets the GPU precision for one test and restores it afterwards, so test order can't
// matter.
class PrecisionGuard {
  public:
    explicit PrecisionGuard(bool use_double) : saved_(hip_uses_double()) {
        hip_set_double(use_double);
    }
    ~PrecisionGuard() { hip_set_double(saved_); }

  private:
    bool saved_;
};

Particles random_cloud(std::size_t n, unsigned seed, double offset_x) {
    Particles p;
    p.resize(n);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> pos(-1.0, 1.0), mass(0.5, 2.0);
    for (std::size_t i = 0; i < n; ++i) {
        p.x[i] = pos(rng) + offset_x;
        p.y[i] = pos(rng);
        p.z[i] = pos(rng);
        p.mass[i] = mass(rng);
    }
    return p;
}

double norm3(double x, double y, double z) {
    return std::sqrt(x * x + y * y + z * z);
}

// Largest per-particle error of a against b, relative to `scale` (one value per particle).
double max_rel_error(const Particles &a, const Particles &b, const std::vector<double> &scale) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double e = norm3(a.ax[i] - b.ax[i], a.ay[i] - b.ay[i], a.az[i] - b.az[i]);
        worst = std::fmax(worst, e / scale[i]);
    }
    return worst;
}

std::vector<double> accel_norms(const Particles &p) {
    std::vector<double> s(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) {
        s[i] = norm3(p.ax[i], p.ay[i], p.az[i]);
    }
    return s;
}

SimParams contact_params() {
    SimParams params;
    params.contact_enabled = true;
    params.contact_k = 20.0;
    params.contact_c = 0.05;
    params.dt = 0.002;
    params.softening = 0.05;
    return params;
}

// Two small planets pressed together and approaching, with a little random motion, so
// contact forces and heat are large (at rest gravity and contact nearly cancel and the
// heat rate is ~1e-11, which would hide errors). Planet 0 sits at x = center - ..., so
// positions are away from the origin, where float rounding is coarser.
Particles approaching_planets(double center_x) {
    const double r = 0.5 * std::cbrt(150.0 / 250.0);
    PlanetSpec a;
    a.n = 150;
    a.radius = r;
    a.particle_radius = 0.06;
    a.mass = 0.15;
    a.center = Vec3{center_x - r + 0.03, 0.0, 0.0}; // overlapping by 0.06
    a.velocity = Vec3{0.25, 0.0, 0.0};
    a.seed = 1;
    PlanetSpec b = a;
    b.center = Vec3{center_x + r - 0.03, 0.1, 0.0};
    b.velocity = Vec3{-0.25, 0.0, 0.0};
    b.seed = 2;
    b.body = 1;
    Particles p;
    make_planet(p, a);
    make_planet(p, b);
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> jitter(-0.05, 0.05);
    for (std::size_t i = 0; i < p.size(); ++i) {
        p.vx[i] += jitter(rng);
        p.vy[i] += jitter(rng);
        p.vz[i] += jitter(rng);
    }
    return p;
}

} // namespace

TEST(GpuPrecision, DefaultIsFloat) {
    // Every suite restores what it changes, so the program's default is still in place.
    EXPECT_FALSE(hip_uses_double());
}

TEST(GpuFloatGravity, MatchesSerialAcrossSizes) {
    REQUIRE_GPU();
    PrecisionGuard guard(false);
    SimParams params;
    params.softening = 0.01;
    // Partial last tiles (255, 257, 1000), exact tiles (256), tiny systems, and a cloud
    // centered at x = 3, away from the origin.
    struct Case {
        std::size_t n;
        double offset;
    };
    const Case cases[] = {{1, 0.0},   {2, 0.0},    {255, 0.0},  {256, 0.0},
                          {257, 0.0}, {1000, 0.0}, {4096, 0.0}, {4096, 3.0}};
    for (const Case &c : cases) {
        SCOPED_TRACE(testing::Message() << "n=" << c.n << " offset=" << c.offset);
        Particles serial = random_cloud(c.n, 11, c.offset);
        Particles gpu = random_cloud(c.n, 11, c.offset);
        compute_gravity_serial(serial, params);
        compute_gravity_hip(gpu, params);
        if (c.n == 1) { // no other particle: exactly zero
            EXPECT_EQ(gpu.ax[0], 0.0);
            continue;
        }
        double err = max_rel_error(gpu, serial, accel_norms(serial));
        std::printf("float gravity n=%zu offset=%g: max relative error %.3g\n", c.n, c.offset, err);
        // Measured up to 3.5e-6, the same at x = 3 (gravity separations are ~1, so this
        // case doesn't need the hi/lo positions; the contact tests at x = 2.5 do).
        EXPECT_LT(err, 4e-5);
        // And above double's ~1e-15: proves the float kernel really ran.
        EXPECT_GT(err, 1e-10);
    }
}

TEST(GpuForces, HandComputedApproachingPair) {
    REQUIRE_GPU();
    // Two unit masses, radii 0.5, centers 0.9 apart at x = 2.5 and 3.4 (away from the
    // origin), closing at 1 (vx = +0.5 and -0.5): overlap 0.1, n = -x for particle 0,
    // vn = (v0 - v1) . n = -1. Force on 0: (k * 0.1 - c * vn) * n = -(2 + 0.3) along x
    // with k = 20, c = 0.3. Heat rate each: 0.5 * c * vn^2 * (1 - c * dt / (2 * m_eff))
    // = 0.15 * (1 - 0.3 * 0.002 / 1) = 0.14991. The particles' gravity on each other
    // is taken from the serial code and added to the expected value.
    for (bool use_double : {false, true}) {
        SCOPED_TRACE(use_double ? "double" : "float");
        PrecisionGuard guard(use_double);
        SimParams params = contact_params();
        params.contact_k = 20.0;
        params.contact_c = 0.3;
        params.dt = 0.002;
        Particles p;
        p.resize(2);
        p.x[0] = 2.5;
        p.x[1] = 3.4;
        p.vx[0] = 0.5;
        p.vx[1] = -0.5;
        p.radius[0] = p.radius[1] = 0.5;
        p.mass[0] = p.mass[1] = 1.0;
        Particles grav = p;
        compute_gravity_serial(grav, params);
        compute_forces_hip(p, params);
        const double contact = 20.0 * 0.1 + 0.3 * 1.0; // k * overlap - c * vn, vn = -1
        EXPECT_NEAR(p.ax[0], grav.ax[0] - contact, 1e-5 * contact);
        EXPECT_NEAR(p.ax[1], grav.ax[1] + contact, 1e-5 * contact);
        EXPECT_NEAR(p.heat_rate[0], 0.14991, 1e-5);
        EXPECT_NEAR(p.heat_rate[1], 0.14991, 1e-5);
    }
}

TEST(GpuForces, MatchCpuContactOnApproachingPlanets) {
    REQUIRE_GPU();
    const SimParams params = contact_params();
    // CPU reference: serial gravity + all-pairs contact, and each part on its own for
    // the error scale (|a_grav| + |a_contact| per particle: the net force nearly cancels).
    Particles base = approaching_planets(2.5);
    Particles grav = base, cpu = base;
    compute_gravity_serial(grav, params);
    compute_gravity_serial(cpu, params);
    compute_contact_bruteforce(cpu, params);
    std::vector<double> scale(base.size());
    double heat_max = 0.0;
    for (std::size_t i = 0; i < base.size(); ++i) {
        double cx = cpu.ax[i] - grav.ax[i], cy = cpu.ay[i] - grav.ay[i],
               cz = cpu.az[i] - grav.az[i];
        scale[i] = norm3(grav.ax[i], grav.ay[i], grav.az[i]) + norm3(cx, cy, cz);
        heat_max = std::fmax(heat_max, cpu.heat_rate[i]);
    }
    ASSERT_GT(heat_max, 1e-4); // the planets really are hitting each other

    for (bool use_double : {true, false}) {
        SCOPED_TRACE(use_double ? "double" : "float");
        PrecisionGuard guard(use_double);
        Particles gpu = base;
        compute_forces_hip(gpu, params);
        double err = max_rel_error(gpu, cpu, scale);
        double heat_err = 0.0;
        for (std::size_t i = 0; i < base.size(); ++i) {
            heat_err = std::fmax(heat_err, std::fabs(gpu.heat_rate[i] - cpu.heat_rate[i]));
        }
        std::printf("forces %s: max relative error %.3g, heat error %.3g of max\n",
                    use_double ? "double" : "float", err, heat_err / heat_max);
        if (use_double) {
            EXPECT_LT(err, 1e-12);
            EXPECT_LT(heat_err, 1e-12 * heat_max);
        } else {
            // Measured 1.2e-4 and 1.7e-7. A missing heat factor is a 10% heat error here
            // and a flipped damper sign moves contact forces by about 2%.
            EXPECT_LT(err, 1e-3);
            EXPECT_GT(err, 1e-10); // float really ran (double gives ~1e-15)
            EXPECT_LT(heat_err, 2e-6 * heat_max);
        }
    }
}

TEST(GpuForces, ContactOffMeansNoHeatRate) {
    REQUIRE_GPU();
    PrecisionGuard guard(false);
    SimParams params = contact_params();
    params.contact_enabled = false;
    Particles p = approaching_planets(0.0);
    for (std::size_t i = 0; i < p.size(); ++i) {
        p.heat_rate[i] = 5.0; // stale values must be cleared
    }
    compute_accelerations(p, params, BACKEND_HIP);
    for (std::size_t i = 0; i < p.size(); ++i) {
        ASSERT_EQ(p.heat_rate[i], 0.0) << "particle " << i;
    }
}

TEST(GpuForces, CoincidentParticlesGiveNoNaN) {
    REQUIRE_GPU();
    for (bool use_double : {false, true}) {
        PrecisionGuard guard(use_double);
        SimParams params = contact_params();
        Particles p;
        p.resize(2);
        p.x[0] = p.x[1] = 1.5; // same point: no direction to push, softening keeps gravity finite
        p.radius[0] = p.radius[1] = 0.5;
        p.mass[0] = p.mass[1] = 1.0;
        compute_forces_hip(p, params);
        EXPECT_EQ(p.ax[0], 0.0);
        EXPECT_EQ(p.heat_rate[0], 0.0);
        EXPECT_TRUE(std::isfinite(p.ax[1]));
    }
}

TEST(GpuForces, FloatCollisionConserves) {
    REQUIRE_GPU();
    PrecisionGuard guard(false);
    // The collision from test_conservation.cpp, on the GPU in float.
    SimParams params = contact_params();
    const double r = 0.5 * std::cbrt(150.0 / 250.0);
    PlanetSpec a;
    a.n = 150;
    a.radius = r;
    a.particle_radius = 0.06;
    a.mass = 0.15;
    a.center = Vec3{-r - 0.05, 0.0, 0.0};
    a.velocity = Vec3{0.25, 0.0, 0.0};
    a.seed = 1;
    PlanetSpec b = a;
    b.center = Vec3{r + 0.05, 0.3, 0.0};
    b.velocity = Vec3{-0.25, 0.0, 0.0};
    b.seed = 2;
    b.body = 1;
    Particles p;
    make_planet(p, a);
    make_planet(p, b);
    ASSERT_EQ(sim_init(p, params, BACKEND_HIP), p.size());
    const Vec3 p0 = total_momentum(p), l0 = total_angular_momentum(p);
    double mom_scale = 0.0, ang_scale = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        mom_scale += p.mass[i] * norm3(p.vx[i], p.vy[i], p.vz[i]);
        ang_scale += p.mass[i] * norm3(p.y[i] * p.vz[i] - p.z[i] * p.vy[i],
                                       p.z[i] * p.vx[i] - p.x[i] * p.vz[i],
                                       p.x[i] * p.vy[i] - p.y[i] * p.vx[i]);
    }
    auto total = [&](const Particles &q) {
        return kinetic_energy(q) + potential_energy(q, params) + contact_energy(q, params) +
               total_heat(q);
    };
    const double e0 = total(p);
    double rise = 0.0, drop = 0.0;
    for (int s = 0; s < 300; ++s) {
        ASSERT_EQ(sim_step(p, params, BACKEND_HIP), p.size()) << "step " << s;
        double d = total(p) - e0;
        rise = std::fmax(rise, d);
        drop = std::fmax(drop, -d);
    }
    const Vec3 p1 = total_momentum(p), l1 = total_angular_momentum(p);
    double dp = norm3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z) / mom_scale;
    double dl = norm3(l1.x - l0.x, l1.y - l0.y, l1.z - l0.z) / ang_scale;
    std::printf("float collision: momentum %.3g, angular momentum %.3g, energy rise %.3g, "
                "drop %.3g (E0 %.6g)\n",
                dp, dl, rise, drop, e0);
    // Measured: momentum 1.8e-9, angular momentum 2.6e-9 (float rounding; double gets
    // ~1e-15), energy up 5.6e-5 and down 7.1e-5, the same contact-onset error the CPU
    // gets in test_conservation.cpp, so the same 1.5e-4 bound.
    EXPECT_LT(dp, 2e-8);
    EXPECT_LT(dl, 3e-8);
    EXPECT_LT(rise, 1.5e-4);
    EXPECT_LT(drop, 1.5e-4);
}

TEST(GpuForces, SwitchingPrecisionMidRunStaysClose) {
    REQUIRE_GPU();
    SimParams params = contact_params();
    Particles ref = approaching_planets(0.0), mixed = ref;
    ASSERT_EQ(sim_init(ref, params, BACKEND_SERIAL), ref.size());
    ASSERT_EQ(sim_init(mixed, params, BACKEND_SERIAL), mixed.size());
    for (bool use_double : {false, true, false}) {
        PrecisionGuard guard(use_double);
        for (int s = 0; s < 5; ++s) {
            ASSERT_EQ(sim_step(ref, params, BACKEND_SERIAL), ref.size());
            ASSERT_EQ(sim_step(mixed, params, BACKEND_HIP), mixed.size());
        }
    }
    double worst = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        worst = std::fmax(
            worst, norm3(mixed.x[i] - ref.x[i], mixed.y[i] - ref.y[i], mixed.z[i] - ref.z[i]));
    }
    std::printf("precision switching: worst position difference %.3g after 15 steps\n", worst);
    EXPECT_LT(worst, 1e-7); // measured 8.2e-9
}

TEST(GpuForces, RelaxationRunsInDoubleAndRestoresFloat) {
    REQUIRE_GPU();
    // Float and double runs must start from the same planets, so relax_planet always uses
    // double on the GPU, and puts the caller's setting back afterwards.
    SimParams params = contact_params();
    PlanetSpec spec;
    spec.n = 150;
    spec.radius = 0.5 * std::cbrt(150.0 / 250.0);
    spec.particle_radius = 0.06;
    spec.mass = 0.15;
    spec.seed = 4;
    Particles in_float, in_double;
    make_planet(in_float, spec);
    make_planet(in_double, spec);
    {
        PrecisionGuard guard(true);
        relax_planet(in_double, params, BACKEND_HIP, 5.0, 0.0, 20); // 20 steps, never calm
    }
    {
        PrecisionGuard guard(false);
        relax_planet(in_float, params, BACKEND_HIP, 5.0, 0.0, 20);
        EXPECT_FALSE(hip_uses_double()); // the caller's float setting is back
    }
    for (std::size_t i = 0; i < in_float.size(); ++i) { // same arithmetic: same bits
        ASSERT_EQ(in_float.x[i], in_double.x[i]) << "particle " << i;
        ASSERT_EQ(in_float.vx[i], in_double.vx[i]) << "particle " << i;
    }
}
