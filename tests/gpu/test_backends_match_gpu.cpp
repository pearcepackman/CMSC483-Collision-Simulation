#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <string>

#include "forces.hpp"
#include "gravity.hpp"
#include "integrator.hpp"
#include "params.hpp"
#include "particles.hpp"
#include "sim.hpp"

// HIP backend vs the serial reference (#25). Built only with ENABLE_HIP and labelled
// `gpu`; each test skips if no GPU is visible. Compare with relative tolerances, never
// `==`, and only over a few steps (gravity is chaotic). See CLAUDE.md.

namespace {

// Skips the calling test when there's no GPU (e.g. HIP_VISIBLE_DEVICES hides them all).
#define REQUIRE_GPU()                                                                              \
    do {                                                                                           \
        std::string info;                                                                          \
        if (!hip_init(&info)) {                                                                    \
            GTEST_SKIP() << info;                                                                  \
        }                                                                                          \
    } while (0)

Particles random_system(std::size_t n, unsigned seed) {
    Particles p;
    p.resize(n);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> pos(-1.0, 1.0), mass(0.5, 2.0);
    for (std::size_t i = 0; i < n; ++i) {
        p.x[i] = pos(rng);
        p.y[i] = pos(rng);
        p.z[i] = pos(rng);
        p.mass[i] = mass(rng);
    }
    return p;
}

SimParams test_params() {
    SimParams params;
    params.softening = 0.01;
    return params;
}

// Every particle's GPU acceleration within `tol` of serial, relative to the size of that
// particle's (serial) acceleration. Same check as tests/test_backends_match.cpp.
void expect_matches_serial(const Particles &gpu, const Particles &serial, double tol) {
    ASSERT_EQ(gpu.size(), serial.size());
    for (std::size_t i = 0; i < serial.size(); ++i) {
        double norm = std::sqrt(serial.ax[i] * serial.ax[i] + serial.ay[i] * serial.ay[i] +
                                serial.az[i] * serial.az[i]);
        EXPECT_LE(std::fabs(gpu.ax[i] - serial.ax[i]), tol * norm) << "particle " << i;
        EXPECT_LE(std::fabs(gpu.ay[i] - serial.ay[i]), tol * norm) << "particle " << i;
        EXPECT_LE(std::fabs(gpu.az[i] - serial.az[i]), tol * norm) << "particle " << i;
    }
}

void expect_rel_near(double actual, double expected, double tol) {
    EXPECT_LE(std::fabs(actual - expected), tol * std::fabs(expected))
        << "actual " << actual << ", expected " << expected;
}

void check_against_serial(std::size_t n, unsigned seed) {
    SCOPED_TRACE(testing::Message() << "n=" << n);
    const SimParams params = test_params();
    Particles serial = random_system(n, seed);
    Particles gpu = random_system(n, seed);
    compute_gravity_serial(serial, params);
    compute_gravity_hip(gpu, params);
    expect_matches_serial(gpu, serial, 1e-12);
}

} // namespace

TEST(GravityHip, TwoParticlesMatchHandComputedForce) {
    REQUIRE_GPU();
    // Mass 2 at the origin, mass 3 at (2, 3, 6): separation 7, so with eps = 0
    // a0 = 3 * (2,3,6) / 343 and a1 = -2 * (2,3,6) / 343.
    Particles p;
    p.resize(2);
    p.mass[0] = 2.0;
    p.mass[1] = 3.0;
    p.x[1] = 2.0;
    p.y[1] = 3.0;
    p.z[1] = 6.0;
    SimParams params;
    params.softening = 0.0;
    compute_gravity_hip(p, params);
    expect_rel_near(p.ax[0], 3.0 * 2.0 / 343.0, 1e-14);
    expect_rel_near(p.ay[0], 3.0 * 3.0 / 343.0, 1e-14);
    expect_rel_near(p.az[0], 3.0 * 6.0 / 343.0, 1e-14);
    expect_rel_near(p.ax[1], -2.0 * 2.0 / 343.0, 1e-14);
    expect_rel_near(p.ay[1], -2.0 * 3.0 / 343.0, 1e-14);
    expect_rel_near(p.az[1], -2.0 * 6.0 / 343.0, 1e-14);
}

TEST(GravityHip, SingleParticleFeelsNoForce) {
    REQUIRE_GPU();
    // With eps = 0, a self-pull would be 0/0 = NaN, so this also proves j == i is skipped.
    Particles p;
    p.resize(1);
    p.mass[0] = 5.0;
    p.x[0] = 1.0;
    p.ax[0] = p.ay[0] = p.az[0] = 7.0; // stale values must be overwritten
    SimParams params;
    params.softening = 0.0;
    compute_gravity_hip(p, params);
    EXPECT_EQ(p.ax[0], 0.0);
    EXPECT_EQ(p.ay[0], 0.0);
    EXPECT_EQ(p.az[0], 0.0);
}

TEST(GravityHip, EmptySystemIsANoOp) {
    REQUIRE_GPU();
    Particles p;
    compute_gravity_hip(p, test_params()); // must not launch a zero-block kernel
    EXPECT_EQ(p.size(), 0u);
}

TEST(GravityHip, MatchesSerialAroundBlockBoundaries) {
    REQUIRE_GPU();
    // The kernel uses 256 threads per block: one short of, exactly, and one past a full
    // block, plus a size that leaves the last block mostly empty.
    const std::size_t sizes[] = {2, 255, 256, 257, 1000};
    for (std::size_t n : sizes) {
        check_against_serial(n, 7);
    }
}

TEST(GravityHip, MatchesSerialOnLargerCloud) {
    REQUIRE_GPU();
    check_against_serial(4096, 3);
}

TEST(GravityHip, ReusesBuffersWhenSizeChanges) {
    REQUIRE_GPU();
    // Grow, shrink, grow past the old capacity: cached GPU buffers must never hand back
    // a previous system's data or overflow.
    check_against_serial(1000, 11);
    check_against_serial(50, 12);
    check_against_serial(3000, 13);
    hip_release();
    check_against_serial(100, 14); // works again after the buffers were freed
}

TEST(GravityHip, OverwritesPreviousAccelerations) {
    REQUIRE_GPU();
    const SimParams params = test_params();
    Particles serial = random_system(300, 9);
    Particles gpu = random_system(300, 9);
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        gpu.ax[i] = gpu.ay[i] = gpu.az[i] = 1e6;
    }
    compute_gravity_serial(serial, params);
    compute_gravity_hip(gpu, params);
    expect_matches_serial(gpu, serial, 1e-12);
}

TEST(GravityHip, DispatcherRoutesHipBackend) {
    REQUIRE_GPU();
    const SimParams params = test_params();
    Particles direct = random_system(200, 13);
    Particles dispatched = random_system(200, 13);
    compute_gravity_hip(direct, params);
    compute_gravity(dispatched, params, BACKEND_HIP);
    expect_matches_serial(dispatched, direct, 1e-12);
}

TEST(GravityHip, FewLeapfrogStepsTrackSerial) {
    REQUIRE_GPU();
    SimParams params = test_params();
    params.dt = 1e-3;
    Particles serial = random_system(500, 21);
    Particles gpu = random_system(500, 21);
    compute_accelerations(serial, params, BACKEND_SERIAL);
    compute_accelerations(gpu, params, BACKEND_HIP);
    for (int step = 0; step < 10; ++step) {
        leapfrog_step(serial, params, BACKEND_SERIAL);
        leapfrog_step(gpu, params, BACKEND_HIP);
    }
    // After 10 short steps the backends still agree closely: each position within 1e-10
    // of serial, relative to that particle's distance from the origin.
    for (std::size_t i = 0; i < serial.size(); ++i) {
        double r = std::sqrt(serial.x[i] * serial.x[i] + serial.y[i] * serial.y[i] +
                             serial.z[i] * serial.z[i]);
        EXPECT_LE(std::fabs(gpu.x[i] - serial.x[i]), 1e-10 * r) << "particle " << i;
        EXPECT_LE(std::fabs(gpu.y[i] - serial.y[i]), 1e-10 * r) << "particle " << i;
        EXPECT_LE(std::fabs(gpu.z[i] - serial.z[i]), 1e-10 * r) << "particle " << i;
    }
}

TEST(GravityHip, SwitchingToHipAndBackMidRunMatches) {
    // The viewer's backend keys (#48): serial -> HIP -> serial must track serial
    // throughout, up to rounding. Few steps (gravity is chaotic).
    REQUIRE_GPU();
    SimParams params;
    Particles reference, switched;
    ASSERT_TRUE(build_scenario("cloud", 500, 8, reference, params));
    ASSERT_TRUE(build_scenario("cloud", 500, 8, switched, params));
    ASSERT_EQ(sim_init(reference, params, BACKEND_SERIAL), reference.size());
    ASSERT_EQ(sim_init(switched, params, BACKEND_SERIAL), switched.size());
    const Backend order[] = {BACKEND_SERIAL, BACKEND_HIP, BACKEND_SERIAL};
    for (Backend b : order) {
        for (int s = 0; s < 5; ++s) {
            ASSERT_EQ(sim_step(reference, params, BACKEND_SERIAL), reference.size());
            ASSERT_EQ(sim_step(switched, params, b), switched.size());
        }
    }
    for (std::size_t i = 0; i < reference.size(); ++i) {
        double r = std::sqrt(reference.x[i] * reference.x[i] + reference.y[i] * reference.y[i] +
                             reference.z[i] * reference.z[i]);
        EXPECT_LE(std::fabs(switched.x[i] - reference.x[i]), 1e-10 * r) << "particle " << i;
        EXPECT_LE(std::fabs(switched.y[i] - reference.y[i]), 1e-10 * r) << "particle " << i;
        EXPECT_LE(std::fabs(switched.z[i] - reference.z[i]), 1e-10 * r) << "particle " << i;
    }
}
