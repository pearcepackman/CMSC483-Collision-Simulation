#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include <omp.h>

#include "gravity.hpp"
#include "params.hpp"
#include "particles.hpp"

// Backends must agree on a single force evaluation. Parallel code may add numbers in a
// different order, so compare with a relative tolerance, never `==` (see CLAUDE.md), and
// never over many steps (gravity is chaotic).

namespace {

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

// Every particle's OpenMP acceleration within `tol` of serial, relative to the size of
// that particle's (serial) acceleration.
void expect_matches_serial(const Particles &omp, const Particles &serial, double tol) {
    ASSERT_EQ(omp.size(), serial.size());
    for (std::size_t i = 0; i < serial.size(); ++i) {
        double norm = std::sqrt(serial.ax[i] * serial.ax[i] + serial.ay[i] * serial.ay[i] +
                                serial.az[i] * serial.az[i]);
        EXPECT_LE(std::fabs(omp.ax[i] - serial.ax[i]), tol * norm) << "particle " << i;
        EXPECT_LE(std::fabs(omp.ay[i] - serial.ay[i]), tol * norm) << "particle " << i;
        EXPECT_LE(std::fabs(omp.az[i] - serial.az[i]), tol * norm) << "particle " << i;
    }
}

// Runs the OpenMP backend with a given thread count, then restores the previous settings
// so tests can't affect each other. Dynamic adjustment is turned off so the runtime
// can't quietly use fewer threads than asked (e.g. under OMP_DYNAMIC=true).
// These tests prove the backends AGREE; they can't tell whether the work actually ran in
// parallel (a deleted #pragma still matches serial). Parallelism is shown by
// bench/bench_gravity and race-freedom by the tsan preset with Archer.
void gravity_omp_with_threads(Particles &p, const SimParams &params, int threads) {
    int saved_threads = omp_get_max_threads();
    int saved_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    compute_gravity_omp(p, params);
    omp_set_num_threads(saved_threads);
    omp_set_dynamic(saved_dynamic);
}

} // namespace

TEST(BackendsMatch, OmpMatchesSerialAtSeveralThreadCounts) {
    const SimParams params = test_params();
    Particles serial = random_system(1000, 3);
    compute_gravity_serial(serial, params);

    const int thread_counts[] = {1, 2, 4, 12};
    for (int threads : thread_counts) {
        SCOPED_TRACE(testing::Message() << threads << " threads");
        Particles omp = random_system(1000, 3);
        gravity_omp_with_threads(omp, params, threads);
        expect_matches_serial(omp, serial, 1e-12);
    }
}

TEST(BackendsMatch, OmpHandlesUnevenAndTinySystems) {
    const SimParams params = test_params();
    struct Case {
        std::size_t n;
        int threads;
    };
    // 37 doesn't split evenly over 4 threads; 2 particles on 12 threads leaves most idle.
    const Case cases[] = {{37, 4}, {2, 12}, {1, 4}, {0, 4}};
    for (const Case &c : cases) {
        SCOPED_TRACE(testing::Message() << "n=" << c.n << " threads=" << c.threads);
        Particles serial = random_system(c.n, 5);
        Particles omp = random_system(c.n, 5);
        compute_gravity_serial(serial, params);
        gravity_omp_with_threads(omp, params, c.threads);
        expect_matches_serial(omp, serial, 1e-12);
    }
}

TEST(BackendsMatch, OmpOverwritesPreviousAccelerations) {
    const SimParams params = test_params();
    Particles serial = random_system(50, 9);
    Particles omp = random_system(50, 9);
    for (std::size_t i = 0; i < omp.size(); ++i) {
        omp.ax[i] = omp.ay[i] = omp.az[i] = 1e6; // stale values must not leak through
    }
    compute_gravity_serial(serial, params);
    gravity_omp_with_threads(omp, params, 4);
    expect_matches_serial(omp, serial, 1e-12);
}

TEST(BackendsMatch, DispatcherRoutesOmpBackend) {
    const SimParams params = test_params();
    Particles direct = random_system(100, 13);
    Particles dispatched = random_system(100, 13);
    compute_gravity_omp(direct, params);
    compute_gravity(dispatched, params, BACKEND_OMP);
    expect_matches_serial(dispatched, direct, 1e-12);
}
