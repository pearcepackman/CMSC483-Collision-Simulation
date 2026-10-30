#include <csignal>

#include <gtest/gtest.h>

// Every module header must compile when included together. (Each .cpp includes its
// own header first, which checks each header compiles on its own.)
#include "config.hpp"
#include "contact.hpp"
#include "forces.hpp"
#include "gravity.hpp"
#include "init.hpp"
#include "integrator.hpp"
#include "io.hpp"
#include "params.hpp"
#include "particles.hpp"
#include "util.hpp"

TEST(Backend, NamesRoundTrip) {
    const Backend all[] = {BACKEND_SERIAL, BACKEND_OMP, BACKEND_HIP};
    for (Backend b : all) {
        Backend parsed = BACKEND_SERIAL;
        ASSERT_TRUE(backend_from_name(backend_name(b), &parsed)) << backend_name(b);
        EXPECT_EQ(parsed, b);
    }
    EXPECT_STREQ(backend_name(BACKEND_SERIAL), "serial");
    EXPECT_STREQ(backend_name(BACKEND_OMP), "omp");
    EXPECT_STREQ(backend_name(BACKEND_HIP), "hip");
}

TEST(Backend, UnknownNameIsRejected) {
    Backend parsed = BACKEND_OMP;
    EXPECT_FALSE(backend_from_name("cuda", &parsed));
    EXPECT_FALSE(backend_from_name("", &parsed));
    EXPECT_FALSE(backend_from_name("Serial", &parsed)); // names are exact and case-sensitive
    EXPECT_FALSE(backend_from_name(nullptr, &parsed));  // e.g. `--backend` with no value
    EXPECT_EQ(parsed, BACKEND_OMP);                     // untouched on failure
}

TEST(Params, DefaultsAreZeroAndContactOff) {
    SimParams params; // no braces on purpose: fields must still be initialized
    EXPECT_EQ(params.dt, 0.0);
    EXPECT_EQ(params.softening, 0.0);
    EXPECT_FALSE(params.contact_enabled);
    Config cfg;
    EXPECT_EQ(cfg.steps, 0);
    EXPECT_EQ(cfg.output_every, 1);
    EXPECT_TRUE(cfg.planets.empty());
}

// Unfinished code must fail loudly (abort), so it can't make a test or run look fine.
// Tests the helper directly so it keeps working as stubs get implemented.
TEST(StubsDeathTest, NotImplementedAborts) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(not_implemented("demo"), testing::KilledBySignal(SIGABRT), "not implemented: demo");
}

TEST(Backend, AvailableMatchesBuild) {
    EXPECT_TRUE(backend_available(BACKEND_SERIAL));
    EXPECT_TRUE(backend_available(BACKEND_OMP));
#ifdef SIM_HAVE_HIP
    EXPECT_TRUE(backend_available(BACKEND_HIP));
#else
    EXPECT_FALSE(backend_available(BACKEND_HIP));
#endif
    EXPECT_FALSE(backend_available(static_cast<Backend>(3)));
}

#ifndef SIM_HAVE_HIP
// Without ENABLE_HIP the HIP branch must abort, not silently skip gravity. (In HIP
// builds it runs on the GPU: tests/gpu/.)
TEST(StubsDeathTest, GravityHipNotBuiltAborts) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    Particles p;
    p.resize(2);
    SimParams params;
    EXPECT_DEATH(compute_gravity(p, params, BACKEND_HIP), "hip backend not built");
}
#endif

// A bad Backend value (e.g. from a cast) must abort, not silently skip gravity and
// leave the previous step's accelerations in place.
TEST(StubsDeathTest, GravityUnknownBackendAborts) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    Particles p;
    p.resize(2);
    SimParams params;
    EXPECT_DEATH(compute_gravity(p, params, static_cast<Backend>(3)), "unknown backend");
}
