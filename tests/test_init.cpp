#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "init.hpp"

namespace {

// A typical planet: 2,000 particles in a unit ball.
PlanetSpec typical_spec() {
    PlanetSpec s;
    s.n = 2000;
    s.radius = 1.0;
    s.particle_radius = 0.04;
    s.mass = 3.0;
    s.seed = 7;
    s.body = 1;
    return s;
}

double distance(const Particles &p, std::size_t i, std::size_t j) {
    double dx = p.x[i] - p.x[j];
    double dy = p.y[i] - p.y[j];
    double dz = p.z[i] - p.z[j];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

} // namespace

TEST(Init, MakesExactlyNParticles) {
    const int counts[] = {1, 13, 500, 2000};
    for (int n : counts) {
        PlanetSpec s = typical_spec();
        s.n = n;
        Particles p;
        make_planet(p, s);
        EXPECT_EQ(p.size(), static_cast<std::size_t>(n)) << "n = " << n;
    }
}

TEST(Init, EveryParticleInsideSphere) {
    PlanetSpec s = typical_spec();
    s.center = Vec3{5.0, -2.0, 0.5};
    Particles p;
    make_planet(p, s);
    ASSERT_EQ(p.size(), 2000u);
    for (std::size_t i = 0; i < p.size(); ++i) {
        double dx = p.x[i] - s.center.x;
        double dy = p.y[i] - s.center.y;
        double dz = p.z[i] - s.center.z;
        double edge = std::sqrt(dx * dx + dy * dy + dz * dz) + s.particle_radius;
        EXPECT_LE(edge, s.radius * (1.0 + 1e-12)) << "particle " << i; // whole particle inside
    }
}

TEST(Init, SameSeedSameOutput) {
    Particles a, b, c;
    PlanetSpec s = typical_spec();
    make_planet(a, s);
    make_planet(b, s);
    s.seed = 8;
    make_planet(c, s);
    ASSERT_EQ(a.size(), b.size());
    ASSERT_EQ(a.size(), c.size());
    bool differs = false;
    for (std::size_t i = 0; i < a.size(); ++i) { // same code, same input: exact match
        EXPECT_EQ(a.x[i], b.x[i]);
        EXPECT_EQ(a.y[i], b.y[i]);
        EXPECT_EQ(a.z[i], b.z[i]);
        if (a.x[i] != c.x[i] || a.y[i] != c.y[i] || a.z[i] != c.z[i]) {
            differs = true;
        }
    }
    EXPECT_TRUE(differs) << "a different seed should give a different planet";
}

// The same seed must give the same planet with every compiler and flag set (gcc and
// clang in CI, -march=native in the release preset), not just within one build. These
// positions came from gcc -O2. If you change how planets are made on purpose,
// regenerate them; if they change by accident, a backend comparison could be testing
// two different planets.
TEST(Init, MatchesStoredPlanet) {
    const double expected[13][3] = {
        {0.0017432454771920272, 0.0020459602779207735, 0.00076951024622746544},
        {-0.56367743418916594, -0.56438876809147209, 0.0026573469790059983},
        {-0.55992465773701106, -0.0018591090926555448, -0.56189250521990897},
        {-0.56420953560000553, 0.0013357178573722148, 0.56399403781541257},
        {-0.56123227702690404, 0.56232843400274068, 0.0031965128168237487},
        {0.0013549229661320857, -0.56461954471589559, -0.56556704049187523},
        {0.0022785304106965605, -0.56570596820460761, 0.56335487811780105},
        {0.00075822073601394552, 0.56164996812281054, -0.56652383566231912},
        {0.00056036637479383884, 0.56150219521345146, 0.55986727861825547},
        {0.56325053452599938, -0.56236176484288847, 0.0013014999142883239},
        {0.55962544014958904, 0.0023163310658625724, -0.56259451100038516},
        {0.56126352976366067, -0.0022567581181522048, 0.56024897558768738},
        {0.55951429506411809, 0.56440626209309996, 5.063768689339909e-05},
    };
    PlanetSpec s = typical_spec();
    s.n = 13;
    s.particle_radius = 0.2;
    Particles p;
    make_planet(p, s);
    ASSERT_EQ(p.size(), 13u);
    for (std::size_t i = 0; i < 13; ++i) { // rounding may differ in the last bits only
        EXPECT_NEAR(p.x[i], expected[i][0], 1e-12) << "particle " << i;
        EXPECT_NEAR(p.y[i], expected[i][1], 1e-12) << "particle " << i;
        EXPECT_NEAR(p.z[i], expected[i][2], 1e-12) << "particle " << i;
    }
}

TEST(Init, MaxOverlapUnderTenPercent) {
    // 0.04 leaves gaps between neighbors. 0.068 is just under the largest radius that
    // fits 2,000 particles (about 0.0688), so the overlap is close to the 10% limit.
    const double radii[] = {0.04, 0.068};
    for (double pr : radii) {
        PlanetSpec s = typical_spec();
        s.particle_radius = pr;
        Particles p;
        make_planet(p, s);
        double max_overlap = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < p.size(); ++i) { // brute force over all pairs
            for (std::size_t j = i + 1; j < p.size(); ++j) {
                double overlap = 2.0 * s.particle_radius - distance(p, i, j);
                if (overlap > max_overlap) {
                    max_overlap = overlap;
                }
            }
        }
        EXPECT_LT(max_overlap, 0.1 * s.particle_radius) << "particle_radius " << pr;
    }
}

TEST(Init, AppendsAndSetsFields) {
    Particles p;
    p.resize(5);
    for (std::size_t i = 0; i < 5; ++i) {
        p.x[i] = 100.0 + static_cast<double>(i); // existing particles must stay untouched
        p.mass[i] = 9.0;
        p.body[i] = 4;
    }
    PlanetSpec s = typical_spec();
    s.n = 500;
    s.velocity = Vec3{0.5, -1.0, 2.0};
    make_planet(p, s);
    ASSERT_EQ(p.size(), 505u);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(p.x[i], 100.0 + static_cast<double>(i));
        EXPECT_EQ(p.mass[i], 9.0);
        EXPECT_EQ(p.body[i], 4);
    }
    double total_mass = 0.0;
    for (std::size_t i = 5; i < p.size(); ++i) {
        EXPECT_DOUBLE_EQ(p.mass[i], s.mass / s.n);
        EXPECT_EQ(p.radius[i], s.particle_radius);
        EXPECT_EQ(p.body[i], s.body);
        EXPECT_EQ(p.vx[i], 0.5);
        EXPECT_EQ(p.vy[i], -1.0);
        EXPECT_EQ(p.vz[i], 2.0);
        EXPECT_EQ(p.ax[i], 0.0); // forces get computed later
        EXPECT_EQ(p.ay[i], 0.0);
        EXPECT_EQ(p.az[i], 0.0);
        total_mass += p.mass[i];
    }
    EXPECT_LE(std::fabs(total_mass - s.mass), 1e-12 * s.mass);
}

// A filled, centered ball: catches keeping the wrong points (a hollow shell, or a
// lopsided cap from skipping the sort) and an off-center planet.
TEST(Init, FillsBallCompactly) {
    const int counts[] = {2, 13, 300, 2000};
    for (int n : counts) {
        PlanetSpec s = typical_spec();
        s.n = n;
        s.center = Vec3{1.0, -2.0, 3.0};
        Particles p;
        make_planet(p, s);
        double cx = 0.0, cy = 0.0, cz = 0.0;
        int inner = 0; // particles within half the radius of the center
        for (std::size_t i = 0; i < p.size(); ++i) {
            double dx = p.x[i] - s.center.x;
            double dy = p.y[i] - s.center.y;
            double dz = p.z[i] - s.center.z;
            cx += dx;
            cy += dy;
            cz += dz;
            if (std::sqrt(dx * dx + dy * dy + dz * dz) <= 0.5 * s.radius) {
                ++inner;
            }
        }
        // Equal masses, so the center of mass is the mean position. Only the jitter
        // (at most 2% of particle_radius) may move it off the center.
        double off = std::sqrt(cx * cx + cy * cy + cz * cz) / n;
        EXPECT_LE(off, 0.02 * s.particle_radius) << "n = " << n;
        if (n >= 300) { // a filled ball has about 1/8 of its particles in the inner half
            EXPECT_GE(inner, n / 10) << "n = " << n;
        }
    }
}

// One particle always fits, even when it nearly fills the planet.
TEST(Init, SingleParticleSitsAtCenter) {
    PlanetSpec s = typical_spec();
    s.n = 1;
    s.particle_radius = 0.5;
    s.center = Vec3{1.0, 2.0, 3.0};
    Particles p;
    make_planet(p, s);
    ASSERT_EQ(p.size(), 1u);
    double jitter = 0.02 * s.particle_radius;
    EXPECT_NEAR(p.x[0], 1.0, jitter);
    EXPECT_NEAR(p.y[0], 2.0, jitter);
    EXPECT_NEAR(p.z[0], 3.0, jitter);
}

TEST(Init, ZeroParticlesIsNoOp) {
    Particles p;
    p.resize(3);
    PlanetSpec s = typical_spec();
    s.n = 0;
    make_planet(p, s);
    EXPECT_EQ(p.size(), 3u);
}

// Just past the largest particle_radius that fits 2,000 particles (about 0.0688): the
// overlap would break the 10% rule, so it must abort. MaxOverlapUnderTenPercent checks
// 0.068 still works, so together they pin the limit.
TEST(InitDeathTest, ImpossibleSpecAborts) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    PlanetSpec s = typical_spec();
    s.particle_radius = 0.07;
    Particles p;
    EXPECT_DEATH(make_planet(p, s),
                 "make_planet: too many particles for this radius and particle_radius .*n=2000");
}

// Each bad spec must hit its own check: the message says which one, so a test can't
// pass because a later check happened to fire.
TEST(InitDeathTest, InvalidSpecAborts) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    Particles p;
    PlanetSpec s = typical_spec();
    s.radius = s.particle_radius; // no room for any particle
    EXPECT_DEATH(make_planet(p, s), "make_planet: radius is too small for particle_radius");
    s = typical_spec();
    s.n = -1;
    EXPECT_DEATH(make_planet(p, s), "make_planet: n must be >= 0");

    const char *bad_number =
        "make_planet: radius, particle_radius and mass must be finite and >= 0";
    s = typical_spec();
    s.radius = nan;
    EXPECT_DEATH(make_planet(p, s), bad_number);
    s = typical_spec();
    s.particle_radius = -0.04;
    EXPECT_DEATH(make_planet(p, s), bad_number);
    s = typical_spec();
    s.particle_radius = inf;
    EXPECT_DEATH(make_planet(p, s), bad_number);
    s = typical_spec();
    s.mass = -1.0;
    EXPECT_DEATH(make_planet(p, s), bad_number);
    s = typical_spec();
    s.mass = inf;
    EXPECT_DEATH(make_planet(p, s), bad_number);
}
