#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>

#include <unistd.h>

#include "config.hpp"
#include "init.hpp"
#include "sim.hpp"

// Layered planets (#50): an iron core inside a rock mantle. Same particle size, heavier
// core particles, chosen as whole lattice shells so the core is centered.

namespace {

// head_on.cfg's planet with an Earth-like core.
PlanetSpec layered_spec() {
    PlanetSpec s;
    s.n = 2000;
    s.radius = 1.0;
    s.particle_radius = 0.065;
    s.mass = 2.0;
    s.seed = 1;
    s.center = Vec3{1.25, -0.5, 3.0}; // not the origin, so centering is really tested
    s.core_radius_fraction = 0.55;
    s.core_density_ratio = 2.5;
    return s;
}

struct Centers {
    Vec3 all, core;
    double mass_all = 0.0, mass_core = 0.0;
};

Centers centers(const Particles &p) {
    Centers c;
    c.all = c.core = Vec3{0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < p.size(); ++i) {
        const double m = p.mass[i];
        c.mass_all += m;
        c.all.x += m * p.x[i];
        c.all.y += m * p.y[i];
        c.all.z += m * p.z[i];
        if (p.material[i] == MATERIAL_IRON) {
            c.mass_core += m;
            c.core.x += m * p.x[i];
            c.core.y += m * p.y[i];
            c.core.z += m * p.z[i];
        }
    }
    c.all = Vec3{c.all.x / c.mass_all, c.all.y / c.mass_all, c.all.z / c.mass_all};
    if (c.mass_core > 0.0) {
        c.core = Vec3{c.core.x / c.mass_core, c.core.y / c.mass_core, c.core.z / c.mass_core};
    }
    return c;
}

double dist(Vec3 a, Vec3 b) {
    return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) +
                     (a.z - b.z) * (a.z - b.z));
}

double dist_from(const Particles &p, std::size_t i, Vec3 c) {
    return dist(Vec3{p.x[i], p.y[i], p.z[i]}, c);
}

} // namespace

TEST(Layered, DefaultsGiveTheUnlayeredPlanetExactly) {
    // No core: every particle rock with mass / n, exactly as before #50, and the same
    // positions whether the defaults are left alone or a ratio is given without a core.
    PlanetSpec s = layered_spec();
    s.core_radius_fraction = 0.0;
    s.core_density_ratio = 1.0;
    PlanetSpec with_ratio = s;
    with_ratio.core_density_ratio = 2.5; // no core, so the ratio changes nothing
    Particles a, b;
    make_planet(a, s);
    make_planet(b, with_ratio);
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        ASSERT_EQ(a.mass[i], s.mass / s.n) << "particle " << i;
        ASSERT_EQ(a.material[i], MATERIAL_ROCK) << "particle " << i;
        ASSERT_EQ(a.x[i], b.x[i]) << "particle " << i;
        ASSERT_EQ(a.y[i], b.y[i]) << "particle " << i;
        ASSERT_EQ(a.z[i], b.z[i]) << "particle " << i;
        ASSERT_EQ(a.mass[i], b.mass[i]) << "particle " << i;
    }
}

TEST(Layered, MassesAddUpAndCoreIsHeavier) {
    const PlanetSpec s = layered_spec();
    Particles p;
    make_planet(p, s);
    int n_core = 0;
    double m_core = 0.0, m_mantle = 0.0, total = 0.0;
    ASSERT_TRUE(planet_particle_masses(s, &n_core, &m_core, &m_mantle));
    int iron = 0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        total += p.mass[i];
        if (p.material[i] == MATERIAL_IRON) {
            ++iron;
            ASSERT_EQ(p.mass[i], m_core) << "particle " << i;
        } else {
            ASSERT_EQ(p.mass[i], m_mantle) << "particle " << i;
        }
    }
    EXPECT_EQ(iron, n_core);
    EXPECT_EQ(m_core, 2.5 * m_mantle);
    EXPECT_NEAR(total, s.mass, 1e-12 * s.mass); // summing 2,000 doubles rounds a little
    // The core's share of the mass follows from the counts. With f = 0.55 of the
    // filled radius, about f^3 of the particles are core: measured 321 of 2,000, 32% of
    // the mass (Earth's core is about 32%).
    const double share = n_core * 2.5 / (s.n - n_core + 2.5 * n_core);
    EXPECT_NEAR(centers(p).mass_core / total, share, 1e-12);
    std::printf("layered n=%d: %d core particles, core mass share %.3f\n", s.n, n_core, share);
    EXPECT_GT(n_core, 200);
    EXPECT_LT(n_core, 500);
}

TEST(Layered, CoreIsWholeShellsAndEverythingIsCentered) {
    const PlanetSpec s = layered_spec();
    Particles p;
    make_planet(p, s);
    Centers c = centers(p);
    // Lattice positions are centered on the mass-weighted mean, so the center of mass
    // is spec.center up to the random jitter (expected ~2e-5 here). Centering on the
    // plain mean instead put it 2.5e-3 off.
    EXPECT_LT(dist(c.all, s.center), 2e-4);
    // The core is whole shells, symmetric about the lattice origin. The outermost
    // mantle shell is only partly filled, so to keep the planet's center of mass in
    // place the core sits slightly off it: measured 4.9e-3, 3.8% of the particle
    // spacing (the plain-mean centering had it 1.1e-2 off).
    EXPECT_LT(dist(c.core, s.center), 0.05 * 2.0 * s.particle_radius);
    // Whole shells: every core particle is closer to the center than every mantle
    // particle, up to the jitter (2% of a particle radius, twice).
    double core_max = 0.0, mantle_min = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < p.size(); ++i) {
        double r = dist_from(p, i, s.center);
        if (p.material[i] == MATERIAL_IRON) {
            core_max = std::fmax(core_max, r);
        } else {
            mantle_min = std::fmin(mantle_min, r);
        }
    }
    EXPECT_LT(core_max, mantle_min + 4.0 * 0.02 * s.particle_radius);
    // Distances alone can't tell a whole shell from a split one (a shell's points are
    // all the same distance out), but the count can: the core must hold exactly the
    // points of the first k FCC shells, one of these totals (checked against a direct
    // count of the lattice).
    const int whole_shells[] = {1,   13,  19,  43,  55,  79,  87,  135, 141, 177,
                                201, 225, 249, 321, 369, 381, 429, 459, 531, 555};
    int iron = 0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        iron += p.material[i] == MATERIAL_IRON ? 1 : 0;
    }
    bool is_whole = false;
    for (int count : whole_shells) {
        is_whole = is_whole || iron == count;
    }
    EXPECT_TRUE(is_whole) << iron << " core particles is not a whole number of shells";
}

TEST(Layered, InvalidCoreSettingsAreRejected) {
    struct Case {
        double f, ratio;
        const char *message;
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const Case cases[] = {
        {-0.1, 2.5, "core_radius_fraction must be >= 0 and < 1"},
        {1.0, 2.5, "core_radius_fraction must be >= 0 and < 1"},
        {nan, 2.5, "core_radius_fraction must be >= 0 and < 1"},
        {0.5, 0.5, "core_density_ratio must be finite and >= 1"},
        {0.5, nan, "core_density_ratio must be finite and >= 1"},
        {0.5, inf, "core_density_ratio must be finite and >= 1"},
    };
    for (const Case &c : cases) {
        PlanetSpec s = layered_spec();
        s.core_radius_fraction = c.f;
        s.core_density_ratio = c.ratio;
        const char *why = nullptr;
        EXPECT_FALSE(planet_spec_ok(s, &why)) << c.f << " " << c.ratio;
        ASSERT_NE(why, nullptr);
        EXPECT_STREQ(why, c.message);
        s.n = 0; // checked even for an empty planet
        EXPECT_FALSE(planet_spec_ok(s, &why)) << c.f << " " << c.ratio;
    }
}

TEST(Layered, PlacePlanetCopiesMaterial) {
    Particles planet, all;
    make_planet(planet, layered_spec());
    place_planet(all, planet, Vec3{5.0, 0.0, 0.0}, Vec3{0.0, 0.0, 0.0});
    ASSERT_EQ(all.size(), planet.size());
    for (std::size_t i = 0; i < all.size(); ++i) {
        ASSERT_EQ(all.material[i], planet.material[i]) << "particle " << i;
    }
}

namespace {

// build_planets' notes (stderr) for one planet, without relaxing it.
std::string build_notes(const PlanetSpec &spec, const SimParams &params) {
    Config cfg;
    cfg.params = params;
    cfg.planets.push_back(spec);
    testing::internal::CaptureStderr();
    build_planets(cfg, BACKEND_SERIAL);
    return testing::internal::GetCapturedStderr();
}

} // namespace

TEST(Layered, TinyCoreIsNoted) {
    PlanetSpec s = layered_spec();
    s.core_radius_fraction = 0.05; // only the center particle
    EXPECT_NE(build_notes(s, SimParams()).find("core_radius_fraction = 0.05 gives only 1 core"),
              std::string::npos);
    s.core_radius_fraction = 0.55;
    EXPECT_EQ(build_notes(s, SimParams()).find("core particle"), std::string::npos);
    s.core_radius_fraction = 0.0; // no core asked for: nothing to note
    EXPECT_EQ(build_notes(s, SimParams()).find("core particle"), std::string::npos);
}

TEST(Layered, DamperNoteUsesTheLightestParticle) {
    // c * dt / m_eff = 1.8 for two mantle-weight particles of the unlayered planet (mass
    // 1e-3), fine. Layered, a mantle particle is lighter (2 / (1679 + 2.5 * 321) =
    // 8.06e-4, 19% lighter), so the same settings give 2.23 and must be noted.
    SimParams params;
    params.contact_enabled = true;
    params.contact_k = 20.0;
    params.dt = 0.002;
    params.contact_c = 1.8 * 0.5e-3 / params.dt;
    PlanetSpec s = layered_spec();
    EXPECT_NE(build_notes(s, params).find("is too strong"), std::string::npos);
    s.core_radius_fraction = 0.0;
    EXPECT_EQ(build_notes(s, params).find("is too strong"), std::string::npos);
}

TEST(Layered, RelaxedCoreStaysInTheMiddle) {
    // smoke.cfg's planet with a core, relaxed with smoke.cfg's settings.
    PlanetSpec s;
    s.n = 250;
    s.radius = 0.5;
    s.particle_radius = 0.06;
    s.mass = 0.25;
    s.seed = 1;
    s.core_radius_fraction = 0.55;
    s.core_density_ratio = 2.5;
    SimParams params;
    params.dt = 0.002;
    params.softening = 0.05;
    params.contact_enabled = true;
    params.contact_k = 20.0;
    params.contact_c = 0.05;
    Particles p;
    make_planet(p, s);
    int n_core = 0;
    double m_core = 0.0, m_mantle = 0.0;
    ASSERT_TRUE(planet_particle_masses(s, &n_core, &m_core, &m_mantle));
    ASSERT_GE(n_core, 13);
    RelaxResult r = relax_planet(p, params, BACKEND_OMP, 5.0, 1e-6, 2000);
    ASSERT_TRUE(r.finite);
    EXPECT_TRUE(r.settled);

    Centers c = centers(p);
    double core_mean = 0.0, mantle_mean = 0.0, max_overlap = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        const bool iron = p.material[i] == MATERIAL_IRON;
        EXPECT_EQ(p.mass[i], iron ? m_core : m_mantle) << "particle " << i;
        (iron ? core_mean : mantle_mean) += dist_from(p, i, c.all);
        for (std::size_t j = i + 1; j < p.size(); ++j) {
            double d = dist(Vec3{p.x[i], p.y[i], p.z[i]}, Vec3{p.x[j], p.y[j], p.z[j]});
            max_overlap = std::fmax(max_overlap, (p.radius[i] + p.radius[j] - d) / p.radius[i]);
        }
    }
    core_mean /= n_core;
    mantle_mean /= static_cast<double>(s.n - n_core);
    const double spacing = 2.0 * s.particle_radius;
    std::printf("relaxed layered planet: core center off by %.3g, core mean radius %.3g, "
                "mantle %.3g, max overlap %.3g of a particle radius\n",
                dist(c.core, c.all), core_mean, mantle_mean, max_overlap);
    // The heavy core stays in the middle: its center within a fifth of a particle
    // spacing of the planet's (measured 1.4e-3, 1% of it), and well inside the mantle.
    // This checks relaxation keeps the layers (no inversion, no blow-up). It can't show
    // a light core would rise: that takes far longer than relaxation runs.
    EXPECT_LT(dist(c.core, c.all), 0.2 * spacing);
    EXPECT_LT(core_mean, 0.6 * mantle_mean);
    // Heavier core particles press harder on each other under the same contact stiffness.
    // make_planet's limit for a fresh planet is 10% overlap; measured 0.12%.
    EXPECT_LT(max_overlap, 0.1);
}
