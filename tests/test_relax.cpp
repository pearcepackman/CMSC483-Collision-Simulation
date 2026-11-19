#include <gtest/gtest.h>

#include <climits>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

#include "cli.hpp"
#include "config.hpp"
#include "diagnostics.hpp"
#include "forces.hpp"
#include "init.hpp"
#include "integrator.hpp"
#include "sim.hpp"

namespace {

// The smoke.cfg planet: 250 particles of mass 1e-3 in a ball of radius 0.5, with the
// contact settings both configs use.
PlanetSpec small_planet() {
    PlanetSpec s;
    s.n = 250;
    s.radius = 0.5;
    s.particle_radius = 0.06;
    s.mass = 0.25;
    s.seed = 1;
    return s;
}

SimParams relax_params() {
    SimParams params;
    params.dt = 0.002;
    params.softening = 0.05;
    params.contact_enabled = true;
    params.contact_k = 20.0;
    params.contact_c = 0.05;
    return params;
}

const double kDrag = 5.0;
const double kThreshold = 1e-6;
const int kMaxSteps = 2000;

Vec3 center_of_mass(const Particles &p) {
    double m = 0.0;
    Vec3 c{0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < p.size(); ++i) {
        m += p.mass[i];
        c.x += p.mass[i] * p.x[i];
        c.y += p.mass[i] * p.y[i];
        c.z += p.mass[i] * p.z[i];
    }
    return Vec3{c.x / m, c.y / m, c.z / m};
}

// Farthest particle center from the center of mass.
double max_center_distance(const Particles &p) {
    Vec3 c = center_of_mass(p);
    double r2 = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        double dx = p.x[i] - c.x, dy = p.y[i] - c.y, dz = p.z[i] - c.z;
        r2 = std::fmax(r2, dx * dx + dy * dy + dz * dz);
    }
    return std::sqrt(r2);
}

// The planet's outer radius: farthest particle edge from the center of mass.
double outer_radius(const Particles &p) {
    return max_center_distance(p) + p.radius[0];
}

Particles fresh_planet() {
    Particles p;
    make_planet(p, small_planet());
    return p;
}

// A planet of one particle at rest: no forces act on it, and its kinetic energy is 0.
Particles one_particle() {
    Particles p;
    p.resize(1);
    p.mass[0] = 1.0;
    p.radius[0] = 0.1;
    return p;
}

// The smoke planet's relaxation with the config values: how many steps it needs.
int smoke_planet_steps() {
    Particles p = fresh_planet();
    return relax_planet(p, relax_params(), BACKEND_SERIAL, kDrag, kThreshold, kMaxSteps).steps;
}

Config one_planet_config() {
    Config cfg;
    cfg.params = relax_params();
    cfg.relax_drag = kDrag;
    cfg.relax_ke_threshold = kThreshold;
    cfg.relax_max_steps = kMaxSteps;
    cfg.planets.push_back(small_planet());
    return cfg;
}

// Runs build_planets and returns what it printed to stderr.
std::string build_stderr(const Config &cfg, Particles *out) {
    testing::internal::CaptureStderr();
    *out = build_planets(cfg, BACKEND_SERIAL);
    return testing::internal::GetCapturedStderr();
}

// Writes a config file into this test's own folder (pid + test name, so parallel test
// runs can't clash) and returns its path.
std::string write_cfg(const std::string &text) {
    const char *name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::filesystem::path dir = std::filesystem::path(::testing::TempDir()) /
                                ("test_relax_" + std::to_string(getpid())) / name;
    std::filesystem::create_directories(dir);
    std::string path = (dir / "run.cfg").string();
    FILE *f = std::fopen(path.c_str(), "wb");
    EXPECT_NE(f, nullptr) << path;
    if (f != nullptr) {
        std::fputs(text.c_str(), f);
        std::fclose(f);
    }
    return path;
}

// One small relaxed planet: steps and dt come from the file unless overridden.
const char *kRunCfg = "output_every = 1\nsteps = 3\ndt = 0.002\nsoftening = 0.05\n"
                      "contact = true\ncontact_k = 20\nrelax_drag = 5\n"
                      "relax_ke_threshold = 1e-6\nrelax_max_steps = 5\n"
                      "planet0.n = 10\nplanet0.radius = 0.5\n"
                      "planet0.particle_radius = 0.06\nplanet0.mass = 0.01\n";

bool parse(std::vector<const char *> args, CliOptions *opts) {
    args.insert(args.begin(), "planet_sim");
    std::string error;
    bool ok = parse_cli(static_cast<int>(args.size()), args.data(), opts, &error);
    EXPECT_TRUE(ok) << error;
    return ok;
}

} // namespace

TEST(Relax, SettlesBelowThresholdAndStaysSettled) {
    Particles p = fresh_planet();
    const SimParams params = relax_params();
    RelaxResult r = relax_planet(p, params, BACKEND_SERIAL, kDrag, kThreshold, kMaxSteps);
    EXPECT_TRUE(r.settled);
    EXPECT_TRUE(r.finite);
    EXPECT_GT(r.steps, 0);
    EXPECT_LT(r.steps, kMaxSteps); // settled, didn't just run out of steps
    EXPECT_LT(kinetic_energy(p), kThreshold);

    // A settled planet stays settled without the drag: run on with the normal forces
    // only. A rule that stopped at a turning point of the planet's sag-and-rebound
    // (kinetic energy passes near zero there) would leave it moving.
    compute_accelerations(p, params, BACKEND_SERIAL);
    double ke_max = 0.0;
    for (int i = 0; i < 500; ++i) {
        leapfrog_step(p, params, BACKEND_SERIAL);
        ke_max = std::fmax(ke_max, kinetic_energy(p));
    }
    EXPECT_LT(ke_max, 10.0 * kThreshold);
}

TEST(Relax, RadiusChangesLessThanTenPercent) {
    Particles p = fresh_planet();
    const double r0 = outer_radius(p);
    relax_planet(p, relax_params(), BACKEND_SERIAL, kDrag, kThreshold, kMaxSteps);
    const double r1 = outer_radius(p);
    EXPECT_LT(std::fabs(r1 - r0) / r0, 0.1) << "r0 = " << r0 << ", r1 = " << r1;
    EXPECT_LT(r1, r0); // gravity squeezes the fresh lattice a little
}

TEST(Relax, StaysAtRestAtTheOrigin) {
    // Relaxing must not push the planet anywhere: forces cancel in pairs and the drag
    // scales every velocity alike, so momentum stays at rounding level.
    Particles p = fresh_planet();
    const Vec3 c0 = center_of_mass(p);
    relax_planet(p, relax_params(), BACKEND_SERIAL, kDrag, kThreshold, kMaxSteps);
    const Vec3 c1 = center_of_mass(p);
    EXPECT_LT(std::fabs(c1.x - c0.x) + std::fabs(c1.y - c0.y) + std::fabs(c1.z - c0.z), 1e-9);
    Vec3 mom = total_momentum(p);
    EXPECT_LT(std::fabs(mom.x) + std::fabs(mom.y) + std::fabs(mom.z), 1e-12);
}

TEST(Relax, StartingAtRestDoesNotCountAsSettled) {
    // A fresh planet has zero kinetic energy, below any threshold. Even with a threshold
    // that is never crossed, relaxation must hold for one dynamical time
    // sqrt(R^3 / (G M)) of the planet, R = farthest center from the center of mass.
    Particles p = fresh_planet();
    const double r = max_center_distance(p);
    const double dt = relax_params().dt;
    const double t_dyn = std::sqrt(r * r * r / 0.25);
    const int hold = static_cast<int>(std::ceil(t_dyn / dt));
    ASSERT_GT(hold, 100); // about 290 steps (0.58 time units) for this planet
    RelaxResult res = relax_planet(p, relax_params(), BACKEND_SERIAL, kDrag, 1e10, kMaxSteps);
    EXPECT_EQ(res.steps, hold);
    EXPECT_EQ(res.hold, hold);
    EXPECT_NEAR(res.t_dyn, t_dyn, 1e-12 * t_dyn);
    EXPECT_TRUE(res.settled);
}

TEST(Relax, BriefDipsDontAddUpToSettled) {
    // Kinetic energy must stay low for one whole dynamical time in a row. An eccentric
    // two-body orbit (no drag, no contact) is slowest at its far point, where kinetic
    // energy dips below 1.03x its minimum for at most 42 steps per orbit, while the hold
    // is 57 steps (R = 0.5, M = 1). The dips add up to 432 steps over 5,000 steps, so a
    // counter that never reset would call this "settled" after a few orbits.
    Particles p;
    p.resize(2);
    p.mass[0] = p.mass[1] = 0.5;
    p.x[0] = -0.5;
    p.x[1] = 0.5;
    p.vy[0] = -0.3; // below circular speed (0.5), so this point is the far end
    p.vy[1] = 0.3;
    SimParams params;
    params.dt = 2.0 * 3.14159265358979323846 / 1000.0;
    const double threshold = 1.03 * kinetic_energy(p);
    RelaxResult r = relax_planet(p, params, BACKEND_SERIAL, 0.0, threshold, 2000);
    EXPECT_EQ(r.steps, 2000);
    EXPECT_FALSE(r.settled);
}

TEST(Relax, ZeroThresholdCanNeverBeMet) {
    // Kinetic energy is never below 0, so even a particle that never moves (kinetic
    // energy exactly 0) doesn't settle: "below" means strictly below.
    Particles p = one_particle();
    RelaxResult r = relax_planet(p, relax_params(), BACKEND_SERIAL, kDrag, 0.0, 20);
    EXPECT_EQ(r.steps, 20);
    EXPECT_FALSE(r.settled);
}

TEST(Relax, StopsAtMaxSteps) {
    Particles p = fresh_planet();
    RelaxResult r = relax_planet(p, relax_params(), BACKEND_SERIAL, kDrag, kThreshold, 50);
    EXPECT_EQ(r.steps, 50);
    EXPECT_FALSE(r.settled);

    // max_steps = 0 takes no step and moves nothing.
    Particles q = fresh_planet();
    Particles before = q;
    r = relax_planet(q, relax_params(), BACKEND_SERIAL, kDrag, kThreshold, 0);
    EXPECT_EQ(r.steps, 0);
    EXPECT_FALSE(r.settled);
    for (std::size_t i = 0; i < q.size(); ++i) {
        EXPECT_EQ(q.x[i], before.x[i]);
        EXPECT_EQ(q.vx[i], before.vx[i]);
    }
}

TEST(Relax, PlanetStartsTheRunCold) {
    // 50 normal steps on a fresh planet heat it up (it sags, so contacts move) ...
    Particles hot;
    make_planet(hot, small_planet());
    SimParams params = relax_params();
    compute_accelerations(hot, params, BACKEND_SERIAL);
    for (int s = 0; s < 50; ++s) {
        leapfrog_step(hot, params, BACKEND_SERIAL);
    }
    ASSERT_GT(total_heat(hot), 0.0);
    // ... but after relaxing, every particle's heat is back to zero.
    Particles p;
    make_planet(p, small_planet());
    relax_planet(p, params, BACKEND_SERIAL, kDrag, kThreshold, 50);
    for (std::size_t i = 0; i < p.size(); ++i) {
        ASSERT_EQ(p.heat[i], 0.0) << "particle " << i;
    }
}

TEST(Relax, SettlingOnTheLastAllowedStepCounts) {
    // With exactly the steps it needs, the planet settles (no off-by-one), and one step
    // fewer is not enough.
    const int need = smoke_planet_steps();
    ASSERT_LT(need, kMaxSteps);
    Particles p = fresh_planet();
    RelaxResult r = relax_planet(p, relax_params(), BACKEND_SERIAL, kDrag, kThreshold, need);
    EXPECT_EQ(r.steps, need);
    EXPECT_TRUE(r.settled);
    Particles q = fresh_planet();
    r = relax_planet(q, relax_params(), BACKEND_SERIAL, kDrag, kThreshold, need - 1);
    EXPECT_FALSE(r.settled);
}

TEST(Relax, HugeHoldDoesNotOverflow) {
    // Two particles 2e6 apart: one dynamical time is about 5e11 steps, more than an int
    // holds. The hold must clamp, not wrap around to 1 (which would call an unrelaxed
    // planet settled after one step).
    Particles p;
    p.resize(2);
    p.mass[0] = p.mass[1] = 0.5;
    p.x[0] = -1e6;
    p.x[1] = 1e6;
    SimParams params;
    params.dt = 0.002;
    RelaxResult r = relax_planet(p, params, BACKEND_SERIAL, kDrag, 1e10, 1000);
    EXPECT_EQ(r.hold, INT_MAX);
    EXPECT_EQ(r.steps, 1000);
    EXPECT_FALSE(r.settled);
}

TEST(Relax, DragHasTheExactRate) {
    // A lone particle feels no force, so only the drag changes its velocity: after k
    // steps, v = v0 * exp(-drag * dt * k), in every component.
    Particles p = one_particle();
    p.vx[0] = 1.0;
    p.vy[0] = -2.0;
    p.vz[0] = 0.5;
    const SimParams params = relax_params();
    const int k = 100;
    relax_planet(p, params, BACKEND_SERIAL, kDrag, 1e-300, k); // threshold never reached
    const double f = std::exp(-kDrag * params.dt * k);
    EXPECT_NEAR(p.vx[0], 1.0 * f, 1e-12);
    EXPECT_NEAR(p.vy[0], -2.0 * f, 1e-12);
    EXPECT_NEAR(p.vz[0], 0.5 * f, 1e-12);
}

TEST(Relax, DragRemovesEnergy) {
    // With contact damping off, the drag is the only thing that removes energy. (With
    // the configs' c = 0.05, contact damping alone also settles this small planet, just
    // more slowly; for the 2,000-particle planet it is about 3x slower.) Same planet and
    // step count with and without drag, then compare the largest kinetic energy over the
    // next 100 undamped steps, since the planet oscillates and one snapshot can mislead.
    SimParams params = relax_params();
    params.contact_c = 0.0;
    double peak[2];
    const double drags[2] = {kDrag, 0.0};
    for (int k = 0; k < 2; ++k) {
        Particles p = fresh_planet();
        relax_planet(p, params, BACKEND_SERIAL, drags[k], 1e-300, 400);
        compute_accelerations(p, params, BACKEND_SERIAL);
        peak[k] = 0.0;
        for (int i = 0; i < 100; ++i) {
            leapfrog_step(p, params, BACKEND_SERIAL);
            peak[k] = std::fmax(peak[k], kinetic_energy(p));
        }
    }
    // Measured: 3.5e-5 with drag, 5.5e-4 without (16x). Not more, because the drag also
    // slows the sag toward the settled shape, so gravity keeps feeding in some motion.
    EXPECT_LT(peak[0], 0.2 * peak[1]) << "with drag " << peak[0] << ", without " << peak[1];
}

TEST(Relax, BlowUpStopsRightAway) {
    // A dt so large that the first step gives NaN/Inf: more steps can't help, so it stops.
    Particles p = fresh_planet();
    SimParams params = relax_params();
    params.dt = 1e300;
    RelaxResult r = relax_planet(p, params, BACKEND_SERIAL, kDrag, kThreshold, kMaxSteps);
    EXPECT_FALSE(r.finite);
    EXPECT_FALSE(r.settled);
    EXPECT_EQ(r.steps, 1);
}

TEST(Relax, OpenMPMatchesSerial) {
    // Relaxation runs on whatever backend the run uses. Over a few steps the OpenMP
    // result must match serial to rounding (compared with a tolerance, as for every
    // backend; over long runs the chaos of gravity would amplify the rounding).
    Particles a = fresh_planet();
    Particles b = fresh_planet();
    relax_planet(a, relax_params(), BACKEND_SERIAL, kDrag, kThreshold, 5);
    relax_planet(b, relax_params(), BACKEND_OMP, kDrag, kThreshold, 5);
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a.x[i], b.x[i], 1e-12);
        EXPECT_NEAR(a.vx[i], b.vx[i], 1e-12 * std::fmax(1e-6, std::fabs(a.vx[i])));
        EXPECT_NEAR(a.vz[i], b.vz[i], 1e-12 * std::fmax(1e-6, std::fabs(a.vz[i])));
    }
}

TEST(Sim, BuildPlanetsRelaxesAtRestThenPlaces) {
    Config cfg = one_planet_config();
    cfg.planets[0].center = Vec3{3.0, -1.0, 0.5};
    cfg.planets[0].velocity = Vec3{-0.25, 0.0, 0.125};
    Particles p;
    EXPECT_EQ(build_stderr(cfg, &p), ""); // settled, normal drag: no note

    // Exactly the planet relaxed on its own at the origin, then shifted and set moving.
    Particles expected = fresh_planet();
    relax_planet(expected, cfg.params, BACKEND_SERIAL, kDrag, kThreshold, kMaxSteps);
    ASSERT_EQ(p.size(), expected.size());
    for (std::size_t i = 0; i < p.size(); ++i) {
        EXPECT_EQ(p.x[i], expected.x[i] + 3.0);
        EXPECT_EQ(p.y[i], expected.y[i] - 1.0);
        EXPECT_EQ(p.vx[i], expected.vx[i] - 0.25);
        EXPECT_EQ(p.vz[i], expected.vz[i] + 0.125);
    }
}

TEST(Sim, BuildPlanetsNotesTheUnsettledPlanet) {
    // planet0 is a single particle (settles at once); planet1 gets far too few steps.
    // The note must name planet1 and give its real kinetic energy and hold.
    Config cfg = one_planet_config();
    cfg.relax_max_steps = 10;
    PlanetSpec single;
    single.n = 1;
    single.radius = 0.2;
    single.particle_radius = 0.1;
    single.mass = 1.0;
    single.body = 0;
    PlanetSpec second = small_planet();
    second.body = 1;
    cfg.planets = {single, second};

    Particles copy = fresh_planet(); // planet1, relaxed the same way on its own
    RelaxResult r = relax_planet(copy, cfg.params, BACKEND_SERIAL, kDrag, kThreshold, 10);
    char expected[256];
    std::snprintf(expected, sizeof(expected),
                  "note: planet1 didn't settle in 10 relax steps: kinetic energy %.3g must "
                  "stay below relax_ke_threshold = %.3g for %d steps in a row",
                  kinetic_energy(copy), kThreshold, r.hold);

    Particles p;
    std::string err = build_stderr(cfg, &p);
    EXPECT_EQ(p.size(), 251u); // a note, not an error: both planets are still built
    EXPECT_NE(err.find(expected), std::string::npos) << err;
    EXPECT_EQ(err.find("planet0"), std::string::npos) << err;
}

TEST(Sim, BuildPlanetsNoNoteWhenSettlingOnTheLastStep) {
    Config cfg = one_planet_config();
    cfg.relax_max_steps = smoke_planet_steps();
    Particles p;
    EXPECT_EQ(build_stderr(cfg, &p), "");
}

TEST(Sim, BuildPlanetsWarnsAboutTooStrongDamper) {
    // Smoke planet: particles of mass 0.25 / 250 = 1e-3, so m_eff = 5e-4 for a pair, and
    // at dt = 0.002, c * dt / m_eff = 4c. Past 2 the explicit damper is unstable (each
    // step flips and grows the approach speed) and the heat it books goes negative.
    Config cfg = one_planet_config();
    cfg.relax_max_steps = 0; // the note is about the settings, no need to run anything
    cfg.params.contact_c = 0.6; // 2.4
    Particles p;
    std::string err = build_stderr(cfg, &p);
    EXPECT_NE(err.find("note: contact_c = 0.6 is too strong at dt = 0.002"), std::string::npos)
        << err;
    EXPECT_NE(err.find("c * dt / m_eff = 2.4"), std::string::npos) << err;
    cfg.params.contact_c = 0.45; // 1.8: stable, no note
    EXPECT_EQ(build_stderr(cfg, &p), "");
    // The lightest pair decides: a second planet of lighter particles (mass 0.1 / 250 =
    // 4e-4, m_eff 2e-4) brings 0.45 to 4.5.
    PlanetSpec light = small_planet();
    light.mass = 0.1;
    light.center = Vec3{3.0, 0.0, 0.0};
    light.body = 1;
    cfg.planets.push_back(light);
    err = build_stderr(cfg, &p);
    EXPECT_NE(err.find("c * dt / m_eff = 4.5"), std::string::npos) << err;
    // Contact off: no damper, no note.
    cfg.params.contact_enabled = false;
    EXPECT_EQ(build_stderr(cfg, &p), "");
}

TEST(Sim, BuildPlanetsWarnsAboutTooStrongDrag) {
    // At drag 500 (drag x dynamical time = 290) the smoke planet creeps: relaxation says
    // "settled" after one hold, yet the planet then moves at up to 900x the threshold.
    // The settled flag can't catch that, so build_planets must warn.
    Config cfg = one_planet_config();
    cfg.relax_drag = 500.0;
    Particles p;
    std::string err = build_stderr(cfg, &p);
    EXPECT_NE(err.find("note: relax_drag = 500 is too strong for planet0"), std::string::npos)
        << err;
    // At the limit's other side, drag 50 (x t_dyn = 29) truly settles: no note.
    cfg.relax_drag = 50.0;
    EXPECT_EQ(build_stderr(cfg, &p), "");
}

TEST(Sim, BuildPlanetsReportsABlowUp) {
    Config cfg = one_planet_config();
    cfg.params.dt = 1e300;
    Particles p;
    std::string err = build_stderr(cfg, &p);
    EXPECT_NE(err.find("planet0 became NaN/Inf during relaxation, after 1 steps"),
              std::string::npos)
        << err;
    EXPECT_EQ(err.find("didn't settle"), std::string::npos) << err; // not the tuning advice
    EXPECT_NE(first_non_finite(p), p.size()); // sim_init will refuse this start
}

TEST(Sim, SetupRunAppliesOverridesBeforeRelaxing) {
    // planet_sim and planet_view both start through setup_run. --dt must reach both the
    // relaxation and the main run: 1e300 makes relaxation blow up (NaN/Inf planets) and
    // comes back as the run's dt.
    std::string path = write_cfg(kRunCfg);
    CliOptions cli;
    ASSERT_TRUE(parse({"--config", path.c_str(), "--dt", "1e300"}, &cli));
    Particles p;
    SimParams params;
    int steps = 1000, output_every = 10;
    std::string error;
    testing::internal::CaptureStderr();
    bool ok = setup_run(cli, BACKEND_SERIAL, true, &p, &params, &steps, &output_every, &error);
    testing::internal::GetCapturedStderr();
    ASSERT_TRUE(ok) << error;
    EXPECT_EQ(params.dt, 1e300);
    EXPECT_NE(first_non_finite(p), p.size());

    // A normal --dt: finite planets, the file's steps and output_every, the flag's dt.
    CliOptions cli2;
    ASSERT_TRUE(parse({"--config", path.c_str(), "--dt", "0.004"}, &cli2));
    steps = 1000;
    output_every = 10;
    testing::internal::CaptureStderr();
    ok = setup_run(cli2, BACKEND_SERIAL, true, &p, &params, &steps, &output_every, &error);
    testing::internal::GetCapturedStderr();
    ASSERT_TRUE(ok) << error;
    EXPECT_EQ(params.dt, 0.004);
    EXPECT_EQ(steps, 3);
    EXPECT_EQ(output_every, 1);
    EXPECT_EQ(p.size(), 10u);
    EXPECT_EQ(first_non_finite(p), p.size());
}

TEST(Sim, SetupRunErrors) {
    std::string error;
    Particles p;
    SimParams params;
    int steps = 1000, output_every = 10;
    CliOptions cli;
    ASSERT_TRUE(parse({"--scenario", "nope"}, &cli));
    EXPECT_FALSE(setup_run(cli, BACKEND_SERIAL, true, &p, &params, &steps, &output_every, &error));
    EXPECT_EQ(error, "unknown scenario 'nope' (orbit, figure8, cloud)");

    // A config with no steps: planet_sim needs them (or --steps), the viewer doesn't.
    std::string text = kRunCfg;
    text.erase(text.find("steps = 3\n"), std::string("steps = 3\n").size());
    std::string path = write_cfg(text);
    CliOptions cli2;
    ASSERT_TRUE(parse({"--config", path.c_str()}, &cli2));
    testing::internal::CaptureStderr();
    EXPECT_FALSE(setup_run(cli2, BACKEND_SERIAL, true, &p, &params, &steps, &output_every, &error));
    EXPECT_NE(error.find("the config sets no steps"), std::string::npos) << error;
    EXPECT_TRUE(setup_run(cli2, BACKEND_SERIAL, false, &p, &params, &steps, &output_every, &error))
        << error;
    testing::internal::GetCapturedStderr();
}
