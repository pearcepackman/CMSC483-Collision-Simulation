#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "cli.hpp"
#include "diagnostics.hpp"
#include "forces.hpp"
#include "integrator.hpp"
#include "sim.hpp"

namespace {

bool parse(std::vector<const char *> args, CliOptions *opts, std::string *error) {
    args.insert(args.begin(), "planet_sim");
    return parse_cli(static_cast<int>(args.size()), args.data(), opts, error);
}

Particles orbit(SimParams *params) {
    Particles p;
    EXPECT_TRUE(build_scenario("orbit", 0, 0, p, *params));
    return p;
}

std::vector<std::string> read_lines(FILE *f) {
    std::rewind(f);
    std::vector<std::string> lines;
    char buf[512];
    while (std::fgets(buf, sizeof buf, f) != nullptr) {
        lines.emplace_back(buf);
    }
    return lines;
}

} // namespace

// ---- Command line ----

TEST(Cli, ParsesEveryFlag) {
    CliOptions o;
    std::string err;
    ASSERT_TRUE(
        parse({"--scenario", "cloud", "--n", "500", "--seed", "7", "--backend", "omp", "--steps",
               "200", "--dt", "0.01", "--out", "frames/run", "--output-every", "20", "--bench"},
              &o, &err))
        << err;
    EXPECT_EQ(o.scenario, "cloud");
    EXPECT_EQ(o.n, 500);
    EXPECT_EQ(o.seed, 7u);
    EXPECT_EQ(o.backend, BACKEND_OMP);
    EXPECT_TRUE(o.has_steps);
    EXPECT_EQ(o.steps, 200);
    EXPECT_TRUE(o.has_dt);
    EXPECT_DOUBLE_EQ(o.dt, 0.01);
    EXPECT_EQ(o.out_dir, "frames/run");
    EXPECT_TRUE(o.has_output_every);
    EXPECT_EQ(o.output_every, 20);
    EXPECT_TRUE(o.bench);
}

TEST(Cli, ConfigWithoutOverridesLeavesThemUnset) {
    CliOptions o;
    std::string err;
    ASSERT_TRUE(parse({"--config", "configs/head_on.cfg"}, &o, &err)) << err;
    EXPECT_EQ(o.config_path, "configs/head_on.cfg");
    EXPECT_FALSE(o.has_steps); // so the config's values win
    EXPECT_FALSE(o.has_dt);
    EXPECT_FALSE(o.has_output_every);
    EXPECT_FALSE(o.bench);
}

TEST(Cli, RejectsBadInput) {
    const std::vector<std::vector<const char *>> bad = {
        {},                                           // neither --config nor --scenario
        {"--config", "a.cfg", "--scenario", "orbit"}, // both
        {"--scenario", "orbit", "--bogus"},           // unknown flag
        {"--scenario"},                               // missing value
        {"--scenario", "orbit", "--steps", "ten"},    // not a number
        {"--scenario", "orbit", "--steps", "-5"},     // negative
        {"--scenario", "orbit", "--dt", "0"},         // dt must be > 0
        {"--scenario", "orbit", "--output-every", "0"},
        {"--scenario", "cloud", "--n", "0"},
        {"--scenario", "orbit", "--backend", "cuda"}, // unknown backend
    };
    for (const auto &args : bad) {
        CliOptions o;
        std::string err;
        EXPECT_FALSE(parse(args, &o, &err)) << "accepted: " << (args.empty() ? "" : args[0]);
        EXPECT_FALSE(err.empty());
    }
}

TEST(Cli, UnknownFlagIsReportedAsUnknownEvenAtTheEnd) {
    // A trailing typo must say "unknown flag", not "missing value".
    CliOptions o;
    std::string err;
    EXPECT_FALSE(parse({"--scenario", "orbit", "--bogus"}, &o, &err));
    EXPECT_NE(err.find("unknown flag '--bogus'"), std::string::npos) << err;
    EXPECT_FALSE(parse({"--scenario"}, &o, &err));
    EXPECT_NE(err.find("missing value after --scenario"), std::string::npos) << err;
}

TEST(Cli, HelpNeedsNothingElse) {
    CliOptions o;
    std::string err;
    ASSERT_TRUE(parse({"--help"}, &o, &err)) << err;
    EXPECT_TRUE(o.help);
}

// ---- NaN / Inf detection ----

TEST(Sim, FindsFirstNonFiniteParticle) {
    Particles p;
    p.resize(5);
    EXPECT_EQ(first_non_finite(p), 5u);
    p.vx[3] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(first_non_finite(p), 3u);
    p.z[1] = std::numeric_limits<double>::infinity();
    EXPECT_EQ(first_non_finite(p), 1u);
}

TEST(Sim, StopsEarlyOnNaN) {
    SimParams params;
    Particles p = orbit(&params);
    p.vy[1] = std::numeric_limits<double>::quiet_NaN();
    RunOptions opts;
    opts.steps = 100;
    RunResult r = run_simulation(p, params, opts, nullptr);
    EXPECT_FALSE(r.finite);
    EXPECT_EQ(r.bad_particle, 1u);
    EXPECT_EQ(r.steps_done, 0); // caught before the first step
}

// ---- The loop ----

TEST(Sim, MatchesHandWrittenLeapfrogLoop) {
    SimParams params;
    Particles via_sim = orbit(&params);
    Particles by_hand = orbit(&params);
    RunOptions opts;
    opts.steps = 500;
    RunResult r = run_simulation(via_sim, params, opts, nullptr);
    ASSERT_TRUE(r.finite);
    EXPECT_EQ(r.steps_done, 500);

    compute_accelerations(by_hand, params, BACKEND_SERIAL);
    for (int s = 0; s < 500; ++s) {
        leapfrog_step(by_hand, params, BACKEND_SERIAL);
    }
    for (std::size_t i = 0; i < 2; ++i) { // same code path, so identical bits
        EXPECT_EQ(via_sim.x[i], by_hand.x[i]);
        EXPECT_EQ(via_sim.y[i], by_hand.y[i]);
        EXPECT_EQ(via_sim.vx[i], by_hand.vx[i]);
        EXPECT_EQ(via_sim.vy[i], by_hand.vy[i]);
    }
}

TEST(Sim, EnergyLogHasOneRowPerOutputStep) {
    SimParams params;
    Particles p = orbit(&params);
    FILE *log = std::tmpfile();
    ASSERT_NE(log, nullptr);
    write_energy_header(log);
    RunOptions opts;
    opts.steps = 100;
    opts.output_every = 10;
    RunResult r = run_simulation(p, params, opts, log);
    ASSERT_TRUE(r.finite);

    std::vector<std::string> lines = read_lines(log);
    std::fclose(log);
    ASSERT_EQ(lines.size(), 1u + 11u); // header + steps 0, 10, ..., 100
    EXPECT_EQ(lines[0], "step,t,kinetic,potential,total,px,py,pz\n");

    // Total energy (column 5) stays within the orbit test's bound of its start (-0.125).
    for (std::size_t row = 1; row < lines.size(); ++row) {
        int step = 0;
        double t = 0, ke = 0, pe = 0, e = 0, px = 0, py = 0, pz = 0;
        ASSERT_EQ(std::sscanf(lines[row].c_str(), "%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &step, &t, &ke,
                              &pe, &e, &px, &py, &pz),
                  8)
            << lines[row];
        EXPECT_EQ(step, static_cast<int>(row - 1) * 10);
        EXPECT_LE(std::fabs(e - (-0.125)), 1e-5 * 0.125);
        EXPECT_LE(std::fabs(ke + pe - e), 1e-14);
    }
}

// ---- Setup ----

TEST(Sim, PlacePlanetShiftsAndAppends) {
    Particles planet;
    planet.resize(2);
    planet.x[1] = 1.0;
    planet.vy[1] = 0.5; // the planet's own internal motion is kept
    planet.mass[0] = planet.mass[1] = 2.0;
    planet.radius[0] = planet.radius[1] = 0.1;
    planet.body[0] = planet.body[1] = 3;

    Particles dst;
    dst.resize(1); // one particle already there
    place_planet(dst, planet, Vec3{10.0, -5.0, 1.0}, Vec3{0.0, 0.0, 2.0});
    ASSERT_EQ(dst.size(), 3u);
    EXPECT_DOUBLE_EQ(dst.x[2], 11.0);
    EXPECT_DOUBLE_EQ(dst.y[2], -5.0);
    EXPECT_DOUBLE_EQ(dst.z[2], 1.0);
    EXPECT_DOUBLE_EQ(dst.vy[2], 0.5);
    EXPECT_DOUBLE_EQ(dst.vz[2], 2.0);
    EXPECT_DOUBLE_EQ(dst.mass[2], 2.0);
    EXPECT_DOUBLE_EQ(dst.radius[2], 0.1);
    EXPECT_EQ(dst.body[2], 3);
    EXPECT_EQ(dst.ax[2], 0.0); // forces get recomputed after setup
}

TEST(Sim, BuiltInScenarios) {
    SimParams params;
    Particles p;
    ASSERT_TRUE(build_scenario("orbit", 0, 0, p, params));
    ASSERT_EQ(p.size(), 2u);
    EXPECT_GT(params.dt, 0.0);
    EXPECT_FALSE(params.contact_enabled);
    double e = kinetic_energy(p) + potential_energy(p, params);
    EXPECT_LE(std::fabs(e - (-0.125)), 1e-14);

    ASSERT_TRUE(build_scenario("figure8", 0, 0, p, params));
    EXPECT_EQ(p.size(), 3u);

    Particles a, b;
    ASSERT_TRUE(build_scenario("cloud", 300, 42, a, params));
    ASSERT_TRUE(build_scenario("cloud", 300, 42, b, params));
    ASSERT_EQ(a.size(), 300u);
    for (std::size_t i = 0; i < a.size(); ++i) { // same seed, same cloud
        EXPECT_EQ(a.x[i], b.x[i]);
        EXPECT_LE(a.x[i] * a.x[i] + a.y[i] * a.y[i] + a.z[i] * a.z[i], 1.0); // in the sphere
    }

    EXPECT_FALSE(build_scenario("nope", 10, 1, p, params));
    EXPECT_FALSE(build_scenario("cloud", 0, 1, p, params));
}
