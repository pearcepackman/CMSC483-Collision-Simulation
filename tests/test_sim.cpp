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
        {"--scenario", "cloud", "--n", "20000000"},   // over the 10M cap (~1 GB of particles)
        {"--scenario", "orbit", "--backend", "cuda"}, // unknown backend
        {"--scenario", "orbit", "--dt", "inf"},       // must be finite
        {"--scenario", "orbit", "--dt", "1e999"},     // overflows to infinity
        {"--scenario", "orbit", "--dt", "nan"},
        {"--scenario", "orbit", "--dt", "-1"},
        {"--scenario", "cloud", "--seed", "-1"},
        {"--scenario", "cloud", "--seed", "99999999999999999999"}, // out of range
        {"--scenario", "orbit", "--out", ""},
    };
    for (const auto &args : bad) {
        CliOptions o;
        std::string err;
        EXPECT_FALSE(parse(args, &o, &err)) << "accepted: " << (args.empty() ? "" : args[0]);
        EXPECT_FALSE(err.empty());
    }
}

TEST(Cli, HipBackendOnlyInHipBuilds) {
    CliOptions o;
    std::string err;
#ifdef SIM_HAVE_HIP
    ASSERT_TRUE(parse({"--scenario", "orbit", "--backend", "hip"}, &o, &err)) << err;
    EXPECT_EQ(o.backend, BACKEND_HIP);
#else
    EXPECT_FALSE(parse({"--scenario", "orbit", "--backend", "hip"}, &o, &err));
    EXPECT_NE(err.find("this build has no hip backend"), std::string::npos) << err;
#endif
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

TEST(Cli, LargeSeedsStayDistinct) {
    // Seeds above 32 bits must not wrap (4294967297 would become 1 in 32 bits).
    CliOptions o;
    std::string err;
    ASSERT_TRUE(parse({"--scenario", "cloud", "--seed", "4294967297"}, &o, &err)) << err;
    EXPECT_EQ(o.seed, 4294967297ULL);
}

TEST(Cli, OverridesReplaceConfigValuesOnlyWhenGiven) {
    SimParams params;
    params.dt = 0.5;
    int steps = 100, output_every = 10;
    CliOptions none;
    std::string err;
    ASSERT_TRUE(apply_overrides(none, true, &params, &steps, &output_every, &err)) << err;
    EXPECT_EQ(params.dt, 0.5); // nothing given: config values stay
    EXPECT_EQ(steps, 100);
    EXPECT_EQ(output_every, 10);

    CliOptions all;
    ASSERT_TRUE(parse({"--config", "x.cfg", "--dt", "0.25", "--steps", "7", "--output-every", "3"},
                      &all, &err))
        << err;
    ASSERT_TRUE(apply_overrides(all, true, &params, &steps, &output_every, &err)) << err;
    EXPECT_EQ(params.dt, 0.25);
    EXPECT_EQ(steps, 7);
    EXPECT_EQ(output_every, 3);
}

TEST(Cli, OverridesRejectUnusableSettings) {
    CliOptions none;
    std::string err;
    SimParams params;
    params.dt = 0.1;
    int steps = 0, output_every = 10;
    // A config that sets no steps would "succeed" without simulating anything.
    EXPECT_FALSE(apply_overrides(none, true, &params, &steps, &output_every, &err));
    EXPECT_NE(err.find("steps"), std::string::npos) << err;

    steps = 10;
    params.dt = 0.0; // config without dt
    EXPECT_FALSE(apply_overrides(none, true, &params, &steps, &output_every, &err));
    EXPECT_NE(err.find("dt"), std::string::npos) << err;
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

TEST(Sim, ChecksEveryPositionAndVelocityField) {
    // Each of the six fields on its own must be caught.
    for (int field = 0; field < 6; ++field) {
        Particles p;
        p.resize(4);
        std::vector<double> *fields[6] = {&p.x, &p.y, &p.z, &p.vx, &p.vy, &p.vz};
        (*fields[field])[2] = std::numeric_limits<double>::quiet_NaN();
        EXPECT_EQ(first_non_finite(p), 2u) << "field " << field;
    }
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

TEST(Sim, StopsAtTheStepThatBlowsUp) {
    // A finite start that overflows during step 1: the check inside the loop must stop
    // the run right there, not carry on or only check every few steps.
    SimParams params;
    Particles p = orbit(&params);
    params.dt = 1e300; // positions overflow to infinity on the first drift
    RunOptions opts;
    opts.steps = 50;
    RunResult r = run_simulation(p, params, opts, nullptr);
    EXPECT_FALSE(r.finite);
    EXPECT_EQ(r.steps_done, 1);
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
        EXPECT_LE(std::fabs(t - step * params.dt), 1e-15 * (1.0 + std::fabs(t)));
        EXPECT_LE(std::fabs(e - (-0.125)), 1e-5 * 0.125);
        EXPECT_LE(std::fabs(ke + pe - e), 1e-14);
        // Momentum starts at exactly zero and the mirror-image pair keeps it there.
        EXPECT_LE(std::fabs(px) + std::fabs(py) + std::fabs(pz), 1e-15);
    }
}

TEST(Sim, LogsTheFinalStepEvenIfNotAMultiple) {
    // 105 steps every 10: rows at 0, 10, ..., 100 and the final step 105, so the
    // energy over the whole run can be measured.
    SimParams params;
    Particles p = orbit(&params);
    FILE *log = std::tmpfile();
    ASSERT_NE(log, nullptr);
    write_energy_header(log);
    RunOptions opts;
    opts.steps = 105;
    opts.output_every = 10;
    ASSERT_TRUE(run_simulation(p, params, opts, log).finite);
    std::vector<std::string> lines = read_lines(log);
    std::fclose(log);
    ASSERT_EQ(lines.size(), 1u + 12u);
    EXPECT_EQ(lines.back().rfind("105,", 0), 0u) << lines.back();
}

TEST(Sim, FinishLogReportsWriteErrors) {
    // /dev/full accepts the open but fails every write ("disk full").
    FILE *f = std::fopen("/dev/full", "w");
    if (f == nullptr) {
        GTEST_SKIP() << "/dev/full not available";
    }
    write_energy_header(f);
    EXPECT_FALSE(finish_log(f)); // must not report success for a truncated file

    FILE *ok = std::tmpfile();
    ASSERT_NE(ok, nullptr);
    write_energy_header(ok);
    EXPECT_TRUE(finish_log(ok));
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
