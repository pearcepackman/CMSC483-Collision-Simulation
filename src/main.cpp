#include <cstdio>
#include <filesystem>
#include <string>

#include <omp.h>

#include "cli.hpp"
#include "config.hpp"
#include "sim.hpp"

// planet_sim: set up particles (config file or built-in scenario), run the simulation,
// write frames and an energy log, or time it with --bench. Exit codes: 0 ok,
// 1 usage or setup error, 2 the simulation produced NaN/Inf.
int main(int argc, char **argv) {
    CliOptions cli;
    std::string error;
    if (!parse_cli(argc, argv, &cli, &error)) {
        std::fprintf(stderr, "planet_sim: %s\n%s", error.c_str(), cli_usage());
        return 1;
    }
    if (cli.help) {
        std::printf("%s", cli_usage());
        return 0;
    }

    // Starting state and settings: from a config file, or a built-in scenario.
    Particles p;
    SimParams params;
    int steps = 1000;
    int output_every = 10;
    if (!cli.config_path.empty()) {
        Config cfg;
        if (!load_config(cli.config_path.c_str(), cfg)) {
            std::fprintf(stderr, "planet_sim: couldn't load config '%s'\n",
                         cli.config_path.c_str());
            return 1;
        }
        params = cfg.params;
        steps = cfg.steps;
        output_every = cfg.output_every;
        p = build_planets(cfg, cli.backend);
    } else if (!build_scenario(cli.scenario.c_str(), cli.n, cli.seed, p, params)) {
        std::fprintf(stderr, "planet_sim: unknown scenario '%s' (orbit, figure8, cloud)\n%s",
                     cli.scenario.c_str(), cli_usage());
        return 1;
    }

    // Command-line flags override the config / scenario defaults.
    if (cli.has_steps) {
        steps = cli.steps;
    }
    if (cli.has_dt) {
        params.dt = cli.dt;
    }
    if (cli.has_output_every) {
        output_every = cli.output_every;
    }
    if (!(params.dt > 0.0) || output_every < 1) {
        std::fprintf(stderr, "planet_sim: dt must be > 0 and output-every >= 1\n");
        return 1;
    }

    RunOptions opts;
    opts.backend = cli.backend;
    opts.steps = steps;
    opts.output_every = output_every;

    // Output files (never in --bench mode, so timing measures only the simulation).
    FILE *energy_log = nullptr;
    if (!cli.bench && !cli.out_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(cli.out_dir, ec);
        std::string log_path = cli.out_dir + "/energy.csv";
        energy_log = ec ? nullptr : std::fopen(log_path.c_str(), "w");
        if (energy_log == nullptr) {
            std::fprintf(stderr, "planet_sim: can't write to '%s'\n", cli.out_dir.c_str());
            return 1;
        }
        write_energy_header(energy_log);
        opts.write_frames = true;
        opts.out_dir = cli.out_dir.c_str();
    }

    RunResult result = run_simulation(p, params, opts, energy_log);
    if (energy_log != nullptr) {
        std::fclose(energy_log);
    }

    if (!result.finite) {
        std::fprintf(stderr, "planet_sim: particle %zu became NaN/Inf after step %d\n",
                     result.bad_particle, result.steps_done);
        return 2;
    }
    if (!result.io_ok) {
        std::fprintf(stderr, "planet_sim: writing a frame to '%s' failed\n", cli.out_dir.c_str());
        return 1;
    }
    if (cli.bench) {
        // One line, easy for scripts/bench.py (#20) to parse.
        double per_step = steps > 0 ? result.seconds / steps : 0.0;
        std::printf("bench backend=%s n=%zu threads=%d steps=%d seconds_per_step=%.6g\n",
                    backend_name(cli.backend), p.size(), omp_get_max_threads(), steps, per_step);
    } else {
        std::printf("ran %d steps of %zu particles (%s) in %.3f s\n", result.steps_done, p.size(),
                    backend_name(cli.backend), result.seconds);
    }
    return 0;
}
