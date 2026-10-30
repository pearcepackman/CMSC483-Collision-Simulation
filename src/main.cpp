#include <cstdio>
#include <filesystem>
#include <new>
#include <string>

#include <omp.h>

#include "cli.hpp"
#include "config.hpp"
#include "sim.hpp"

namespace {

// The run itself, separated from main() so an allocation failure for a huge --n can be
// reported cleanly instead of crashing.
int run(const CliOptions &cli) {
    // Starting state and settings: from a config file, or a built-in scenario.
    Particles p;
    SimParams params;
    int steps = 1000;
    int output_every = 10;
    const bool from_config = !cli.config_path.empty();
    if (from_config) {
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
    if ((cli.has_n || cli.has_seed) && (from_config || cli.scenario != "cloud")) {
        std::fprintf(stderr, "planet_sim: note: --n and --seed only apply to --scenario "
                             "cloud; ignoring them\n");
    }

    // Command-line flags override the config / scenario defaults.
    std::string error;
    if (!apply_overrides(cli, from_config, &params, &steps, &output_every, &error)) {
        std::fprintf(stderr, "planet_sim: %s\n", error.c_str());
        return 1;
    }

#ifdef SIM_HAVE_HIP
    // Pick the GPU up front, so "no GPU" is a clean error instead of an abort mid-run.
    // The device goes to stderr so the --bench line on stdout stays one parseable line.
    if (cli.backend == BACKEND_HIP) {
        std::string info;
        if (!hip_init(&info)) {
            std::fprintf(stderr, "planet_sim: --backend hip: %s\n", info.c_str());
            return 1;
        }
        std::fprintf(stderr, "planet_sim: hip device: %s\n", info.c_str());
    }
#endif

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
    const bool log_ok = energy_log == nullptr || finish_log(energy_log);

    if (!result.finite) {
        std::fprintf(stderr, "planet_sim: particle %zu became NaN/Inf after step %d\n",
                     result.bad_particle, result.steps_done);
        return 2;
    }
    if (!result.io_ok) {
        std::fprintf(stderr, "planet_sim: writing a frame to '%s' failed\n", cli.out_dir.c_str());
        return 1;
    }
    if (!log_ok) {
        std::fprintf(stderr, "planet_sim: writing '%s/energy.csv' failed (disk full?)\n",
                     cli.out_dir.c_str());
        return 1;
    }
    // Report what actually ran (opts, not cli), so a wrong setup can't print a right label.
    // CPU threads computing gravity: OpenMP's team, 1 for serial, 0 for the GPU (so a
    // script grouping by thread count can't mistake a GPU run for a 1-thread CPU run).
    const int threads = opts.backend == BACKEND_OMP      ? omp_get_max_threads()
                        : opts.backend == BACKEND_SERIAL ? 1
                                                         : 0;
    if (cli.bench) {
        // One line, easy for scripts/bench.py (#20) to parse.
        double per_step = steps > 0 ? result.seconds / steps : 0.0;
        std::printf("bench backend=%s n=%zu threads=%d steps=%d seconds_per_step=%.6g\n",
                    backend_name(opts.backend), p.size(), threads, steps, per_step);
    } else {
        std::printf("ran %d steps of %zu particles (%s, %d threads) in %.3f s\n", result.steps_done,
                    p.size(), backend_name(opts.backend), threads, result.seconds);
    }
    return 0;
}

} // namespace

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
    try {
        return run(cli);
    } catch (const std::bad_alloc &) {
        std::fprintf(stderr, "planet_sim: out of memory (try a smaller --n)\n");
        return 1;
    }
}
