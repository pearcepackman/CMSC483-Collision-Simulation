#include "cli.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {

// strtoll / strtod with checks that the whole string was a number and it fit
// (errno == ERANGE means it overflowed, e.g. "1e999" or a 30-digit seed).
bool to_long(const char *s, long long *out) {
    char *end = nullptr;
    errno = 0;
    *out = std::strtoll(s, &end, 10);
    return end != s && *end == '\0' && errno != ERANGE;
}

bool to_double(const char *s, double *out) {
    char *end = nullptr;
    errno = 0;
    *out = std::strtod(s, &end);
    return end != s && *end == '\0' && errno != ERANGE;
}

// Flags that take a value. Checked before looking for the value, so a typo like
// `--bogus` reports "unknown flag", not "missing value".
bool takes_value(const char *flag) {
    const char *flags[] = {"--config", "--scenario", "--out",          "--backend", "--n",
                           "--seed",   "--steps",    "--output-every", "--dt"};
    for (const char *f : flags) {
        if (std::strcmp(flag, f) == 0) {
            return true;
        }
    }
    return false;
}

} // namespace

const char *cli_usage() {
    return "usage: planet_sim (--config <file> | --scenario orbit|figure8|cloud [--n N] "
           "[--seed S])\n"
           "                  [--backend serial|omp] [--steps N] [--dt h] [--out <dir>]\n"
           "                  [--output-every k] [--bench] [--help]\n"
           "  --config        scenario file (flags given here override it); needs the config\n"
           "                  loader (#13), until then it stops with 'not implemented'\n"
           "  --scenario      built-in gravity-only setup: orbit, figure8, or cloud of N "
           "particles\n"
           "  --out           write frames and energy.csv into <dir> (created if needed)\n"
           "  --output-every  frame + energy row every k steps (energy is O(N^2) to compute)\n"
           "  --bench         time the step loop and print one 'bench ...' line; writes nothing\n"
           "exit codes: 0 ok, 1 usage or setup error, 2 simulation produced NaN/Inf\n";
}

bool parse_cli(int argc, const char *const *argv, CliOptions *opts, std::string *error) {
    *opts = CliOptions();
    for (int i = 1; i < argc; ++i) {
        const char *flag = argv[i];
        if (std::strcmp(flag, "--help") == 0) {
            opts->help = true;
            return true;
        }
        if (std::strcmp(flag, "--bench") == 0) {
            opts->bench = true;
            continue;
        }
        if (!takes_value(flag)) {
            *error = std::string("unknown flag '") + flag + "'";
            return false;
        }
        if (i + 1 >= argc) {
            *error = std::string("missing value after ") + flag;
            return false;
        }
        const char *value = argv[++i];
        long long n = 0;
        double d = 0.0;
        if (std::strcmp(flag, "--config") == 0) {
            opts->config_path = value;
        } else if (std::strcmp(flag, "--scenario") == 0) {
            opts->scenario = value;
        } else if (std::strcmp(flag, "--out") == 0) {
            if (value[0] == '\0') {
                *error = "--out needs a folder name";
                return false;
            }
            opts->out_dir = value;
        } else if (std::strcmp(flag, "--backend") == 0) {
            if (!backend_from_name(value, &opts->backend)) {
                *error = std::string("unknown backend '") + value + "' (serial or omp)";
                return false;
            }
            if (opts->backend == BACKEND_HIP) { // compute_gravity would abort
                *error = "the hip backend isn't built yet (#25); use serial or omp";
                return false;
            }
        } else if (std::strcmp(flag, "--n") == 0) {
            // 10M particles is ~1 GB of particle data, and O(N^2) gravity is already
            // far too slow to run at that size.
            if (!to_long(value, &n) || n < 1 || n > 10000000) {
                *error =
                    std::string("--n needs a whole number from 1 to 10000000, got '") + value + "'";
                return false;
            }
            opts->n = static_cast<int>(n);
            opts->has_n = true;
        } else if (std::strcmp(flag, "--seed") == 0) {
            if (!to_long(value, &n) || n < 0) {
                *error = std::string("--seed needs a whole number >= 0, got '") + value + "'";
                return false;
            }
            opts->seed = static_cast<std::uint64_t>(n);
            opts->has_seed = true;
        } else if (std::strcmp(flag, "--steps") == 0) {
            if (!to_long(value, &n) || n < 0 || n > 2000000000) {
                *error = std::string("--steps needs a whole number >= 0, got '") + value + "'";
                return false;
            }
            opts->steps = static_cast<int>(n);
            opts->has_steps = true;
        } else if (std::strcmp(flag, "--output-every") == 0) {
            if (!to_long(value, &n) || n < 1 || n > 2000000000) {
                *error =
                    std::string("--output-every needs a whole number >= 1, got '") + value + "'";
                return false;
            }
            opts->output_every = static_cast<int>(n);
            opts->has_output_every = true;
        } else if (std::strcmp(flag, "--dt") == 0) {
            if (!to_double(value, &d) || !std::isfinite(d) || !(d > 0.0)) {
                *error = std::string("--dt needs a finite number > 0, got '") + value + "'";
                return false;
            }
            opts->dt = d;
            opts->has_dt = true;
        }
    }
    if (opts->config_path.empty() == opts->scenario.empty()) {
        *error = "give exactly one of --config <file> or --scenario <name>";
        return false;
    }
    return true;
}

bool apply_overrides(const CliOptions &cli, bool from_config, SimParams *params, int *steps,
                     int *output_every, std::string *error) {
    if (cli.has_dt) {
        params->dt = cli.dt;
    }
    if (cli.has_steps) {
        *steps = cli.steps;
    }
    if (cli.has_output_every) {
        *output_every = cli.output_every;
    }
    if (from_config && !cli.has_steps && *steps < 1) {
        *error = "the config sets no steps (add 'steps = N' or pass --steps N)";
        return false;
    }
    if (!std::isfinite(params->dt) || !(params->dt > 0.0)) {
        *error = "dt must be a finite number > 0 (set it in the config or pass --dt)";
        return false;
    }
    if (*output_every < 1) {
        *error = "output-every must be >= 1";
        return false;
    }
    return true;
}
