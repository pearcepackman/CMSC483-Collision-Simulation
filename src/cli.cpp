#include "cli.hpp"

#include <cstdlib>
#include <cstring>

namespace {

// strtol / strtod with a check that the whole string was a number.
bool to_long(const char *s, long *out) {
    char *end = nullptr;
    *out = std::strtol(s, &end, 10);
    return end != s && *end == '\0';
}

bool to_double(const char *s, double *out) {
    char *end = nullptr;
    *out = std::strtod(s, &end);
    return end != s && *end == '\0';
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
           "  --config        scenario file (flags given here override it)\n"
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
        long n = 0;
        double d = 0.0;
        if (std::strcmp(flag, "--config") == 0) {
            opts->config_path = value;
        } else if (std::strcmp(flag, "--scenario") == 0) {
            opts->scenario = value;
        } else if (std::strcmp(flag, "--out") == 0) {
            opts->out_dir = value;
        } else if (std::strcmp(flag, "--backend") == 0) {
            if (!backend_from_name(value, &opts->backend)) {
                *error = std::string("unknown backend '") + value + "' (serial, omp or hip)";
                return false;
            }
        } else if (std::strcmp(flag, "--n") == 0) {
            if (!to_long(value, &n) || n < 1 || n > 100000000) {
                *error = std::string("--n needs a whole number >= 1, got '") + value + "'";
                return false;
            }
            opts->n = static_cast<int>(n);
        } else if (std::strcmp(flag, "--seed") == 0) {
            if (!to_long(value, &n) || n < 0) {
                *error = std::string("--seed needs a whole number >= 0, got '") + value + "'";
                return false;
            }
            opts->seed = static_cast<unsigned>(n);
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
            if (!to_double(value, &d) || !(d > 0.0)) {
                *error = std::string("--dt needs a number > 0, got '") + value + "'";
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
