#pragma once

#include <cstdint>
#include <string>

#include "gravity.hpp"
#include "params.hpp"

// planet_sim's command-line options. The has_* flags record which settings were given
// on the command line, so they can override a config file.
struct CliOptions {
    std::string config_path; // --config <file>
    std::string scenario;    // --scenario orbit|figure8|cloud
    int n = 1000;            // --n: particle count for the cloud scenario
    bool has_n = false;
    std::uint64_t seed = 1; // --seed: for the cloud scenario (64-bit, matches the RNG)
    bool has_seed = false;
    Backend backend = BACKEND_SERIAL;
    int steps = 0;
    bool has_steps = false;
    double dt = 0.0;
    bool has_dt = false;
    int output_every = 0;
    bool has_output_every = false;
    std::string out_dir; // --out: folder for frames + energy.csv (empty = no output)
    bool bench = false;  // --bench: time the step loop, write nothing
    bool help = false;   // --help
};

// Parses argv into opts. Returns false and sets *error on an unknown flag, a missing or
// malformed value, or when not exactly one of --config / --scenario is given.
bool parse_cli(int argc, const char *const *argv, CliOptions *opts, std::string *error);

// Applies the command-line overrides (--dt, --steps, --output-every) on top of the
// settings from a config file or scenario, then checks the result is runnable. Returns
// false and sets *error if dt isn't > 0, output_every < 1, or (from_config) the config
// set no steps and --steps wasn't given (a run of 0 steps would "succeed" doing nothing).
bool apply_overrides(const CliOptions &cli, bool from_config, SimParams *params, int *steps,
                     int *output_every, std::string *error);

// The usage text printed for --help and on errors.
const char *cli_usage();
