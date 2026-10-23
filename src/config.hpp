#pragma once

#include <vector>

#include "init.hpp"
#include "params.hpp"

// Everything a run needs, loaded from a scenario file in configs/.
struct Config {
    SimParams params;
    std::vector<PlanetSpec> planets;
    int steps = 0;        // number of timesteps to run
    int output_every = 1; // write a frame every this many steps
    // Relaxation of each planet before impact (#15). relax_max_steps = 0 skips it.
    double relax_ke_threshold = 0.0;
    int relax_max_steps = 0;
};

// Loads a `key = value` scenario file into cfg. Returns false and prints the
// offending line on an unknown key or a bad value. Keys missing from the file keep
// the defaults already in cfg.
bool load_config(const char *path, Config &cfg);
