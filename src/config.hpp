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
    // Relaxation of each planet before impact (#15, relax_planet in init.hpp):
    // relax_max_steps = 0 skips it. relax_drag is the velocity drag rate (per unit time).
    double relax_ke_threshold = 0.0;
    int relax_max_steps = 0;
    double relax_drag = 0.0;
};

// Loads a `key = value` scenario file into cfg (format: docs/notes.md, examples in
// configs/). One pair per line, `#` starts a comment. Keys:
//   dt, softening, contact (true/false), contact_k, contact_c, steps, output_every,
//   relax_ke_threshold, relax_max_steps, relax_drag
//   planetN.n, .radius, .particle_radius, .mass, .center, .velocity, .seed
// (center and velocity are three numbers separated by spaces; planet N gets body id N).
// Returns false, prints the reason to stderr and leaves cfg unchanged on any error.
// Errors on one line ("path:line: reason" plus the line): an unknown or repeated key,
// no '=', or a value that isn't entirely one finite number of the right type. Errors
// about the whole file ("path: reason"): it can't be opened or isn't a regular file,
// no output_every, no planets, planets not numbered 0, 1, 2, ... with no gaps, a
// planet missing n, radius, particle_radius or mass, a planet make_planet can't build
// (planet_spec_ok), softening < 0, contact on with contact_k <= 0, contact_c < 0,
// negative relax_* values, or relax_max_steps > 0 with relax_ke_threshold 0. Other keys
// missing from the file keep the values already in cfg. dt, steps and output_every are
// range-checked by apply_overrides, after command-line flags may have replaced them.
bool load_config(const char *path, Config &cfg);
