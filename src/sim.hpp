#pragma once

#include <cstdio>

#include "config.hpp"
#include "gravity.hpp"
#include "params.hpp"
#include "particles.hpp"

struct RunOptions {
    Backend backend = BACKEND_SERIAL;
    int steps = 0;
    int output_every = 1;      // frame + energy row at step 0 and every k steps
    bool write_frames = false; // call write_frame() into out_dir
    const char *out_dir = nullptr;
};

struct RunResult {
    int steps_done = 0;
    bool finite = true;           // false: a position or velocity became NaN/Inf
    std::size_t bad_particle = 0; // the first non-finite particle, if !finite
    bool io_ok = true;            // false: writing a frame failed
    double seconds = 0.0;         // wall time of the step loop
};

// Runs the main loop. Calls compute_accelerations() once first (the leapfrog contract),
// then takes opts.steps leapfrog steps. At step 0 and every output_every steps it writes
// a frame (if enabled) and an energy row (if energy_log isn't null); potential energy is
// O(N^2) and serial, so it's only computed then. After every step it checks for NaN/Inf
// and stops early if it finds any.
RunResult run_simulation(Particles &p, const SimParams &params, const RunOptions &opts,
                         FILE *energy_log);

// Index of the first particle with a NaN/Inf position or velocity, or p.size() if none.
std::size_t first_non_finite(const Particles &p);

// The energy log: a header, then one row per output step:
//   step,t,kinetic,potential,total,px,py,pz
void write_energy_header(FILE *f);
void write_energy_row(FILE *f, int step, double t, const Particles &p, const SimParams &params);

// Appends `planet` to dst, shifted by center and moving with velocity (added to each
// particle's own velocity). Masses, radii and body ids are kept.
void place_planet(Particles &dst, const Particles &planet, Vec3 center, Vec3 velocity);

// Builds the starting particles from a config: each planet is generated at rest at the
// origin, relaxed if cfg.relax_max_steps > 0, then moved to its center and given its
// velocity. (Relaxing a moving planet, or two planets together, gives wrong results.)
Particles build_planets(const Config &cfg, Backend backend);

// Built-in gravity-only setups that need no config file, with their own default params:
//   "orbit"   two equal masses in a circular orbit (period 2*pi)
//   "figure8" the three-body figure-eight orbit
//   "cloud"   n random equal-mass particles at rest in a unit sphere, from seed
// Returns false for an unknown name or n < 1.
bool build_scenario(const char *name, int n, unsigned seed, Particles &p, SimParams &params);
