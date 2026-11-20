#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

#include "cli.hpp"
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

// Checks the starting state for NaN/Inf, then computes the accelerations for it (the
// leapfrog contract: a(t = 0) before the first step). Returns the first non-finite
// particle, or p.size() if the state is ready to step; on a bad state it computes nothing.
std::size_t sim_init(Particles &p, const SimParams &params, Backend backend);

// One leapfrog step on `backend`, then the NaN/Inf check. Returns the first non-finite
// particle, or p.size(). The backend may differ from the previous call (the viewer
// switches live): all state lives in Particles on the host, and every backend computes
// the same accelerations, so a step only depends on the backend through rounding.
std::size_t sim_step(Particles &p, const SimParams &params, Backend backend);

// Runs the main loop with sim_init and sim_step: takes opts.steps leapfrog steps. At
// step 0, every output_every steps, and the final step it writes a frame (if enabled,
// numbered 0, 1, 2, ...) and an energy row (if energy_log isn't null); potential energy
// is O(N^2) and serial, so it's only computed then. After every step it checks for
// NaN/Inf and stops early if it finds any.
RunResult run_simulation(Particles &p, const SimParams &params, const RunOptions &opts,
                         FILE *energy_log);

// Index of the first particle with a NaN/Inf position or velocity, or p.size() if none.
std::size_t first_non_finite(const Particles &p);

// The energy log: a header, then one row per output step:
//   step,t,kinetic,potential,contact,heat,total,px,py,pz
// with total = kinetic + potential + contact (energy in compressed contact springs) +
// heat. It stays nearly constant through a collision. What's left is a small error that
// appears as contacts begin, shrinking roughly in proportion to dt (docs/notes.md, #59).
void write_energy_header(FILE *f);
void write_energy_row(FILE *f, int step, double t, const Particles &p, const SimParams &params);

// Closes a log file. Returns false if any write failed or the close did (e.g. disk full),
// so a cut-off energy.csv can't pass as complete.
bool finish_log(FILE *f);

// Appends `planet` to dst, shifted by center and moving with velocity (added to each
// particle's own velocity). Masses, radii, body ids and heat are kept.
void place_planet(Particles &dst, const Particles &planet, Vec3 center, Vec3 velocity);

// Builds the starting particles from a config: each planet is generated at rest at the
// origin, relaxed with cfg.params if cfg.relax_max_steps > 0, then moved to its center
// and given its velocity. (Relaxing a moving planet, or two planets together, gives
// wrong results.) cfg.params.dt must already be valid: apply command-line overrides
// before calling this (setup_run does). Notes on stderr, after which the run goes on: a
// planet that doesn't settle within relax_max_steps, one that became NaN/Inf while
// relaxing (sim_init then refuses to start), or a relax_drag so strong for the planet
// (drag x dynamical time > MAX_RELAX_DRAG_T_DYN) that "settled" can't be trusted, or a
// contact damper too strong for the lightest pair of particles (c * dt / m_eff >= 2).
// ax/ay/az of the result are not valid (sim_init).
Particles build_planets(const Config &cfg, Backend backend);

// The starting state of a run, shared by planet_sim and planet_view so both build it
// the same way: either a config file (loaded, then the command-line overrides applied,
// then its planets built and relaxed on `backend` with those final params, so --dt
// reaches relaxation) or a built-in scenario (built, then the overrides applied).
// *steps and *output_every come in holding the scenario defaults and go out with the
// config's values and the overrides. require_steps: a config must set steps unless
// --steps is given (planet_sim; the viewer runs until closed). Returns false and sets
// *error if the config won't load, the scenario is unknown, or an override is bad. A
// GPU backend must be initialized first (relaxation runs on it). ax/ay/az of *p are
// not valid yet: call sim_init.
bool setup_run(const CliOptions &cli, Backend backend, bool require_steps, Particles *p,
               SimParams *params, int *steps, int *output_every, std::string *error);

// Built-in gravity-only setups that need no config file, with their own default params:
//   "orbit"   two equal masses in a circular orbit (period 2*pi)
//   "figure8" the three-body figure-eight orbit
//   "cloud"   n random equal-mass particles at rest in a unit sphere, from seed
// Returns false for an unknown name or n < 1.
bool build_scenario(const char *name, int n, std::uint64_t seed, Particles &p, SimParams &params);
