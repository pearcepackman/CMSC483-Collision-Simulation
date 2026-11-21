#pragma once

#include "gravity.hpp"
#include "params.hpp"
#include "particles.hpp"

// One planet to generate: a ball of particles at a position and velocity.
struct PlanetSpec {
    int n = 0;                    // particle count
    double radius = 0.0;          // planet radius
    double particle_radius = 0.0; // radius of each particle
    double mass = 0.0;            // total planet mass (all particles together)
    // Layered planets (#50): particles whose lattice position is within
    // core_radius_fraction of the radius the particle centers fill are iron core, the
    // rest rock mantle. Same particle size; core particles are core_density_ratio times
    // heavier. 0 / 1 (the defaults) = no core: every particle has mass / n, as before.
    double core_radius_fraction = 0.0; // in [0, 1)
    double core_density_ratio = 1.0;   // >= 1 (iron is about 2.5x denser than rock)
    Vec3 center = {0.0, 0.0, 0.0};
    Vec3 velocity = {0.0, 0.0, 0.0};
    unsigned seed = 0; // same seed -> identical planet
    int body = 0;      // id written into Particles::body
};

// Appends spec.n particles filling a sphere, reproducible from spec.seed. Aborts with
// the reason if planet_spec_ok would reject spec.
void make_planet(Particles &p, const PlanetSpec &spec);

// True if make_planet can build spec. Otherwise false, and *why (if not null) gets the
// reason, e.g. "too many particles for this radius and particle_radius". Lets callers
// such as load_config report a bad spec cleanly instead of aborting.
bool planet_spec_ok(const PlanetSpec &spec, const char **why);

// Particle masses make_planet will use for spec: how many particles are core, and the
// mass of one core and one mantle particle (equal without a core). Returns false if
// spec is invalid (see planet_spec_ok). Used for the damper-stability note, which needs
// the lightest particle.
bool planet_particle_masses(const PlanetSpec &spec, int *n_core, double *m_core, double *m_mantle);

// What relax_planet did.
struct RelaxResult {
    int steps = 0;        // steps taken
    int hold = 0;         // steps kinetic energy had to stay below the threshold in a row
    double t_dyn = 0.0;   // the planet's dynamical time sqrt(R^3 / (G M)) (hold = t_dyn / dt)
    bool settled = false; // kinetic energy stayed below the threshold for hold steps
    bool finite = true;   // false: kinetic energy became NaN/Inf, so it stopped early
};

// Settles a freshly generated planet into a stable ball: leapfrog steps with the
// normal forces (params), each followed by v *= exp(-drag * dt) for every particle,
// until the total kinetic energy has stayed below ke_threshold for one dynamical time
// sqrt(R^3 / (G M)) of the planet (R: farthest particle center from the center of
// mass), or max_steps steps are done, or kinetic energy stops being finite. A single
// low reading isn't enough, because a sagging planet's kinetic energy passes near zero
// at every turning point. The hold is a time, so a smaller dt needs more steps.
// ke_threshold = 0 can never be met. Computes the accelerations itself before the first
// step. A drag much stronger than 1 / t_dyn only makes the planet creep: its kinetic
// energy then stays low while it is still far from settled (see MAX_RELAX_DRAG_T_DYN).
// p must hold ONE planet at rest (velocity zero): relax each planet on its own,
// then give it its velocity and append it to the main Particles. Otherwise bulk
// motion counts as kinetic energy and the planets pull on each other. Afterwards
// ax/ay/az are stale (the drag ran after the last force evaluation), so the caller
// must recompute accelerations before the main run (sim_init does). Heat from the
// contact dampers during relaxation is set back to zero at the end (unless it stopped
// on NaN/Inf), so the planet starts the run cold.
RelaxResult relax_planet(Particles &p, const SimParams &params, Backend backend, double drag,
                         double ke_threshold, int max_steps);

// Above this drag * t_dyn, relaxation can report "settled" for a planet that isn't. On
// the smoke planet (t_dyn = 0.58) it still truly settled at 145 and failed at 203 (it
// moved at up to 900x the threshold afterwards); this is half of that.
const double MAX_RELAX_DRAG_T_DYN = 100.0;
