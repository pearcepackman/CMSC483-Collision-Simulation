#pragma once

#include "gravity.hpp"
#include "params.hpp"
#include "particles.hpp"

// One planet to generate: a ball of particles at a position and velocity.
struct PlanetSpec {
    int n = 0;                    // particle count
    double radius = 0.0;          // planet radius
    double particle_radius = 0.0; // radius of each particle
    double mass = 0.0;            // total planet mass, split evenly across particles
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

// Runs damped steps until kinetic energy drops below ke_threshold or max_steps
// is reached, so a freshly generated planet settles into a stable ball.
// p must hold ONE planet at rest (velocity zero): relax each planet on its own,
// then give it its velocity and append it to the main Particles. Otherwise bulk
// motion counts as kinetic energy and the planets pull on each other.
// Returns the number of steps taken.
int relax_planet(Particles &p, const SimParams &params, Backend backend, double ke_threshold,
                 int max_steps);
