#pragma once

#include "gravity.hpp"
#include "params.hpp"
#include "particles.hpp"

// One leapfrog (kick-drift-kick) step of size params.dt. ax/ay/az must already
// hold the accelerations for the current positions: call compute_accelerations()
// once before the first step, and again whenever anything other than leapfrog_step
// changes positions, velocities, masses, the particle count or the force settings
// (e.g. after appending relaxed planets or switching off relaxation damping).
// Changing params.dt between steps is fine.
void leapfrog_step(Particles &p, const SimParams &params, Backend backend);
