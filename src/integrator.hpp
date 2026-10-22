#pragma once

#include "gravity.hpp"
#include "params.hpp"
#include "particles.hpp"

// One leapfrog (kick-drift-kick) step of size params.dt. ax/ay/az must already
// hold the accelerations for the current positions: call compute_accelerations()
// once before the first step.
void leapfrog_step(Particles &p, const SimParams &params, Backend backend);
