#pragma once

#include "gravity.hpp"
#include "params.hpp"
#include "particles.hpp"

// All forces for one evaluation, in a fixed order: gravity on the chosen backend
// (overwrites ax/ay/az), then contact if enabled (adds to ax/ay/az).
void compute_accelerations(Particles &p, const SimParams &params, Backend backend);
