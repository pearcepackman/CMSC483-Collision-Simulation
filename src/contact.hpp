#pragma once

#include "params.hpp"
#include "particles.hpp"

// Soft-sphere contact (spring + damper) between overlapping particles, checking
// all pairs. Adds to ax/ay/az, so run it after gravity.
void compute_contact_bruteforce(Particles &p, const SimParams &params);
