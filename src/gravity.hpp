#pragma once

#include "params.hpp"
#include "particles.hpp"

// Which implementation computes gravity. Every backend takes the same inputs and
// produces the same outputs (within floating-point tolerance).
enum Backend { BACKEND_SERIAL, BACKEND_OMP, BACKEND_HIP };

const char *backend_name(Backend b); // "serial", "omp" or "hip"

// Parses a backend name (exact, case-sensitive). Returns false and leaves *out
// untouched if the name is unknown or null.
bool backend_from_name(const char *name, Backend *out);

// Gravity on the chosen backend. Overwrites ax/ay/az.
void compute_gravity(Particles &p, const SimParams &params, Backend b);

// Reference all-pairs gravity with softening. Overwrites ax/ay/az. Every other
// backend is tested against this one.
void compute_gravity_serial(Particles &p, const SimParams &params);

// OpenMP all-pairs gravity. Overwrites ax/ay/az.
void compute_gravity_omp(Particles &p, const SimParams &params);

// Total gravitational potential energy (softened, matching the forces).
double potential_energy(const Particles &p, const SimParams &params);
