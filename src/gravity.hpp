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

// Reference all-pairs gravity with softening (G = 1). Overwrites ax/ay/az:
//   a_i = sum_{j != i} m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
// Every other backend is tested against this one. Softening must be > 0 if two
// particles can sit at the same position (eps = 0 there divides by zero -> NaN).
void compute_gravity_serial(Particles &p, const SimParams &params);

// OpenMP all-pairs gravity. Overwrites ax/ay/az.
void compute_gravity_omp(Particles &p, const SimParams &params);

// Total gravitational potential energy, softened to match the forces above:
//   U = -sum_{i<j} m_i m_j / sqrt(|r_j - r_i|^2 + eps^2)
// O(N^2) and serial, so log it occasionally, not every step. With eps = 0, two
// particles at the same position give -inf.
double potential_energy(const Particles &p, const SimParams &params);
