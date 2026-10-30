#pragma once

#include <string>

#include "params.hpp"
#include "particles.hpp"

// Which implementation computes gravity. Every backend takes the same inputs and
// produces the same outputs (within floating-point tolerance).
enum Backend { BACKEND_SERIAL, BACKEND_OMP, BACKEND_HIP };

const char *backend_name(Backend b); // "serial", "omp" or "hip"

// Parses a backend name (exact, case-sensitive). Returns false and leaves *out
// untouched if the name is unknown or null.
bool backend_from_name(const char *name, Backend *out);

// True if this build includes the backend: serial and OpenMP always, HIP only when
// built with ENABLE_HIP (the `hip` preset).
bool backend_available(Backend b);

// Gravity on the chosen backend. Overwrites ax/ay/az.
void compute_gravity(Particles &p, const SimParams &params, Backend b);

// Reference all-pairs gravity with softening (G = 1). Overwrites ax/ay/az:
//   a_i = sum_{j != i} m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
// Every other backend is tested against this one. Softening must be > 0 if two
// particles can sit at the same position (eps = 0 there divides by zero -> NaN).
void compute_gravity_serial(Particles &p, const SimParams &params);

// OpenMP all-pairs gravity. Overwrites ax/ay/az.
void compute_gravity_omp(Particles &p, const SimParams &params);

// Which GPU to run on, as a pure function so it can be tested without a GPU. Among
// `count` devices, the one with the most compute units whose architecture is in
// `built_archs` (comma-separated, e.g. "gfx1200,gfx1036": the GPUs the kernel was
// compiled for). archs[i] may carry feature flags ("gfx1200:xnack-"); only the part
// before ':' must match a listed name exactly. Returns -1 if no device qualifies.
int pick_gpu(int count, const int *compute_units, const char *const *archs,
             const char *built_archs);

// ---- HIP backend: src/gravity_hip.hip, only built with ENABLE_HIP (SIM_HAVE_HIP) ----

// Picks the GPU with pick_gpu() among the visible devices (HIP_VISIBLE_DEVICES is
// respected), so a discrete card wins over a CPU's integrated GPU, and a GPU the kernel
// wasn't compiled for is never chosen. Returns false if there is no usable GPU. *info
// gets the device description, or the error. Safe to call more than once.
bool hip_init(std::string *info);

// All-pairs gravity on the GPU in double precision: same formula and summation order
// as serial. Overwrites ax/ay/az. Copies positions and masses to the GPU and the
// accelerations back on every call. Calls hip_init if needed; aborts on any HIP error.
void compute_gravity_hip(Particles &p, const SimParams &params);

// Frees the GPU buffers compute_gravity_hip keeps between calls.
void hip_release();

// Total gravitational potential energy, softened to match the forces above:
//   U = -sum_{i<j} m_i m_j / sqrt(|r_j - r_i|^2 + eps^2)
// O(N^2) and serial, so log it occasionally, not every step. With eps = 0, two
// particles at the same position give -inf.
double potential_energy(const Particles &p, const SimParams &params);
