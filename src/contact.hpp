#pragma once

#include "params.hpp"
#include "particles.hpp"

// Soft-sphere contact (spring + damper) between overlapping particles, in two versions
// (below) that give the same result. Each adds to ax/ay/az, so run it after gravity.
// For particles i and j that overlap by delta = r_i + r_j - |r_i - r_j| > 0, the force
// on i is
//   F_i = (k * delta - c * (v_i - v_j) . n) * n,   n = (r_i - r_j) / |r_i - r_j|
// and a_i += F_i / m_i. F_j = -F_i exactly. k = contact_k (force per unit overlap),
// c = contact_c (force per unit approach speed). Two particles at exactly the same
// position have no direction to push in and get no contact force.
// It also sets (overwrites) heat_rate[i]: half the damper power c * vn^2 of each of i's
// contacts, times 1 - c * dt / (2 m_eff) (see contact.cpp), so the two particles of a
// pair heat up equally. leapfrog_step turns the rate into heat.
// Every mass must be > 0: each particle's force is divided by its own mass, even when
// it touches nothing (0 / 0 = NaN).
// The force is not clamped: while a pair separates, the damper term can outweigh the
// spring and pull the two together for the last part of the contact. That is the plain
// spring-dashpot model, and the restitution formula in docs/kickstart.md assumes it.
// Explicit-step limits (docs/notes.md): energy grows if c * dt / m_eff >= 2, and pairs
// faster than about (r_i + r_j) * sqrt(k / m_eff) pass through each other.

// Energy stored in compressed contact springs: the sum over overlapping pairs, each pair
// once, of 0.5 * k * delta^2. Same overlap rules as the force (a pair exactly touching
// or at the same point counts 0), and 0 when contact is off. O(N^2) like
// potential_energy, so only compute it when writing an energy-log row (#59).
double contact_energy(const Particles &p, const SimParams &params);

// Checks every pair, O(N^2): the reference the grid version is tested against.
// Keep it simple; don't speed it up.
void compute_contact_bruteforce(Particles &p, const SimParams &params);

// The same forces and heat_rate as compute_contact_bruteforce (up to rounding: the pairs
// are summed in a different order), in about O(N): particles are binned into a hashed
// grid of cells 2 * max radius wide (grid.hpp) and each particle checks only its own
// cell and the 26 around it. This is what compute_accelerations uses. table_size = 0
// picks the normal size; tests pass a small one to force cells to share slots.
// Limits: the cell size comes from the single largest radius, so mixing very different
// particle sizes makes the small particles' cells crowded (one particle 50x larger than
// the rest made it slower than all pairs; see docs/notes.md). It reuses one grid between
// calls, so call it from one thread at a time (splitting its loop inside is fine, #27).
// Radii must be >= 0 (make_planet ensures it); the two versions differ for negative ones.
void compute_contact_grid(Particles &p, const SimParams &params, std::size_t table_size = 0);
