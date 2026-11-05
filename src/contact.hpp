#pragma once

#include "params.hpp"
#include "particles.hpp"

// Soft-sphere contact (spring + damper) between overlapping particles, checking
// all pairs. Adds to ax/ay/az, so run it after gravity. For particles i and j that
// overlap by delta = r_i + r_j - |r_i - r_j| > 0, the force on i is
//   F_i = (k * delta - c * (v_i - v_j) . n) * n,   n = (r_i - r_j) / |r_i - r_j|
// and a_i += F_i / m_i. F_j = -F_i exactly. k = contact_k (force per unit overlap),
// c = contact_c (force per unit approach speed). Two particles at exactly the same
// position have no direction to push in and get no contact force.
// Every mass must be > 0: each particle's force is divided by its own mass, even when
// it touches nothing (0 / 0 = NaN).
// The force is not clamped: while a pair separates, the damper term can outweigh the
// spring and pull the two together for the last part of the contact. That is the plain
// spring-dashpot model, and the restitution formula in docs/kickstart.md assumes it.
// Explicit-step limits (docs/notes.md): energy grows if c * dt / m_eff >= 2, and pairs
// faster than about (r_i + r_j) * sqrt(k / m_eff) pass through each other.
void compute_contact_bruteforce(Particles &p, const SimParams &params);
