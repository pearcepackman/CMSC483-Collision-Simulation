#pragma once

#include "particles.hpp"

// Conserved-quantity checks used by tests and the run loop to catch
// physics bugs (drifting momentum, rising energy).
double total_mass(const Particles &p);
Vec3 total_momentum(const Particles &p);
// Sum of m (r x v) about the origin. Gravity and contact are central forces, so it's
// conserved up to rounding.
Vec3 total_angular_momentum(const Particles &p);
double kinetic_energy(const Particles &p);
// Sum of p.heat: the energy contact damping has turned into heat so far.
double total_heat(const Particles &p);
