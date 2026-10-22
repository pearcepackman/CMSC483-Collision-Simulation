#pragma once

#include "particles.hpp"

// Conserved-quantity checks used by tests and the run loop to catch
// physics bugs (drifting momentum, rising energy).
double total_mass(const Particles &p);
Vec3 total_momentum(const Particles &p);
double kinetic_energy(const Particles &p);
