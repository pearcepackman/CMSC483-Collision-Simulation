#include "integrator.hpp"

#include "forces.hpp"

// Kick-drift-kick leapfrog. Unlike Euler, its energy error stays bounded over long
// runs instead of growing every orbit (see docs/notes.md).
void leapfrog_step(Particles &p, const SimParams &params, Backend backend) {
    const std::size_t n = p.size();
    const double dt = params.dt;
    const double half_dt = 0.5 * dt;

    // Kick: half a step of velocity using the accelerations at the current positions.
    for (std::size_t i = 0; i < n; ++i) {
        p.vx[i] += p.ax[i] * half_dt;
        p.vy[i] += p.ay[i] * half_dt;
        p.vz[i] += p.az[i] * half_dt;
    }

    // Drift: a full step of position using the half-step velocities.
    for (std::size_t i = 0; i < n; ++i) {
        p.x[i] += p.vx[i] * dt;
        p.y[i] += p.vy[i] * dt;
        p.z[i] += p.vz[i] * dt;
    }

    // Forces at the new positions. These also serve as the next step's first kick.
    compute_accelerations(p, params, backend);

    // Kick: the second half step of velocity.
    for (std::size_t i = 0; i < n; ++i) {
        p.vx[i] += p.ax[i] * half_dt;
        p.vy[i] += p.ay[i] * half_dt;
        p.vz[i] += p.az[i] * half_dt;
    }
}
