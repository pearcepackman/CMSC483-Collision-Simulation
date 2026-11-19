#include "contact.hpp"

#include <cmath>

// Same loop shape as gravity_serial.cpp: each particle sums the push from every other,
// so each pair is computed twice. The two sides see exactly negated dx and dv, so their
// forces cancel bit for bit, and a parallel version can split the outer loop with no
// two threads writing the same particle.
void compute_contact_bruteforce(Particles &p, const SimParams &params) {
    const std::size_t n = p.size();
    const double k = params.contact_k;
    const double c = params.contact_c;
    const double dt = params.dt; // only for the heat (see below)

    for (std::size_t i = 0; i < n; ++i) {
        double fx = 0.0, fy = 0.0, fz = 0.0;
        double q = 0.0; // heating rate of particle i
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) {
                continue;
            }
            double dx = p.x[i] - p.x[j];
            double dy = p.y[i] - p.y[j];
            double dz = p.z[i] - p.z[j];
            double reach = p.radius[i] + p.radius[j];
            double d2 = dx * dx + dy * dy + dz * dz;
            // Not touching, or at the same point (no direction to push in).
            if (d2 >= reach * reach || d2 == 0.0) {
                continue;
            }
            double d = std::sqrt(d2);
            double nx = dx / d, ny = dy / d, nz = dz / d; // unit vector from j to i
            double overlap = reach - d;
            // Normal relative velocity: negative while the pair is approaching.
            double vn =
                (p.vx[i] - p.vx[j]) * nx + (p.vy[i] - p.vy[j]) * ny + (p.vz[i] - p.vz[j]) * nz;
            double f = k * overlap - c * vn; // spring pushes apart, damper resists vn
            fx += f * nx;
            fy += f * ny;
            fz += f * nz;
            // The damper turns c * vn^2 of power into heat. vn is from the half-step
            // velocities, and the damper force acts through two half kicks (this step's
            // last, the next step's first), which shrink vn on average by the factor
            // (1 - c dt / (2 m_eff)). With it, the heat matches the kinetic energy the
            // damper removes: exactly for a damper acting alone over a contact that begins
            // and ends apart, to about 1e-5 of the starting kinetic energy in the bounce tests, 1.3% short
            // in the head-on collision (docs/notes.md). It is < 0 only past
            // c dt / m_eff = 2, where the damper is unstable anyway (build_planets warns).
            // This pair is visited once from each side, so each side takes half.
            double m_eff = p.mass[i] * p.mass[j] / (p.mass[i] + p.mass[j]);
            q += 0.5 * c * vn * vn * (1.0 - 0.5 * c * dt / m_eff);
        }
        p.ax[i] += fx / p.mass[i];
        p.ay[i] += fy / p.mass[i];
        p.az[i] += fz / p.mass[i];
        p.heat_rate[i] = q;
    }
}
