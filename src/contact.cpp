#include "contact.hpp"

#include <cmath>

#include "grid.hpp"

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
            // and ends apart, to about 1e-5 of the starting kinetic energy in the bounce
            // tests, and 1.3% short in the head-on collision (docs/notes.md). It is < 0
            // only past c dt / m_eff = 2, where the damper is unstable anyway
            // (build_planets warns).
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

// Particles that touch are less than 2 * max radius apart, so a cell of that side puts
// them in the same or neighboring cells. The tiny margin is only a safeguard: review
// scanned pairs just below every cell boundary up to 2^40 cells and found none two
// cells apart even without it (subtraction and division round the same way), so no
// test can fail if it's removed.
static double contact_cell_size(const Particles &p) {
    double r_max = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        if (p.radius[i] > r_max) {
            r_max = p.radius[i];
        }
    }
    return 2.0 * r_max * (1.0 + 1e-9);
}

// The grid's arrays, kept between calls so a step allocates nothing (they only grow).
// One call at a time, like the HIP gravity buffers: two threads calling
// compute_contact_grid at once would rebuild the grid under each other (review saw a
// crash). Splitting the loop inside one call (#27) is fine.
static CellList g_grid;

// Same result as compute_contact_bruteforce, but particle i only looks at the particles
// in its own cell and the 26 around it. The per-pair math is written out again on
// purpose instead of shared, so the test that compares the two checks two independent
// copies. Each pair is still computed from both sides, so i writes only its own
// ax/ay/az and heat_rate (#27 can split the outer loop across threads with no races).
void compute_contact_grid(Particles &p, const SimParams &params, std::size_t table_size) {
    const std::size_t n = p.size();
    const double k = params.contact_k;
    const double c = params.contact_c;
    const double dt = params.dt;
    const double cell_size = contact_cell_size(p);
    if (n == 0) {
        return;
    }
    if (!(cell_size > 0.0)) { // every radius is 0: nothing can touch
        for (std::size_t i = 0; i < n; ++i) {
            p.heat_rate[i] = 0.0;
        }
        return;
    }
    if (table_size == 0) {
        table_size = cell_table_size(n);
    }
    build_cell_list(p, cell_size, table_size, g_grid);
    const CellList &g = g_grid;

    for (std::size_t i = 0; i < n; ++i) {
        double fx = 0.0, fy = 0.0, fz = 0.0;
        double q = 0.0;
        for (std::int64_t ox = -1; ox <= 1; ++ox) {
            for (std::int64_t oy = -1; oy <= 1; ++oy) {
                for (std::int64_t oz = -1; oz <= 1; ++oz) {
                    const std::int64_t cx = g.cx[i] + ox, cy = g.cy[i] + oy, cz = g.cz[i] + oz;
                    const std::size_t s = cell_slot(cx, cy, cz, g.table_size);
                    for (std::size_t a = g.start[s]; a < g.start[s + 1]; ++a) {
                        const std::size_t j = g.order[a];
                        // Another cell sharing this slot, or i itself (the d2 == 0 check
                        // below would skip i too; this just says so).
                        if (g.cx[j] != cx || g.cy[j] != cy || g.cz[j] != cz || j == i) {
                            continue;
                        }
                        double dx = p.x[i] - p.x[j];
                        double dy = p.y[i] - p.y[j];
                        double dz = p.z[i] - p.z[j];
                        double reach = p.radius[i] + p.radius[j];
                        double d2 = dx * dx + dy * dy + dz * dz;
                        if (d2 >= reach * reach || d2 == 0.0) {
                            continue;
                        }
                        double d = std::sqrt(d2);
                        double nx = dx / d, ny = dy / d, nz = dz / d;
                        double overlap = reach - d;
                        double vn = (p.vx[i] - p.vx[j]) * nx + (p.vy[i] - p.vy[j]) * ny +
                                    (p.vz[i] - p.vz[j]) * nz;
                        double f = k * overlap - c * vn;
                        fx += f * nx;
                        fy += f * ny;
                        fz += f * nz;
                        double m_eff = p.mass[i] * p.mass[j] / (p.mass[i] + p.mass[j]);
                        q += 0.5 * c * vn * vn * (1.0 - 0.5 * c * dt / m_eff);
                    }
                }
            }
        }
        p.ax[i] += fx / p.mass[i];
        p.ay[i] += fy / p.mass[i];
        p.az[i] += fz / p.mass[i];
        p.heat_rate[i] = q;
    }
}
