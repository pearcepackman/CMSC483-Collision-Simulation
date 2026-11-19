#include "init.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "diagnostics.hpp"
#include "forces.hpp"
#include "integrator.hpp"

// Each particle is nudged by at most this fraction of its radius, so the lattice
// isn't perfectly regular. Small enough that neighbors can't overlap much.
static const double JITTER_FRACTION = 0.02;
// Largest allowed overlap of two fresh particles, as a fraction of particle_radius.
static const double MAX_OVERLAP_FRACTION = 0.1;

// A bad PlanetSpec would give a planet that explodes or is the wrong size, so stop
// loudly instead (like not_implemented).
[[noreturn]] static void planet_spec_error(const PlanetSpec &spec, const char *why) {
    std::fprintf(stderr, "make_planet: %s (n=%d, radius=%g, particle_radius=%g)\n", why, spec.n,
                 spec.radius, spec.particle_radius);
    std::abort();
}

// A point of the face-centered cubic (FCC) lattice, the densest regular packing.
// Measured in half cube sides, the FCC points are exactly the integer (ix, iy, iz)
// with an even sum, and key = ix^2 + iy^2 + iz^2 is the squared distance from the
// origin, exact as an integer.
struct LatticePoint {
    long long key;
    int ix, iy, iz;
};

// Closer to the origin first; ties by coordinates. Integers only, so every compiler
// and flag set (e.g. FMA under -march=native) picks the same points in the same order.
static bool closer_to_origin(const LatticePoint &a, const LatticePoint &b) {
    if (a.key != b.key) {
        return a.key < b.key;
    }
    if (a.ix != b.ix) {
        return a.ix < b.ix;
    }
    if (a.iy != b.iy) {
        return a.iy < b.iy;
    }
    return a.iz < b.iz;
}

// The n FCC points closest to the origin, closest first.
static std::vector<LatticePoint> closest_fcc_points(int n) {
    // There is one point per 2 cubic units, so a ball of radius cbrt(3n / (2 pi))
    // holds about n. Start a bit bigger and grow until it holds at least n.
    const double pi = 3.14159265358979323846;
    int h = static_cast<int>(std::ceil(std::cbrt(3.0 * n / (2.0 * pi)))) + 2;
    std::vector<LatticePoint> pts;
    for (;;) {
        // Every point with key <= h^2 lies inside the [-h, h] cube, so the ball of
        // radius h is complete and its n closest points are the overall n closest.
        const long long limit = static_cast<long long>(h) * h;
        pts.clear();
        for (int ix = -h; ix <= h; ++ix) {
            for (int iy = -h; iy <= h; ++iy) {
                for (int iz = -h; iz <= h; ++iz) {
                    if ((ix + iy + iz) % 2 != 0) {
                        continue;
                    }
                    long long key = static_cast<long long>(ix) * ix +
                                    static_cast<long long>(iy) * iy +
                                    static_cast<long long>(iz) * iz;
                    if (key <= limit) {
                        pts.push_back(LatticePoint{key, ix, iy, iz});
                    }
                }
            }
        }
        if (pts.size() >= static_cast<std::size_t>(n)) {
            break;
        }
        h += 2;
    }
    std::sort(pts.begin(), pts.end(), closer_to_origin);
    pts.resize(static_cast<std::size_t>(n));
    return pts;
}

// Uniform in [-1, 1) from the generator's raw bits. std::mt19937_64 gives the same
// numbers everywhere but std::uniform_real_distribution doesn't (libstdc++ on Linux
// and libc++ on macOS differ), so this keeps a seed's planet the same on both.
static double uniform_pm1(std::mt19937_64 &rng) {
    return static_cast<double>(rng() >> 11) * 0x1.0p-52 - 1.0;
}

// Where a planet's particles go: the FCC points, their mean (subtracted to center the
// planet) and the scale from lattice units to distance.
struct PlanetLayout {
    std::vector<LatticePoint> pts;
    double mx = 0.0, my = 0.0, mz = 0.0;
    double half_side = 0.0;
    double jitter = 0.0;
};

// Checks spec and, if it's usable, fills *layout. Returns nullptr on success or the
// reason the spec can't be built. make_planet and planet_spec_ok share it, so a config
// file and the generator always agree on what is valid.
static const char *layout_planet(const PlanetSpec &spec, PlanetLayout *layout) {
    if (spec.n < 0) {
        return "n must be >= 0";
    }
    if (!std::isfinite(spec.radius) || !std::isfinite(spec.particle_radius) ||
        !std::isfinite(spec.mass) || spec.particle_radius < 0.0 || spec.mass < 0.0) {
        return "radius, particle_radius and mass must be finite and >= 0";
    }
    if (spec.n == 0) {
        return nullptr;
    }
    // Contact divides by each particle's mass, so a massless particle would get a NaN
    // acceleration even when it touches nothing.
    if (spec.mass == 0.0) {
        return "mass must be > 0 when n > 0";
    }

    // Centers stay within r_max, so each whole particle is inside the sphere even
    // after its jitter.
    const double pr = spec.particle_radius;
    const double jitter = JITTER_FRACTION * pr;
    const double r_max = spec.radius - pr - jitter;
    if (r_max < 0.0) {
        return "radius is too small for particle_radius";
    }

    std::vector<LatticePoint> pts = closest_fcc_points(spec.n);
    const std::size_t n = pts.size();

    // Center the points on their mean. The outer shell is usually only partly filled,
    // so without this the planet's center of mass would sit off spec.center.
    long long sx = 0, sy = 0, sz = 0;
    for (std::size_t i = 0; i < n; ++i) {
        sx += pts[i].ix;
        sy += pts[i].iy;
        sz += pts[i].iz;
    }
    const double mx = static_cast<double>(sx) / spec.n;
    const double my = static_cast<double>(sy) / spec.n;
    const double mz = static_cast<double>(sz) / spec.n;
    double extent = 0.0; // farthest centered point, in half cube sides
    for (std::size_t i = 0; i < n; ++i) {
        double dx = pts[i].ix - mx;
        double dy = pts[i].iy - my;
        double dz = pts[i].iz - mz;
        extent = std::max(extent, std::sqrt(dx * dx + dy * dy + dz * dz));
    }

    // Largest spacing that keeps every center within r_max. A single particle has
    // extent 0 and just sits at the center.
    const double half_side = extent > 0.0 ? r_max / extent : 0.0;
    if (n > 1) {
        // Nearest neighbors are sqrt(2) half sides apart; jitter can bring two closer
        // by 2 * jitter.
        double closest = std::sqrt(2.0) * half_side - 2.0 * jitter;
        if (2.0 * pr - closest >= MAX_OVERLAP_FRACTION * pr) {
            return "too many particles for this radius and particle_radius";
        }
    }

    layout->pts = pts;
    layout->mx = mx;
    layout->my = my;
    layout->mz = mz;
    layout->half_side = half_side;
    layout->jitter = jitter;
    return nullptr;
}

bool planet_spec_ok(const PlanetSpec &spec, const char **why) {
    PlanetLayout layout;
    const char *err = layout_planet(spec, &layout);
    if (why != nullptr) {
        *why = err;
    }
    return err == nullptr;
}

void make_planet(Particles &p, const PlanetSpec &spec) {
    PlanetLayout layout;
    const char *err = layout_planet(spec, &layout);
    if (err != nullptr) {
        planet_spec_error(spec, err);
    }
    if (spec.n == 0) {
        return;
    }
    const std::vector<LatticePoint> &pts = layout.pts;
    const std::size_t n = pts.size();
    const double mx = layout.mx, my = layout.my, mz = layout.mz;
    const double half_side = layout.half_side;
    const double jitter = layout.jitter;
    const double pr = spec.particle_radius;

    const std::size_t base = p.size();
    p.resize(base + n);
    std::mt19937_64 rng(spec.seed);
    for (std::size_t i = 0; i < n; ++i) {
        double jx, jy, jz;
        do { // rejection sampling: a random offset inside a ball of radius 1
            jx = uniform_pm1(rng);
            jy = uniform_pm1(rng);
            jz = uniform_pm1(rng);
        } while (jx * jx + jy * jy + jz * jz > 1.0);
        p.x[base + i] = (pts[i].ix - mx) * half_side + jitter * jx + spec.center.x;
        p.y[base + i] = (pts[i].iy - my) * half_side + jitter * jy + spec.center.y;
        p.z[base + i] = (pts[i].iz - mz) * half_side + jitter * jz + spec.center.z;
        p.vx[base + i] = spec.velocity.x;
        p.vy[base + i] = spec.velocity.y;
        p.vz[base + i] = spec.velocity.z;
        p.mass[base + i] = spec.mass / spec.n;
        p.radius[base + i] = pr;
        p.body[base + i] = spec.body;
    }
}

// The dynamical time sqrt(R^3 / (G M)) of the planet in p (G = 1), with R the farthest
// particle center from the center of mass. Roughly the time a particle takes to cross
// the planet under its gravity, so the planet's sag-and-rebound oscillation fits inside
// it. 0 for an empty or massless p, or a single particle.
static double dynamical_time(const Particles &p) {
    double m = 0.0, cx = 0.0, cy = 0.0, cz = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        m += p.mass[i];
        cx += p.mass[i] * p.x[i];
        cy += p.mass[i] * p.y[i];
        cz += p.mass[i] * p.z[i];
    }
    if (!(m > 0.0)) {
        return 0.0;
    }
    cx /= m;
    cy /= m;
    cz /= m;
    double r2 = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        double dx = p.x[i] - cx, dy = p.y[i] - cy, dz = p.z[i] - cz;
        r2 = std::max(r2, dx * dx + dy * dy + dz * dz);
    }
    return std::sqrt(r2 * std::sqrt(r2) / m);
}

// Steps that make up t_dyn, at least 1. Clamped in double before the cast: a tiny dt or
// a huge planet would overflow int (undefined behavior; on x86 it gave INT_MIN, so a
// hold of 1 and an unrelaxed planet reported as settled).
static int steps_for(double t_dyn, double dt) {
    if (!(dt > 0.0)) {
        return 1;
    }
    double s = std::ceil(t_dyn / dt);
    if (!(s < static_cast<double>(INT_MAX))) {
        return INT_MAX;
    }
    return std::max(1, static_cast<int>(s));
}

RelaxResult relax_planet(Particles &p, const SimParams &params, Backend backend, double drag,
                         double ke_threshold, int max_steps) {
    RelaxResult r;
    // One factor for every step: each velocity loses this fraction of itself, the same
    // as a drag force -drag * m * v over one step.
    const double keep = std::exp(-drag * params.dt);
    // Settled means below the threshold for a whole dynamical time, not just once: the
    // planet sags and rebounds, and kinetic energy passes near zero at every turning
    // point (a 2,000-particle planet dipped to 3e-4 at step 150, then rose to 1.1e-3).
    // This also covers the start, where a planet at rest has zero kinetic energy.
    r.t_dyn = dynamical_time(p);
    r.hold = steps_for(r.t_dyn, params.dt);
    compute_accelerations(p, params, backend); // leapfrog needs a(t) before the first step
    int calm = 0;                              // consecutive steps that ended below the threshold
    while (r.steps < max_steps && calm < r.hold) {
        leapfrog_step(p, params, backend);
        // The drag changes velocities after the step's forces were computed, so the
        // contact damper term in ax/ay/az is from the pre-drag velocities. That is a
        // small error in a phase whose only job is to settle the planet, and fixing it
        // would mean a second force evaluation per step.
        for (std::size_t i = 0; i < p.size(); ++i) {
            p.vx[i] *= keep;
            p.vy[i] *= keep;
            p.vz[i] *= keep;
        }
        ++r.steps;
        double ke = kinetic_energy(p);
        if (!std::isfinite(ke)) {
            r.finite = false; // blew up (dt far too large?): more steps won't help
            return r;
        }
        calm = ke < ke_threshold ? calm + 1 : 0;
    }
    r.settled = calm >= r.hold;
    return r;
}
