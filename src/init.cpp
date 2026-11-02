#include "init.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "util.hpp"

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

void make_planet(Particles &p, const PlanetSpec &spec) {
    if (spec.n < 0) {
        planet_spec_error(spec, "n must be >= 0");
    }
    if (!std::isfinite(spec.radius) || !std::isfinite(spec.particle_radius) ||
        !std::isfinite(spec.mass) || spec.particle_radius < 0.0 || spec.mass < 0.0) {
        planet_spec_error(spec, "radius, particle_radius and mass must be finite and >= 0");
    }
    if (spec.n == 0) {
        return;
    }

    // Centers stay within r_max, so each whole particle is inside the sphere even
    // after its jitter.
    const double pr = spec.particle_radius;
    const double jitter = JITTER_FRACTION * pr;
    const double r_max = spec.radius - pr - jitter;
    if (r_max < 0.0) {
        planet_spec_error(spec, "radius is too small for particle_radius");
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
            planet_spec_error(spec, "too many particles for this radius and particle_radius");
        }
    }

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

// Stub until #15.
int relax_planet(Particles &, const SimParams &, Backend, double, int) {
    not_implemented("relax_planet");
}
