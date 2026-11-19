#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "contact.hpp"
#include "diagnostics.hpp"
#include "forces.hpp"
#include "grid.hpp"
#include "init.hpp"
#include "integrator.hpp"
#include "params.hpp"
#include "particles.hpp"

namespace {

SimParams contact_params() {
    SimParams params;
    params.dt = 0.002;
    params.softening = 0.05;
    params.contact_enabled = true;
    params.contact_k = 20.0;
    params.contact_c = 0.05;
    return params;
}

// Random velocities in [-v, v], so the dampers (and heat) have something to do.
void shake(Particles &p, double v, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-v, v);
    for (std::size_t i = 0; i < p.size(); ++i) {
        p.vx[i] = u(rng);
        p.vy[i] = u(rng);
        p.vz[i] = u(rng);
    }
}

// The smoke planet (250 particles, radius 0.5), squeezed by 5% so neighbors overlap.
Particles squeezed_planet(Vec3 center, unsigned seed) {
    PlanetSpec s;
    s.n = 250;
    s.radius = 0.5;
    s.particle_radius = 0.06;
    s.mass = 0.25;
    s.seed = seed;
    Particles p;
    make_planet(p, s);
    for (std::size_t i = 0; i < p.size(); ++i) {
        p.x[i] = 0.95 * p.x[i] + center.x;
        p.y[i] = 0.95 * p.y[i] + center.y;
        p.z[i] = 0.95 * p.z[i] + center.z;
    }
    return p;
}

void append(Particles &dst, const Particles &src) {
    const std::size_t base = dst.size();
    dst.resize(base + src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        dst.x[base + i] = src.x[i];
        dst.y[base + i] = src.y[i];
        dst.z[base + i] = src.z[i];
        dst.vx[base + i] = src.vx[i];
        dst.vy[base + i] = src.vy[i];
        dst.vz[base + i] = src.vz[i];
        dst.mass[base + i] = src.mass[i];
        dst.radius[base + i] = src.radius[i];
    }
}

// n particles in a cube of side `box` around center, radii in [r_min, r_max], masses in
// [0.5, 2], random velocities.
Particles cloud(std::size_t n, double box, Vec3 center, double r_min, double r_max, unsigned seed) {
    Particles p;
    p.resize(n);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> pos(-0.5 * box, 0.5 * box), rad(r_min, r_max),
        mass(0.5, 2.0), vel(-1.0, 1.0);
    for (std::size_t i = 0; i < n; ++i) {
        p.x[i] = center.x + pos(rng);
        p.y[i] = center.y + pos(rng);
        p.z[i] = center.z + pos(rng);
        p.radius[i] = rad(rng);
        p.mass[i] = mass(rng);
        p.vx[i] = vel(rng);
        p.vy[i] = vel(rng);
        p.vz[i] = vel(rng);
    }
    return p;
}

// Runs both contact versions on copies of `base` (with nonzero starting accelerations,
// which both must add to, and stale heat rates, which both must overwrite) and checks
// they agree to 1e-12 of the largest force and heat rate. Returns the number of
// particles with a nonzero contact force, so a test can make sure contacts happened.
std::size_t expect_grid_matches(const Particles &base, const SimParams &params,
                                std::size_t table_size = 0) {
    Particles a = base, b = base;
    for (std::size_t i = 0; i < base.size(); ++i) {
        a.ax[i] = b.ax[i] = 0.25 * static_cast<double>(i % 7);
        a.heat_rate[i] = b.heat_rate[i] = 7.0;
    }
    compute_contact_bruteforce(a, params);
    compute_contact_grid(b, params, table_size);
    double f_max = 0.0, q_max = 0.0;
    std::size_t touched = 0;
    for (std::size_t i = 0; i < base.size(); ++i) {
        double m = a.mass[i];
        double fx = m * (a.ax[i] - 0.25 * static_cast<double>(i % 7));
        f_max = std::fmax(f_max, std::fabs(fx));
        f_max = std::fmax(f_max, std::fabs(m * a.ay[i]));
        f_max = std::fmax(f_max, std::fabs(m * a.az[i]));
        q_max = std::fmax(q_max, std::fabs(a.heat_rate[i]));
        if (fx != 0.0 || a.ay[i] != 0.0 || a.az[i] != 0.0) {
            ++touched;
        }
    }
    for (std::size_t i = 0; i < base.size(); ++i) {
        double m = a.mass[i];
        EXPECT_LE(m * std::fabs(a.ax[i] - b.ax[i]), 1e-12 * f_max) << "particle " << i;
        EXPECT_LE(m * std::fabs(a.ay[i] - b.ay[i]), 1e-12 * f_max) << "particle " << i;
        EXPECT_LE(m * std::fabs(a.az[i] - b.az[i]), 1e-12 * f_max) << "particle " << i;
        EXPECT_LE(std::fabs(a.heat_rate[i] - b.heat_rate[i]), 1e-12 * q_max) << "particle " << i;
    }
    return touched;
}

} // namespace

// ---- The grid itself ----

TEST(Grid, CellCoordIsFloorAndClamped) {
    EXPECT_EQ(cell_coord(0.0, 1.0), 0);
    EXPECT_EQ(cell_coord(0.99, 1.0), 0);
    EXPECT_EQ(cell_coord(1.0, 1.0), 1);
    EXPECT_EQ(cell_coord(-0.01, 1.0), -1); // floor, not truncation toward 0
    EXPECT_EQ(cell_coord(-1.0, 1.0), -1);
    EXPECT_EQ(cell_coord(-2.5, 0.5), -5);
    const std::int64_t limit = std::int64_t{1} << 40;
    EXPECT_EQ(cell_coord(1e300, 1.0), limit);
    EXPECT_EQ(cell_coord(-1e300, 1.0), -limit);
    EXPECT_EQ(cell_coord(std::numeric_limits<double>::infinity(), 1.0), limit);
    EXPECT_EQ(cell_coord(-std::numeric_limits<double>::infinity(), 1.0), -limit);
    EXPECT_EQ(cell_coord(std::nan(""), 1.0), 0);
}

TEST(Grid, TableSizeIsAPowerOfTwoAtLeastTwiceN) {
    EXPECT_EQ(cell_table_size(0), 1u);
    EXPECT_EQ(cell_table_size(1), 2u);
    EXPECT_EQ(cell_table_size(3), 8u);
    EXPECT_EQ(cell_table_size(4), 8u);
    EXPECT_EQ(cell_table_size(5), 16u);
    EXPECT_EQ(cell_table_size(4000), 8192u);
}

TEST(Grid, EveryParticleIsListedOnceUnderItsSlot) {
    Particles p = cloud(300, 2.0, Vec3{-0.3, 0.1, 0.0}, 0.05, 0.1, 3);
    const std::size_t sizes[] = {1, 2, 64, cell_table_size(300)};
    for (std::size_t table : sizes) {
        SCOPED_TRACE(table);
        CellList g;
        build_cell_list(p, 0.2, table, g);
        ASSERT_EQ(g.start.size(), table + 1);
        EXPECT_EQ(g.start[0], 0u);
        EXPECT_EQ(g.start[table], p.size());
        std::vector<int> seen(p.size(), 0);
        for (std::size_t s = 0; s < table; ++s) {
            ASSERT_LE(g.start[s], g.start[s + 1]);
            for (std::size_t a = g.start[s]; a < g.start[s + 1]; ++a) {
                std::size_t i = g.order[a];
                ASSERT_LT(i, p.size());
                ++seen[i];
                EXPECT_EQ(g.cx[i], cell_coord(p.x[i], 0.2));
                EXPECT_EQ(g.cy[i], cell_coord(p.y[i], 0.2));
                EXPECT_EQ(g.cz[i], cell_coord(p.z[i], 0.2));
                EXPECT_EQ(cell_slot(g.cx[i], g.cy[i], g.cz[i], table), s);
                if (a > g.start[s]) {
                    EXPECT_LT(g.order[a - 1], i); // index order inside a slot
                }
            }
        }
        for (std::size_t i = 0; i < p.size(); ++i) {
            EXPECT_EQ(seen[i], 1) << "particle " << i;
        }
    }
}

TEST(Grid, ReusedListShrinksCorrectly) {
    // Building for fewer particles after more must not keep stale entries.
    CellList g;
    build_cell_list(cloud(300, 2.0, Vec3{0, 0, 0}, 0.05, 0.1, 4), 0.2, 1024, g);
    Particles small = cloud(5, 2.0, Vec3{0, 0, 0}, 0.05, 0.1, 5);
    build_cell_list(small, 0.2, 16, g);
    EXPECT_EQ(g.start.size(), 17u);
    EXPECT_EQ(g.start[16], 5u);
    EXPECT_EQ(g.order.size(), 5u);
}

// ---- Grid contact matches brute force ----

TEST(GridContact, MatchesBruteForceOnAPlanet) {
    Particles p = squeezed_planet(Vec3{0.2, -0.1, 0.3}, 1);
    shake(p, 0.5, 7);
    EXPECT_GT(expect_grid_matches(p, contact_params()), 200u);
}

TEST(GridContact, MatchesBruteForceOnTwoPlanetsInContact) {
    // Centers 0.9 apart with radius ~0.475 each: they overlap where they meet, and
    // move toward each other.
    Particles p = squeezed_planet(Vec3{-0.45, 0.0, 0.0}, 1);
    Particles q = squeezed_planet(Vec3{0.45, 0.02, -0.01}, 2);
    shake(p, 0.1, 8);
    shake(q, 0.1, 9);
    for (std::size_t i = 0; i < p.size(); ++i) {
        p.vx[i] += 1.0;
        q.vx[i] -= 1.0;
    }
    append(p, q);
    EXPECT_GT(expect_grid_matches(p, contact_params()), 400u);
}

TEST(GridContact, MatchesBruteForceWithMixedRadii) {
    // Radii from 0.01 to 0.2: the cell size comes from the largest, so small particles
    // must still find big neighbors several of their own sizes away.
    Particles p = cloud(400, 2.0, Vec3{0.0, 0.0, 0.0}, 0.01, 0.2, 11);
    SimParams params = contact_params();
    params.contact_c = 0.3;
    EXPECT_GT(expect_grid_matches(p, params), 100u);
}

TEST(GridContact, MatchesBruteForceFarFromTheOrigin) {
    // Negative coordinates (floor, not truncation) and stragglers so far away that
    // their cell coordinates are clamped. Two stragglers at 1e15 touch each other.
    Particles p = cloud(200, 1.5, Vec3{-3e4, 7e3, -12.5}, 0.05, 0.12, 12);
    Particles far;
    far.resize(4);
    far.x[0] = 1e15;
    far.x[1] = 1e15 + 0.125; // radii 0.1 each: overlap 0.075
    far.y[2] = -2e14;
    far.z[3] = 3e13;
    for (std::size_t i = 0; i < 4; ++i) {
        far.radius[i] = 0.1;
        far.mass[i] = 1.0;
    }
    far.vx[0] = 0.5;
    append(p, far);
    expect_grid_matches(p, contact_params());
    // The far pair does push apart (both versions found it).
    Particles b = p;
    compute_contact_grid(b, contact_params());
    EXPECT_LT(b.ax[200], 0.0);
    EXPECT_GT(b.ax[201], 0.0);
}

TEST(GridContact, MatchesBruteForceOnCellBoundaries) {
    // A lattice with spacing 0.199 and radius 0.1: every neighbor pair overlaps by
    // 0.001, and many particles sit exactly on multiples of the spacing.
    Particles p;
    p.resize(6 * 6 * 6);
    std::size_t idx = 0;
    for (int a = 0; a < 6; ++a) {
        for (int b = 0; b < 6; ++b) {
            for (int c = 0; c < 6; ++c) {
                p.x[idx] = 0.199 * a - 0.5;
                p.y[idx] = 0.199 * b;
                p.z[idx] = -0.199 * c;
                p.radius[idx] = 0.1;
                p.mass[idx] = 1.0;
                ++idx;
            }
        }
    }
    shake(p, 0.2, 13);
    EXPECT_GT(expect_grid_matches(p, contact_params()), p.size() / 2);
}

TEST(GridContact, MatchesBruteForceWhenEveryCellSharesASlot) {
    // A table of 1 or 2 slots puts many different cells in the same slot; the loop must
    // skip particles from cells it didn't ask for, or it would count pairs twice.
    Particles p = cloud(300, 2.0, Vec3{0.0, 0.0, 0.0}, 0.05, 0.15, 14);
    const std::size_t sizes[] = {1, 2, 4};
    for (std::size_t table : sizes) {
        SCOPED_TRACE(table);
        EXPECT_GT(expect_grid_matches(p, contact_params(), table), 50u);
    }
}

TEST(GridContact, TinyAndDegenerateInputs) {
    SimParams params = contact_params();
    Particles none;
    compute_contact_grid(none, params); // nothing to do, and no crash

    Particles one = cloud(1, 1.0, Vec3{0, 0, 0}, 0.1, 0.1, 15);
    expect_grid_matches(one, params);

    Particles pair = cloud(2, 0.1, Vec3{0, 0, 0}, 0.1, 0.1, 16); // overlapping
    EXPECT_EQ(expect_grid_matches(pair, params), 2u);

    Particles same = cloud(2, 1.0, Vec3{0, 0, 0}, 0.1, 0.1, 17);
    same.x[1] = same.x[0];
    same.y[1] = same.y[0];
    same.z[1] = same.z[0];
    EXPECT_EQ(expect_grid_matches(same, params), 0u); // no direction: no force, no NaN

    Particles points = cloud(10, 1.0, Vec3{0, 0, 0}, 0.0, 0.0, 18); // radius 0
    points.heat_rate[3] = 7.0;
    compute_contact_grid(points, params);
    EXPECT_EQ(points.heat_rate[3], 0.0);
    EXPECT_EQ(expect_grid_matches(points, params), 0u);
}

TEST(GridContact, ShortCollisionMatchesBruteForceRun) {
    // 20 full steps of two small planets colliding: compute_accelerations (grid contact)
    // against the same leapfrog written out here with brute-force contact. Rounding
    // differences grow during a collision, so this compares after a few steps only,
    // relative to the planets' size and speed.
    Particles p = squeezed_planet(Vec3{-0.45, 0.0, 0.0}, 1);
    Particles q = squeezed_planet(Vec3{0.45, 0.02, -0.01}, 2);
    for (std::size_t i = 0; i < p.size(); ++i) {
        p.vx[i] = 1.0;
        q.vx[i] = -1.0;
    }
    append(p, q);
    SimParams params = contact_params();
    Particles grid = p, brute = p;

    compute_accelerations(grid, params, BACKEND_SERIAL);
    compute_gravity(brute, params, BACKEND_SERIAL);
    compute_contact_bruteforce(brute, params);
    const double dt = params.dt;
    for (int step = 0; step < 20; ++step) {
        leapfrog_step(grid, params, BACKEND_SERIAL);
        for (std::size_t i = 0; i < brute.size(); ++i) {
            brute.vx[i] += brute.ax[i] * 0.5 * dt;
            brute.vy[i] += brute.ay[i] * 0.5 * dt;
            brute.vz[i] += brute.az[i] * 0.5 * dt;
            brute.x[i] += brute.vx[i] * dt;
            brute.y[i] += brute.vy[i] * dt;
            brute.z[i] += brute.vz[i] * dt;
        }
        compute_gravity(brute, params, BACKEND_SERIAL);
        compute_contact_bruteforce(brute, params);
        for (std::size_t i = 0; i < brute.size(); ++i) {
            brute.heat[i] += brute.heat_rate[i] * dt;
            brute.vx[i] += brute.ax[i] * 0.5 * dt;
            brute.vy[i] += brute.ay[i] * 0.5 * dt;
            brute.vz[i] += brute.az[i] * 0.5 * dt;
        }
    }
    ASSERT_GT(total_heat(brute), 1e-4); // the planets really collided
    double worst_x = 0.0, worst_v = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        worst_x = std::fmax(worst_x, std::fabs(grid.x[i] - brute.x[i]));
        worst_x = std::fmax(worst_x, std::fabs(grid.y[i] - brute.y[i]));
        worst_x = std::fmax(worst_x, std::fabs(grid.z[i] - brute.z[i]));
        worst_v = std::fmax(worst_v, std::fabs(grid.vx[i] - brute.vx[i]));
        worst_v = std::fmax(worst_v, std::fabs(grid.vy[i] - brute.vy[i]));
        worst_v = std::fmax(worst_v, std::fabs(grid.vz[i] - brute.vz[i]));
    }
    EXPECT_LT(worst_x, 1e-10); // planet radius ~0.5
    EXPECT_LT(worst_v, 1e-9);  // speeds ~1
    EXPECT_NEAR(total_heat(grid), total_heat(brute), 1e-10 * total_heat(brute));
}
