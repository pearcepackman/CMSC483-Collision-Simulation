#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "look.hpp"

// The viewer's colors and sprite sizes (#47).

namespace {

const LookScales kScales{2.0, 2.0}; // speed scale 2, heat scale 0.5 * 2^2

Rgb heat_color(double heat_per_mass, double mass = 1.0) {
    return particle_color(COLOR_HEAT, 0, MATERIAL_ROCK, mass, 0.0, heat_per_mass * mass, kScales);
}

bool same(Rgb a, Rgb b) {
    return a.r == b.r && a.g == b.g && a.b == b.b;
}

} // namespace

TEST(ViewerLook, ModeNamesRoundTripAndCycle) {
    ColorMode m = COLOR_HEAT;
    for (int k = 0; k < COLOR_MODE_COUNT; ++k) {
        ColorMode parsed = COLOR_PLAIN;
        ASSERT_TRUE(color_mode_from_name(color_mode_name(m), &parsed)) << color_mode_name(m);
        EXPECT_EQ(parsed, m);
        m = next_color_mode(m);
    }
    EXPECT_EQ(m, COLOR_HEAT); // C goes through all five and wraps
    EXPECT_STREQ(color_mode_name(COLOR_HEAT), "heat");
    ColorMode untouched = COLOR_SPEED;
    EXPECT_FALSE(color_mode_from_name("Heat", &untouched)); // exact names only
    EXPECT_FALSE(color_mode_from_name("", &untouched));
    EXPECT_FALSE(color_mode_from_name(nullptr, &untouched));
    EXPECT_EQ(untouched, COLOR_SPEED);
}

TEST(ViewerLook, ScaleIsImpactOrEscapeSpeed) {
    // Two single-particle bodies: masses 2 and 1, radii 0.5 each, so the mutual escape
    // speed is sqrt(2 * 3 / 1) = sqrt(6) = 2.449.
    Particles p;
    p.resize(2);
    p.mass[0] = 2.0;
    p.mass[1] = 1.0;
    p.radius[0] = p.radius[1] = 0.5;
    p.x[1] = 5.0;
    p.body[1] = 1;
    LookScales s = look_scales(p);
    EXPECT_NEAR(s.speed, std::sqrt(6.0), 1e-12);
    EXPECT_NEAR(s.heat_per_mass, 0.5 * 6.0, 1e-12);

    // Coming in faster than that: the impact speed (relative speed 4) wins.
    p.vx[0] = 1.0;
    p.vx[1] = -3.0;
    EXPECT_NEAR(look_scales(p).speed, 4.0, 1e-12);

    // One body (the cloud scenario): its own surface escape speed sqrt(2 M / R).
    Particles one;
    one.resize(1);
    one.mass[0] = 2.0;
    one.radius[0] = 0.25;
    EXPECT_NEAR(look_scales(one).speed, std::sqrt(2.0 * 2.0 / 0.25), 1e-12);
}

TEST(ViewerLook, ScaleFallsBackWithoutMassOrSize) {
    Particles none;
    EXPECT_EQ(look_scales(none).speed, 1.0);
    Particles point; // one particle of radius 0: no size to escape from
    point.resize(1);
    point.mass[0] = 1.0;
    EXPECT_EQ(look_scales(point).speed, 1.0);
    Particles massless;
    massless.resize(3);
    EXPECT_EQ(look_scales(massless).speed, 1.0);
}

TEST(ViewerLook, HeatGetsBrighterAndEndsWhite) {
    // Cold: the dim material color. Then always brighter as heat rises, up to white at
    // the scale (about the impact's kinetic energy per unit mass) and beyond.
    Rgb cold = heat_color(0.0);
    Rgb rock = particle_color(COLOR_MATERIAL, 0, MATERIAL_ROCK, 1.0, 0.0, 0.0, kScales);
    EXPECT_NEAR(cold.r, 0.3f * rock.r, 1e-6);
    EXPECT_LT(luma(cold), 0.5f * luma(rock));
    float last = luma(cold);
    for (double x = 0.001; x <= 5.0; x *= 1.5) {
        float l = luma(heat_color(x * kScales.heat_per_mass));
        EXPECT_GE(l, last) << "heat " << x << " x the scale";
        last = l;
    }
    Rgb hot = heat_color(1.0 * kScales.heat_per_mass);
    EXPECT_FLOAT_EQ(hot.r, 1.0f);
    EXPECT_FLOAT_EQ(hot.g, 1.0f);
    EXPECT_FLOAT_EQ(hot.b, 1.0f);
    EXPECT_TRUE(same(heat_color(100.0 * kScales.heat_per_mass), hot)); // clamps
    // Below 0.003x the scale there is no glow yet; at 0.1x it is clearly glowing.
    EXPECT_TRUE(same(heat_color(0.002 * kScales.heat_per_mass), cold));
    EXPECT_GT(luma(heat_color(0.1 * kScales.heat_per_mass)), 2.0f * luma(cold));
    EXPECT_LT(luma(heat_color(0.1 * kScales.heat_per_mass)), luma(hot));
}

TEST(ViewerLook, HeatIsPerUnitMass) {
    // An iron particle 2.5x heavier with 2.5x the heat is exactly as hot.
    Rgb light = particle_color(COLOR_HEAT, 0, MATERIAL_ROCK, 1.0, 0.0, 0.4, kScales);
    Rgb heavy_same_per_mass = particle_color(COLOR_HEAT, 0, MATERIAL_ROCK, 2.5, 0.0, 1.0, kScales);
    EXPECT_TRUE(same(light, heavy_same_per_mass));
    Rgb heavy_same_total = particle_color(COLOR_HEAT, 0, MATERIAL_ROCK, 2.5, 0.0, 0.4, kScales);
    EXPECT_LT(luma(heavy_same_total), luma(light));
    // Massless (shouldn't happen) is just cold, not NaN.
    Rgb massless = particle_color(COLOR_HEAT, 0, MATERIAL_ROCK, 0.0, 0.0, 1.0, kScales);
    EXPECT_TRUE(same(massless, heat_color(0.0)));
}

TEST(ViewerLook, MaterialAndPlanetColorsAreDistinct) {
    Rgb c[4];
    int k = 0;
    for (int body : {0, 1}) {
        for (int material : {MATERIAL_ROCK, MATERIAL_IRON}) {
            c[k++] = particle_color(COLOR_MATERIAL, body, material, 1.0, 0.0, 0.0, kScales);
        }
    }
    for (int a = 0; a < 4; ++a) {
        for (int b = a + 1; b < 4; ++b) {
            EXPECT_FALSE(same(c[a], c[b])) << a << " vs " << b;
        }
    }
    EXPECT_FALSE(same(particle_color(COLOR_PLANET, 0, 0, 1, 0, 0, kScales),
                      particle_color(COLOR_PLANET, 1, 0, 1, 0, 0, kScales)));
    // Bodies alternate: planet 2 looks like planet 0. Material doesn't change planet mode.
    EXPECT_TRUE(same(particle_color(COLOR_PLANET, 2, MATERIAL_IRON, 1, 0, 0, kScales),
                     particle_color(COLOR_PLANET, 0, MATERIAL_ROCK, 1, 0, 0, kScales)));
}

TEST(ViewerLook, SpeedRampClampsAndRises) {
    Rgb still = particle_color(COLOR_SPEED, 0, 0, 1.0, 0.0, 0.0, kScales);
    Rgb at_scale = particle_color(COLOR_SPEED, 0, 0, 1.0, kScales.speed, 0.0, kScales);
    Rgb faster = particle_color(COLOR_SPEED, 0, 0, 1.0, 10.0 * kScales.speed, 0.0, kScales);
    EXPECT_GT(luma(still), 0.0f); // slow particles stay visible
    EXPECT_GT(luma(at_scale), luma(still));
    EXPECT_TRUE(same(faster, at_scale)); // clamps at the top
    EXPECT_TRUE(same(particle_color(COLOR_SPEED, 0, 0, 1.0, -1.0, 0.0, kScales), still));
}

TEST(ViewerLook, SpriteIsRealSizeUntilTooSmall) {
    // Near: glow_factor x radius. 0.041 x 3 = 0.123 world units at depth 2 with 800 px per
    // unit is 49 px, well above the 1.5 px minimum.
    EXPECT_FLOAT_EQ(sprite_half_size(0.041, 3.0, 2.0, 800.0, 1.5), 0.123f);
    // Far: the same particle at depth 200 would be 0.49 px, so it is enlarged to exactly
    // 1.5 px: 1.5 * 200 / 800 = 0.375 world units.
    EXPECT_FLOAT_EQ(sprite_half_size(0.041, 3.0, 200.0, 800.0, 1.5), 0.375f);
    // Behind the camera or no projection: no minimum.
    EXPECT_FLOAT_EQ(sprite_half_size(0.041, 3.0, -1.0, 800.0, 1.5), 0.123f);
}

TEST(ViewerLook, HeatGlowFollowsTheRamp) {
    EXPECT_EQ(heat_glow(1.0, 0.0, kScales), 0.0);
    EXPECT_EQ(heat_glow(1.0, 0.002 * kScales.heat_per_mass, kScales), 0.0); // below 0.003x
    EXPECT_NEAR(heat_glow(1.0, 1.0 * kScales.heat_per_mass, kScales), 1.0, 1e-12);
    EXPECT_EQ(heat_glow(1.0, 50.0 * kScales.heat_per_mass, kScales), 1.0); // clamps
    // Halfway on the log ramp: 10^(-1.25) of the scale.
    EXPECT_NEAR(heat_glow(1.0, std::pow(10.0, -1.25) * kScales.heat_per_mass, kScales), 0.5, 1e-12);
    // Per unit mass, and no NaN from a massless particle.
    EXPECT_EQ(heat_glow(2.5, 2.5 * 0.1, kScales), heat_glow(1.0, 0.1, kScales));
    EXPECT_EQ(heat_glow(0.0, 1.0, kScales), 0.0);
}

TEST(ViewerLook, AbsurdBodyIdsAreIgnored) {
    // Two real bodies plus one particle with an id far past any planet count: the
    // scale is the two real bodies' (and nothing allocates billions of entries).
    Particles p;
    p.resize(3);
    p.mass[0] = 2.0;
    p.mass[1] = 1.0;
    p.mass[2] = 5.0;
    p.radius[0] = p.radius[1] = p.radius[2] = 0.5;
    p.x[1] = 5.0;
    p.body[1] = 1;
    p.body[2] = 1000000000;
    EXPECT_NEAR(look_scales(p).speed, std::sqrt(6.0), 1e-12);
}
