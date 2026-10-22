#include <gtest/gtest.h>

#include "diagnostics.hpp"
#include "particles.hpp"

TEST(Particles, ResizeKeepsAllFieldsTheSameLength) {
    Particles p;
    p.resize(5);
    EXPECT_EQ(p.size(), 5u);
    for (const std::vector<double> *field :
         {&p.x, &p.y, &p.z, &p.vx, &p.vy, &p.vz, &p.ax, &p.ay, &p.az, &p.mass, &p.radius}) {
        ASSERT_EQ(field->size(), 5u);
        for (double v : *field) {
            EXPECT_EQ(v, 0.0);
        }
    }
    ASSERT_EQ(p.body.size(), 5u);
    for (int b : p.body) {
        EXPECT_EQ(b, 0);
    }
}

TEST(Diagnostics, EmptySystemIsAllZero) {
    Particles p;
    EXPECT_EQ(total_mass(p), 0.0);
    EXPECT_EQ(kinetic_energy(p), 0.0);
    Vec3 P = total_momentum(p);
    EXPECT_EQ(P.x, 0.0);
    EXPECT_EQ(P.y, 0.0);
    EXPECT_EQ(P.z, 0.0);
}

TEST(Diagnostics, OpposingParticlesHaveZeroMomentum) {
    Particles p;
    p.resize(2);
    p.mass[0] = 2.0;
    p.vx[0] = 3.0;
    p.vy[0] = -1.0;
    p.mass[1] = 1.0;
    p.vx[1] = -6.0;
    p.vy[1] = 2.0;

    // Momentum sums cancel to ~0, so compare against the size of the terms being
    // summed (|m v| ~ 6 here), never exact zero: rounding makes real sums ~1e-15.
    const double scale = 6.0;
    Vec3 P = total_momentum(p);
    EXPECT_NEAR(P.x, 0.0, 1e-12 * scale);
    EXPECT_NEAR(P.y, 0.0, 1e-12 * scale);
    EXPECT_NEAR(P.z, 0.0, 1e-12 * scale);
    EXPECT_DOUBLE_EQ(total_mass(p), 3.0);
}

TEST(Diagnostics, KineticEnergyIsHalfMVSquared) {
    Particles p;
    p.resize(1);
    p.mass[0] = 4.0;
    p.vx[0] = 1.0;
    p.vy[0] = 2.0;
    p.vz[0] = 2.0; // |v|^2 = 9

    EXPECT_DOUBLE_EQ(kinetic_energy(p), 0.5 * 4.0 * 9.0);
}
