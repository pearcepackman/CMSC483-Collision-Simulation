#pragma once

#include <cstddef>
#include <vector>

struct Vec3 {
    double x, y, z;
};

// Struct-of-Arrays particle storage: one array per component, so a loop over
// a single field reads contiguous memory (SIMD on CPU, coalesced loads on GPU).
// Keep this layout; every backend depends on it.
struct Particles {
    std::vector<double> x, y, z;    // position
    std::vector<double> vx, vy, vz; // velocity
    std::vector<double> ax, ay, az; // acceleration from the last force evaluation
    std::vector<double> mass;
    std::vector<double> radius;
    std::vector<int> body; // which planet the particle started in (for coloring)

    std::size_t size() const { return x.size(); }

    // New particles are zero-initialized.
    void resize(std::size_t n) {
        for (std::vector<double> *field :
             {&x, &y, &z, &vx, &vy, &vz, &ax, &ay, &az, &mass, &radius}) {
            field->resize(n, 0.0);
        }
        body.resize(n, 0);
    }
};
