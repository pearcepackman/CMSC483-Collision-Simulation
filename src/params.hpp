#pragma once

// Physics and run settings shared by every module. Functions take
// `const SimParams &` so adding a field never changes their signatures.
// Units: simulation units with G = 1. Defaults mean "unset", not physical values:
// a run always sets them from its config or test.
struct SimParams {
    double dt = 0.0;              // timestep
    double softening = 0.0;       // gravity softening length (epsilon)
    bool contact_enabled = false; // false = gravity only (e.g. the two-body orbit test)
    double contact_k = 0.0;       // contact spring stiffness
    double contact_c = 0.0;       // contact damping coefficient
};
