#include "gravity.hpp"

#include <cstring>

#include "util.hpp"

const char *backend_name(Backend b) {
    switch (b) {
    case BACKEND_SERIAL:
        return "serial";
    case BACKEND_OMP:
        return "omp";
    case BACKEND_HIP:
        return "hip";
    }
    return "unknown";
}

bool backend_from_name(const char *name, Backend *out) {
    if (name == nullptr) {
        return false;
    }
    const Backend all[] = {BACKEND_SERIAL, BACKEND_OMP, BACKEND_HIP};
    for (Backend b : all) {
        if (std::strcmp(name, backend_name(b)) == 0) {
            *out = b;
            return true;
        }
    }
    return false;
}

void compute_gravity(Particles &p, const SimParams &params, Backend b) {
    switch (b) {
    case BACKEND_SERIAL:
        compute_gravity_serial(p, params);
        return;
    case BACKEND_OMP:
        compute_gravity_omp(p, params);
        return;
    case BACKEND_HIP:
        not_implemented("compute_gravity_hip"); // #24, #25
    }
    // Only reachable with a value outside the enum (e.g. a bad cast). Fail loudly
    // rather than silently keep the previous step's accelerations.
    not_implemented("compute_gravity: unknown backend");
}
