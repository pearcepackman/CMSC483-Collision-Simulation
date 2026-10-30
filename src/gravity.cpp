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

bool backend_available(Backend b) {
    switch (b) {
    case BACKEND_SERIAL:
    case BACKEND_OMP:
        return true;
    case BACKEND_HIP:
#ifdef SIM_HAVE_HIP
        return true;
#else
        return false;
#endif
    }
    return false;
}

namespace {

// True if `arch` (before any ':') is one of the comma-separated names in `list`.
bool arch_in_list(const char *arch, const char *list) {
    const std::size_t arch_len = std::strcspn(arch, ":");
    const char *item = list;
    while (*item != '\0') {
        const std::size_t item_len = std::strcspn(item, ",");
        if (item_len == arch_len && std::strncmp(item, arch, arch_len) == 0) {
            return true;
        }
        item += item_len;
        if (*item == ',') {
            ++item;
        }
    }
    return false;
}

} // namespace

int pick_gpu(int count, const int *compute_units, const char *const *archs,
             const char *built_archs) {
    int best = -1;
    for (int d = 0; d < count; ++d) {
        if (arch_in_list(archs[d], built_archs) &&
            (best < 0 || compute_units[d] > compute_units[best])) {
            best = d;
        }
    }
    return best;
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
#ifdef SIM_HAVE_HIP
        compute_gravity_hip(p, params);
        return;
#else
        // The CLI rejects --backend hip in these builds, so only a direct call gets here.
        not_implemented("hip backend not built (configure with: cmake --preset hip)");
#endif
    }
    // Only reachable with a value outside the enum (e.g. a bad cast). Fail loudly
    // rather than silently keep the previous step's accelerations.
    not_implemented("compute_gravity: unknown backend");
}
