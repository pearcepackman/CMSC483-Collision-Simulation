#include "forces.hpp"

#include "contact.hpp"

void compute_accelerations(Particles &p, const SimParams &params, Backend backend) {
    compute_gravity(p, params, backend);
    if (params.contact_enabled) {
        compute_contact_bruteforce(p, params); // also sets heat_rate
    } else {
        for (std::size_t i = 0; i < p.size(); ++i) {
            p.heat_rate[i] = 0.0; // only contact heats, so no stale rate from earlier
        }
    }
}
