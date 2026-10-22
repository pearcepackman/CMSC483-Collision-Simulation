#include "forces.hpp"

#include "contact.hpp"

void compute_accelerations(Particles &p, const SimParams &params, Backend backend) {
    compute_gravity(p, params, backend);
    if (params.contact_enabled) {
        compute_contact_bruteforce(p, params);
    }
}
