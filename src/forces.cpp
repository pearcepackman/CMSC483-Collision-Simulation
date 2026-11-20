#include "forces.hpp"

#include "contact.hpp"

void compute_accelerations(Particles &p, const SimParams &params, Backend backend) {
#ifdef SIM_HAVE_HIP
    if (backend == BACKEND_HIP) {
        // One GPU kernel does gravity, contact and heat_rate (#66): the all-pairs loop
        // already visits every pair, so contact costs almost nothing there, and nothing
        // is left running on one CPU thread.
        compute_forces_hip(p, params);
        return;
    }
#endif
    compute_gravity(p, params, backend);
    if (params.contact_enabled) {
        compute_contact_grid(p, params); // also sets heat_rate
    } else {
        for (std::size_t i = 0; i < p.size(); ++i) {
            p.heat_rate[i] = 0.0; // only contact heats, so no stale rate from earlier
        }
    }
}
