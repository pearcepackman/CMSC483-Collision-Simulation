#include "sim.hpp"

#include <cmath>
#include <cstring>
#include <random>

#include <omp.h>

#include "contact.hpp"
#include "diagnostics.hpp"
#include "forces.hpp"
#include "init.hpp"
#include "integrator.hpp"
#include "io.hpp"

std::size_t first_non_finite(const Particles &p) {
    for (std::size_t i = 0; i < p.size(); ++i) {
        if (!std::isfinite(p.x[i]) || !std::isfinite(p.y[i]) || !std::isfinite(p.z[i]) ||
            !std::isfinite(p.vx[i]) || !std::isfinite(p.vy[i]) || !std::isfinite(p.vz[i])) {
            return i;
        }
    }
    return p.size();
}

void write_energy_header(FILE *f) {
    std::fprintf(f, "step,t,kinetic,potential,contact,heat,total,px,py,pz\n");
}

void write_energy_row(FILE *f, int step, double t, const Particles &p, const SimParams &params) {
    double ke = kinetic_energy(p);
    double pe = potential_energy(p, params);
    double spring = contact_energy(p, params);
    double heat = total_heat(p);
    Vec3 mom = total_momentum(p);
    std::fprintf(f, "%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", step, t, ke, pe,
                 spring, heat, ke + pe + spring + heat, mom.x, mom.y, mom.z);
}

bool finish_log(FILE *f) {
    bool ok = std::ferror(f) == 0;
    if (std::fclose(f) != 0) { // fclose flushes, so a full disk often shows up here
        ok = false;
    }
    return ok;
}

std::size_t sim_init(Particles &p, const SimParams &params, Backend backend) {
    std::size_t bad = first_non_finite(p);
    if (bad == p.size()) {
        compute_accelerations(p, params, backend);
    }
    return bad;
}

std::size_t sim_step(Particles &p, const SimParams &params, Backend backend) {
    leapfrog_step(p, params, backend);
    return first_non_finite(p);
}

RunResult run_simulation(Particles &p, const SimParams &params, const RunOptions &opts,
                         FILE *energy_log) {
    RunResult result;
    int frame = 0; // frames are numbered 0, 1, 2, ... in output order
    // Frame + energy row for one output step. Energy is O(N^2) and serial, so only here.
    auto output = [&](int step) {
        if (opts.write_frames && !write_frame(p, opts.out_dir, frame)) {
            result.io_ok = false;
        }
        ++frame;
        if (energy_log != nullptr) {
            write_energy_row(energy_log, step, step * params.dt, p, params);
        }
    };

    result.bad_particle = sim_init(p, params, opts.backend);
    if (result.bad_particle != p.size()) { // bad input: don't even start
        result.finite = false;
        return result;
    }
    output(0);

    double t0 = omp_get_wtime();
    for (int step = 1; step <= opts.steps && result.io_ok; ++step) {
        result.bad_particle = sim_step(p, params, opts.backend);
        result.steps_done = step;
        if (result.bad_particle != p.size()) {
            result.finite = false;
            break;
        }
        // Also output the final step, so the log always covers the whole run.
        if (step % opts.output_every == 0 || step == opts.steps) {
            output(step);
        }
    }
    result.seconds = omp_get_wtime() - t0;
    return result;
}

void place_planet(Particles &dst, const Particles &planet, Vec3 center, Vec3 velocity) {
    const std::size_t base = dst.size();
    dst.resize(base + planet.size());
    for (std::size_t i = 0; i < planet.size(); ++i) {
        dst.x[base + i] = planet.x[i] + center.x;
        dst.y[base + i] = planet.y[i] + center.y;
        dst.z[base + i] = planet.z[i] + center.z;
        dst.vx[base + i] = planet.vx[i] + velocity.x;
        dst.vy[base + i] = planet.vy[i] + velocity.y;
        dst.vz[base + i] = planet.vz[i] + velocity.z;
        dst.mass[base + i] = planet.mass[i];
        dst.radius[base + i] = planet.radius[i];
        dst.body[base + i] = planet.body[i];
        dst.heat[base + i] = planet.heat[i]; // heat_rate isn't copied: sim_init recomputes it
    }
}

// Notes (not errors) when the contact damper is too strong for the lightest pair of
// particles: past c * dt / m_eff = 2 the explicit damper makes each contact's approach
// speed flip and grow every step instead of dying out, and the heat it books goes
// negative. The run would blow up, so say why first.
static void note_strong_damper(const Config &cfg) {
    if (!cfg.params.contact_enabled || cfg.planets.empty()) {
        return;
    }
    double m_min = 0.0;
    for (const PlanetSpec &spec : cfg.planets) {
        if (spec.n <= 0) {
            continue; // make_planet rejects it; nothing to divide by here
        }
        double m = spec.mass / static_cast<double>(spec.n);
        if (m_min == 0.0 || m < m_min) {
            m_min = m;
        }
    }
    if (!(m_min > 0.0)) {
        return;
    }
    const double alpha = cfg.params.contact_c * cfg.params.dt / (0.5 * m_min);
    if (alpha >= 2.0) {
        std::fprintf(stderr,
                     "note: contact_c = %g is too strong at dt = %g: two of the lightest "
                     "particles (mass %g) have c * dt / m_eff = %.3g, which must stay below 2, "
                     "or contacts bounce harder every step and blow up; lower contact_c or dt\n",
                     cfg.params.contact_c, cfg.params.dt, m_min, alpha);
    }
}

Particles build_planets(const Config &cfg, Backend backend) {
    note_strong_damper(cfg);
    Particles all;
    for (const PlanetSpec &spec : cfg.planets) {
        PlanetSpec at_rest = spec; // generate and relax at the origin, not moving
        at_rest.center = Vec3{0.0, 0.0, 0.0};
        at_rest.velocity = Vec3{0.0, 0.0, 0.0};
        Particles planet;
        make_planet(planet, at_rest);
        if (cfg.relax_max_steps > 0) {
            RelaxResult r = relax_planet(planet, cfg.params, backend, cfg.relax_drag,
                                         cfg.relax_ke_threshold, cfg.relax_max_steps);
            // Notes, not errors: the run goes on (sim_init then stops a NaN/Inf start).
            if (!r.finite) {
                std::fprintf(stderr,
                             "planet%d became NaN/Inf during relaxation, after %d steps "
                             "(dt too large?)\n",
                             spec.body, r.steps);
            } else if (!r.settled) {
                std::fprintf(stderr,
                             "note: planet%d didn't settle in %d relax steps: kinetic energy "
                             "%.3g must stay below relax_ke_threshold = %.3g for %d steps in "
                             "a row (one dynamical time; a smaller dt needs more steps); "
                             "raise relax_max_steps\n",
                             spec.body, r.steps, kinetic_energy(planet), cfg.relax_ke_threshold,
                             r.hold);
            }
            if (r.finite && cfg.relax_drag * r.t_dyn > MAX_RELAX_DRAG_T_DYN) {
                std::fprintf(stderr,
                             "note: relax_drag = %g is too strong for planet%d (drag x "
                             "dynamical time = %.0f, above %.0f): the planet only creeps, so "
                             "it can look settled when it isn't; lower relax_drag\n",
                             cfg.relax_drag, spec.body, cfg.relax_drag * r.t_dyn,
                             MAX_RELAX_DRAG_T_DYN);
            }
        }
        place_planet(all, planet, spec.center, spec.velocity);
    }
    return all;
}

bool setup_run(const CliOptions &cli, Backend backend, bool require_steps, Particles *p,
               SimParams *params, int *steps, int *output_every, std::string *error) {
    Config cfg;
    const bool from_config = !cli.config_path.empty();
    if (from_config) {
        if (!load_config(cli.config_path.c_str(), cfg)) {
            *error = "couldn't load config '" + cli.config_path + "'";
            return false;
        }
        *params = cfg.params;
        *steps = cfg.steps;
        *output_every = cfg.output_every;
    } else if (!build_scenario(cli.scenario.c_str(), cli.n, cli.seed, *p, *params)) {
        *error = "unknown scenario '" + cli.scenario + "' (orbit, figure8, cloud)";
        return false;
    }
    if (!apply_overrides(cli, from_config && require_steps, params, steps, output_every, error)) {
        return false;
    }
    // Planets last: relaxation steps them, so it must use the checked, overridden dt
    // (a --dt applies to it too).
    if (from_config) {
        cfg.params = *params;
        *p = build_planets(cfg, backend);
    }
    return true;
}

bool build_scenario(const char *name, int n, std::uint64_t seed, Particles &p, SimParams &params) {
    const double pi = 3.14159265358979323846;
    p = Particles();
    params = SimParams(); // gravity only, no contact
    if (std::strcmp(name, "orbit") == 0) {
        // Two masses of 0.5, separation 1, speed 0.5: circular with G = 1, period 2*pi.
        p.resize(2);
        p.mass[0] = p.mass[1] = 0.5;
        p.x[0] = -0.5;
        p.x[1] = 0.5;
        p.vy[0] = -0.5;
        p.vy[1] = 0.5;
        p.body[1] = 1;
        params.dt = 2.0 * pi / 1000.0;
        return true;
    }
    if (std::strcmp(name, "figure8") == 0) {
        // Chenciner & Montgomery (2000): three equal masses on one figure-eight, G = m = 1.
        p.resize(3);
        p.mass[0] = p.mass[1] = p.mass[2] = 1.0;
        p.x[0] = -0.97000436;
        p.y[0] = 0.24308753;
        p.x[1] = 0.97000436;
        p.y[1] = -0.24308753;
        p.vx[2] = -0.93240737;
        p.vy[2] = -0.86473146;
        p.vx[0] = p.vx[1] = 0.93240737 / 2.0;
        p.vy[0] = p.vy[1] = 0.86473146 / 2.0;
        p.body[1] = 1;
        p.body[2] = 2;
        params.dt = 6.32591398 / 2000.0; // 2,000 steps per period
        return true;
    }
    if (std::strcmp(name, "cloud") == 0 && n >= 1) {
        // n equal masses (total 1) at rest, uniform in a unit sphere: collapses under
        // its own gravity. For benchmarks at any N.
        p.resize(static_cast<std::size_t>(n));
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        for (std::size_t i = 0; i < p.size(); ++i) {
            double x, y, z;
            do { // rejection sampling: keep points inside the sphere
                x = u(rng);
                y = u(rng);
                z = u(rng);
            } while (x * x + y * y + z * z > 1.0);
            p.x[i] = x;
            p.y[i] = y;
            p.z[i] = z;
            p.mass[i] = 1.0 / n;
        }
        params.softening = 0.05;
        params.dt = 0.001;
        return true;
    }
    return false;
}
