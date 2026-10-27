#include "sim.hpp"

#include <cmath>
#include <cstring>
#include <random>

#include <omp.h>

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
    std::fprintf(f, "step,t,kinetic,potential,total,px,py,pz\n");
}

void write_energy_row(FILE *f, int step, double t, const Particles &p, const SimParams &params) {
    double ke = kinetic_energy(p);
    double pe = potential_energy(p, params);
    Vec3 mom = total_momentum(p);
    std::fprintf(f, "%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", step, t, ke, pe, ke + pe,
                 mom.x, mom.y, mom.z);
}

bool finish_log(FILE *f) {
    bool ok = std::ferror(f) == 0;
    if (std::fclose(f) != 0) { // fclose flushes, so a full disk often shows up here
        ok = false;
    }
    return ok;
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

    result.bad_particle = first_non_finite(p);
    if (result.bad_particle != p.size()) { // bad input: don't even start
        result.finite = false;
        return result;
    }
    compute_accelerations(p, params, opts.backend); // leapfrog needs a(t = 0)
    output(0);

    double t0 = omp_get_wtime();
    for (int step = 1; step <= opts.steps && result.io_ok; ++step) {
        leapfrog_step(p, params, opts.backend);
        result.steps_done = step;
        result.bad_particle = first_non_finite(p);
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
    }
}

Particles build_planets(const Config &cfg, Backend backend) {
    Particles all;
    for (const PlanetSpec &spec : cfg.planets) {
        PlanetSpec at_rest = spec; // generate and relax at the origin, not moving
        at_rest.center = Vec3{0.0, 0.0, 0.0};
        at_rest.velocity = Vec3{0.0, 0.0, 0.0};
        Particles planet;
        make_planet(planet, at_rest);
        if (cfg.relax_max_steps > 0) {
            relax_planet(planet, cfg.params, backend, cfg.relax_ke_threshold, cfg.relax_max_steps);
        }
        place_planet(all, planet, spec.center, spec.velocity);
    }
    return all;
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
