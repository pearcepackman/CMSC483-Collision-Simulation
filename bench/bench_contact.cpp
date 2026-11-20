// Times one contact evaluation: all pairs (compute_contact_bruteforce) against the
// hashed grid (compute_contact_grid), on planets of N = 1000, 2000, 4000, ... up to
// N_max. Both run on one thread. Usage: bench_contact [N_max=32000] [rounds=5]
// Build with the `release` preset for real numbers:
//   ./build/release/bench_contact 32000
// Each planet has the head-on planets' packing (particle radius 0.065, radius scaled so
// the density stays the same), squeezed by 8% so every particle overlaps its neighbors
// (larger planets come out of make_planet a little less tightly packed), with random
// velocities so the dampers work too. Rounds interleave the two versions, so CPU clock
// changes hit both alike. The grid time includes building the grid.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <omp.h>

#include "contact.hpp"
#include "init.hpp"

namespace {

bool parse_arg(int argc, char **argv, int index, long min_value, long fallback, long *out) {
    if (argc <= index) {
        *out = fallback;
        return true;
    }
    char *end = nullptr;
    long value = std::strtol(argv[index], &end, 10);
    if (end == argv[index] || *end != '\0' || value < min_value) {
        std::fprintf(stderr, "bad argument '%s' (need an integer >= %ld)\n", argv[index],
                     min_value);
        return false;
    }
    *out = value;
    return true;
}

Particles packed_planet(int n) {
    PlanetSpec s;
    s.n = n;
    s.radius = std::cbrt(n / 2000.0); // 2,000 particles fill radius 1 (configs/head_on.cfg)
    s.particle_radius = 0.065;
    s.mass = 2.0 * n / 2000.0;
    s.seed = 1;
    Particles p;
    make_planet(p, s);
    std::mt19937_64 rng(2);
    std::uniform_real_distribution<double> v(-0.1, 0.1);
    for (std::size_t i = 0; i < p.size(); ++i) {
        p.x[i] *= 0.92;
        p.y[i] *= 0.92;
        p.z[i] *= 0.92;
        p.vx[i] = v(rng);
        p.vy[i] = v(rng);
        p.vz[i] = v(rng);
    }
    return p;
}

double time_once(Particles &p, const SimParams &params, bool grid) {
    for (std::size_t i = 0; i < p.size(); ++i) {
        p.ax[i] = p.ay[i] = p.az[i] = 0.0;
    }
    double t0 = omp_get_wtime();
    if (grid) {
        compute_contact_grid(p, params);
    } else {
        compute_contact_bruteforce(p, params);
    }
    return omp_get_wtime() - t0;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

void print_cpu_model() {
    FILE *f = std::fopen("/proc/cpuinfo", "r");
    char line[512];
    while (f != nullptr && std::fgets(line, sizeof line, f) != nullptr) {
        const char *colon = std::strchr(line, ':');
        if (std::strncmp(line, "model name", 10) == 0 && colon != nullptr) {
            std::printf("CPU:%s", colon + 1);
            break;
        }
    }
    if (f != nullptr) {
        std::fclose(f);
    }
}

} // namespace

int main(int argc, char **argv) {
    long n_arg = 0, rounds_arg = 0;
    if (!parse_arg(argc, argv, 1, 1000, 32000, &n_arg) ||
        !parse_arg(argc, argv, 2, 1, 5, &rounds_arg)) {
        std::fprintf(stderr, "usage: bench_contact [N_max=32000 (>= 1000)] [rounds=5]\n");
        return 1;
    }
    const int rounds = static_cast<int>(rounds_arg);
    SimParams params;
    params.dt = 0.002;
    params.contact_enabled = true;
    params.contact_k = 20.0;
    params.contact_c = 0.05;

    print_cpu_model();
    std::printf("one thread | median of %d interleaved rounds | grid time includes the build\n\n",
                rounds);
    std::printf("%8s %12s %12s %9s %16s %11s %10s\n", "N", "all pairs s", "grid s", "speedup",
                "grid us/particle", "max diff", "touching");
    for (long n = 1000; n <= n_arg; n *= 2) {
        Particles brute = packed_planet(static_cast<int>(n));
        Particles grid = brute;
        time_once(grid, params, true); // warm-up (the grid's arrays get allocated here)
        std::vector<double> tb, tg;
        for (int r = 0; r < rounds; ++r) {
            tb.push_back(time_once(brute, params, false));
            tg.push_back(time_once(grid, params, true));
        }
        // The two must agree (up to summation order), or the timing means nothing: every
        // force component and the heat rate, each relative to its largest value.
        double f_max = 0.0, f_diff = 0.0, q_max = 0.0, q_diff = 0.0;
        std::size_t touching = 0; // particles with a contact force: the timing must have work
        for (std::size_t i = 0; i < brute.size(); ++i) {
            touching += (brute.ax[i] != 0.0 || brute.ay[i] != 0.0 || brute.az[i] != 0.0) ? 1 : 0;
            f_max = std::fmax(f_max, std::fabs(brute.ax[i]));
            f_max = std::fmax(f_max, std::fabs(brute.ay[i]));
            f_max = std::fmax(f_max, std::fabs(brute.az[i]));
            f_diff = std::fmax(f_diff, std::fabs(brute.ax[i] - grid.ax[i]));
            f_diff = std::fmax(f_diff, std::fabs(brute.ay[i] - grid.ay[i]));
            f_diff = std::fmax(f_diff, std::fabs(brute.az[i] - grid.az[i]));
            q_max = std::fmax(q_max, std::fabs(brute.heat_rate[i]));
            q_diff = std::fmax(q_diff, std::fabs(brute.heat_rate[i] - grid.heat_rate[i]));
        }
        double b = median(tb), g = median(tg);
        double rel = f_max > 0.0 ? f_diff / f_max : 0.0;
        double rel_q = q_max > 0.0 ? q_diff / q_max : 0.0;
        std::printf("%8ld %12.5f %12.6f %8.0fx %16.3f %11.1e %10zu\n", n, b, g, b / g,
                    1e6 * g / static_cast<double>(n), std::fmax(rel, rel_q), touching);
        if (!(rel <= 1e-12) || !(rel_q <= 1e-12) || !(q_max > 0.0) || touching < brute.size() / 2) {
            std::fprintf(stderr,
                         "grid and all-pairs contact disagree (or too few contacts) at N = %ld\n",
                         n);
            return 1;
        }
    }
    return 0;
}
