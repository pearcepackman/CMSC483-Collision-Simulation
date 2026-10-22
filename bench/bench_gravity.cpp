// Times one gravity evaluation, serial vs OpenMP at several thread counts.
// Usage: bench_gravity [N=10000] [rounds=9]   (N >= 2, rounds >= 1)
// Build with the `release` preset for real numbers:
//   OMP_PROC_BIND=close OMP_PLACES=cores ./build/release/bench_gravity 10000
// Each round times every configuration once, in turn, so CPU clock changes (boost,
// power saving, other programs) hit all of them alike instead of biasing one.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <omp.h>

#include "gravity.hpp"

namespace {

// Parses argv[index] as an integer >= min_value, or returns fallback if it's absent.
// Returns false (and prints why) on junk, negatives or values below min_value.
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

Particles random_system(std::size_t n) {
    Particles p;
    p.resize(n);
    std::mt19937_64 rng(1); // fixed seed: every run times the same system
    std::uniform_real_distribution<double> pos(-1.0, 1.0), mass(0.5, 2.0);
    for (std::size_t i = 0; i < n; ++i) {
        p.x[i] = pos(rng);
        p.y[i] = pos(rng);
        p.z[i] = pos(rng);
        p.mass[i] = mass(rng);
    }
    return p;
}

struct Config {
    Backend backend;
    int threads;
    std::vector<double> times;
};

double time_once(Particles &p, const SimParams &params, const Config &c) {
    omp_set_num_threads(c.threads);
    double t0 = omp_get_wtime();
    compute_gravity(p, params, c.backend);
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
        if (std::strncmp(line, "model name", 10) == 0) {
            std::printf("CPU:%s", std::strchr(line, ':') + 1);
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
    if (!parse_arg(argc, argv, 1, 2, 10000, &n_arg) ||
        !parse_arg(argc, argv, 2, 1, 9, &rounds_arg)) {
        std::fprintf(stderr, "usage: bench_gravity [N=10000] [rounds=9]\n");
        return 1;
    }
    const std::size_t n = static_cast<std::size_t>(n_arg);
    const int rounds = static_cast<int>(rounds_arg);
    SimParams params;
    params.softening = 0.01;
    Particles p = random_system(n);

    std::vector<Config> configs = {{BACKEND_SERIAL, 1, {}}};
    const int counts[] = {1, 2, 4, 6, 8, 12, 16, 24, 32};
    for (int threads : counts) {
        if (threads <= omp_get_num_procs()) {
            configs.push_back({BACKEND_OMP, threads, {}});
        }
    }
    for (const Config &c : configs) {
        time_once(p, params, c); // warm-up: thread pool, caches, CPU clock
    }
    for (int r = 0; r < rounds; ++r) {
        for (Config &c : configs) {
            c.times.push_back(time_once(p, params, c));
        }
    }

    print_cpu_model();
    std::printf("logical CPUs: %d | N = %zu | median of %d interleaved rounds\n\n",
                omp_get_num_procs(), n, rounds);
    std::printf("%-8s %8s %10s %12s %16s %11s\n", "backend", "threads", "s/eval", "vs serial",
                "vs omp 1-thread", "efficiency");
    const double serial = median(configs[0].times);
    const double omp1 = median(configs[1].times);
    for (const Config &c : configs) {
        double t = median(c.times);
        if (c.backend == BACKEND_SERIAL) { // efficiency only means something for OpenMP
            std::printf("%-8s %8d %10.5f %11.2fx %15.2fx %11s\n", "serial", c.threads, t,
                        serial / t, omp1 / t, "-");
        } else {
            std::printf("%-8s %8d %10.5f %11.2fx %15.2fx %10.0f%%\n", "omp", c.threads, t,
                        serial / t, omp1 / t, 100.0 * omp1 / t / c.threads);
        }
    }
    return 0;
}
