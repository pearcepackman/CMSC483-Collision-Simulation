// CI canary for the tsan job: a deliberately racy OpenMP loop. ThreadSanitizer (with
// Archer) MUST report a data race here. If it doesn't, race checking is misconfigured
// and the tsan job fails instead of passing silently. Not part of the build or tests.
#include <cstdio>

int main() {
    long total = 0;
#pragma omp parallel for default(none) shared(total)
    for (int i = 0; i < 100000; ++i) {
        total += i; // every thread reads and writes `total` unsynchronized: a data race
    }
    std::printf("%ld\n", total);
    return 0;
}
