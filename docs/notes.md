# Design Notes

A running log of the parallel-computing decisions in this project: what we chose, what we compared, and the evidence. This becomes the presentation. Add an entry whenever a decision is made or a benchmark settles a question. Include the numbers and the hardware.

**Entry format:** date · who · decision · why · evidence (benchmark, test, link to PR).

---

## Data layout

### Struct-of-Arrays particle storage (Oct 2026)
**Decision:** store particles as separate arrays per field (`x[]`, `y[]`, `z[]`, `vx[]`, …) in [src/particles.hpp](../src/particles.hpp), not an array of `{x, y, z, vx, ...}` structs.
**Why:** the force loop reads one field across many particles. With SoA those reads are contiguous, so the CPU can use SIMD and a GPU warp's loads coalesce into a few memory transactions. With an array of structs, each load pulls in fields the loop doesn't need.
**Evidence:** _TODO: benchmark AoS vs SoA in the OpenMP backend (milestone 3)._

---

## Module interfaces

### Fixed signatures before any physics (Oct 2026, #6)
**Decision:** every module (gravity, contact, integrator, init, io, config) got its header and stub bodies first, so both of us code against signatures that don't move.
- **One `SimParams` struct** ([src/params.hpp](../src/params.hpp)) is passed as `const SimParams &` to every physics function. Adding a setting never changes a signature, so one person's change can't break the other's code.
- **Acceleration contract:** gravity *overwrites* `ax/ay/az`; contact *adds* to them. [`compute_accelerations()`](../src/forces.hpp) is the one place that runs them in that order, and the integrator and planet relaxation both call it.
- **Backends are a plain enum** (`BACKEND_SERIAL / OMP / HIP`) chosen at run time, so every backend shares one interface and can be swapped by a flag.
- **Stubs abort** with `not implemented: <name>` instead of returning quietly, so an unfinished module can't make a test or run look correct.

**Why:** two people writing modules that call each other need a stable contract on day one; the overwrite/add rule is the kind of detail that otherwise causes silent double-counted forces.

### GPU tests live in their own binary (Oct 2026, #6 review)
**Decision:** CPU tests are every `tests/*.cpp`, built into one `sim_tests` binary. GPU tests go in `tests/gpu/*.cpp` (the CPU glob doesn't recurse, so it skips them) and build into a separate `sim_tests_gpu` binary only when `ENABLE_HIP` is on, with every test labelled `gpu`.
**Why:** CTest labels apply per binary, and CI has no GPU. One mixed binary would either fail to link without HIP or run GPU tests that `-LE gpu` can't filter out.

---

## Correctness strategy

### Compare backends with tolerances, not equality (Oct 2026)
**Decision:** backend-agreement tests compare a single force evaluation within a relative tolerance (~1e-12 for `double`), never `==`, and never after thousands of steps.
**Why:** floating-point addition isn't associative. Parallel reductions add the same numbers in a different order, so correct backends differ in the last few bits. And gravitational N-body systems are chaotic: those tiny differences grow until long runs diverge even when both are correct. Long runs are compared by conserved quantities (energy, momentum) instead.
**Evidence:** _TODO: plot of serial vs OpenMP divergence over time (milestone 3). It makes a good slide._

### Test at 1 and 4 threads in CI (Oct 2026)
**Decision:** CI runs the full suite with `OMP_NUM_THREADS=1` and `OMP_NUM_THREADS=4`.
**Why:** a data race can't happen with one thread, so a suite that passes at 1 thread and fails at 4 points straight at a race.

### Sanitizers in CI (Oct 2026)
**Decision:** AddressSanitizer + UBSan on every PR; ThreadSanitizer (with LLVM's Archer for OpenMP awareness) from milestone 3.
**Why:** memory bugs and data races often don't crash. They corrupt results silently or fail only sometimes. Sanitizers make them fail loudly and point at the line.

---

## Gravity

### The serial reference computes every pair twice (Oct 2026, #7)
**Decision:** [`compute_gravity_serial()`](../src/gravity_serial.cpp) loops "for each particle i, sum the pull of every other j", so each pair is computed twice (once from each side), with softening `eps` and G = 1.
**Why:** it's the simplest correct version, and it adds the numbers in the same order the OpenMP and GPU backends will (one particle per thread), so backend-match tests compare against an identically structured sum. Using Newton's 3rd law (each pair once, applied to both particles) halves the work but adds in a different order and needs atomics in parallel; #31 measures that trade-off.
**Evidence** ([tests/test_gravity.cpp](../tests/test_gravity.cpp)): 2- and 3-particle cases match hand-computed forces and potential to 1e-14 relative; 20 random particles match the same sum done in `long double` to 1e-13; the force equals minus the slope of the potential (central differences); and sum(m·a) over 100 random particles cancels to within 1e-12 of sum(m·|a|). During review, 19 deliberately broken copies of the gravity code (sign flips, swapped axes, single precision, missing pairs) were all caught by these tests.

### OpenMP gravity: first speedup (Oct 2026, #19)
**Decision:** [`compute_gravity_omp()`](../src/gravity_omp.cpp) is the serial loop with `#pragma omp parallel for schedule(static) default(none)` on the outer loop. Each thread owns a block of particles and is the only writer of their accelerations; positions and masses are read-only, so there are no races and no atomics. Each particle is summed by one thread in the same order as serial, so results match the reference.
**Evidence (correctness):** [tests/test_backends_match.cpp](../tests/test_backends_match.cpp) matches serial within 1e-12 relative at 1, 2, 4 and 12 threads, including N = 37 on 4 threads and N = 2 on 12. ThreadSanitizer with Archer (OpenMP-aware) reports 0 races; on a deliberately racy copy (shared running sum) it reports 9, so the check is real.
**Evidence (speed):** AMD Ryzen 5 7600X (6 cores / 12 threads), N = 10,000, `release` build, median of 9 interleaved rounds ([bench/bench_gravity.cpp](../bench/bench_gravity.cpp)). Reproduce with:
```bash
cmake --preset release && cmake --build --preset release
OMP_PROC_BIND=close OMP_PLACES=cores ./build/release/bench_gravity 10000
```

| OpenMP threads | s/eval | speedup vs 1-thread OpenMP | efficiency |
|---|---|---|---|
| 1 | 0.286 | 1.00x | 100% |
| 2 | 0.153 | 1.88x | 94% |
| 4 | 0.070 | 4.08x | 102% |
| 6 | 0.049 | 5.88x | 98% |
| 8 | 0.070 | 4.08x | 51% |
| 12 | 0.050 | 5.73x | 48% |

Across four runs, 6 threads gave 5.84–6.03x over 1-thread OpenMP. Serial took 0.28–0.33 s; it's the noisiest number, so the table compares against 1-thread OpenMP (the bench prints both).

**Why it looks like this:**
- **Near-perfect up to 6 threads:** each particle's work is independent and identical, there's no shared write, and N² math dominates memory traffic. One thread per physical core, no contention.
- **8 threads is *slower* than 6:** `schedule(static)` hands every thread an equal block, but with 8 threads on 6 cores, two cores run two threads each at about half speed. The whole loop waits for those slow blocks, so 8 threads finish no sooner than 4. This is the load-imbalance question #22 (static vs dynamic) investigates.
- **12 threads ≈ 6:** the second hardware thread per core (SMT) shares the same floating-point units, and this loop already keeps them busy, so it adds nothing.
- **Measurement lesson:** the first runs showed "123% efficiency." That was noise, not magic: with the `powersave` governor, CPU boost and other programs running, the clock speed drifts between measurements. The benchmark now interleaves configurations each round and reports speedup against 1-thread OpenMP as well as serial.
- **1-thread OpenMP is 3–16% faster than serial** (varies run to run) with identical math and bit-identical results. The compiled code shows why: serial's inner loop re-loads the `y`, `z` and `mass` array addresses from the `Particles` struct on every iteration (the compiler can't prove the writes to `ax/ay/az` don't change them), while the OpenMP version copies them into local pointers that stay in registers. The serial reference is deliberately left unoptimized (CLAUDE.md).

_Upcoming: pair-twice (no races) vs Newton's-3rd-law with atomics (half the work); OpenMP `schedule(static)` vs `dynamic`; HIP naive vs shared-memory tiled kernel; `float` vs `double` (consumer GPUs run `double` many times slower)._

---

## Time integration

### Leapfrog (kick-drift-kick) instead of Euler (Oct 2026, #8)
**Decision:** [`leapfrog_step()`](../src/integrator.cpp) advances the simulation as: half-step velocity kick, full-step position drift, recompute forces, half-step kick. Forces are computed once per step.
**Why:** leapfrog is *symplectic*: it preserves the geometry of orbital motion, so its energy error wobbles but never builds up. Euler updates positions and velocities from the old state only, so every orbit adds a little energy, and orbits spiral outward. Leapfrog costs the same one force evaluation per step.
**Evidence:** the same two-body circular orbit (G = 1, period 2π, 1,000 steps per orbit, 1,000 orbits) through both methods:

| Worst energy error so far | after 1 orbit | after 10 | after 100 | after 1,000 | separation at the end (starts at 1) |
|---|---|---|---|---|---|
| **Leapfrog** | 3.9e-10 | 3.9e-10 | 3.9e-10 | 3.9e-10 | 1.00 |
| **Euler** | 6.8e-2 | 3.3e-1 | 6.6e-1 | 8.4e-1 | 6.42 |

Relative energy error |E − E₀|/|E₀|, worst value seen up to that point (checked every step). The numbers are identical with gcc `-O2` and `-O3 -march=native`. Leapfrog's error wobbles within each orbit, peaking at 3.9e-10, and that peak never grows: it's a fixed property of the method at this step size, not accumulated error. Euler is off by 7% after one orbit and keeps getting worse: the orbit spirals outward to 6× its starting separation. [tests/test_two_body.cpp](../tests/test_two_body.cpp) enforces < 1e-5 with no drift; during review it caught 8 deliberately broken integrators, including one losing 1e-11 of its speed per step.

---

## Contact

_Upcoming: uniform grid (cell list) construction in parallel; load balance across uneven cells._

## Scaling results

_Upcoming: runtime vs N (log-log), strong scaling, efficiency, Amdahl's-law fit. Record CPU/GPU model and core count with every result._
