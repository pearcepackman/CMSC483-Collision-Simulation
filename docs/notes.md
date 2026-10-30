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

### HIP gravity: naive double kernel (Oct 2026, #24, #25)
**Decision:** [`compute_gravity_hip()`](../src/gravity_hip.hip) runs one GPU thread per particle (256 per block), each looping over all others in global memory with the same formula and summation order as serial, in `double`. The GPU copies of the arrays (Struct-of-Arrays) are kept between calls and grow only when N does; positions and masses go up and accelerations come back on **every call**. HIP is optional: `cmake --preset hip` (`ENABLE_HIP`, `gfx1200`); other builds report `--backend hip` as not built.
**Device choice:** ROCm lists two devices on the dev PC: the RX 9060 XT (`gfx1200`) and the Ryzen 7600X's integrated GPU (`gfx1036`). `hip_init()` picks, with the pure function `pick_gpu()`, the visible device with the most compute units **among the architectures the kernel was compiled for** (`CMAKE_HIP_ARCHITECTURES`, passed in as `SIM_HIP_ARCHS`), so the discrete card wins regardless of ordering and `HIP_VISIBLE_DEVICES` overrides. The `hip` preset compiles for **both** chips: review found that with ROCm 7.2, if *any* visible GPU lacks code for a kernel, the first launch aborts inside the HIP runtime (exit 134, no error our checks can catch), even when the chosen GPU has code. A GPU outside the list is never chosen; if it's the only one, or there's no GPU at all, `planet_sim` exits 1 with a message and GPU tests skip. `pick_gpu` is unit-tested on the CPU ([tests/test_gpu_select.cpp](../tests/test_gpu_select.cpp)), since on this PC "first device" and "best device" are the same card and GPU tests alone can't tell them apart. `hipSetDevice` is per host thread, so `compute_gravity_hip` sets it on every call. In the `--bench` line, hip runs report `threads=0` (no CPU threads compute gravity).
**Practical size:** one kernel launch per step, measured 0.89 s/step at N = 65,536 and 7.4 s at 200,000 (O(N²)). The CLI's 10M cap would mean hours per launch on the GPU that also drives the display, so keep `--backend hip` to roughly N ≤ 1e5 until #26 and #32 speed it up.
**Why double first:** it matches the serial reference to 1e-12, so correctness is checked strictly before any precision trade-off. `float` is #32.
**Evidence (correctness):** [tests/gpu/test_backends_match_gpu.cpp](../tests/gpu/test_backends_match_gpu.cpp) (label `gpu`): hand-computed pair to 1e-14; one particle with eps = 0 gets exactly zero (self-pull skipped, else NaN); N = 2, 255, 256, 257, 1000 and 4096 match serial within 1e-12 relative; cached buffers survive grow/shrink/free; 10 leapfrog steps track serial positions to 1e-10. Eight deliberately broken kernels (no self-skip, blocks rounded down, float math, swapped axis, stale mass after reuse, eps not squared, missing copy-back, buffers never regrown) were all caught: six by tests, one by the HIP error check, one by `-Werror`.
**Evidence (speed):** whole `planet_sim --bench` step (gravity + leapfrog, including the copies), `cloud` scenario, OpenMP from the `release` build with 6 threads pinned to cores:

| N | OpenMP 6 threads (s/step) | HIP naive double (s/step) | HIP vs OpenMP |
|---|---|---|---|
| 2,000 | 0.0018 | 0.0030 | 0.6x (slower) |
| 10,000 | 0.044 | 0.034 | 1.3x |
| 30,000 | 0.41 | 0.20 | 2.1x |

**Why it looks like this:**
- **Double precision is slow on this card.** Consumer RDNA GPUs run `double` at a small fraction of `float` speed. A scratch probe of the kernel alone at N = 10,000 took 25 ms in `double` vs 2.0 ms in `float`, so in `double` the GPU only modestly beats 6 CPU cores. #32 measures the `float` version properly.
- **Small N loses.** At 2,000 particles the fixed costs (kernel launch, seven host↔device copies per step) outweigh the math; the GPU only pays off as N² grows. #30 removes the per-step copies.
- **No data reuse yet.** Every thread reads every other particle from global memory; #26 (shared-memory tiles) loads each block of particles once and shares it.

**Build gotcha (Arch ROCm 7.2):** the ROCm clang guesses its install root as `/opt/rocm/lib` and can't find `hip/hip_runtime.h`; the `hip` preset passes `--rocm-path=/opt/rocm` and pins the compiler and ROCm root so it works from any shell.

_Upcoming: pair-twice (no races) vs Newton's-3rd-law with atomics (half the work); OpenMP `schedule(static)` vs `dynamic`; HIP naive vs shared-memory tiled kernel; `float` vs `double` on the GPU._

---

## The simulation program

### `planet_sim`: loop, outputs, built-in scenarios (Oct 2026, #14)
**Decisions:**
- **One loop** ([`run_simulation()`](../src/sim.cpp)): `compute_accelerations()` once (the leapfrog contract), then leapfrog steps. A frame and an energy row at step 0, every `output_every` steps, and the final step (so the log always covers the whole run); potential energy is O(N²) and serial, so it's never computed on other steps. Frames are numbered 0, 1, 2, ... in output order.
- **Command-line overrides** live in [`apply_overrides()`](../src/cli.cpp), which also refuses unusable runs: a config with no `steps` (a 0-step run would "succeed" doing nothing), a non-finite or non-positive `dt`. Typos like `--dt inf` are usage errors (exit 1), not "the physics exploded" (exit 2).
- **NaN/Inf check after every step** (one O(N) pass, negligible next to O(N²) gravity). The run stops at the first bad particle and exits with code 2, so "the physics exploded" can't look like success.
- **Built-in scenarios** (`--scenario orbit|figure8|cloud`) so the program runs before the config loader (#13) and planet generator (#9) exist: benchmarks (#20) and tests (#16) can start now. Collisions come from config files.
- **Planets are built at rest at the origin**, relaxed, then placed with `place_planet()` (the #15 contract), and forces are recomputed for the combined system before stepping.
- **`--bench`** writes nothing and prints one line for scripts: `bench backend=omp n=5000 threads=6 steps=20 seconds_per_step=0.0121`.
- **Exit codes:** 0 ok, 1 usage/setup error, 2 NaN/Inf.

**Evidence:** [tests/test_sim.cpp](../tests/test_sim.cpp) checks the loop gives the same bits as hand-written leapfrog steps, the energy log's rows and conservation, NaN detection, CLI parsing; CTest also runs the real `planet_sim` binary and checks exit codes and output ([tests/CMakeLists.txt](../tests/CMakeLists.txt)). First end-to-end OpenMP number through the real program: a 5,000-particle cloud, 0.077 s/step serial vs 0.012 s/step with 6 threads (6.4x), Ryzen 5 7600X, `release`.

**Lessons:**
- The first version of the program-level tests passed two error cases *for the wrong reason* (CMake split the argument list, so every run failed with "missing value"). Each error test now matches its specific message, which then caught a real bug: a trailing unknown flag was reported as "missing value".
- Review deliberately broke 34 copies of this code and **17 still passed the tests**: the gaps were at the program level (overrides, the NaN exit code, a mid-run NaN, `--dt inf`). After adding those tests (including one that blows up during step 1 and a `/dev/full` "disk full" test for the energy log), they're covered. Also caught: CI still ran `planet_sim` with no arguments, which became a usage error and blocked the PR. Branch protection did its job.

### `planet_view`: live viewer with backend switching (Oct 2026, #45, #48)
**Decisions:**
- **One step at a time:** `run_simulation` is now built on [`sim_init()` and `sim_step()`](../src/sim.cpp), and the viewer calls the same two functions, so it runs exactly the physics `planet_sim` runs (same setup flags, same config/relaxation path).
- **Simulation and drawing share one thread.** Each frame runs steps until a time budget (12 ms by default, `[`/`]` to change) is used, always at least one, then draws. No locks or races. A slow backend slows the window too, which is what the demo should show: at 10k particles serial drops to a few fps.
- **Backend switching needs no data movement (yet).** All state lives in host `Particles`, and the HIP backend copies positions in and accelerations out on every call, so `1`/`2`/`3` just change which backend the next step uses. Once #30 keeps data on the GPU, switching away from HIP must copy it back first.
- **raylib 5.5**, pinned and built from source behind `ENABLE_VIEWER` (default off), so CI's normal jobs and anyone without X11/OpenGL never touch it. Particles are drawn as small camera-facing squares in one batch; the glow look is #47.

**Evidence:** `SimStep.*` in [tests/test_sim.cpp](../tests/test_sim.cpp): stepping one at a time gives the same bits as `run_simulation`; serial → OpenMP → serial matches serial throughout; a GPU version (serial → HIP → serial) matches within 1e-10 relative. CI's `viewer` job runs `planet_view` for 60 frames on a virtual screen (Xvfb, software OpenGL).

Live numbers on the dev PC (`demo` preset, 10,000-particle `cloud`, 1280×800 window):

| Backend | ms/step | fps |
|---|---|---|
| serial | 278 | 4 |
| OpenMP (12 threads) | 46 | 21 |
| HIP (naive double) | 26 | 38 |

These match `planet_sim --bench` for the same N, so drawing costs little next to the physics at this size.

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
