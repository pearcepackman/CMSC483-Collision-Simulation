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
- Acceleration contract: gravity *overwrites* `ax/ay/az`; contact *adds* to them. [`compute_accelerations()`](../src/forces.hpp) is the one place that runs them in that order, and the integrator and planet relaxation both call it.
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
- Near-perfect up to 6 threads: each particle's work is independent and identical, there's no shared write, and N² math dominates memory traffic. One thread per physical core, no contention.
- **8 threads is *slower* than 6:** `schedule(static)` hands every thread an equal block, but with 8 threads on 6 cores, two cores run two threads each at about half speed. The whole loop waits for those slow blocks, so 8 threads finish no sooner than 4. This is the load-imbalance question #22 (static vs dynamic) investigates.
- 12 threads ≈ 6: the second hardware thread per core (SMT) shares the same floating-point units, and this loop already keeps them busy, so it adds nothing.
- Measurement lesson: the first runs showed "123% efficiency." That was noise, not magic: with the `powersave` governor, CPU boost and other programs running, the clock speed drifts between measurements. The benchmark now interleaves configurations each round and reports speedup against 1-thread OpenMP as well as serial.
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
- Double precision is slow on this card. Consumer RDNA GPUs run `double` at a small fraction of `float` speed. A scratch probe of the kernel alone at N = 10,000 took 25 ms in `double` vs 2.0 ms in `float`, so in `double` the GPU only modestly beats 6 CPU cores. #32 measures the `float` version properly.
- Small N loses. At 2,000 particles the fixed costs (kernel launch, seven host↔device copies per step) outweigh the math; the GPU only pays off as N² grows. #30 removes the per-step copies.
- No data reuse yet. Every thread reads every other particle from global memory; #26 (shared-memory tiles) loads each block of particles once and shares it.

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
- Exit codes: 0 ok, 1 usage/setup error, 2 NaN/Inf.

**Evidence:** [tests/test_sim.cpp](../tests/test_sim.cpp) checks the loop gives the same bits as hand-written leapfrog steps, the energy log's rows and conservation, NaN detection, CLI parsing; CTest also runs the real `planet_sim` binary and checks exit codes and output ([tests/CMakeLists.txt](../tests/CMakeLists.txt)). First end-to-end OpenMP number through the real program: a 5,000-particle cloud, 0.077 s/step serial vs 0.012 s/step with 6 threads (6.4x), Ryzen 5 7600X, `release`.

**Lessons:**
- The first version of the program-level tests passed two error cases *for the wrong reason* (CMake split the argument list, so every run failed with "missing value"). Each error test now matches its specific message, which then caught a real bug: a trailing unknown flag was reported as "missing value".
- Review deliberately broke 34 copies of this code and **17 still passed the tests**: the gaps were at the program level (overrides, the NaN exit code, a mid-run NaN, `--dt inf`). After adding those tests (including one that blows up during step 1 and a `/dev/full` "disk full" test for the energy log), they're covered. Also caught: CI still ran `planet_sim` with no arguments, which became a usage error and blocked the PR. Branch protection did its job.

### Frame files: one CSV per output step (Nov 2026, #10)
**Decision:** [`write_frame()`](../src/io.cpp) writes `<out>/frame_00042.csv` with a header line `id,x,y,z,body` and one row per particle. Doubles use `%.17g`, which is enough digits to read back the exact same double. The file is opened in binary mode so lines end in `\n` on every OS (Windows text mode would write `\r\n`). [`read_frame()`](../src/io.cpp) is strict about structure: wrong header, wrong number of columns, a non-number, a body that doesn't fit in an `int`, ids not 0, 1, 2, ... in order, a blank line anywhere but the very end, a NUL byte, or a line too long for its 256-byte buffer (real rows are under 100 characters) all return false and leave the caller's particles untouched. It accepts what an editor might do to a file: `\r\n` line endings, no newline after the last row, or one blank line at the end. Within a field it is as lenient as `sscanf` (a leading space, a `+` sign or a hex float like `0x1p3` still read), which is harmless for files `write_frame` wrote.
**Why:** CSV with a header is what the render script (#11) and anyone poking at a run can open directly in pandas or a spreadsheet, and it matches `energy.csv`. Only positions and body ids go in, because that's all rendering needs; replay (#49) can add columns later. A strict reader means a truncated or hand-edited frame fails loudly instead of rendering as a planet with missing particles.
**Evidence:** a frame is about 68 bytes per particle (136 KB at 2,000 particles, 687 KB at 10,000). Writing one takes about 1 ms at 2,000 and 6 ms at 10,000 (Intel Core Ultra 7 155H, WSL, gcc `-O2`), small next to a step at those sizes, and frames are only written every `--output-every` steps. [tests/test_io.cpp](../tests/test_io.cpp) checks a bit-for-bit round trip (including -0.0, subnormals, NaN and inf, and a real two-planet frame) and each malformed case; two CTest runs of the real `planet_sim --out` check the frame count, including the extra frame at a final step that isn't a multiple of `--output-every`. A full disk is tested by pointing the frame's file name at Linux's `/dev/full` (a device that fails every write) with a symlink, and a test in [tests/test_sim.cpp](../tests/test_sim.cpp) checks that a frame that can't be written stops the run. We ran 21 deliberately broken copies of `io.cpp` and `sim.cpp` against the tests; 17 made a test fail and one more (dropping the NUL-byte check) is caught by AddressSanitizer in CI as a read before the start of the line buffer. The three that pass don't change behavior on Linux: writing in text mode (only differs on Windows), skipping the `ferror` check (with `/dev/full` the final `fclose` fails too), and skipping the null-path check (glibc's `fopen` already returns null).

### `render.py`: quick-look video from frame files (Nov 2026, #11)
**Decision:** [`scripts/render.py`](../scripts/render.py) reads a folder of frames, checks each one the way `read_frame()` does (exact header, 5 columns, ids 0, 1, 2, ... in order, whole-number bodies, no comment lines, no blank lines except one at the end; `\r\n` and nan/inf are fine), and draws a matplotlib 3D scatter per frame colored by planet. matplotlib's `FFMpegWriter` sends each image straight into ffmpeg (H.264, `yuv420p` so browsers and phones play it), with no PNG files in between. ffmpeg itself comes from the `imageio-ffmpeg` pip package, so `pip install -r scripts/requirements.txt` is the whole setup on every machine, with no root needed. All 12 packages are pinned, matplotlib's own dependencies included: the first version pinned only the three we import, pip then picked pyparsing 3.3.3, which is newer than matplotlib 3.10.3 and prints deprecation warnings on every run. A later pyparsing could break CI with no change on our side.
**Camera:** the axes are fixed for the whole video, so the view doesn't zoom as things move. A first pass over the frames finds one cube that holds every particle except the farthest 0.5% on each side of each axis, and particles outside it are clipped. The first version used every particle, and in a gravity-only test run a few particles flung out to about 30 planet radii shrank the planets to a dot for the whole video.
**Evidence:** two generated planets of 2,000 particles each, 121 frames (32 MB of CSV), render at about 9 frames per second into a 276 KB, 4-second video (Intel Core Ultra 7 155H, WSL). That's slow next to the viewer but fine for a debug tool, and `--every k` skips frames for long runs. [scripts/test_render.py](../scripts/test_render.py) (15 tests) checks loading and rejecting frames, numeric frame order, the camera box (widest axis, stragglers, nan/inf), the video's frame count and length, and clean errors for bad paths; CI's `render` job runs them and renders a real `planet_sim` run. Review ran 27 broken copies of the script and 14 passed the old tests. The tests added since catch 8 of those 14. Five more only change how the video looks (no clipping, one color for every planet, old points not erased, axis limits not set, a pixel format some players can't open), which you see by watching the uploaded video. The last one switches the box to interpolated percentiles, which moves it by under 1%.

### Scenario files: `key = value` with numbered planets (Nov 2026, #13)
**Decision:** [`load_config()`](../src/config.cpp) reads one `key = value` per line, with `#` comments, and planets as `planet0.n`, `planet1.radius`, and so on. Planet N gets body id N, so frames color it the same way every run. Center and velocity are three numbers separated by spaces. Any error (unknown or repeated key, a value that isn't entirely one finite number of the right type, planets numbered with a gap, a planet missing `n`, `radius`, `particle_radius` or `mass`) prints `file:line: reason` with the line and leaves the `Config` untouched. `output_every` is required, and so is at least one planet. Each planet is also run through [`planet_spec_ok()`](../src/init.hpp), the same checks `make_planet` aborts on, so a negative `n` or a `particle_radius` too big for the planet is a clean error naming the planet (exit 1), not a crash. `softening`, `contact_c` and the `relax_*` values must be at least 0, and `contact = true` needs `contact_k > 0`. `dt`, `steps` and `output_every` are left to `apply_overrides`, because command-line flags can still replace them. A UTF-8 byte order mark at the start of the file is skipped, and a directory is rejected as "not a regular file". The files are [configs/head_on.cfg](../configs/head_on.cfg) and [configs/smoke.cfg](../configs/smoke.cfg).
**Why:** the issue asked for plain `key = value`, and a numbered prefix keeps every line self-contained, while `[planet]` sections would make a key's meaning depend on where it sits in the file. Unknown keys must fail, because a typo like `stpes = 5000` would otherwise silently run with the default. Repeated keys fail for the same reason: with two `dt` lines, only one of them takes effect, and nothing says which. `output_every` has no safe default. With the old default of 1, every step writes a frame and computes the O(N²) potential energy, which costs about as much as the step itself. `strtod` accepts `nan` and `inf`, so the loader rejects non-finite values itself. The first version left range checks to `make_planet`, but `make_planet` aborts (exit 134 and a core dump, or a closed viewer window), and a wrong number in a config is the most common mistake people will make. Sharing one check function keeps the loader and the generator from disagreeing. Windows PowerShell 5.1's `Out-File` writes the byte order mark, and without the skip it turned the first key into `ï»¿output_every`.
**Choosing the config values:** both files use particle mass 1e-3, `k = 20`, `c = 0.05` and `dt = 0.002` from the kickstart preview. The spring period `2π·sqrt(m_eff/k)` is 0.031, so `dt` is T/15.7 (limit T/10), and `c·dt/m_eff = 0.2` (limit 2, see Contact). head_on uses 2,000 particles per planet with `particle_radius = 0.065`, where FCC neighbors just touch. That is 94% of the 0.0688 limit. Its planets start at a relative speed of 1, half the pair's escape speed of 2 at contact, and gravity speeds them up to about 1.84 (92% of escape speed) by the time they touch. Their total energy is ½·1·1 − 2·2/5 = −0.3, so they are bound and should merge. smoke started with the same 0.065 at radius 0.5 and `make_planet` refused it: the margin `radius - particle_radius` takes a larger share of a small ball. The limit for 250 particles at radius 0.5 is between 0.062 and 0.065, so smoke uses 0.06.
**Evidence:** [tests/test_config.cpp](../tests/test_config.cpp) (14 tests) loads every key from a sample file, checks that missing keys keep their values and that Windows line endings and a byte order mark load, and checks each error message, including the line number after comments and blank lines. It also loads both shipped files. Each test writes its files in a folder named after the process id and the test, because the first version shared one file and two test runs at once (two presets, or `ctest -j`) failed about 3,700 times each in 300 repeats; now 0. `Cli.SmokeConfigRuns` and `Cli.HeadOnConfigRuns` build and run both files, and the smoke test caught the 0.065 problem above. `Cli.BadPlanetConfigIsSetupError` checks that a planet `make_planet` can't build gives exit 1. In review, 10 broken copies of the loader (line numbers that skip comments, no space check between vector numbers, no seed upper bound, no underflow check, no byte order mark skip, no planet or range checks, and others) each failed a test.
**Smoke timing:** 200 steps of 500 particles (Intel Core Ultra 7 155H, 22 threads, WSL, RelWithDebInfo) take 0.18 s serial, 0.10 to 0.12 s with OpenMP on 4 threads (about 1.7x), and 1.06 to 1.25 s on all 22 threads, about 6x slower than serial. Only gravity is parallel so far (contact is still serial), and at 500 particles each of 22 threads gets about 23 particles per step, so starting and joining the threads every step costs more than the work it splits. CI should run the smoke test on the serial backend or a few threads.
**Energy:** the full smoke run's logged energy only falls, by 0.0029. The first 300 steps of head_on (4,000 particles, about 12 s on 22 threads), logged every 25 steps as the file says, show two clear rises in kinetic plus gravitational energy, +0.065 at step 125 and +0.026 at step 275 (and one of 7e-8 at step 25). Counting the energy stored in compressed contact springs (`½·k·δ²` per overlapping pair, computed from the frames) removes both: that total only falls, from -5.366 to -5.498. The unrelaxed planets sag under their own gravity and spring back, and relaxation (#15) will damp that out. `energy.csv` does not include spring energy yet.

### `planet_view`: live viewer with backend switching (Oct 2026, #45, #48)
**Decisions:**
- One step at a time: `run_simulation` is now built on [`sim_init()` and `sim_step()`](../src/sim.cpp), and the viewer calls the same two functions, so it runs exactly the physics `planet_sim` runs (same setup flags, same config/relaxation path).
- Simulation and drawing share one thread. Each frame runs steps until a time budget (12 ms by default, `[`/`]` to change) is used, always at least one, then draws. No locks or races. A slow backend slows the window too, which is what the demo should show: at 10k particles serial drops to a few fps.
- Backend switching needs no data movement (yet). All state lives in host `Particles`, and the HIP backend copies positions in and accelerations out on every call, so `1`/`2`/`3` just change which backend the next step uses. Once #30 keeps data on the GPU, switching away from HIP must copy it back first.
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

## Planets

### FCC lattice with a small seeded jitter (Nov 2026, #9)
**Decision:** [`make_planet()`](../src/init.cpp) takes the `n` points of a face-centered cubic (FCC) lattice closest to the origin, shifts them so their mean is at the planet's center, scales the lattice to the largest spacing that keeps every center within `radius - particle_radius - jitter`, and moves each particle by a random offset of at most 2% of `particle_radius`. The seed only controls that offset. If that spacing would let neighbors overlap by 10% of `particle_radius` or more, `make_planet` aborts with the spec in the message instead of building a planet that explodes on its first contact step.
**Why:** a purely random fill can't meet the "overlap under 10%" rule at any useful density, and random filling with a minimum distance stops at about 38% packing and is O(N²) without a grid. FCC is the densest regular packing (74%), so the planet starts close to its settled size and relaxation (#15) has less to fix. A simple cubic grid packs only 52% and has flat planes that shear easily.
**Same seed, same planet, on every build:** measured in half cube sides, FCC points are the integer triples with an even sum, so the code picks and orders the `n` closest points by exact integer distance. A first version sorted on floating-point distances, and the release preset's `-march=native` turns `x*x + y*y + z*z` into fused multiply-adds (FMA, one rounding instead of two). That reordered points that tie within a shell, and 1,003 of 2,000 particles landed somewhere else than in the dev build. With integer keys the two builds now differ by at most 1.1e-16. The jitter comes from the raw 64-bit output of `std::mt19937_64`, which the standard fixes exactly, not from `std::uniform_real_distribution`, which libstdc++ (Linux) and libc++ (macOS) implement differently. `Init.MatchesStoredPlanet` checks a stored 13-particle planet, so CI's gcc and clang builds must agree.
**Centering:** the last lattice shell is usually only partly filled, and the first version left the planet's center of mass up to 2% of the radius off center at a few hundred particles (47% for n = 2). Subtracting the mean brings that down to the jitter: 8.5e-6 of the radius at n = 2,000 and 3.6e-4 at n = 2.
**Evidence:** for 2,000 particles in a unit ball, the largest `particle_radius` that passes is about 0.0688 (worst overlap 9.7% of `particle_radius`); 0.069 aborts. At 0.068 the worst overlap is 7.2%, and at 0.065 neighbors just touch (-2.6%, a small gap). The tests pin the limit from both sides (0.068 must work, 0.07 must abort). We also ran 21 deliberately broken copies of `make_planet` (overlap limit raised to 20%, jitter in a cube instead of a ball, farthest points kept, no centering, wrong seed, and so on) against the tests, and every one made a test fail.

---

## Contact

### Brute-force spring-dashpot contact (Nov 2026, #12)
**Decision:** [`compute_contact_bruteforce()`](../src/contact.cpp) applies `F = (k·δ − c·v_n)·n̂` to every overlapping pair, where `δ = r_i + r_j − d` and `v_n` is the relative velocity along the line between the centers. Each particle sums the force from all the others (every pair computed twice), like the serial gravity, and adds `F / m_i` to its acceleration. The force is not clamped to "push only", and every particle mass must be greater than 0 (`make_planet` now rejects a planet with mass 0).
**Why:** computing each pair from both sides doubles the arithmetic, but the two sides see exactly negated inputs, so the pair forces cancel bit for bit, and each thread can own a range of particles in the OpenMP and grid versions (#23) without write races. Clamping would stop the damper from pulling the pair together near the end of a contact (about 28% of the peak push in the test bounce), but the formula `c = 2ζ·sqrt(m_eff·k)` for a target restitution assumes the unclamped model. A massless particle gets 0/0 = NaN from the division, even when it touches nothing.
**Limits of the explicit step:** two particles, measured with the planned `k = 20`, `dt = 0.002`, particle mass `1e-3` (`m_eff = 5e-4`) and no gravity. Each kick multiplies the normal relative speed by about `1 − c·dt/m_eff`, so energy grows once `c·dt/m_eff ≥ 2`:

| c | c·dt/m_eff | energy after the bounce / before |
|---|---|---|
| 0.05 (planned) | 0.2 | 0.15 |
| 0.2 | 0.8 | about 0 |
| 0.5 | 2.0 | 1.17 |
| 1.0 | 4.0 | 9.5 |

A particle with many neighbors feels several dampers at once, so the real limit on `c` is lower than this. The spring force tops out at `k·(r_i + r_j)` when the centers meet, so pairs faster than about `(r_i + r_j)·sqrt(k/m_eff)` ≈ 32 pass through each other. With radius 0.0805, relative speeds of 1, 10 and 31 bounced, and 40 and 100 went through. The fastest planned impact is 2.6.
**Evidence:** in a two-particle head-on bounce with gravity (unit masses, radius 0.5, `k = 1e4`, `dt = 1e-4`, about 440 steps per spring period), energy is kept to 1.2e-5 without damping, and damping set for restitution 0.5 loses 88% of the starting kinetic energy against 87.5% predicted. During review we ran 20 deliberately broken copies of the contact loop against the tests (wrong damper sign, `=` instead of `+=`, one radius only, the other particle's mass, a clamped force, half the pairs, and so on). 4 passed the first version of the tests: normal velocity from x only, a flipped z term, `fz` using `ny`, and touching counted as contact. Tests along a direction with three different components, and an approaching pair at exactly touching distance, now catch all 20.

_Upcoming: uniform grid (cell list) construction in parallel; load balance across uneven cells._

## Scaling results

_Upcoming: runtime vs N (log-log), strong scaling, efficiency, Amdahl's-law fit. Record CPU/GPU model and core count with every result._
