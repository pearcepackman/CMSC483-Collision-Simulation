# Planet Collision Simulator

[![CI](https://github.com/pearcepackman/CMSC483-Collision-Simulation/actions/workflows/ci.yml/badge.svg)](https://github.com/pearcepackman/CMSC483-Collision-Simulation/actions/workflows/ci.yml)

Smash two particle-based planets together and use parallel computing to make an O(N²) physics problem fast enough to watch. CMSC 483 (Parallel Computing) project, Fall 2026.

Each planet is a ball of thousands of particles. Every timestep, every particle pulls on every other one (gravity), overlapping particles push apart and lose energy (soft-sphere contact), and a leapfrog integrator advances positions and velocities. The same simulation runs on three swappable backends, **serial**, **OpenMP** (CPU threads) and **HIP** (AMD GPU), so we can measure exactly how much each one speeds things up.

> Demo video coming with milestone 2.

## Status

| # | Milestone | Status |
|---|---|---|
| 0 | Repo, build system, CI | Done |
| 1 | Serial two-body orbit (leapfrog, energy conservation) | In progress: gravity + leapfrog done; planet generation, frame output, rendering left |
| 2 | Serial planet collision + first video | In progress: `planet_sim` loop and CLI done; planets, contact, config, relaxation, video left |
| 3 | OpenMP backend + speedup graphs | In progress: OpenMP gravity ~6x faster on 6 cores |
| 4 | GPU backend (HIP on AMD) | Not started |
| 5 | Grid-based contact, impact scenarios, full benchmarks | Not started |
| 6 | Stretch goals (Barnes-Hut, atomics comparison) + presentation | Not started |

Full plan: [PLAN.md](PLAN.md).

## Quickstart

**Requirements:** CMake ≥ 3.25, Ninja, a C++17 compiler with OpenMP, Python 3 (for the formatter). ROCm/HIP optional (only for the GPU backend, needs a supported AMD GPU).

```bash
# Ubuntu / Debian (primary platform)
sudo apt install build-essential cmake ninja-build clang libomp-dev python3-venv

# macOS (secondary: no GPU backend, no ThreadSanitizer, no AddressSanitizer on current Apple clang)
brew install cmake ninja libomp
```

Build, test, run:

```bash
cmake --preset dev && cmake --build --preset dev
ctest --preset dev
./build/dev/planet_sim --scenario figure8 --steps 20000
```

GoogleTest downloads automatically during the first configure.

`planet_sim` usage (`--help` for all flags):

```bash
planet_sim --scenario orbit|figure8|cloud [--n N] [--seed S]   # built-in setups, no files needed
planet_sim --config configs/head_on.cfg --out frames/run       # planets from a scenario file
           [--backend serial|omp] [--steps N] [--dt h] [--output-every k]
planet_sim --scenario cloud --n 10000 --backend omp --bench    # one "bench ..." timing line
```

`--out` writes `frame_NNNNN.csv` files and `energy.csv` (`step,t,kinetic,potential,total,px,py,pz`). Exit codes: 0 ok, 1 usage/setup error, 2 the simulation produced NaN/Inf. `--config` and frame output need #13 and #10.

Measure the OpenMP speedup (release build, threads pinned to cores):

```bash
cmake --preset release && cmake --build --preset release
OMP_PROC_BIND=close OMP_PLACES=cores ./build/release/bench_gravity 10000
```

### Build presets

| Preset | Use it for |
|---|---|
| `dev` | Day-to-day work: optimized with debug info, warnings are errors |
| `release` | Benchmarks: `-O3 -march=native` |
| `asan` | Finding memory bugs and undefined behaviour (AddressSanitizer + UBSan, Linux) |
| `tsan` | Finding data races in parallel code (ThreadSanitizer, Linux) |

Each preset builds into its own `build/<preset>/` folder. Run `cmake --preset <name> && cmake --build --preset <name> && ctest --preset <name>`.

### Formatting

Code is formatted with a pinned clang-format so local runs and CI always agree:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
git ls-files '*.cpp' '*.hpp' '*.hip' | xargs .venv/bin/clang-format -i
```

## Repository layout

```
src/        simulator source (particles, physics, backends, CLI)
tests/      GoogleTest suites: physics correctness and backend agreement
docs/       tooling decisions, design notes (presentation), kickstart guide for new work
scripts/    rendering, plotting and benchmark scripts (Python)        [coming]
configs/    impact scenarios (head-on, grazing, Theia-like)           [coming]
bench/      micro-benchmarks (bench_gravity) and committed results
```

## How we know it's correct

Parallel physics code can look fine while being wrong, so CI checks it on every pull request:

- **Physics tests:** orbits stay closed, momentum and energy are conserved, no NaNs.
- **Backend agreement:** OpenMP and GPU forces match the serial reference within a floating-point tolerance.
- **Run at 1 and 4 threads:** catches races that only appear with multiple threads.
- **Sanitizers:** AddressSanitizer + UBSan for memory bugs; ThreadSanitizer for data races.
- **gcc and clang**, warnings treated as errors, and a GPU (HIP) compile check.

Details: [docs/tooling.md](docs/tooling.md). Design decisions and benchmark findings: [docs/notes.md](docs/notes.md).

## Team

- Pearce Packman
- Kyle ([@kpoudyal06](https://github.com/kpoudyal06))
