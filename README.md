# Planet Collision Simulator

[![CI](https://github.com/pearcepackman/CMSC483-Collision-Simulation/actions/workflows/ci.yml/badge.svg)](https://github.com/pearcepackman/CMSC483-Collision-Simulation/actions/workflows/ci.yml)

Smash two particle-based planets together and use parallel computing to make an O(N²) physics problem fast enough to watch. CMSC 483 (Parallel Computing) project, Fall 2026.

Each planet is a ball of thousands of particles. Every timestep, every particle pulls on every other one (gravity), overlapping particles push apart and lose energy (soft-sphere contact), and a leapfrog integrator advances positions and velocities. The same simulation runs on three swappable backends, **serial**, **OpenMP** (CPU threads) and **HIP** (AMD GPU), so we can measure exactly how much each one speeds things up.

> Demo video coming with milestone 2.

## Status

| # | Milestone | Status |
|---|---|---|
| 0 | Repo, build system, CI | In progress |
| 1 | Serial two-body orbit (leapfrog, energy conservation) | Not started |
| 2 | Serial planet collision + first video | Not started |
| 3 | OpenMP backend + speedup graphs | Not started |
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
./build/dev/planet_sim
```

GoogleTest downloads automatically during the first configure.

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
docs/       tooling decisions and the parallel-design notes behind the presentation
scripts/    rendering, plotting and benchmark scripts (Python)        [coming]
configs/    impact scenarios (head-on, grazing, Theia-like)           [coming]
bench/      committed benchmark results                               [coming]
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
