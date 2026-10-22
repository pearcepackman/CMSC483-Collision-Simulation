# Planet Collision Simulator: Project Plan

Parallel computing class project (Fall 2026). Two-person team, heavy use of AI-assisted development (Claude Code).

**One-line pitch:** Smash two particle-based planets together and show how parallel computing (OpenMP on CPU, then CUDA on GPU) turns an O(N²) physics problem from "go get lunch" into "watch it live."

---

## 1. What we're building

Each planet is a ball of particles (start with ~5,000, scale to 50,000+). Every step of the simulation:

1. **Gravity:** every particle pulls on every other particle.
2. **Contact:** particles that overlap push apart (a spring) and lose a bit of energy (a damper), so planets hold their shape instead of passing through each other like ghosts.
3. **Integrate:** update velocities and positions, advance time, repeat.

Output frames get rendered into a video of the impact: a debris splash, fragments flung into orbit, and material settling into a new body.

### What we are NOT building

Real research codes (like SWIFT, used for the Moon-forming impact papers) use **SPH**, which adds pressure, temperature and material equations of state. That's much harder to get right. We use the simpler **"rubble pile" model**: gravity plus soft-sphere contact. It looks convincing, it's a real technique, and it's achievable.

---

## 2. Why this is a good parallel computing project

These points are the backbone of the final presentation.

| Point | Why it matters |
|---|---|
| **Gravity is O(N²)** | 50,000 particles = ~2.5 billion pair interactions per step, times thousands of steps. Serial is painfully slow, so the speedup is real and measurable. |
| **Each particle's force is independent** | Thread *i* computes the total force on particle *i*. That's natural data parallelism for both OpenMP and CUDA. |
| **A real design trade-off** | Newton's 3rd law lets you compute each pair once (half the work) but then two threads write to the same particle, which means atomics or locks. Computing each pair twice avoids races. Benchmark both and explain the winner. |
| **Memory layout matters** | Struct-of-Arrays (separate `x[]`, `y[]`, `z[]` arrays) vs Array-of-Structs. SoA enables SIMD on CPU and coalesced memory access on GPU. |
| **Contact uses a different pattern** | Collisions only involve nearby particles, so a uniform grid (cell list) makes contact detection ~O(N). Building the grid in parallel is its own problem. |
| **Algorithm vs hardware (stretch)** | Barnes-Hut (octree) cuts gravity to O(N log N). "Smarter algorithm on 1 core vs brute force on a GPU" is a great slide. |
| **Scaling analysis** | Strong scaling (fixed N, more threads), weak scaling, Amdahl's law, where speedup flattens and why. |

---

## 3. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | **C++17**, written in a plain "C-style" (structs, arrays, simple functions) | CUDA is C++, test frameworks are C++, and `std::vector` saves pain. Avoid fancy templates/OOP so the parallel parts stay readable. |
| Build | **CMake** | Standard for C++/CUDA; makes CUDA an optional component (`-DENABLE_CUDA=ON`). |
| CPU parallel | **OpenMP** | Easiest way to parallelize loops; supports scheduling experiments. |
| GPU parallel | **CUDA** | The main speedup story. Optional build so laptops without NVIDIA GPUs still work. |
| Tests | **GoogleTest** (pulled in with CMake `FetchContent`) | Physics correctness checks run in CI. |
| Analysis & plots | **Python** (numpy, matplotlib) | Benchmark graphs, energy plots. |
| Visualization | Python/matplotlib frames → **ffmpeg** video; **ParaView** for nicer 3D renders | No graphics code in the simulator itself. Live rendering (raylib/OpenGL) is a stretch goal only. |
| GPU hardware | Own NVIDIA GPU if either of us has one; otherwise UMBC HPCF **chip** cluster | HPCF accounts are free for UMBC students and chip has GPU nodes. Request an account early. |

### Open-source references (study, don't copy)

- **REBOUND** (C, GPLv3, OpenMP): a collisional N-body code with leapfrog integrators and Barnes-Hut for gravity and collisions. Great reference for structure, and its Python package (`pip install rebound`) can **validate our results** in tests (e.g. compare a 3-body orbit).
- **NVIDIA "Fast N-Body Simulation with CUDA"** (GPU Gems 3, ch. 31): the classic tiled shared-memory CUDA kernel for all-pairs gravity.
- **MarcVivas/N-body** (GitHub): a progression from serial → OpenMP → GPU → Barnes-Hut. Basically our roadmap.
- **missing-user/barnes-hut** (GitHub): OpenMP Barnes-Hut with a good write-up on static vs dynamic scheduling.
- **SWIFT + WoMa**: the real giant-impact research stack (SPH). Use only for inspiration and "here's how NASA-level work does it" slides.

> **Check with the professor** how much outside code is allowed. Default assumption: we write our own simulator and only use OSS as reference/validation. REBOUND is GPLv3, so don't paste its code into our repo.

---

## 4. Physics cheat sheet (simple version)

Use **simulation units** with `G = 1` to avoid giant/tiny numbers. All values `double` on CPU; GPU may test `float` vs `double` as a benchmark point.

- **Gravity with softening:** `a_i = Σ_j m_j * (r_j - r_i) / (|r_j - r_i|² + ε²)^(3/2)`
  `ε` (softening) prevents infinite force when particles get very close.
- **Contact (soft sphere):** if two particles of radius `R` overlap by `δ = 2R - |r_ij| > 0`:
  `F = k * δ * n̂  -  c * (v_rel · n̂) * n̂`
  (`k` = spring stiffness, `c` = damping, `n̂` = unit vector between them)
- **Integrator: Leapfrog (kick-drift-kick).** Simple, and it conserves energy well over long runs, unlike plain Euler.
  1. `v += a * dt/2`  2. `x += v * dt`  3. recompute `a`  4. `v += a * dt/2`
- **Timestep:** small enough that contact springs don't explode. Rule of thumb: `dt` ≲ 1/10 of the spring's oscillation period `2π·sqrt(m/k)`.
- **Making a planet:** fill a sphere with particles (random or lattice), then run a short "relax" phase with damping so it settles into a stable ball before the impact.

---

## 5. Repo layout

```
planet-collision/
├── CMakeLists.txt
├── README.md
├── PLAN.md                  ← this file
├── CLAUDE.md                ← rules for Claude Code (see section 9)
├── .clang-format
├── .github/workflows/ci.yml
├── src/
│   ├── main.cpp             ← CLI: load config, run sim, write output
│   ├── particles.hpp        ← SoA particle storage
│   ├── init.cpp             ← planet generation + relaxation
│   ├── gravity_serial.cpp   ← reference implementation (never delete)
│   ├── gravity_omp.cpp
│   ├── gravity_cuda.cu      ← only built with ENABLE_CUDA
│   ├── contact.cpp          ← uniform grid + contact forces
│   ├── integrator.cpp       ← leapfrog
│   ├── diagnostics.cpp      ← energy, momentum, angular momentum
│   └── io.cpp               ← write frames (CSV first, binary later)
├── tests/
│   ├── test_two_body.cpp
│   ├── test_conservation.cpp
│   └── test_backends_match.cpp
├── configs/                 ← scenario files (head-on, grazing, Theia-like)
├── scripts/
│   ├── render.py            ← frames → images → video
│   ├── plot_energy.py
│   └── bench.py             ← run benchmarks, make speedup graphs
└── bench/results/           ← committed CSVs of benchmark runs
```

**Key rule:** one interface, multiple backends. `compute_gravity(particles)` has a serial, OpenMP and CUDA version selected by a flag (`--backend serial|omp|cuda`). Same inputs, same outputs, swappable. This makes testing and benchmarking easy.

---

## 6. Milestones

Dates assume final presentation in early December. Adjust to the real due date.

| # | Target | Milestone | Done when |
|---|---|---|---|
| 0 | Oct 22–25 | **Repo + CI skeleton** | CMake builds a hello-world, GoogleTest runs, CI is green |
| 1 | by Nov 1 | **Serial two-body orbit** | Leapfrog orbit closes; energy drift test passes |
| 2 | by Nov 8 | **Serial planet collision** | Two relaxed planets collide; first video exists (small N ≈ 2–5k) |
| 3 | by Nov 15 | **OpenMP backend** | Matches serial within tolerance; first speedup graph |
| 4 | by Nov 22 | **CUDA backend** | Matches serial within tolerance; runs on GPU (own machine or chip) |
| 5 | by Nov 29 | **Grid contact + scenarios + benchmarks** | Head-on vs grazing impact videos; full scaling graphs |
| 6 | Dec | **Stretch + presentation** | Barnes-Hut or Newton's-3rd-law/atomics comparison; slides done |

### Suggested split

- **Partner A — engine:** integrator, gravity backends (OpenMP, CUDA), performance work.
- **Partner B — world + proof:** planet generation, contact/grid, scenarios, rendering, benchmark scripts.
- **Both:** tests, CI, presentation. Each person should be able to explain every parallel decision; that's what gets asked in Q&A.

---

## 7. Testing ("make sure shit doesn't explode")

Physics bugs are sneaky. The simulation can look fine while being wrong. These tests catch that:

1. **Two-body orbit:** two particles in a circular orbit stay circular; with ~1,000 steps per orbit, relative energy error stays small (start with < 1e-5) and does *not* grow steadily over 1,000 orbits. Leapfrog's energy error oscillates instead of drifting; plain Euler would fail this test, which is a nice demo.
2. **Momentum conservation:** with no external forces, total momentum stays constant to ~machine precision (gravity and contact forces are equal and opposite).
3. **Angular momentum conservation:** same idea, for an off-center collision.
4. **Energy sanity:** in a collision, energy may *decrease* (damping) but must never *increase* beyond a small tolerance. Rising energy = timestep too big or a bug.
5. **Backends match:** compare the *forces from a single evaluation* (and positions after only a few steps) between serial, OpenMP and CUDA.
   *Note:* parallel runs add numbers in a different order, so results won't be bit-identical. Floating-point addition isn't associative. Compare with a relative tolerance (e.g. ~1e-12 for double forces, looser for `float`), not `==`.
   *Don't* compare after thousands of steps: many-body gravity is chaotic, so tiny rounding differences grow until runs diverge even when both are correct. Compare long runs by conserved quantities (energy, momentum) instead. Both points make good presentation slides.
6. **No NaNs/Infs:** fail immediately if any position or velocity becomes NaN.
7. **Determinism:** with a fixed random seed, the serial backend gives the exact same result every run.
8. **(Optional) REBOUND cross-check:** a Python test compares a 3-body run against REBOUND's output.

Tests needing a GPU are labeled `gpu` in CTest so CI can skip them.

---

## 8. CI pipeline (GitHub Actions)

> Detailed tooling decisions (presets, sanitizers, TSan, secret scanning, Linux vs macOS) live in [docs/tooling.md](docs/tooling.md).

GitHub's free hosted runners have **no GPU**. So CI compiles the CUDA code but only *runs* CPU tests. GPU tests run manually on a GPU machine (own PC or chip).

**`.github/workflows/ci.yml` jobs, on every push and pull request:**

1. **format** — `clang-format --dry-run --Werror` on all sources.
2. **build-and-test (CPU)** — matrix of `gcc` and `clang`:
   - Build with warnings as errors (`-Wall -Wextra -Werror`).
   - `ctest -LE gpu` with `OMP_NUM_THREADS=1` **and** `OMP_NUM_THREADS=4` (catches race conditions that only appear with multiple threads).
3. **sanitizers** — Debug build with AddressSanitizer + UndefinedBehaviorSanitizer; run the test suite. Catches out-of-bounds writes and memory bugs, the classic C++ "explodes later" bugs.
4. **cuda-compile** — run inside an `nvidia/cuda` devel container with `-DENABLE_CUDA=ON`; build only. Catches CUDA syntax and type errors even without a GPU.
5. **physics smoke test** — run a tiny collision (~500 particles, 200 steps) and fail if energy drift exceeds the threshold or any NaN appears.

**Branch rules:** protect `main`, require CI to pass, use pull requests even as a two-person team. Claude Code works on a branch, CI checks it, a human reviews and merges.

**Optional later:** a `workflow_dispatch` job for GPU tests on a self-hosted runner, if anyone wants to wire one up. Not required.

---

## 9. Working with Claude Code

Put a `CLAUDE.md` at the repo root with rules like these:

- Build: `cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`
- Test: `ctest --test-dir build --output-on-failure`
- **Never delete or "optimize" `gravity_serial.cpp`.** It is the reference every other backend is tested against.
- Every new backend or physics change needs a test, and all tests must pass before a commit.
- Don't change physics constants or tolerances in tests just to make them pass. Flag it instead.
- Keep particle data in Struct-of-Arrays layout.
- Keep each PR to one milestone task.

**Prompting tips:**

- Give Claude Code **one milestone at a time** ("Implement milestone 1 from PLAN.md"), not the whole plan at once.
- Ask it to **explain the parallel decisions** it made (scheduling, data layout, races). Paste those explanations into a `docs/notes.md`; they become presentation material.
- When something looks wrong, ask it to **write a failing test first**, then fix the bug.
- Read the diffs. AI-written parallel code can have races that pass tests by luck.

---

## 10. Benchmarks to collect (for the presentation)

- Runtime per step vs N (1k → 100k) for serial, OpenMP and CUDA. Log-log plot shows O(N²).
- OpenMP strong scaling: 1, 2, 4, 8, 16+ threads at fixed N, with speedup and efficiency.
- OpenMP `schedule(static)` vs `schedule(dynamic)`, especially for the contact step (uneven work per grid cell).
- CUDA: naive kernel vs shared-memory tiled kernel; `float` vs `double`.
- Pair-twice vs Newton's-3rd-law-with-atomics.
- Stretch: brute force vs Barnes-Hut crossover point.

Record hardware (CPU model, core count, GPU model) with every result, and commit raw CSVs to `bench/results/`.

---

## 11. Open questions for the team

- [ ] When is the presentation due, and what's required (report, demo, code)?
- [ ] Does the class require a specific framework (OpenMP / MPI / CUDA)? If MPI is required, add a domain-decomposition milestone.
- [ ] Does either of us have an NVIDIA GPU? If not, request HPCF (chip) accounts this week.
- [ ] What's allowed regarding outside code and AI use? Get it in writing.
