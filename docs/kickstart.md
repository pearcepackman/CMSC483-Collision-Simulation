# Kickstart: building the world side (M1–M2)

For Kyle (and the Claude Code session he works with). Everything needed to pick up issues #9–#13, #15 and #18 without re-deriving what's already been figured out. Written Oct 22, 2026.

---

## 1. Where things stand

**Done and merged** (serial physics core, first parallel backend, CI):

| Issue | What it gives you |
|---|---|
| #6 | Every module's header with fixed signatures and stub bodies (stubs abort with `not implemented: <name>`) |
| #7 | `compute_gravity_serial()` + `potential_energy()`: the reference gravity, tested to 1e-14 |
| #8 | `leapfrog_step()`: kick-drift-kick integrator, energy error 3.9e-10 over 1,000 orbits |
| #19 | `compute_gravity_omp()`: ~6x faster on 6 cores, bit-identical to serial |
| #21 | CI race checking (ThreadSanitizer + Archer) |

**Yours, all unblocked now:** #9 planet generation, #10 frame output, #12 contact forces, #13 config loader. Then #11 render script (after #10), #15 relaxation (after #9 + #12), #18 first collision video (after those + Pearce's #14 simulation loop, in progress).

**Proof it works:** a throwaway preview (outside the repo) already ran two lattice planets colliding using the repo's real gravity + leapfrog, with quick versions of contact, generation and relaxation. Section 4 has the parameters that worked. Ask Pearce for the videos.

---

## 2. How we work

**One-time setup** (Arch shown; Ubuntu equivalents in [docs/tooling.md](tooling.md)):
```bash
sudo pacman -S cmake ninja clang openmp github-cli
gh auth login
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt   # pinned clang-format
cmake --preset dev && cmake --build --preset dev && ctest --preset dev
```

**Per issue:**
1. `git switch main && git pull && git switch -c <short-name>`
2. In Claude Code: **`/plan-issue <N>`**. It reads the issue, checks dependencies, asks the questions that matter, and posts an approved plan on the issue.
3. Implement **tests first** (see the posted plan), then the code.
4. **`/pre-push`**: senior review (+ adversarial review for physics/parallel code), local CI, docs check, commit message.
5. Commit, push, open the PR with `Closes #N`. Claude keeps the PR description current.
6. **Merge only when CI is green.** `main` is protected: all 5 checks (format, gcc, clang, asan, tsan) must pass. Reviews aren't required.

**Rules that bite** ([CLAUDE.md](../CLAUDE.md) has all of them):
- Claude never commits, branches or merges. You do all git work (it can edit issues and PR descriptions).
- Never touch `src/gravity_serial.cpp` beyond bug fixes: it's the reference everything is tested against.
- Compare floating-point results with **relative tolerances**, never `==`. Don't loosen a tolerance to make a test pass; find out why it fails.
- Plain C-style C++ (structs, arrays, free functions), Struct-of-Arrays particle data.
- Every OpenMP region uses `default(none)`.

---

## 3. Contracts your code must keep

These are fixed in the headers from #6. Breaking one causes silent physics bugs, not crashes.

| Contract | Where | Why it matters |
|---|---|---|
| **Gravity overwrites `ax/ay/az`; contact *adds* to them** | `forces.hpp` | `compute_contact_bruteforce()` must use `+=`. Overwriting would erase gravity. |
| `compute_accelerations(p, params, backend)` = gravity, then contact if `params.contact_enabled` | `forces.cpp` | The one place force order is defined. Relaxation and the sim loop both call it (via `leapfrog_step`). |
| **Leapfrog needs fresh accelerations:** call `compute_accelerations()` before the first `leapfrog_step()` and after anything else changes positions, velocities, masses, particle count or force settings | `integrator.hpp` | Otherwise the first step uses stale forces. |
| **Relax each planet on its own, at rest**, then give it its velocity and append it to the main `Particles` | `init.hpp` | If it's moving, bulk motion counts as kinetic energy and relaxation never "settles". If both planets are present, they pull on each other. |
| `make_planet()` **appends** `spec.n` particles and sets `Particles::body = spec.body` | `init.hpp` | Two planets go into one `Particles`; `body` colors them in renders. |
| All physics settings go in `SimParams` (add fields freely, with defaults) | `params.hpp` | Adding a field never changes a function signature. Defaults mean "unset", not physical values. |
| Frames: `%.17g` for doubles; `write_frame()` does **not** create the folder; `read_frame()` resizes `p` | `io.hpp` | `%.17g` makes the round-trip test exact. The sim loop (#14) creates the folder. |
| `load_config()`: unknown key = error; keys missing from the file keep the defaults already in `cfg` | `config.hpp` | A typo must not silently become a default. |
| `Particles` has 12 arrays (`x y z vx vy vz ax ay az mass radius body`); `resize()` grows all of them | `particles.hpp` | Never push to one array alone. |

---

## 4. Collision parameters that worked (preview)

Simulation units, G = 1. These ran stable (no NaN, energy only ever decreased) for 6,000 steps with 1,458 particles.

| Setting | Value | How it was chosen |
|---|---|---|
| Particle mass | `1e-3` | ~1,000 particles per radius-1 planet → planet mass ≈ 1 |
| Lattice spacing | `0.161` | simple cubic grid inside radius 1 gives ~1,000 particles; particles start just touching |
| Particle radius | `spacing / 2 = 0.0805` | |
| Contact stiffness `k` | `20` | gravity squeezes overlaps to a few % of the radius |
| Contact damping `c` | `0.05` | coefficient of restitution ~0.5 ("rubble, not rubber") |
| Timestep `dt` | `0.002` | contact spring period `2π·sqrt(m_eff/k)` with `m_eff = m/2` is ≈ 0.031, so ~15 steps per oscillation (PLAN's rule: ≥ 10) |
| Gravity softening | `0.05` | ≈ particle radius |
| Relaxation | 1,500 steps, multiply velocities by `exp(-5·dt)` after each step | settled both planets (KE ~1e-4) |

Formulas, if you change something:
- Spring period: `T = 2π·sqrt(m_eff / k)`, keep `dt ≤ T/10`.
- Damping for a target restitution `e`: `ζ = -ln(e) / sqrt(π² + ln²(e))`, `c = 2ζ·sqrt(m_eff·k)` (e = 0.5 → c ≈ 0.043).
- Escape speed of the pair: `v_esc = sqrt(2·G·(M1 + M2) / (R1 + R2))` ≈ 1.29 for the preview planets. Impact at 0.9 merged; at 2.6 (2x escape) it sprayed debris.

**Preview code (never reviewed or tested; needs the tests your issues list):**

Contact force, `#12` (pair-twice, adds to acceleration):
```cpp
// for each i, for each j != i:
double dx = x[i] - x[j], dy = y[i] - y[j], dz = z[i] - z[j];
double reach = r[i] + r[j], d2 = dx*dx + dy*dy + dz*dz;
if (d2 >= reach * reach || d2 == 0.0) continue;          // not touching
double d = sqrt(d2), nx = dx/d, ny = dy/d, nz = dz/d;     // unit vector j -> i
double overlap = reach - d;
double vn = (vx[i]-vx[j])*nx + (vy[i]-vy[j])*ny + (vz[i]-vz[j])*nz;
double f = k*overlap - c*vn;                              // spring pushes apart, damper resists approach
fx += f*nx; fy += f*ny; fz += f*nz;                       // then ax[i] += fx / m[i], etc.
```

Lattice planet, `#9`: loop a cubic grid `i, j, k ∈ [-ceil(R/s), ceil(R/s)]`, keep points with `x²+y²+z² ≤ R²`, add jitter of ±1% of the spacing from a seeded RNG (so it isn't perfectly symmetric). Note: `make_planet` takes `n` in `PlanetSpec`; a lattice gives "whatever fits", so either derive spacing from `n` (`s = (4/3·π·R³ / n)^(1/3)`) and accept a count near `n`, or fill randomly with rejection of overlaps. Decide in `/plan-issue 9`; the issue's test wants "correct count".

Relaxation, `#15`: `compute_accelerations()` once, then loop `leapfrog_step()` + damp velocities until kinetic energy < threshold or `max_steps`. `relax_planet` needs the threshold and step limit from somewhere: likely new `Config`/`SimParams` fields (fine to add).

---

## 5. Per-issue notes

- **#9 planets:** "same seed gives identical output": `std::mt19937` is identical everywhere, but `std::uniform_real_distribution` can differ between standard libraries (gcc vs clang/macOS). Fine for tests on one machine; if results must match across machines, map the raw `mt19937` output to `[0,1)` yourself. Test "max initial overlap < 10% of particle radius" is cheap: check every pair for small N.
- **#10 frames:** one CSV per frame, `frame_00042.csv`, columns `id,x,y,z,body`. Round-trip test: write, `read_frame`, compare. With `%.17g` the values come back bit-identical.
- **#11 render:** matplotlib 3D scatter → PNGs → ffmpeg (both installed on Pearce's machine). Color by `body`. `matplotlib.animation.FFMpegWriter` writes an MP4 directly without temporary PNGs. Pin versions in `scripts/requirements.txt`.
- **#12 contact:** tests from the issue: separated → zero force; overlapping → repulsive, equal and opposite; head-on bounce loses energy with damping, keeps it without. Check `+=` (contract above). The O(N²) brute force is expected here; the grid comes in #23.
- **#13 config:** `key = value`, unknown key → `false` + the offending line. Planets need a numbered or sectioned syntax (`planet0.radius = 1.0`?); decide in `/plan-issue 13`.
- **#15 relax:** see section 4. Test: KE below threshold after relaxing, radius changed < 10%.
- **#18 video + energy plot:** `planet_sim --config <file> --out frames/<run>` writes `frames/<run>/energy.csv` with columns `step,t,kinetic,potential,total,px,py,pz` (one row per frame). Total energy must never rise, only drop from damping. Rising energy means `dt` is too big or a bug.
- **#20 benchmark harness:** `planet_sim --scenario cloud --n <N> --backend serial|omp --steps <k> --bench` prints exactly one line, e.g. `bench backend=omp n=5000 threads=6 steps=20 seconds_per_step=0.0121`. Set threads with `OMP_NUM_THREADS`, and pin with `OMP_PROC_BIND=close OMP_PLACES=cores` (see notes.md on why). Works today, no other issues needed.
- **`Config` relaxation fields (added in #14):** `relax_ke_threshold` and `relax_max_steps` (0 = skip). #13 parses them, #15 tunes them. `build_planets()` in `src/sim.cpp` already calls `make_planet` at rest, then `relax_planet` if `relax_max_steps > 0`, then places the planet.

---

## 6. Lessons already learned (don't repeat them)

- **CI's compilers are older than Arch's** (ubuntu-24.04: gcc 13, clang 18, libomp 18). Something passing locally can fail in CI. That's why CI gates merging.
- **Benchmarks lie unless careful:** CPU clock speed drifts (boost, power saving, other programs). `bench/bench_gravity.cpp` interleaves configurations and takes medians; copy that approach.
- **Tests must be able to fail:** reviews deliberately broke the code (swapped axes, dropped pairs, energy leaks) to prove the tests catch it. Pick test values that make bugs visible (e.g. a separation like (2,3,6) whose components all differ).
- **Thread sanitizer needs Archer** for OpenMP: `OMP_TOOL_LIBRARIES=$(find /usr/lib -name 'libarcher.so*' | head -n 1) ctest --preset tsan`.
- **Design decisions go in [docs/notes.md](notes.md)** with numbers: it becomes the presentation.
