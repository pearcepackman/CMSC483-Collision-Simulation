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

_Upcoming: pair-twice (no races) vs Newton's-3rd-law with atomics (half the work); OpenMP `schedule(static)` vs `dynamic`; HIP naive vs shared-memory tiled kernel; `float` vs `double` (consumer GPUs run `double` many times slower)._

## Contact

_Upcoming: uniform grid (cell list) construction in parallel; load balance across uneven cells._

## Scaling results

_Upcoming: runtime vs N (log-log), strong scaling, efficiency, Amdahl's-law fit. Record CPU/GPU model and core count with every result._
