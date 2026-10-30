# Tooling & CI Decisions

How the repo is built, checked, and tested, and why. Complements [PLAN.md](../PLAN.md) sections 3, 5 and 8.

**Primary dev platform: Linux (x86_64).** macOS works for editing and serial/OpenMP runs, but the GPU backend (ROCm/HIP) and reliable race detection are Linux-only. CI runs on Linux. The GPU machine is Pearce's PC (AMD Radeon RX 9060 XT, `gfx1200`).

---

## 1. Language: C++17, written C-style. No separate C code.

All `.cpp` files, written with plain structs, arrays and free functions. Reasons:

- HIP (`.hip`) is a C++ dialect, and GoogleTest is C++. A C core would still need C++ around it.
- `std::vector` gives bounds-checkable, auto-freed arrays, so there's no `malloc`/`free` bookkeeping.
- Mixing C and C++ means `extern "C"` glue and two sets of compiler flags, which buys us nothing here.

If we ever do want a `.c` file, CMake handles it (`project(... LANGUAGES C CXX)`). It's just not needed.

---

## 2. Build system: CMake + Ninja + presets

For a web developer, the closest mapping is:

| Web world | Here |
|---|---|
| `package.json` scripts | `CMakePresets.json` (named build configs) |
| `npm install` | CMake `FetchContent` (downloads GoogleTest at configure time) |
| `npm run build` | `cmake --preset dev && cmake --build --preset dev` |
| `npm test` / Jest | `ctest --preset dev` / GoogleTest |
| Prettier | `clang-format` |
| ESLint | `clang-tidy` + compiler warnings (`-Wall -Wextra -Werror`) |
| `node_modules/`, `dist/` | `build/` (gitignored, safe to delete any time) |

- **CMake ≥ 3.25**, **Ninja** generator (faster than Make, better parallel builds).
- **Presets** we'll define:

  | Preset | Compiler | Purpose |
  |---|---|---|
  | `dev` | default | Day-to-day: RelWithDebInfo, `-Werror` |
  | `release` | default | Benchmarks: `-O3 -march=native` |
  | `asan` | clang | Debug + AddressSanitizer + UBSan (memory bugs, UB) |
  | `tsan` | clang + libomp | ThreadSanitizer (data races in OpenMP code) |
  | `hip` | ROCm clang (HIP) + system C++ compiler | `-DENABLE_HIP=ON`, `CMAKE_HIP_ARCHITECTURES=gfx1200;gfx1036` (the RX 9060 XT and the Ryzen's integrated GPU: ROCm aborts a launch if any visible GPU lacks code, see docs/notes.md), compiler and ROCm root pinned to `/opt/rocm`. `ctest --preset hip` also runs the `gpu`-labelled tests (they skip if no GPU is visible) |

- **GoogleTest** via `FetchContent`, pinned to **v1.17.0** (latest confirmed release; needs C++17). Pin a tag, never `main`.
- Export `compile_commands.json` (`CMAKE_EXPORT_COMPILE_COMMANDS=ON`) so editors (clangd / VS Code) understand the code.

### Linux setup (Ubuntu/Debian)

```bash
sudo apt install build-essential cmake ninja-build clang libomp-dev python3-venv
pip install clang-format==<pinned>   # pinned version: see section 3
# GPU backend: only on a machine with a ROCm-supported AMD GPU.
# Arch: sudo pacman -S hip-runtime-amd rocminfo rocm-cmake   (~7 GB, mostly ROCm's LLVM;
#       the full rocm-hip-sdk adds math/ML libraries we don't use)
# Check: /opt/rocm/bin/rocminfo | grep -E "Marketing|gfx"   -> should list gfx1200
```

**ROCm on Arch, two gotchas:** `/etc/profile.d/rocm.sh` sets `ROCM_PATH` only in login shells, and without it ROCm's clang guesses its root as `/opt/rocm/lib` and can't find `hip/hip_runtime.h`. The `hip` preset pins `CMAKE_HIP_COMPILER`, `CMAKE_HIP_COMPILER_ROCM_ROOT` and `--rocm-path=/opt/rocm`, so CMake builds work from any shell; for running `hipcc` by hand, open a new login shell or pass `--rocm-path=/opt/rocm`. ROCm also lists the CPU's integrated GPU as a device; the backend picks the one with the most compute units among the architectures it was built for (see docs/notes.md). A different GPU needs its `gfx` name (from `rocminfo`) added to `CMAKE_HIP_ARCHITECTURES` in the preset.

### macOS setup (secondary)

```bash
brew install cmake ninja libomp
export OpenMP_ROOT=$(brew --prefix)/opt/libomp   # Apple clang has no OpenMP; this points CMake at Homebrew's
```

On macOS, `gcc` is actually Apple clang. There's no ROCm and no usable TSan for OpenMP code. As of Apple clang 17 / macOS 26.6, **ASan deadlocks at startup** (even on an empty program), so the `asan` preset only works on Linux. UBSan alone works.

### Remote Linux box

If the Linux machine is remote (e.g. working on Pearce's PC from a laptop), use **VS Code Remote-SSH**. The Claude Code extension runs on the remote side, so builds, tests and `/pre-push` all run on Linux.

---

## 3. Formatting & linting

- **clang-format**: one `.clang-format` at the root (based on LLVM style, 100-column limit). **Pin the version** (e.g. via the `clang-format` pip package) and use the same version in CI. Different clang-format versions format code differently, which would cause CI failures that look random.
- **clang-tidy**: start with a small check set and grow it:
  `bugprone-*`, `performance-*`, `openmp-use-default-none`, `openmp-exception-escape`, `concurrency-*`.
  `openmp-use-default-none` matters a lot here: it forces every OpenMP loop to say which variables are shared and which are private, and getting that wrong is the #1 source of OpenMP races.
- **Compiler warnings** `-Wall -Wextra -Wshadow -Wconversion -Werror` on our code only, not on GoogleTest.

---

## 4. CI (GitHub Actions)

Runs on every push and PR. GitHub's hosted runners have **no GPU**, so HIP is compile-only in CI.

| Job | What it does | Catches |
|---|---|---|
| `format` | `clang-format --dry-run --Werror` (pinned version) | Unformatted code |
| `build-test` | Matrix of gcc and clang: build with `-Werror`, then `ctest -LE gpu` with `OMP_NUM_THREADS=1` **and** `4` | Compile errors, test failures, races that only show up with several threads |
| `asan` | clang Debug + ASan + UBSan, full test suite | Out-of-bounds writes, use-after-free, signed overflow, misaligned access |
| `tsan` | clang + libomp + Archer (found with `find /usr/lib -name 'libarcher.so*'`, loaded via `OMP_TOOL_LIBRARIES`), `tsan` preset at 4 threads | Data races in OpenMP loops. **Blocking** (#21). It first runs a **canary**, [.github/ci/tsan_canary.cpp](../.github/ci/tsan_canary.cpp), a deliberately racy loop; if TSan doesn't report it, the job fails, so a broken race checker can't pass silently. Locally: `OMP_TOOL_LIBRARIES=$(find /usr/lib -name 'libarcher.so*' \| head -n 1) ctest --preset tsan`. |
| `tidy` | clang-tidy on changed files | Missing `default(none)`, common bug patterns |
| `hip-compile` | In the `rocm/dev-ubuntu-24.04:7.2.4` container (same ROCm as the dev PC): `hip` preset (explicit architectures, since CMake can't detect a GPU that isn't there), CPU tests with `-LE gpu`, and `planet_sim --backend hip` must exit 1 with "no usable GPU found" | HIP syntax/type errors and warnings (`-Werror` applies to HIP too), a HIP build breaking CPU code, a crash instead of a clean error without a GPU |
| `smoke` | ~500 particles, 200 steps: fail on NaN or energy drift above threshold | "It compiles but the physics exploded" |

Extras:
- `concurrency: cancel-in-progress` so a new push cancels the older run.
- Upload `bench/` CSVs and smoke-test energy logs as artifacts on failure.
- Labels: tests needing a GPU get the `gpu` CTest label. GPU tests run manually on a GPU machine.
- **CI's toolchain is older than Arch's:** ubuntu-24.04 has gcc 13, clang 18 and libomp 18, versus much newer versions on the dev machine. Local runs passing doesn't guarantee CI passes, so CI is the final check.
- **LeakSanitizer suppression** ([tests/lsan_suppressions.txt](../tests/lsan_suppressions.txt), loaded by the `asan` test preset): libomp 18 never frees its thread-team structures, and LSan reported that as a leak once tests changed the OpenMP thread count (PR #42). Only allocations inside `libomp.so` are ignored; leaks in our code still fail. LSan reads the file only when it finds a leak, so a typo in the path only shows up when something leaks.

### Secret scanning

There's little to leak in this project (no API keys). The real risks are an **SSH key or password** pasted into a script, and **huge files** (frames, videos, `build/`) committed by accident. So:

- Enable GitHub **secret scanning + push protection** in repo settings (free on public repos).
- `/pre-push` runs `gitleaks` if installed (otherwise a regex check), plus a large-file/junk check.
- A dedicated CI secrets job isn't worth it.

### Branch protection

**On (Oct 2026):** `main` only accepts changes whose CI passed. Required checks: `format`, `build-test (g++)`, `build-test (clang++)`, `asan` and `tsan`. It applies to admins too (no "merge anyway" override). No review approvals are required (team decision: only CI gates merging). Force-pushes to and deletion of `main` are blocked. When `hip-compile` (#24) and `smoke` (#17) land, add them to the required list (repo Settings → Branches, or `gh api -X PUT repos/{owner}/{repo}/branches/main/protection`). Emergency: an admin can temporarily turn protection off in Settings → Branches.

---

## 5. `.gitignore` essentials

```
build/
out/
compile_commands.json
*.mp4
*.gif
frames/
bench/raw/
.venv/
.DS_Store
```

Commit `bench/results/*.csv` (the curated results), not raw run output.

---

## 6. Pre-push workflow (Claude Code)

`/pre-push` (in `.claude/skills/pre-push/`) runs before every push:

1. Scope the diff against `main`
2. Senior review + adversarial review (subagents) → findings presented for sign-off
3. Fix the approved findings
4. Secrets + junk-file scan
5. Add missing tests
6. Run the local equivalent of CI (format, build, tests at 1 and 4 threads, ASan, TSan on Linux, smoke)
7. Propose a commit message: `Added:` / `Edited:` / `Added/Edited:` / `Fixed:` + ~50 chars

---

## Sources

- [CMake forum: finding OpenMP with clang on macOS](https://discourse.cmake.org/t/how-to-find-openmp-with-clang-on-macos/8860)
- [openCARP Q&A: CMake can't find OpenMP on macOS](https://opencarp.org/q2a/1515/cmake-could-not-find-openmp-on-macos)
- [Homebrew libomp formula](https://formulae.brew.sh/formula/libomp)
- [LLVM Archer (TSan support for OpenMP)](https://llvm.googlesource.com/llvm-project/+/dda73336ad22bd0b5ecda17040c50fb10fcbe5fb/openmp/tools/archer)
- [Archer slides (VI-HPS)](https://www.vi-hps.org/cms/upload/material/tw27/Archer.pdf)
- [Debian googletest tracker (1.18 not yet released)](https://tracker.debian.org/pkg/googletest)
- [Julia discourse: CUDA in GitHub CI (no GPUs on hosted runners)](https://discourse.julialang.org/t/how-to-use-cuda-in-github-ci/77855)
- [Phoronix: ROCm 7.0.2 officially supports the Radeon RX 9060 series](https://www.phoronix.com/news/AMD-ROCm-7.0.2-Released)
