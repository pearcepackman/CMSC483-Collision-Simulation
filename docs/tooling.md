# Tooling & CI Decisions

How the repo is built, checked, and tested, and why. Complements [PLAN.md](../PLAN.md) sections 3, 5 and 8.

**Primary dev platform: Linux (x86_64).** macOS works for editing and serial/OpenMP runs, but CUDA and reliable race detection are Linux-only. CI runs on Linux.

---

## 1. Language: C++17, written C-style. No separate C code.

All `.cpp` files, written with plain structs, arrays and free functions. Reasons:

- CUDA (`.cu`) is a C++ dialect, and GoogleTest is C++. A C core would still need C++ around it.
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

- **CMake ≥ 3.24**, **Ninja** generator (faster than Make, better parallel builds).
- **Presets** we'll define:

  | Preset | Compiler | Purpose |
  |---|---|---|
  | `dev` | default | Day-to-day: RelWithDebInfo, `-Werror` |
  | `release` | default | Benchmarks: `-O3 -march=native` |
  | `asan` | clang | Debug + AddressSanitizer + UBSan (memory bugs, UB) |
  | `tsan` | clang + libomp | ThreadSanitizer (data races in OpenMP code) |
  | `cuda` | nvcc | `-DENABLE_CUDA=ON` |

- **GoogleTest** via `FetchContent`, pinned to **v1.17.0** (latest confirmed release; needs C++17). Pin a tag, never `main`.
- Export `compile_commands.json` (`CMAKE_EXPORT_COMPILE_COMMANDS=ON`) so editors (clangd / VS Code) understand the code.

### Linux setup (Ubuntu/Debian)

```bash
sudo apt install build-essential cmake ninja-build clang libomp-dev python3-venv
pip install clang-format==<pinned>   # pinned version: see section 3
# CUDA: install the toolkit only if the machine has an NVIDIA GPU (nvidia-smi works)
```

### macOS setup (secondary)

```bash
brew install cmake ninja libomp
export OpenMP_ROOT=$(brew --prefix)/opt/libomp   # Apple clang has no OpenMP; this points CMake at Homebrew's
```

On macOS, `gcc` is actually Apple clang. There's no CUDA and no usable TSan for OpenMP code.

### Remote Linux box

If the Linux machine is remote (lab box, HPCF), use **VS Code Remote-SSH**. The Claude Code extension runs on the remote side, so builds, tests and `/pre-push` all run on Linux.

---

## 3. Formatting & linting

- **clang-format**: one `.clang-format` at the root (based on LLVM style, 100-column limit). **Pin the version** (e.g. via the `clang-format` pip package) and use the same version in CI. Different clang-format versions format code differently, which would cause CI failures that look random.
- **clang-tidy**: start with a small check set and grow it:
  `bugprone-*`, `performance-*`, `openmp-use-default-none`, `openmp-exception-escape`, `concurrency-*`.
  `openmp-use-default-none` matters a lot here: it forces every OpenMP loop to say which variables are shared and which are private, and getting that wrong is the #1 source of OpenMP races.
- **Compiler warnings** `-Wall -Wextra -Wshadow -Wconversion -Werror` on our code only, not on GoogleTest.

---

## 4. CI (GitHub Actions)

Runs on every push and PR. GitHub's hosted runners have **no GPU**, so CUDA is compile-only in CI.

| Job | What it does | Catches |
|---|---|---|
| `format` | `clang-format --dry-run --Werror` (pinned version) | Unformatted code |
| `build-test` | Matrix of gcc and clang: build with `-Werror`, then `ctest -LE gpu` with `OMP_NUM_THREADS=1` **and** `4` | Compile errors, test failures, races that only show up with several threads |
| `asan` | clang Debug + ASan + UBSan, full test suite | Out-of-bounds writes, use-after-free, signed overflow, misaligned access |
| `tsan` | clang + libomp (Archer), `TSAN_OPTIONS=ignore_noninstrumented_modules=1` | Data races in OpenMP loops. **Start as non-blocking** (`continue-on-error`) until we confirm it's clean, then make it required. |
| `tidy` | clang-tidy on changed files | Missing `default(none)`, common bug patterns |
| `cuda-compile` | Inside `nvidia/cuda:<ver>-devel-ubuntu24.04`, `-DENABLE_CUDA=ON`, explicit `CMAKE_CUDA_ARCHITECTURES` (CMake can't detect a GPU that isn't there) | CUDA syntax/type errors |
| `smoke` | ~500 particles, 200 steps: fail on NaN or energy drift above threshold | "It compiles but the physics exploded" |

Extras:
- `concurrency: cancel-in-progress` so a new push cancels the older run.
- Upload `bench/` CSVs and smoke-test energy logs as artifacts on failure.
- Labels: tests needing a GPU get the `gpu` CTest label. GPU tests run manually on a GPU machine.

### Secret scanning

There's little to leak in this project (no API keys). The real risks are an **SSH key or HPCF password** pasted into a script, and **huge files** (frames, videos, `build/`) committed by accident. So:

- Enable GitHub **secret scanning + push protection** in repo settings (free on public repos).
- `/pre-push` runs `gitleaks` if installed (otherwise a regex check), plus a large-file/junk check.
- A dedicated CI secrets job isn't worth it.

### Branch protection

Protect `main`: require PRs, require `format`, `build-test`, `asan`, `cuda-compile` and `smoke` to pass.

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
- [stden/cpp-cuda image (nvcc builds without a GPU)](https://hub.docker.com/r/stden/cpp-cuda)
