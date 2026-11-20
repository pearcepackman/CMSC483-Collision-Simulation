# CLAUDE.md

Planet collision N-body simulator (C++17, OpenMP, HIP on AMD GPUs). The plan is [PLAN.md](PLAN.md); tooling decisions are in [docs/tooling.md](docs/tooling.md). Primary dev platform is Linux; the GPU machine is Pearce's PC (Radeon RX 9060 XT, ROCm). macOS lacks ROCm and reliable TSan.

## Commands

```bash
cmake --preset dev && cmake --build --preset dev     # configure + build
ctest --preset dev                                    # tests (excludes `gpu` label)
OMP_NUM_THREADS=4 ctest --preset dev                  # tests with threads
cmake --preset asan && cmake --build --preset asan && ctest --preset asan   # memory/UB checks
cmake --preset tsan && cmake --build --preset tsan && ctest --preset tsan   # race checks (Linux)
git ls-files '*.cpp' '*.hpp' '*.hip' | xargs clang-format --dry-run --Werror
.venv/bin/python -m unittest discover scripts         # Python script tests
.venv/bin/python scripts/render.py frames/<run> -o run.mp4   # frames to video
```

Formatter: `python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt` (pinned clang-format). Scripts: `.venv/bin/pip install -r scripts/requirements.txt` (pinned numpy, matplotlib, imageio-ffmpeg).

## Git: what Claude may do

Pearce allows Claude to do the everyday branch work so a feature goes from plan to an open PR without hand-offs:

- **Allowed:** `git fetch`/`pull` on `main`, creating a feature branch from an up-to-date `main` (name it after the issue, e.g. `59-energy-tests`), `git add` of the files the work changed, commits, pushing that branch, opening a PR with `gh pr create`, and keeping its description current (one `Closes #N` line per issue it finishes, plus a short summary). Also issues, milestones, labels, and comments on issues and PRs.
- **Commit messages and PR text** follow the repo's format (`Added:` / `Edited:` / `Added/Edited:` / `Fixed:`) and **never mention Claude, Claude Code or AI**, and never add `Co-Authored-By` or "Generated with" lines.
- **Never, unless asked in the current message:** merging or closing PRs (Pearce reviews and merges), pushing to `main`, force-pushing, rebasing, resetting, stashing, tagging, deleting branches, or changing branch protection. Only commit after the local checks in `/pre-push` pass.

`.claude/settings.json` still asks before the risky commands.

## Rules

- **Never delete or "optimize" `gravity_serial.cpp`.** It's the reference every other backend is tested against.
- Every new backend or physics change needs a test, and all tests must pass before a commit.
- Don't change physics constants or test tolerances just to make tests pass. Flag it instead.
- Keep particle data in Struct-of-Arrays layout (`src/particles.hpp`).
- Every OpenMP parallel region uses `default(none)`, so each variable is explicitly shared or private.
- Compare backends with relative tolerances, never `==`, and only over a few steps (gravity is chaotic).
- Tests that need a GPU get the CTest label `gpu`.
- Keep each PR to one feature; it may close several related issues in the same milestone (one `Closes #N` line per issue). Run `/pre-push` before pushing.
- Write plain C-style C++: structs, arrays, free functions. No clever templates.
- Keep the progress log current: every PR that changes what the project can do or measure adds a dated entry to [docs/progress.md](docs/progress.md) (what was built, what it shows about parallel computing, the numbers), with a screenshot in `docs/media/` or a plot in `docs/figures/` when there's something to see. Take viewer screenshots on the real display (`--frames N --screenshot docs/media/<name>.png`), not under Xvfb, whose software rendering skews the timings. Numbers in captions must match the image. Design decisions and full measurements still go in `docs/notes.md`.
- Docs are read by the course staff and anyone who finds the repo, so write them the way a student engineer would: plain sentences and specific numbers. No em-dashes, no marketing words ("seamless", "robust", "powerful", "journey", "showcase"), no "**Label:** text" bullet lists, no emoji. This is about style only: keep the honest notes on how we use Claude Code (PLAN.md section 9, tooling.md, kickstart.md), since the course may require disclosing AI use.
