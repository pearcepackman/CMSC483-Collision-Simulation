# CLAUDE.md

Planet collision N-body simulator (C++17, OpenMP, CUDA). The plan is [PLAN.md](PLAN.md); tooling decisions are in [docs/tooling.md](docs/tooling.md). Primary dev platform is Linux; macOS lacks CUDA and reliable TSan.

## Commands

```bash
cmake --preset dev && cmake --build --preset dev     # configure + build
ctest --preset dev                                    # tests (excludes `gpu` label)
OMP_NUM_THREADS=4 ctest --preset dev                  # tests with threads
cmake --preset asan && cmake --build --preset asan && ctest --preset asan   # memory/UB checks
cmake --preset tsan && cmake --build --preset tsan && ctest --preset tsan   # race checks (Linux)
git ls-files '*.cpp' '*.hpp' '*.cu' '*.cuh' | xargs clang-format --dry-run --Werror
```

Formatter: `python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt` (pinned clang-format).

## Git: hard rule

**IMPORTANT: Never change git or pull-request state unless the user asks for that specific action in their current message.** No commits, staging, branches, checkouts/switches, merges, rebases, resets, stashes, tags, pushes or pulls, and no creating, editing, merging, closing or commenting on PRs (including through `gh api`). Approving a plan, a review finding, or a skill's output is **not** permission; ask separately. Instead, say what you'd run and let the user run it or tell you to.

Allowed without asking: reading state (`git status`, `log`, `diff`, `show`, `fetch`; `gh pr view`, `gh run`, `gh issue view/list`), and GitHub planning: creating and editing issues, milestones and labels, and commenting on issues. `.claude/settings.json` backs this up with permission prompts.

## Rules

- **Never delete or "optimize" `gravity_serial.cpp`.** It's the reference every other backend is tested against.
- Every new backend or physics change needs a test, and all tests must pass before a commit.
- Don't change physics constants or test tolerances just to make tests pass. Flag it instead.
- Keep particle data in Struct-of-Arrays layout (`src/particles.hpp`).
- Compare backends with relative tolerances, never `==`, and only over a few steps (gravity is chaotic).
- Tests that need a GPU get the CTest label `gpu`.
- Keep each PR to one milestone task. Run `/pre-push` before pushing.
- Write plain C-style C++: structs, arrays, free functions. No clever templates.
