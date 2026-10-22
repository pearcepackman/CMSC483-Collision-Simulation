---
name: pre-push
description: Pre-push gate for this repo. Scopes the diff against main, runs a senior review plus an adversarial review, gets the user's sign-off on findings, fixes approved ones, scans for secrets and junk files, adds missing tests, runs the local equivalent of CI, and proposes a commit message. Use when the user says /pre-push, "ready to push", "check before I push", or similar.
---

# Pre-push gate

Run these phases in order. **Phase 3 is a hard stop:** present findings and end your turn so the user can sign off. Don't fix anything before they answer.

Read `CLAUDE.md` and `docs/tooling.md` first for project rules and build commands. They override anything generic here.

## Phase 1: Scope the diff

```bash
git fetch origin main --quiet 2>/dev/null || true
BASE=$(git merge-base HEAD origin/main 2>/dev/null || git merge-base HEAD main)
git diff --stat $BASE            # committed + uncommitted changes vs main
git ls-files --others --exclude-standard   # untracked files (new files count too)
git branch --show-current
```

Sort the changed files into: **physics/engine** (`src/`), **parallel code** (anything with `#pragma omp`, `.cu`, atomics), **tests**, **build/CI** (`CMakeLists.txt`, presets, `.github/`), **scripts** (`scripts/*.py`), **docs**.

Tell the user in 2–3 lines what's in scope. If the branch is `main`, warn that `main` should be protected and suggest a branch name.

## Phase 2: Review

**Trivial diff** (docs-only, or under ~30 changed lines with no parallel code): review it yourself and skip the subagents.

**Otherwise**, launch two subagents **in parallel, in a single message** (Agent tool, `general-purpose`). Give each the base SHA, the file list, and tell it to read the full diff with `git diff $BASE` plus untracked files. Each returns findings as: severity (`blocker` / `should-fix` / `nit`), `file:line`, the problem, a concrete failure scenario, and a proposed fix.

**Agent A: Senior reviewer.** Review like a senior HPC/C++ engineer. Checklist:
- **OpenMP:** every shared write is safe (reduction, private, or per-thread buffer); `default(none)` used; no hidden shared temporaries; scheduling choice makes sense; no false sharing on hot arrays.
- **CUDA:** bounds check on `idx < n`; every API call and kernel launch is error-checked; `__syncthreads()` isn't inside divergent branches; host/device copies are correct and not redundant.
- **Numerics:** softening applied; no division by zero or `sqrt` of a negative; `float` vs `double` is deliberate; no accumulation order that silently changes serial results.
- **Project rules:** `gravity_serial.cpp` is not deleted or "optimized"; particle data stays Struct-of-Arrays; test tolerances and physics constants are not loosened; backends share one interface.
- **C++ hygiene:** no raw `new`/`malloc` without a reason; no out-of-bounds indexing; integer overflow on `N*N` (use `size_t`/`int64_t`).
- **CMake/CI:** no hardcoded paths; CUDA stays optional; warnings aren't silenced to make the build pass.
- Simplicity: C-style and readable beats clever.

**Agent B: Adversarial reviewer.** Assume the change is broken and try to prove it. For each suspicion, build a **concrete** input or thread interleaving that breaks it: N=0, N=1, two particles at the same position, huge velocities, `OMP_NUM_THREADS` > N, a non-multiple-of-block-size N for CUDA. Where cheap, confirm by writing a throwaway test or program in the scratchpad (not the repo) and running it. Report only what you could demonstrate or argue concretely, and mark each `CONFIRMED` (reproduced) or `PLAUSIBLE` (argued, not run). Also attack the tests themselves: would they still pass if the code were wrong?

When both return, merge duplicates, drop anything you can't back up, and rank most severe first.

## Phase 3: Present findings and STOP

Show a numbered table:

| # | Sev | Where | Problem | Proposed fix |
|---|---|---|---|---|

Under it, one line per item explaining the failure scenario in plain language. The user is new to C++/HPC, so briefly explain jargon (e.g. "false sharing: two threads writing neighbouring memory slow each other down").

Then ask: **"Which should I fix? Reply e.g. `all`, `1 3 4`, `all but 5`, or `none`."**

**End your turn here.** Don't continue until the user answers.

If there are no findings, say so and continue to Phase 4 without stopping.

## Phase 4: Fix approved findings

Fix only what the user approved. If a fix changes behaviour, write the failing test first, then fix. Never change a physics constant or test tolerance to make something pass. Flag it to the user instead.

## Phase 5: Secrets and junk scan

Scan the changed and untracked files only:

```bash
# Preferred, if installed:
gitleaks git --log-opts="$BASE..HEAD" --no-banner --redact
gitleaks dir . --no-banner --redact     # also covers uncommitted files
```

If `gitleaks` isn't installed, grep the diff for: `-----BEGIN .*PRIVATE KEY`, `ssh-rsa `/`ssh-ed25519 ` with key material, `password\s*=`, `token\s*=`, `AKIA[0-9A-Z]{16}`, `ghp_[A-Za-z0-9]{36}`, `.env` files, hostnames combined with credentials (HPCF logins).

**Junk check** (more likely than secrets in this repo). Flag any of these about to be committed: `build/`, `out/`, `compile_commands.json`, `*.mp4`, `*.gif`, `*.png` frame dumps, `bench/raw/`, binaries or object files, or any file over 1 MB. Suggest `.gitignore` entries.

Report findings. Ask before deleting or untracking anything.

## Phase 6: Tests for what changed

Per `CLAUDE.md`: every new backend or physics change needs a test. For each behaviour change in `src/`, check a test covers it. Typical needs:
- New/changed backend → add a case to `test_backends_match` (single force evaluation, relative tolerance, not `==`).
- Integrator/force change → conservation tests (energy, momentum, angular momentum) still meaningful.
- New code path → NaN/Inf guard exercised.
- GPU-only tests → CTest label `gpu`.

Docs, scripts and CI-only changes usually need no new C++ tests; say so explicitly. Write the missing tests, then go on to Phase 7.

## Phase 7: Run the local equivalent of CI

Use presets if `CMakePresets.json` exists; otherwise use the commands in `CLAUDE.md`. If there's no `CMakeLists.txt` yet (before milestone 0), say the build/test phase is skipped and why.

Run in this order and stop at the first failure:

1. **Format:** `clang-format --dry-run --Werror` on changed `*.cpp *.hpp *.cu *.cuh`. If it fails, run `clang-format -i` on those files and tell the user.
2. **Build:** `cmake --preset dev && cmake --build --preset dev` (warnings are errors).
3. **Tests:** run `ctest --preset dev -LE gpu --output-on-failure` twice, with `OMP_NUM_THREADS=1` and `OMP_NUM_THREADS=4`.
4. **ASan/UBSan:** `asan` preset build + `ctest -LE gpu`.
5. **TSan** (Linux only, only if the diff touches parallel code): `tsan` preset with `TSAN_OPTIONS=ignore_noninstrumented_modules=1`, `OMP_NUM_THREADS=4`.
6. **CUDA:** if `nvcc` exists, build the `cuda` preset. If `nvidia-smi` also works, run `ctest -L gpu`. Otherwise say it's skipped (CI compiles it).
7. **Smoke test:** the tiny collision run (see CI `smoke` job). Fail on NaN or energy drift.
8. **Python:** if `scripts/` changed, at least `python3 -m py_compile` on the changed scripts.

Platform rules: on macOS (`uname` = Darwin), skip TSan and CUDA and say so. If OpenMP isn't found, set `OpenMP_ROOT=$(brew --prefix)/opt/libomp`. Report a skipped step as skipped. Never report it as passed.

If a failure is caused by the diff, fix it and re-run from step 1. If it isn't caused by the diff, or fixing it means touching tolerances or constants, stop and tell the user.

## Phase 8: Summary and commit message

Give a short summary as a checklist: review (N found / N fixed / N skipped), secrets/junk, tests added, and each CI step (passed / failed / skipped + reason).

Then propose **one** commit subject line in exactly one of these forms:

```
Added: <summary>
Edited: <summary>
Added/Edited: <summary>
Fixed: <summary>
```

- `Added:` only new files/features. `Edited:` only changes to existing code/docs. `Added/Edited:` both. `Fixed:` the main point is a bug fix (wins over the others).
- Summary is **around 50 characters**, specific, no trailing period (e.g. `Added: OpenMP gravity backend with match test`).
- Put it in a code block so it's easy to copy.

Don't commit or push unless the user asks. If they ask you to commit, use that subject line.
