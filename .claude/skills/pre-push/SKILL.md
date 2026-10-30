---
name: pre-push
description: Pre-push gate for this repo. Scopes the diff against main, runs a senior review (plus an adversarial review for parallel/physics code, or on request), gets the user's sign-off on findings, fixes approved ones, scans for secrets and junk files, adds missing tests, updates stale docs (README status, design notes, plan), runs the local equivalent of CI, and proposes a commit message. Use when the user says /pre-push, "ready to push", "check before I push", or similar.
argument-hint: "[adversarial | no-adversarial]"
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

Sort the changed files into: **physics/engine** (`src/`), **parallel code** (anything with `#pragma omp`, `.hip`, atomics), **tests**, **build/CI** (`CMakeLists.txt`, presets, `.github/`), **scripts** (`scripts/*.py`), **docs**.

Tell the user in 2–3 lines what's in scope. If the branch is `main`, warn that `main` should be protected and suggest a branch name.

**Find the issue(s) this work is for.** A PR may close several issues (see `CLAUDE.md`), so collect all of: a number at the start of the branch name (`6-module-interfaces`), every `Closes #N` / `Fixes #N` line in the branch's open PR (`gh pr view --json body -q .body`), and any issue the user named. If none turn up, ask the user. Load each with `gh issue view N --comments`, including any plan posted by `/plan-issue`. Phase 3 checks the diff against all of them together.

## Phase 2: Review

Arguments: `$ARGUMENTS`

**Trivial diff** (docs-only, or under ~30 changed lines with no parallel code): review it yourself and skip the subagents.

**Otherwise**, launch Agent A. Decide whether to also launch Agent B (the adversarial reviewer is slow and uses a lot of usage):

| Arguments contain | Agent B |
|---|---|
| `adversarial` | Always run it |
| `no-adversarial` | Never run it |
| neither (default) | Run it only if the diff touches **parallel code** (`#pragma omp`, `.hip`, atomics) or **physics** in `src/` (gravity, contact, integrator, planet init). Skip it for setup, build, CI, docs, scripts and diagnostics-only changes. |

Say in one line whether Agent B is running and why (e.g. "Adversarial: skipped, no parallel/physics code; run `/pre-push adversarial` to force it").

When both run, launch them **in parallel, in a single message** (Agent tool, `general-purpose`). Give each the base SHA, the file list, and tell it to read the full diff with `git diff $BASE` plus untracked files. Each returns findings as: severity (`blocker` / `should-fix` / `nit`), `file:line`, the problem, a concrete failure scenario, and a proposed fix.

**Agent A: Senior reviewer.** Review like a senior HPC/C++ engineer. Checklist:
- **OpenMP:** every shared write is safe (reduction, private, or per-thread buffer); `default(none)` used; no hidden shared temporaries; scheduling choice makes sense; no false sharing on hot arrays.
- **HIP (GPU):** bounds check on `idx < n`; every `hip*` API call and kernel launch is error-checked (`hipGetLastError` after launches); `__syncthreads()` isn't inside divergent branches; host/device copies are correct and not redundant; `float` vs `double` on the GPU is deliberate.
- **Numerics:** softening applied; no division by zero or `sqrt` of a negative; `float` vs `double` is deliberate; no accumulation order that silently changes serial results.
- **Project rules:** `gravity_serial.cpp` is not deleted or "optimized"; particle data stays Struct-of-Arrays; test tolerances and physics constants are not loosened; backends share one interface.
- **C++ hygiene:** no raw `new`/`malloc` without a reason; no out-of-bounds indexing; integer overflow on `N*N` (use `size_t`/`int64_t`).
- **CMake/CI:** no hardcoded paths; HIP stays optional; warnings aren't silenced to make the build pass.
- **Test and CI gaps (always report these as findings):** for every new or changed behaviour (function, flag, exit code, error message, backend, build option, executable), name the test that would fail if it broke. If none would, that's a finding. Also check that **CI runs** that test: a test in a preset or label CI never builds/runs protects nothing. Flag tests that pass for the wrong reason (e.g. two backends that give identical bits, so a comparison can't detect a mix-up) and new artifacts CI never exercises (a new executable, preset, script or runtime mode).
- Simplicity: C-style and readable beats clever.

**Agent B: Adversarial reviewer.** Assume the change is broken and try to prove it. For each suspicion, build a **concrete** input or thread interleaving that breaks it: N=0, N=1, two particles at the same position, huge velocities, `OMP_NUM_THREADS` > N, a non-multiple-of-block-size N for the GPU kernel. Where cheap, confirm by writing a throwaway test or program in the scratchpad (not the repo) and running it. Report only what you could demonstrate or argue concretely, and mark each `CONFIRMED` (reproduced) or `PLAUSIBLE` (argued, not run). Also attack the tests themselves: would they still pass if the code were wrong?

When both return, merge duplicates, drop anything you can't back up, and rank most severe first.

## Phase 3: Present findings and STOP

Show a numbered table:

| # | Sev | Where | Problem | Proposed fix |
|---|---|---|---|---|

Under it, one line per item explaining the failure scenario in plain language. The user is new to C++/HPC, so briefly explain jargon (e.g. "false sharing: two threads writing neighbouring memory slow each other down").

Always add a **Test & CI coverage** table, even when the review found nothing else: one row per new or changed behaviour → the test that catches it breaking (or **none**) → the CI job that runs that test (or **not in CI**). Every **none** / **not in CI** row is also a `should-fix` in the findings table, with the test or CI step to add (or the open issue that will add it, e.g. #17 for the smoke test). The goal: CI keeps up with the code, so `main` stays protected as features land.

If an issue was found in Phase 1, add an **Issue coverage** table: each "Done when" box → done / missing / not checkable here (e.g. "reviewed by Kyle"), plus any changes outside the issue's scope or posted plan. Missing boxes and scope creep also go in the findings table as `should-fix`.

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

If `gitleaks` isn't installed, grep the diff for: `-----BEGIN .*PRIVATE KEY`, `ssh-rsa `/`ssh-ed25519 ` with key material, `password\s*=`, `token\s*=`, `AKIA[0-9A-Z]{16}`, `ghp_[A-Za-z0-9]{36}`, `.env` files, hostnames combined with credentials (SSH logins).

**Junk check** (more likely than secrets in this repo). Flag any of these about to be committed: `build/`, `out/`, `compile_commands.json`, `*.mp4`, `*.gif`, `*.png` frame dumps, `bench/raw/`, binaries or object files, or any file over 1 MB. Suggest `.gitignore` entries.

Report findings. Ask before deleting or untracking anything.

## Phase 6: Tests for what changed

Write the tests and CI steps for every gap the user approved in Phase 3 (the Test & CI coverage table), then re-check the table: each row should now name a test and a CI job. Where cheap, prove a new test bites by breaking the code on purpose (a quick mutant in the scratchpad or a temporary edit you revert) and watching the test fail.

Per `CLAUDE.md`: every new backend or physics change needs a test. For each behaviour change in `src/`, check a test covers it. Typical needs:
- New/changed backend → add a case to `test_backends_match` (single force evaluation, relative tolerance, not `==`).
- Integrator/force change → conservation tests (energy, momentum, angular momentum) still meaningful.
- New code path → NaN/Inf guard exercised.
- GPU-only tests → CTest label `gpu`.

Docs, scripts and CI-only changes usually need no new C++ tests; say so explicitly. Write the missing tests.

**CI coverage:** a test only protects `main` if CI runs it. If the change adds something CI should exercise (a new executable or test binary, a CTest label, a sanitizer or runtime setting, a script, a new parallel code path that TSan should see), check `.github/workflows/ci.yml` covers it. If not, add it in this PR when it's small, or name the open issue that will (e.g. #21 for TSan, #17 for the smoke test) in the Phase 8 summary. CI runs ubuntu-24.04 (gcc 13, clang 18, libomp 18), older than the dev machine, so a local pass can still fail in CI (see docs/tooling.md). If this change **adds or renames a CI job**, it gates nothing until it's in `main`'s required checks: say so in the Phase 8 summary and ask the user to approve adding it (`docs/tooling.md`, "Branch protection"). Then go on to Phase 6b.

## Phase 6b: Docs for what changed

Docs drift silently, so check each against the diff and the issue(s) from Phase 1, and update what's stale:

- **README.md:** the milestone **Status** table, the quickstart/commands if they changed, the repository layout if top-level folders were added, and the team list. Status counts only issues closed as **completed** (or completed by this PR); issues closed as "not planned" don't count. A milestone is "Done" when every remaining issue is completed, "In progress" once at least one is, "Not started" otherwise. Check with `gh issue list --milestone "<name>" --state all --json number,state,stateReason`.
- **docs/notes.md:** every design decision or measurement this work made gets an entry (date · decision · why · evidence, with numbers and hardware for benchmarks). This file becomes the presentation.
- **docs/journey.md:** if the work changes what the project can do or measure, add a dated entry (what was built, the parallel-computing angle, evidence). When there's something visible (viewer, render, plot), capture it: viewer screenshots on the real display with `--frames N --screenshot docs/media/<name>.png` (not Xvfb, which skews timings), plots into `docs/figures/`. Read each image back and make the caption's numbers match it.
- **PLAN.md:** the repo layout (§5) if files were added or renamed; anything the work proved wrong.
- **docs/tooling.md** for build/CI/tooling changes; **CLAUDE.md** if commands or rules changed.

List what you updated (or "docs already current") in the Phase 8 summary.

## Phase 7: Run the local equivalent of CI

Use presets if `CMakePresets.json` exists; otherwise use the commands in `CLAUDE.md`. If there's no `CMakeLists.txt` yet (before milestone 0), say the build/test phase is skipped and why.

Run in this order and stop at the first failure:

1. **Format:** `clang-format --dry-run --Werror` on changed `*.cpp *.hpp *.hip`. If it fails, run `clang-format -i` on those files and tell the user.
2. **Build:** `cmake --preset dev && cmake --build --preset dev` (warnings are errors).
3. **Tests:** run `ctest --preset dev -LE gpu --output-on-failure` twice, with `OMP_NUM_THREADS=1` and `OMP_NUM_THREADS=4`.
4. **ASan/UBSan:** `asan` preset build + `ctest -LE gpu`.
5. **TSan** (Linux only, only if the diff touches parallel code): `tsan` preset (it sets `TSAN_OPTIONS` and `OMP_NUM_THREADS=4`), with Archer loaded the way CI does: `OMP_TOOL_LIBRARIES=$(find /usr/lib -name 'libarcher.so*' | head -n 1) ctest --preset tsan`. Without Archer, TSan doesn't understand OpenMP and its results mean little.
6. **HIP (GPU):** if `hipcc` exists, build the `hip` preset. If `rocminfo` also lists a `gfx` GPU agent, run `ctest -L gpu`. Otherwise say it's skipped (CI compiles it; GPU tests run on Pearce's PC).
7. **Smoke test:** the tiny collision run (see CI `smoke` job). Fail on NaN or energy drift. Until the `smoke` job exists (milestone 2), report this step as skipped (N/A).
8. **Python:** if `scripts/` changed, at least `python3 -m py_compile` on the changed scripts.

Platform rules: on macOS (`uname` = Darwin), skip TSan and HIP and say so. If OpenMP isn't found, set `OpenMP_ROOT=$(brew --prefix)/opt/libomp`. Apple clang 17 on macOS 26 has an ASan runtime that deadlocks at startup, even on an empty `main`. On macOS, first compile and run a trivial `-fsanitize=address` program with a 5-second timeout. If it hangs, skip the ASan step and report it as skipped because of the toolchain (CI covers it on Linux). Run every test command with a timeout so a hang can't stall the gate. macOS has no `timeout` command, so use `perl -e 'alarm shift; exec @ARGV' <seconds> <cmd...>` there, and `timeout <seconds> <cmd...>` on Linux. Report a skipped step as skipped. Never report it as passed.

If a failure is caused by the diff, fix it and re-run from step 1. If it isn't caused by the diff, or fixing it means touching tolerances or constants, stop and tell the user.

## Phase 8: Summary and commit message

Give a short summary as a checklist: review (N found / N fixed / N skipped), secrets/junk, tests added, docs updated (Phase 6b), and each CI step (passed / failed / skipped + reason).

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

**PR description:** if the branch has an open PR, update its description with `gh pr edit --body-file` (allowed by the git rule in `CLAUDE.md`): one `Closes #N` line per issue from Phase 1 that this work finishes, then a short summary of what's done and anything still in progress. Keep any text the user wrote. If there's no PR yet, give the description for the user to paste when they open it.

Don't commit, push, touch branches, or create/merge/close PRs. The user does that.
