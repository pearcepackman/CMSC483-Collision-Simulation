---
name: plan-issue
description: Pre-work planning for one GitHub issue in this repo. Loads the issue, checks its dependencies and assignee, compares it against the code and docs, asks the user the questions that change the implementation, proposes splitting it if needed, writes a self-contained implementation plan split into parallel work streams, has an adversarial subagent attack the plan, then stops for approval. After approval it creates/edits issues and posts the plan as an issue comment. Use when the user says /plan-issue <number>.
argument-hint: "[issue-number]"
disable-model-invocation: true
allowed-tools: Bash(gh issue view *) Bash(gh issue list *) Bash(gh label list *) Bash(gh api user *) Bash(git status *) Bash(git log *) Bash(git diff *)
---

# Plan an issue

Plan issue **#$0**. Run the phases in order. **Phase 5 is a hard stop:** present the plan and end your turn. Until the user approves, only read: no file edits, no GitHub writes.

**No git work while planning.** Branches, commits, pushes and PRs happen during implementation, as the git section of `CLAUDE.md` allows.

The user is new to C++/HPC: explain jargon in a few words the first time it appears.

## Context (loaded before you start)

Issue:

```!
gh issue view $0 --json number,title,state,milestone,assignees,labels,body,comments -q '"#\(.number) [\(.state)] \(.title)\nMilestone: \(.milestone.title // "none") | Assignees: \([.assignees[].login] | join(",")) | Labels: \([.labels[].name] | join(","))\n\n\(.body)\n\n## Comments (\(.comments | length))\n\([.comments[] | "--- \(.author.login):\n\(.body)"] | join("\n"))"'
```

Who you're working for (GitHub login):

```!
gh api user -q .login
```

All issues (for dependencies, numbering and duplicates):

```!
gh issue list --state all --limit 200 --json number,title,state,milestone,assignees -q '.[] | "#\(.number) [\(.state)] \(.title) | \(.milestone.title // "no milestone") | \([.assignees[].login] | join(","))"'
```

## Phase 1: Check the issue

Report in a short list:

- **State:** if it's closed, say so and stop.
- **Assignee:** if it isn't the login above, say whose it is and ask whether to continue before doing anything else.
- **Depends on:** the state of each issue listed. If any are open, say the implementation is blocked on them (planning can still go ahead).
- **Blocks:** the issues waiting on this one. Read their bodies (`gh issue view N`): this issue's design has to serve them, especially any interface they'll code against.

## Phase 2: Compare against the repo

Read only what the issue touches: the files in its **Files:** line and their neighbours, the relevant sections of `PLAN.md`, `docs/notes.md`, and `docs/tooling.md` for build/CI work. Don't sweep the whole repo.

List any **drift**: places where the issue no longer matches the code or plan (a file already exists, a name changed, a decision was made later). Drift is a question for Phase 3, not something to silently resolve.

**Fast path:** if the whole change fits in one sentence and Phase 3 has nothing to ask, skip Phases 3–4, give a 3-line plan (what changes, the test, the verification command) and stop for a "go".

## Phase 3: Ask questions

Ask with `AskUserQuestion` (up to 4 per call), only about decisions that change the implementation: interfaces others depend on, scope, trade-offs, drift. Put your recommendation first and mark it "(Recommended)". Don't ask anything `CLAUDE.md`, `PLAN.md` or the issue already answers. If there's nothing to ask, say so and move on.

## Phase 4: Should it be split?

Propose a split only if the issue is clearly more than one PR's worth (independent "Done when" items in unrelated areas), or parts belong to the other person. Ownership per `PLAN.md` §6: engine and GPU work → pearcepackman, world/scenarios/rendering → kpoudyal06.

For each proposed new issue, write it in the repo's issue format, ready to create:

```
<title>
<one-paragraph description>

**Done when**
- [ ] ...

**Files:** ...

**Depends on:** #...
**Blocks:** #...
```

Plus its milestone, labels (from `gh label list`) and assignee, and which existing issues' **Depends on / Blocks** lines would change. If no split is needed, say so in one line.

## Phase 5: Write the plan and STOP

The plan must stand on its own: someone in a fresh session should be able to implement it from the plan alone. Sections:

1. **Goal**: one or two lines.
2. **Out of scope**: what this issue deliberately doesn't do.
3. **Design**: files to add/change; function signatures in a code block; data layout (Struct-of-Arrays per `CLAUDE.md`). Keep to the plain C-style C++ the repo uses.
4. **Tests first**: one test per "Done when" box, written before the code. Backend comparisons use relative tolerances over a few steps, never `==`. GPU tests get the CTest label `gpu`.
5. **Risks**: races, numerics, interface lock-in, only those that apply.
6. **Docs**: a `docs/notes.md` entry if a design decision is made.
7. **Verification**: the exact commands that must pass (presets, tests at `OMP_NUM_THREADS=1` and `4`, `asan`, `tsan` if parallel code, `hip` if GPU code, format check).
8. **Done-when map**: a table mapping each "Done when" box to the plan step that satisfies it.
9. **Work split**: the plan cut into streams that can be built at the same time because they touch **different files** (e.g. a C++ function + its test, a CI job, a Python script), each with its files, what it needs from the other streams (a function signature fixed in §3 is enough), and how it's verified on its own. Say which stream goes first if one blocks the others, and when the work is too small to be worth splitting (one stream is fine).

## Phase 5b: Adversarial plan review

Before showing the plan, launch **one** subagent (Agent tool, `general-purpose`) to attack it. Give it the plan text, the issue numbers, and the files the plan touches; tell it to read the code, change nothing in the repo, and write any throwaway programs only in the scratchpad. Ask it to assume the plan is wrong and try to break it:

- **Correctness:** edge inputs (N = 0 or 1, coincident particles, huge dt, a NaN mid-run), numerics and tolerances, a design that can't meet a "Done when" box, a test that would pass even if the code were wrong.
- **Usability and UX:** confusing flags, error messages that don't say how to fix the problem, output formats that break existing readers (scripts, CSV headers, the viewer), defaults that surprise.
- **Security and robustness:** untrusted input (config files, command-line values, file paths: overwriting files, path traversal, huge allocations), CI steps that could leak secrets or run untrusted code.
- **Fit:** breaks something the other person owns, conflicts with `CLAUDE.md` rules, CI that wouldn't actually run the new tests, missing docs or progress-log entry.

It returns findings as severity (`blocker` / `should-fix` / `nit`), the plan step, the problem, a concrete failure scenario, and a fix, marked `CONFIRMED` (it ran something) or `PLAUSIBLE`. Fold every finding you agree with into the plan before showing it. List the ones you rejected, each with one line on why.

Skip this phase only if the user passes `no-adversarial` or the plan is the Phase 2 fast path; say so in one line.

Then ask: **"Approve, edit, or reject the plan? If approved, I'll post it as a comment on #$0 [and create the split issues]."**

**End your turn here.**

## Phase 6: After approval

Do only what was approved:

- Create split issues with `gh issue create --title ... --body-file ... --milestone ... --label ... --assignee ...`, then update the **Depends on / Blocks** lines of the affected issues with `gh issue edit N --body-file ...`.
- Post the final plan (with the user's edits) as a comment: `gh issue comment $0 --body-file <file>`. Write temp files to the scratchpad, not the repo.

Report the links, then say the issue is ready to implement. If the session is long, suggest `/clear` and starting fresh with "implement the plan in the last comment on #$0".

## Phase 7: Implementing (only when the user asks)

If the plan has two or more streams (§9) and the user wants it built, the streams can run as **parallel subagents** (Agent tool, launched in one message), one per stream, each given the plan, its stream, and its file list:

- They all work in the **same working tree** (the feature branch), on their own files only. The agents themselves don't run git: no worktrees, branches, stashes or commits, so no `isolation: "worktree"`. You commit once everything is integrated and verified.
- Each writes its tests first, builds, and runs its own tests, then reports what it changed and what it verified.
- You then integrate: read every change, run the plan's full Verification list yourself, and fix anything that breaks between streams. A stream's "it passed" is not proof until your own run shows it.

Parallel agents cost more in total than doing the work yourself, so use them when the streams are big enough that waiting on them in sequence would be slow, and say in one line that you're doing it.
