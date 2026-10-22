---
name: plan-issue
description: Pre-work planning for one GitHub issue in this repo. Loads the issue, checks its dependencies and assignee, compares it against the code and docs, asks the user the questions that change the implementation, proposes splitting it if needed, and writes a self-contained implementation plan, then stops for approval. After approval it only creates/edits issues and posts the plan as an issue comment. Never touches git. Use when the user says /plan-issue <number>.
argument-hint: "[issue-number]"
disable-model-invocation: true
allowed-tools: Bash(gh issue view *) Bash(gh issue list *) Bash(gh label list *) Bash(gh api user *) Bash(git status *) Bash(git log *) Bash(git diff *)
---

# Plan an issue

Plan issue **#$0**. Run the phases in order. **Phase 5 is a hard stop:** present the plan and end your turn. Until the user approves, only read: no file edits, no GitHub writes.

**Never do git work** (branches, commits, pushes, PRs). The user does all of it. See the git rule in `CLAUDE.md`.

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

Then ask: **"Approve, edit, or reject the plan? If approved, I'll post it as a comment on #$0 [and create the split issues]."**

**End your turn here.**

## Phase 6: After approval

Do only what was approved:

- Create split issues with `gh issue create --title ... --body-file ... --milestone ... --label ... --assignee ...`, then update the **Depends on / Blocks** lines of the affected issues with `gh issue edit N --body-file ...`.
- Post the final plan (with the user's edits) as a comment: `gh issue comment $0 --body-file <file>`. Write temp files to the scratchpad, not the repo.

Report the links, then say the issue is ready to implement. If the session is long, suggest `/clear` and starting fresh with "implement the plan in the last comment on #$0".
