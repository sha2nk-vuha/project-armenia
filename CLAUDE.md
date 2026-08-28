# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A collection of Claude Code skills sourced from `mattpocock/skills` on GitHub. Skills are prompt-driven instruction sets (SKILL.md files) that Claude Code users invoke via `/skill-name`. This repo is the installed skill tree — not application code.

## Skill management

`skills-lock.json` is the lockfile. It records each skill's source path on GitHub and a `computedHash` for integrity verification. Treat it like a `package-lock.json` — edit it when adding, removing, or updating skills, and keep it in sync with the files under `agent/skills/` and `.agents/skills/`.

Both `agent/skills/` and `.agents/skills/` mirror the same skill tree. `.agents/skills/` is the directory Claude Code reads at runtime; `agent/skills/` is the working copy.

## Skill structure

Each skill lives in its own subdirectory. The required file is `SKILL.md` with YAML frontmatter:

```yaml
---
description: "One-line trigger description (omit for user-invoked skills)"
disable-model-invocation: true   # optional — makes the skill user-invoked only
---
```

Supporting files (e.g. `GLOSSARY.md`, `AGENT-BRIEF.md`) are placed in the same directory and referenced from `SKILL.md` via relative links. They are loaded on demand (progressive disclosure), not upfront.

**Model-invoked** skills carry a `description` so Claude can fire them autonomously. Every description adds context load — only use model-invocation when the skill must trigger on its own or be reached by another skill.

**User-invoked** skills set `disable-model-invocation: true` and have a human-facing description. They cost no context load but require the user to remember they exist.

## Skill categories (from skills-lock.json)

| Category | Skills |
|---|---|
| Engineering | `codebase-design`, `diagnosing-bugs`, `domain-modeling`, `grill-with-docs`, `implement`, `improve-codebase-architecture`, `resolving-merge-conflicts`, `tdd`, `to-issues`, `to-prd`, `triage`, `setup-matt-pocock-skills` |
| Productivity | `grilling`, `grill-me`, `handoff`, `teach`, `writing-great-skills` |
| Misc | `git-guardrails-claude-code`, `migrate-to-shoehorn`, `scaffold-exercises`, `setup-pre-commit` |
| Personal | `edit-article`, `obsidian-vault`, `ask-matt` |
| In-progress | `decision-mapping`, `loop-me`, `review`, `writing-beats`, `writing-fragments`, `writing-shape` |
| Deprecated | `design-an-interface`, `qa`, `request-refactor-plan`, `ubiquitous-language` |

## First-time setup

Run `/setup-matt-pocock-skills` before using the engineering skills. It configures the issue tracker, triage label vocabulary, and domain doc layout for this repo, then writes the `## Agent skills` block into `CLAUDE.md`.

## Writing and editing skills

Read `agent/skills/writing-great-skills/SKILL.md` before authoring or editing any skill — it defines the vocabulary and principles the other skills are built on (leading words, information hierarchy, context load vs cognitive load, completion criteria, failure modes).

Key invariants:
- Keep each meaning in a single source of truth.
- Push reference behind context pointers (external files) to keep `SKILL.md` legible.
- Every step must end on a checkable completion criterion.
- Run the no-op test on every line: does it change behaviour versus the model default? If not, delete it.

## Agent skills

### Issue tracker

Issues and PRDs live as local markdown files under `.scratch/<feature-slug>/` (no external tracker). See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles, using the default label strings (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
