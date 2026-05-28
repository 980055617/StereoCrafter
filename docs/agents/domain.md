# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Research Objective

The research goal is to reduce inference cost in StereoCrafter/SVD-style video
inpainting while preserving the quality of the original model. The current first
target is replacing `attn1` self-attention blocks with Mamba so inference becomes
faster and uses less memory. If that succeeds, the broader goal is to replace
other transformer blocks with Mamba as well, while keeping quality close to the
`origin` reference model.

When proposing changes, optimize for this objective:

- Preserve or recover `origin` output quality first.
- Measure speed and memory improvements against `origin`.
- Do not treat quality-only hacks as success if they undermine the Mamba
  replacement goal.
- Do not edit files with `origin` in their name; they are reference baselines.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root, or
- **`CONTEXT-MAP.md`** at the repo root if it exists; it points at one `CONTEXT.md` per context. Read each one relevant to the topic.
- **`docs/adr/`**; read ADRs that touch the area you're about to work in. In multi-context repos, also check `src/<context>/docs/adr/` for context-scoped decisions.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The producer skill (`/grill-with-docs`) creates them lazily when terms or decisions actually get resolved.

## File structure

Single-context repo (most repos):

```text
/
|-- CONTEXT.md
|-- docs/adr/
|   |-- 0001-event-sourced-orders.md
|   `-- 0002-postgres-for-write-model.md
`-- src/
```

Multi-context repo (presence of `CONTEXT-MAP.md` at the root):

```text
/
|-- CONTEXT-MAP.md
|-- docs/adr/                          # system-wide decisions
`-- src/
    |-- ordering/
    |   |-- CONTEXT.md
    |   `-- docs/adr/                  # context-specific decisions
    `-- billing/
        |-- CONTEXT.md
        `-- docs/adr/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/grill-with-docs`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders), but worth reopening because..._
