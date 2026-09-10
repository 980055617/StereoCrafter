## StereoCrafter Agent Notes

The workspace root is `/home/kawa/master_project`. Start with `../AGENTS.md`
and `../CONTEXT-MAP.md`; this file only contains StereoCrafter-specific notes
for agents that start inside this subdirectory.

### Domain docs

Read `CONTEXT.md` for StereoCrafter vocabulary.

### Experiment notes

- `docs/agents/0160-overfit-diagnosis.md`
- `docs/agents/model-change-log.md`

For every model, training, inference, evaluation, architecture, or debugging
task, read the relevant `docs/agents/` notes before making claims or edits. If
the task has no matching note yet, create one under `docs/agents/`.

Update `docs/agents/model-change-log.md` with the question, change,
commands/checkpoints, metrics, interpretation, and next step. If the result
changes the recommended direction for the 0160 overfit work, also update
`docs/agents/0160-overfit-diagnosis.md`.

### Rules

Do not edit files with `origin` in their name; they are reference baselines.

Do not delete, modify, move, prune, or otherwise touch these origin/reference
weight directories:

- `weights/StereoCrafter/`
- `weights/stable-video-diffusion-img2vid-xt-1-1/`
- `weights/DepthCrafter/`
