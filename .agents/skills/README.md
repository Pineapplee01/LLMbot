# Repo-Local Codex Skills

These skills split research work into narrow roles. Prefer explicit invocation when a task could fit more than one role.

| Skill | Use For | Do Not Use For |
| --- | --- | --- |
| `$implementer` | Scoped method, code, config, or operational doc changes. | Experiment execution, result interpretation, final review, or experimental claims. |
| `$test_guardian` | Tests, validation, regression reproduction, and governance gates. | Method redesign, feature implementation, experiment running, or result interpretation. |
| `$experiment_runner` | Experiment configs, manifests, run commands, and artifact organization. | Method/code changes, analysis conclusions, or critical review. |
| `$analysis_writer` | Tables, summaries, error analysis, and evidence-bounded interpretation. | Missing metadata, unsupported conclusions, experiment running, or code changes. |
| `$reviewer` | Read-only critical review of diffs, evidence, boundaries, and claim validity. | Implementation, experiment execution, or rewriting results. |

## Explicit Invocation

Use explicit `$skill` invocation when:

- The task crosses implementation, validation, experiment, analysis, or review boundaries.
- The next owner must not perform adjacent work.
- Research evidence or claim boundaries are involved.
- A handoff is needed between roles.

Examples:

- `$implementer: add the manifest flag without running experiments`
- `$test_guardian: validate changed scope and regression tests`
- `$experiment_runner: prepare the seed-1 run manifest and command`
- `$analysis_writer: summarize validated metrics without new claims`
- `$reviewer: audit claim validity against manifest and metrics`

## Common Handoff Patterns

- `$implementer -> $test_guardian -> $reviewer`
- `$experiment_runner -> $test_guardian -> $analysis_writer -> $reviewer`
- `$analysis_writer -> $reviewer -> $experiment_runner` when evidence is missing
- `$reviewer -> $implementer -> $test_guardian` when review finds engineering defects
