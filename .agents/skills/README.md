# Repo-Local Codex Skills

These skills split BotDetection research work into narrow roles. Prefer explicit invocation when a task could fit more than one role.

## Framework Boundary

- Superpowers controls design and implementation gates.
- OMX orchestrates approved work and long-running execution.
- Repo-local skills define who owns each research job after scope is clear.

Do not let OMX, Ralph, Team mode, or a single agent collapse implementation, validation, experiment running, analysis writing, and final review into one claim path.

## Skills

| Skill | Use For | Do Not Use For |
| --- | --- | --- |
| `$implementer` | Approved scoped code, config, governance, or operational-doc changes in legal edit zones. | Experiment execution, result interpretation, final review, unsupported claims, or unapproved new files. |
| `$test_guardian` | CLI-arg validation, smoke checks, manifest/artifact checks, regression reproduction, and governance drift gates. | Method redesign, feature implementation, full experiment running, result interpretation, or creating test files by default. |
| `$experiment_runner` | Experiment commands, run metadata, manifests, provenance, and artifact organization. | Method/code changes, analysis conclusions, final review, or manual artifact editing. |
| `$analysis_writer` | Tables, summaries, error analysis, and cautious interpretation from validated metrics and manifests. | Missing metadata, unsupported conclusions, experiment running, code changes, or claim upgrades. |
| `$reviewer` | Read-only critical review of diffs, evidence, edit boundaries, reproducibility, and claim safety. | Implementation, experiment execution, rewriting results, or softening unsupported claims. |

## Explicit Invocation

Use explicit `$skill` invocation when:

- The task crosses implementation, validation, experiment, analysis, or review boundaries.
- The next owner must not perform adjacent work.
- Research evidence, external-code verification, or claim boundaries are involved.
- A handoff is needed between roles.
- An OMX workflow such as `$ralplan`, `$ralph`, or `$team` needs a narrow owner for one research job.

Examples:

- `$implementer: update the manifest flag after the Superpowers design gate`
- `$test_guardian: validate CLI args and governance drift without adding tests`
- `$experiment_runner: prepare the seed-1 run manifest and command`
- `$analysis_writer: summarize validated metrics without new claims`
- `$reviewer: audit claim validity against manifest and metrics`

## Common Handoff Patterns

- `superpowers:brainstorming -> $implementer -> $test_guardian -> $reviewer`
- `$experiment_runner -> $test_guardian -> $analysis_writer -> $reviewer`
- `$analysis_writer -> $reviewer -> $experiment_runner` when evidence is missing
- `$reviewer -> $implementer -> $test_guardian` when review finds engineering defects
