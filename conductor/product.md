# LLMbot Product Context

Date: 2026-07-04

## Product

`LLMbot/` is the active operator mainline for bot-detection research pipeline
work in `G:\Research\BotDetection`. Active Python source lives in the flat
`LLMbot/code/` directory.

The product is not a hosted application. It is a research-grade Python pipeline
for staged social-bot detection experiments, artifact generation, and
claim-safe comparison work.

## Primary Users

- Research implementers who change `LLMbot/code/` source and must preserve CLI,
  manifest, artifact, and comparison contracts.
- Experiment operators who run `python main.py ...`, `python precompute.py ...`,
  or curated `run_*.py` queue scripts.
- Reviewers who audit whether results, manifests, and docs support a research
  claim without relying on memory or ad hoc naming.

## Current Mainline Goal

Keep `LLMbot/` understandable and governable while the implementation is still
in a staged extraction window:

- keep `LLMbot/main.py` as the compatibility entrypoint into
  `LLMbot/code/main.py`
- keep `code/parser_args.py -> code/main.py -> code/stage_runner.py ->
  code/stage_registry.py` as the main entry and orchestration spine
- keep public task names, public flags, manifests, and artifact paths canonical
- keep algorithm code, orchestration code, runner scripts, and generated
  artifact surfaces distinct
- keep the active source directory flat for now; do not add `stages/`,
  `methods/`, or similar taxonomy directories before deletion and function
  consolidation

## Non-Goals

- Do not make first-stage governance a large code refactor.
- Do not split `code/trainer_legacy_impl.py`, `code/precompute.py`,
  `code/router.py`, `code/estimators.py`, or `code/model_building.py` into
  deeper package directories in this phase.
- Do not hand-edit `LLMbot/experiments/`, `LLMbot/server_logs/`, datasets,
  checkpoints, saved artifacts, or generated manifests.
- Do not upgrade research claims from documentation changes alone.

## Claim Boundary

Documentation may describe implemented code paths, public/internal stage
boundaries, and artifact contracts. It must not claim that a method is an
official reproduction, claim-grade result, or validated multi-seed conclusion
unless manifests, metrics, traces, and local verification evidence support that
claim.

## First-Stage Success Criteria

- The repo has a short governance entrypoint in `conductor/`.
- Canonical terms are captured in `UBIQUITOUS_LANGUAGE.md`.
- Existing code-development docs name the same main entry, orchestration layer,
  algorithm layer, precompute context, runner-script surface, and artifact
  surface.
- Active Python source location is documented as `LLMbot/code/`, with root
  compatibility entrypoints preserved.
