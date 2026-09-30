# GLANCE Utility Plus BotMoE Selector For Routed Prompt Experts

Date: 2026-06-03
Status: approved-in-thread
Scope: `LLMbot/` active mainline only

## Goal

Add a new routed-node prompt-expert experiment line that combines:

1. GLANCE-style utility advantage semantics for deciding whether a routed node
   should change the frozen base prediction.
2. A BotMoE-style sparse selector over existing prompt experts:
   `graph_following`, `graph_follower`, `tweet`, `conflict`.

The outcome must be directly comparable against the current high-base frozen
SimTeG line on canonical TwiBot-20 splits.

## Design Boundary

- `abstain/base` is not a selectable expert.
- `abstain/base` stays under a separate GLANCE-style keep/change gate.
- BotMoE selector only chooses among the four routed prompt experts.
- No prompt-cache rebuild is required for the first implementation.
- No backbone change and no router retraining protocol change.

## Mainline Design

### A. GLANCE utility line

Reuse the current `oracle_advantage` semantics already present in
`trainer_glance.py`:

`loss_gnn - loss_refiner - beta`

Use this to define `utility_positive`, and keep the explicit refiner gate on the
strict routed path:

- `joint_refiner_target_mode=keep_change`
- `joint_refiner_gate_target=utility_positive`

This gate owns the decision:

- keep base prediction
- or allow prompt-expert correction

### B. BotMoE selector line

Add a new prompt-expert fusion mode in `joint_router_refinement`:

- `botmoe_selector`

This selector should follow the BotMoE-style sparse MoE contract:

- node-conditioned gate logits
- top-k sparse expert selection
- weighted combination of selected experts
- auxiliary balancing loss from expert importance and expert load

Mapped to the current routed-node setting:

- experts:
  `graph_following`, `graph_follower`, `tweet`, `conflict`
- gate input:
  routed-node context from existing refiner-side features
  (`z_gnn`, `metadata_structured`, structural side channel)
- gate output:
  sparse weights over the four prompt experts
- fused representation:
  weighted sum of projected expert embeddings

### C. Combined path

The combined experiment path is:

1. frozen router selects routed nodes
2. explicit GLANCE utility gate decides keep/change on routed nodes
3. if change is taken, BotMoE selector chooses sparse prompt experts
4. fused prompt representation feeds routed refiner classifier

This keeps responsibilities separated:

- gate learns "should change?"
- selector learns "which expert if changing?"

## Edit Zones

- `LLMbot/trainer_glance.py`
- `LLMbot/parser_args.py`
- `LLMbot/README.md`
- `docs/code/parser.md`
- `docs/code/research.md`
- `docs/ARCHITECTURE.md`
- `LLMbot/experiments.md`
- `docs/code.md` only if maintainability risk wording changes materially

## Out Of Scope

- `LLMbot/baseline/`
- `LLMbot/code/`
- prompt-cache regeneration
- new source files for model code
- backbone training changes
- full BotMoE end-to-end reproduction
- metadata as a selectable expert

## CLI Surface

Planned additions or reuse:

- reuse:
  - `--joint_refiner_explicit_gate`
  - `--joint_refiner_target_mode keep_change`
  - `--joint_refiner_gate_target utility_positive`
  - `--joint_refiner_weight_mode`
- add:
  - `--joint_prompt_expert_fusion botmoe_selector`
  - BotMoE-specific selector args if needed:
    - selector top-k
    - noisy gating on/off
    - aux loss weight

## Artifact Contract

Manifest and metrics should record:

- `joint_prompt_expert_fusion=botmoe_selector`
- GLANCE utility gate settings
- selector top-k
- selector aux-loss weight
- routed selected-expert distribution
- mean selector entropy / max weight
- gate precision/recall against utility-positive target
- routed fix / break / net gain metrics

Per-node rows should include:

- `gate_prob`
- `gate_decision`
- selector weights over four experts
- selected top expert

## Experiment Matrix

Compare on the same high-base routed setting:

1. base frozen SimTeG
2. `raw_concat_follower_triplet`
3. utility gate only
4. BotMoE selector only
5. utility gate + BotMoE selector

Primary metrics:

- Accuracy
- Macro-F1
- fixed wrong
- broken correct
- net gain
- wrong-node fix rate
- correct-node break rate
- selected-expert distribution on routed test nodes

## Risks

1. Selector may still collapse to one global expert if classification loss
   dominates sparse-gating supervision.
2. Utility gate may become too conservative and route nearly all nodes to keep.
3. Shared implementation in `trainer_glance.py` is high-risk because many
   existing prompt-expert modes already coexist there.

## Validation

- `python main.py --help`
- `python -m py_compile LLMbot/main.py LLMbot/parser_args.py LLMbot/trainer_glance.py`
- local smoke for new fusion mode on prompt-expert payload
- server experiment on canonical high-base frozen SimTeG reuse root
- artifact existence checks in `LLMbot/experiments.md`

## Handoff

- main thread: shared-file implementation in `trainer_glance.py`
- delegated lanes: CLI/doc sync and experiment registry / server command
