# Routed LLM Evidence Refinement

Claim role: routed-only LLM evidence, prompt-expert cache consumption,
GLANCE-style refinement, frozen-router reuse, and prompt-expert quality audit.

Canonical vocabulary:

- `prompt_expert_bundle`
- `evidence_card`
- `refiner_features`
- `oracle_advantage`
- `utility_reward`
- `gate_prob`

Primary symbols:

- `llm_evidence_refiner.py` now owns the standalone evidence-refiner utility in
  this package.
- `trainer_glance.py`, `precompute.py`, and `prompt.py` remain top-level
  because they are large cross-claim surfaces.

Shared dependencies:

- `precompute.py`
- `prompt.py`
- `trainer_glance.py`
- `utils/`

Do not move yet:

- `trainer_glance.py`
- `precompute.py`
- `prompt.py`
