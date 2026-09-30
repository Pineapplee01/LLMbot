# Routed LLM Evidence Refinement

Claim role: routed-only LLM evidence, prompt-expert cache consumption,
GLANCE-style refinement, frozen-router reuse, and prompt-expert quality audit.

Current owner modules:

- `trainer_glance.py`
- `precompute.py`
- `prompt.py`
- `llm_evidence_refiner.py`

Canonical vocabulary:

- `prompt_expert_bundle`
- `evidence_card`
- `refiner_features`
- `oracle_advantage`
- `utility_reward`
- `gate_prob`

Migration note: `precompute.py` is an independent bounded context. Keep it
source-complete until prompt/evidence cache schemas are separately governed.
