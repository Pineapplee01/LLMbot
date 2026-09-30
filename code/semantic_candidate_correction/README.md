# Semantic Candidate Correction

Claim role: semantic-only correction gates, candidate semantic evidence, and
prompt/cache construction that does not change graph routing semantics.

Canonical vocabulary:

- `semantic_gate_score`
- `candidate_output_path`
- `correction_delta`
- `semantic_evidence`

Primary symbols:

- Semantic correction gates currently live in `trainer_semantic.py`.
- Prompt/cache construction overlaps with `precompute.py` and `prompt.py`.

Shared dependencies:

- `trainer_semantic.py`
- `precompute.py`
- `prompt.py`
- `utils/`

Do not move yet:

- `trainer_semantic.py`
- `precompute.py`
- `prompt.py`
