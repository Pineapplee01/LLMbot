# Semantic Candidate Correction

Claim role: semantic-only correction gates, candidate semantic evidence, and
prompt/cache construction that does not change graph routing semantics.

Current owner modules:

- `trainer_semantic.py`
- `precompute.py`
- `prompt.py`

Canonical vocabulary:

- `semantic_gate_score`
- `candidate_output_path`
- `correction_delta`
- `semantic_evidence`

Migration note: this claim overlaps with routed LLM evidence through
`precompute.py`; split cache generation only after cache schema names are fixed.
