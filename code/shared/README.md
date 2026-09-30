# Shared Code Surface

`LLMbot/code/shared/` records cross-claim code that should not be copied into
multiple claim packages. Shared code is imported through the current flat
modules or through explicit future shared modules, not duplicated.

Shared modules are:

- orchestration: `main.py`, `parser_args.py`, `stage_registry.py`,
  `stage_runner.py`, `trainer.py`
- artifacts/runtime: `artifact_contracts.py`, `runtime_env.py`
- data and metrics: `dataloader.py`, `utils/__init__.py`,
  `utils/calibration.py`, `utils/losses.py`, `utils/metrics.py`,
  `utils/misc.py`
- model primitives: `model_building.py`, `GNNs.py`, `hypergnn.py`,
  `subgroups.py`

Claim-level moves should first add package imports and root compatibility
shims, then move one claim at a time. If two claims need the same helper, keep
that helper in shared code and reference it from the claim map instead of
duplicating it.
