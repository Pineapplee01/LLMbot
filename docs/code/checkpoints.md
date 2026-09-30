# LLMbot Checkpoint And Weight Artifact Governance

This document records the current checkpoint, weight, and tensor-artifact
save/load behavior under the active `LLMbot/` mainline as inspected on
2026-05-15.

It is an engineering governance document. It is not experiment evidence, does
not revise external literature, and does not make a model-performance claim.

## Scope

Active audit scope:

- `LLMbot/utils/__init__.py`
- `LLMbot/main.py`
- `LLMbot/trainer.py`
- `LLMbot/model_building.py`
- `LLMbot/data/*.py`
- `LLMbot/trainers/*.py`
- `LLMbot/scripts/oracle_retrieval_upper_bound.py`

Deprecated surfaces under `LLMbot/baseline/` and `LLMbot/code/` are not active
mainline contracts. They may be read for forensic comparison or migration, but
new checkpoint policy must be defined from `LLMbot/`.

This document intentionally does not rename, move, delete, or reinterpret any
existing local checkpoint or artifact.

## Current Save And Load Model

The active code currently has four save families:

1. Manifest-backed run artifacts under an experiment root, such as
   `<experiment_root>/frozen/...` and `<experiment_root>/stages/<stage>/...`.
2. Legacy distillation checkpoints under
   `<experiment_name>/checkpoints/{LM,GNN,MLP,...}/best.pkl` and
   `<experiment_name>/MLP_KD/seed_<seed>_best.pkl`.
3. Trainer-specific roots, especially
   `LLMbot/checkpoints/dual_router/<group>/seed_<seed>/` and
   `LLMbot/saved_artifacts/<group>/seed_<seed>/`.
4. Feature/cache tensors, including embeddings, soft labels, split tensors,
   diagnostic outputs, logits, probabilities, and score vectors.

The active code currently has two load families:

1. Preferred loads through `LLMbot/utils.safe_torch_load`, which attempts
   `torch.load(..., weights_only=True)` when the installed PyTorch supports it.
2. Broader local loads through `torch.load(..., weights_only=False)` or local
   wrappers that force that mode. These are trusted-local-artifact paths and
   should be narrowed or explicitly documented before claim-grade use.

The main naming problem is that `.pt` currently means many things: loadable
weights, logits/probabilities, embeddings, score vectors, split indices, and
miscellaneous diagnostics. File names and manifests must therefore carry the
semantic type.

## Current Helpers

| Helper | Location | Current behavior | Governance assessment |
| --- | --- | --- | --- |
| `safe_torch_load` | `LLMbot/utils/__init__.py:30` | Calls `torch.load(..., weights_only=True)` and falls back for older PyTorch. | Preferred loader for tensor/checkpoint reads. |
| `write_torch` | `LLMbot/utils/__init__.py:90` | Ensures the parent directory and calls `torch.save`. | Useful writer, but it does not enforce naming or schema. |
| `build_experiment_root` | `LLMbot/utils/__init__.py:164` | Builds `<artifact_root or experiment_name>/seed_<seed>/`. | Good run root shape when paired with manifests. |
| `build_stage_dir` | `LLMbot/utils/__init__.py:172` | Builds `<experiment_root>/stages/<stage>/`. | Useful but `stage` should not leak into new paper-facing names. |
| `save_stage_artifacts` | `LLMbot/utils/__init__.py:176` | Writes tensors as `.pt` and dicts as `.json`. | Convenient, but accepts caller-provided names without policy checks. |
| `prepare_path` | `LLMbot/utils/__init__.py:296` | Creates legacy `checkpoints/{LM,GNN,MLP,LM_pretrain,GNN_pretrain}` and `MLP_KD`. | Legacy distillation layout; do not expand for new work. |

## Save Surface Matrix

| Surface | Write site | Save root | Filename pattern | Payload schema | Current role |
| --- | --- | --- | --- | --- | --- |
| Phase A graph detector | `LLMbot/trainer.py:302` | `<experiment_root>/frozen/g0/` | `checkpoint.pt` | `model`, `input_adapter`, `input_pipeline`, `model_config`, `selection_metrics` | Active, legacy-named preparation checkpoint. |
| Phase A graph outputs | `LLMbot/trainer.py:303` | `<experiment_root>/frozen/g0/` | `outputs.pt` | logits, probabilities, predictions, labels, node representations | Active, legacy-named preparation artifact. |
| Phase A semantic-source cell | `LLMbot/main.py:293` | semantic-source matrix cell dir | `outputs.pt` | copied/written Phase A outputs | Active matrix artifact; parent path carries context. |
| Phase A checkpoint copy | `LLMbot/main.py:297` and `LLMbot/main.py:299` | semantic-source matrix cell dir | `checkpoint.pt` | copied checkpoint or fallback JSON-like torch payload | Active compatibility copy. |
| GATS gate outputs | `LLMbot/trainer.py:578` | `<experiment_root>/frozen/gates/gats_faithful/` | `outputs.pt` | gate/calibration outputs | Active but legacy-named gate artifact. |
| LM-only head outputs | `LLMbot/trainer.py:2097` | `<experiment_root>/frozen/lm_only_head/` | `outputs.pt` | LM-only logits/probabilities/predictions | Active gate artifact. |
| LM-only head checkpoint | `LLMbot/trainer.py:2098` | `<experiment_root>/frozen/lm_only_head/` | `checkpoint.pt` | scaler/classifier payload and temperature | Active gate checkpoint. |
| Semantic finetune embeddings | `LLMbot/trainer.py:1368` | `<experiment_root>/stages/semantic_finetune/` | `embeddings.pt` | semantic embedding tensor | Active artifact; generic but parent-scoped. |
| Semantic finetune outputs | `LLMbot/trainer.py:1369` | `<experiment_root>/stages/semantic_finetune/` | `outputs.pt` | logits/probabilities/predictions | Active artifact; generic but parent-scoped. |
| Semantic finetune classifier | `LLMbot/trainer.py:1370` | `<experiment_root>/stages/semantic_finetune/` | `classifier.pt` | classifier state dict | Active checkpoint-like payload. |
| Semantic finetune model | `LLMbot/trainer.py:1376` | `<experiment_root>/stages/semantic_finetune/` | `model.pt` | `model`, `model_config` | Active but too generic for new independent use. |
| Stage artifact bundles | `LLMbot/trainer.py:3408`, `3584`, `3672` | `<experiment_root>/stages/<stage>/` | caller-provided names such as `all_node_outputs.pt` | mixed tensor/dict artifacts | Active; caller names must carry artifact type. |
| Estimator router checkpoint | `LLMbot/trainer.py:2245` | estimator stage dir | `router_checkpoint.pt` | `estimator.state_dict_payload()` | Active but ambiguous; router kind is not explicit. |
| Residual-risk selector checkpoint | `LLMbot/trainer.py:2277` | estimator stage dir | `residual_risk_selector_checkpoint.pt` | `estimator.state_dict_payload()` | Active and clear. |
| LOGIN uncertainty-module checkpoint | `LLMbot/trainer.py` | estimator stage dir | `login_router_checkpoint.pt` | `estimator.state_dict_payload()` | Active; official-code-verified for the node-selection uncertainty submodule only, not the full LOGIN loop. |
| Vertical/minimal diagnostics | `LLMbot/trainer.py:3420` to `3424` | diagnostic dir | `gnn_logits_all.pt`, `gnn_probs_all.pt`, `gnn_pred_all.pt`, `gnn_entropy_all.pt`, `lm_pred_test.pt` | logits, probabilities, predictions, entropy | Active diagnostics; clear enough but not checkpoints. |
| LM pretrain checkpoint | `LLMbot/trainer.py:801`, `837` | `<experiment>/checkpoints/LM_pretrain/` | `best.pkl` | `model`, `optimizer`, `scheduler` | Legacy active path; not research-grade naming. |
| LM checkpoint | `LLMbot/trainer.py:945` | `<experiment>/checkpoints/LM/` | `best.pkl` | `model`, `optimizer`, `scheduler` | Legacy active path; not research-grade naming. |
| GNN checkpoint | `LLMbot/trainer.py:4288` | `<experiment>/checkpoints/GNN/` | `best.pkl` | `model`, `optimizer`, `scheduler`, `model_config` | Legacy active path; not research-grade naming. |
| MLP KD checkpoint | `LLMbot/trainer.py:4560` | `<experiment>/MLP_KD/` | `seed_<seed>_best.pkl` | `model`, `model_params` | Legacy active path; partially seeded but still vague. |
| MLP checkpoint | `LLMbot/trainer.py:4607` | `<experiment>/checkpoints/MLP/` | `best.pkl` | `model`, `optimizer`, `scheduler` | Legacy active path; not research-grade naming. |
| Iterative embeddings | `LLMbot/trainer.py:867`, `987` | intermediate data dir | `embeddings_iter_<iter>.pt` | embedding tensor | Legacy iterative cache; not a checkpoint. |
| Iterative soft labels | `LLMbot/trainer.py:868`, `985`, `4322`, `4631` | intermediate data dir | `soft_labels_iter_<iter>.pt` | soft-label tensor | Legacy iterative cache; not a checkpoint. |
| Raw LM embeddings | `LLMbot/trainer.py:963` | raw data dir | `embeddings_<model>.pt` | embedding tensor | Feature cache; not a checkpoint. |
| Text trainer default checkpoint | `LLMbot/trainers/text_trainer.py:850` | caller-provided path | `best_text.pt` by default | `model_state_dict`, validation score, epoch, metrics, config metadata | Active trainer path; naming partly descriptive but omits seed/metric unless parent path supplies it. |
| Text matrix checkpoint | `LLMbot/trainers/text_trainer.py:1206` | text matrix run dir | `best_text_seed<seed>.pt` | text trainer checkpoint payload | Active trainer path; includes seed but not metric/split/schema. |
| Fusion trainer checkpoint | `LLMbot/trainers/fusion_trainer.py:462` | caller-provided path | `best_model.pt` by default | `model_state_dict`, validation score, validation metrics | Active trainer path; too generic. |
| Dual-router split tensor | `LLMbot/trainers/dual_router_trainer.py:321` | `LLMbot/saved_artifacts/<group>/seed_<seed>/` | `split_indices.pt` | train/valid/test index tensors | Active artifact; not a checkpoint. |
| Dual-router stage outputs | `LLMbot/trainers/dual_router_trainer.py:415` | `LLMbot/saved_artifacts/<group>/seed_<seed>/` | `<name>.pt` | mixed output payloads | Active artifact writer; caller names must carry type. |
| Dual-router semantic expert | `LLMbot/trainers/dual_router_trainer.py:642` | `LLMbot/checkpoints/dual_router/<group>/seed_<seed>/` | `semantic_expert.pt` | state dict | Active checkpoint; readable but parent supplies seed/group. |
| Dual-router graph expert | `LLMbot/trainers/dual_router_trainer.py:839` | `LLMbot/checkpoints/dual_router/<group>/seed_<seed>/` | `<stage_name>_expert.pt` | state dict | Active checkpoint; may inherit vague `stage_*` names. |
| Dual-router pairwise router | `LLMbot/trainers/dual_router_trainer.py:974` | `LLMbot/checkpoints/dual_router/<group>/seed_<seed>/` | `pairwise_router.pt` | state dict | Active checkpoint; readable but parent supplies seed/group. |
| Precompute fragility proxies | `LLMbot/data/precompute.py:374` | precompute output dir | `per_node_fragility_proxies.pt` | layer mapping and fragility tensors | Active feature/diagnostic artifact. |
| Qwen layer embeddings | `LLMbot/data/precompute.py:442` | precompute output dir | `qwen3_emb_L<layer>.pt` | embedding tensor | Active feature cache; layer notation should be normalized later. |

## Load Surface Matrix

| Surface | Load site | Loader | `weights_only=False`? | Trust boundary and risk |
| --- | --- | --- | --- | --- |
| General utility reads | `LLMbot/utils/__init__.py:96` | `safe_torch_load` | No by default | Preferred load path for local tensor/checkpoint artifacts. |
| Dataset tensors | `LLMbot/utils/__init__.py:257` to `265` | `safe_torch_load` | No by default | Dataset tensors are local evidence inputs; still should not be hand-edited. |
| Distilled knowledge | `LLMbot/utils/__init__.py:288` to `292` | `safe_torch_load` | No by default | Legacy iterative caches; read compatibility should remain. |
| Data loader fallback | `LLMbot/data/loader.py:16` to `18` | `torch.load` with `weights_only=True`, fallback without flag | Fallback only | Acceptable PyTorch-version fallback; no explicit broad load unless old PyTorch lacks the flag. |
| Feature path resolution | `LLMbot/model_building.py:406` | `safe_torch_load` | No by default | Preferred path for feature tensors. |
| Graph detector checkpoint | `LLMbot/model_building.py:737` | `safe_torch_load` | No by default | Preferred checkpoint read. |
| Phase A checkpoint metadata | `LLMbot/main.py:264` | `safe_torch_load` | No by default | Preferred read for manifest-backed checkpoint. |
| Phase A outputs | `LLMbot/trainer.py:356`, `541` | `safe_torch_load` | No by default | Preferred read for outputs. |
| Legacy LM/GNN/MLP checkpoints | `LLMbot/trainer.py:787`, `848`, `966`, `1020`, `1051`, `4299`, `4362`, `4379`, `4515`, `4617`, `4662` | `safe_torch_load` | No by default | Safer loader, but filenames and schemas remain legacy. |
| Semantic embeddings and OOF outputs | `LLMbot/trainer.py:1715`, `1717`, `1795`, `1981`, `2016` | `safe_torch_load` | No by default | Preferred load path. |
| LM-only outputs | `LLMbot/trainer.py:3769`, `3910` | `safe_torch_load` | No by default | Preferred load path. |
| Text trainer checkpoint reload | `LLMbot/trainers/text_trainer.py:873` | `torch.load` | Yes | Trusted local checkpoint only; payload appears dict-like and should be narrowed later. |
| Fusion trainer checkpoint reload | `LLMbot/trainers/fusion_trainer.py:471` | `torch.load` | Yes | Trusted local checkpoint only; payload appears dict-like and should be narrowed later. |
| Dual-router explanation embeddings | `LLMbot/trainers/dual_router_trainer.py:468` | `torch.load` | Yes | Trusted local cache only; accepts tensor or dict with `embeddings`. |
| Oracle retrieval all artifacts | `LLMbot/scripts/oracle_retrieval_upper_bound.py:49` | local `_safe_torch_load` | Yes when supported | Trusted local diagnostic artifacts only; broad compatibility reader for mixed legacy schemas. |

## Naming Decision Matrix

| Name or pattern | Current status | Future policy | Rationale |
| --- | --- | --- | --- |
| `best.pkl` | Legacy active | Read-compatible only; do not introduce new writes for new experiment surfaces. | Hides model family, seed, metric, split, and schema. |
| `seed_<seed>_best.pkl` | Legacy active | Read-compatible only; avoid new writes. | Adds seed but still hides metric, split, and schema. |
| `checkpoint.pt` | Active inside manifest-backed directories | Allowed only when the parent directory is method-specific and manifest-backed. | Generic filename is acceptable only with a strong parent context. |
| `model.pt` | Active semantic finetune artifact | Avoid for new independent checkpoints; prefer method-scoped `*_checkpoint.pt`. | Too generic outside parent directory. |
| `classifier.pt` | Active semantic finetune artifact | Allowed inside a manifest-backed method directory; otherwise use `*_classifier_checkpoint.pt`. | Clear in context, weak outside context. |
| `outputs.pt` | Active artifact | Allowed only in method-specific manifest directories; otherwise use `*_outputs.pt`. | Generic name needs parent context. |
| `embeddings.pt` | Active artifact | Allowed only in method-specific manifest directories; otherwise use `*_embeddings.pt`. | Generic name needs parent context. |
| `router_checkpoint.pt` | Active estimator artifact | Replace in future writes with method-scoped names. | Router can mean uncertainty, residual risk, pairwise action routing, or other policies. |
| `residual_risk_selector_checkpoint.pt` | Active estimator artifact | Recommended pattern. | Names method scope and artifact type. |
| `login_router_checkpoint.pt` | Active estimator artifact | Keep only with manifest scope `hard_node_selection_only`. | Must not imply complete LOGIN reproduction. |
| `<stage_name>_expert.pt` | Active dual-router artifact | Avoid if `stage_name` is vague or includes legacy `stage_*`; prefer semantic method names. | Can reproduce the older stage-name drift. |
| `frozen/*` | Active legacy-named directories | Read-compatible only for existing pipelines; do not use in new paper-facing names. | Engineering-era vocabulary, not method vocabulary. |
| `g0/*` | Active legacy-named directories | Read-compatible only; do not introduce new `g0_*` names. | Too local and not self-explanatory. |
| `stage_*` | Present in concepts and some paths | Do not create new checkpoint or artifact names with this prefix. | Overloaded and not research-method-specific. |
| `*_checkpoint.pt` | Partially used | Preferred for loadable state payloads. | Makes `.pt` role explicit. |
| `*_outputs.pt` | Partially used | Preferred for logits/probabilities/predictions when parent path is not enough. | Separates outputs from weights. |
| `*_embeddings.pt` | Partially used | Preferred for node/text embeddings. | Separates features from weights. |
| `*_scores.pt` | Partially used | Preferred for scalar score vectors. | Separates risk/uncertainty scores from checkpoints. |

## Current Naming Risks

### High: `best.pkl` Is Not Research-Grade

The legacy distillation path repeatedly writes `best.pkl` under parent folders
such as `checkpoints/LM`, `checkpoints/GNN`, and `checkpoints/MLP`. The parent
folder carries some meaning, but the filename does not record backbone, seed,
selection split, selection metric, schema, or whether optimizer/scheduler state
is present.

Future code should keep old readers but stop expanding this pattern.

### High: `.pt` Carries Too Many Artifact Types

The active code uses `.pt` for checkpoints, outputs, embeddings, splits, soft
labels, entropy, diagnostics, and score vectors. This is normal in PyTorch, but
it makes filenames and manifests part of the artifact contract. New names must
encode the artifact type.

### Medium: Multiple Roots Coexist

The code writes to at least these active roots:

- `<experiment_root>/frozen/...`
- `<experiment_root>/stages/...`
- `<experiment_name>/checkpoints/...`
- `<experiment_name>/MLP_KD/...`
- `LLMbot/checkpoints/dual_router/...`
- `LLMbot/saved_artifacts/...`
- dataset or precompute output directories

New work should prefer a manifest-backed experiment root. Legacy roots should
remain readable but should not become the template for new outputs.

### Medium: Broad `weights_only=False` Loads Remain

Most active loads use `safe_torch_load`, but text/fusion trainers, dual-router
explanation embeddings, and the oracle diagnostic runner still use broad
`weights_only=False` loads. Some compatibility may be necessary, but each such
path should be treated as trusted-local-only and narrowed when practical.

### Medium: `stage`, `frozen`, And `g0` Leak Into Artifact Semantics

`frozen/g0` and stage-local paths are still active for compatibility. They
should remain readable, but new code should use method/backbone/source names
instead of propagating implementation-era terms.

## Migration Rule

1. Do not rename, move, delete, or rewrite historical local artifacts.
2. Keep old readers for `best.pkl`, `checkpoint.pt`, `outputs.pt`, `frozen/*`,
   and `g0/*` until the relevant runs are migrated or archived.
3. New writes must use lowercase snake_case.
4. New writes must make the artifact type explicit unless the parent directory
   is already method-specific and manifest-backed.
5. New checkpoint payloads should record or sit next to a manifest recording
   `checkpoint_schema`, `selection_metric`, `selection_split`,
   `source_checkpoint_path` when copied or derived, and the source command when
   available.
6. New code must not introduce `best.pkl`, `stage_*`, `frozen_*`, `g0_*`,
   `new_*`, or `final_*` names.
7. If a path must load with `weights_only=False`, it must be documented as a
   trusted-local-artifact path and should validate the expected payload shape.

## Recommended New Naming

Use these patterns for future active-mainline writes:

```text
<method>_<backbone>_seed<seed>_<split>_<metric>_checkpoint.pt
<method>_<backbone>_seed<seed>_outputs.pt
<method>_<semantic_source>_seed<seed>_embeddings.pt
<method>_<backbone>_seed<seed>_scores.pt
```

Examples:

```text
rgcn_roberta_seed1_val_macro_f1_checkpoint.pt
roberta_text_seed1_val_macro_f1_checkpoint.pt
residual_risk_rgcn_seed1_scores.pt
rgcn_oracle_retrieval_seed1_outputs.pt
```

Inside a strongly scoped manifest-backed directory, shorter fixed names remain
acceptable:

```text
checkpoint.pt
outputs.pt
embeddings.pt
manifest.json
```

The shorter names are acceptable only when the directory name and manifest
fully define method, backbone, seed, split, metric, and schema.

## Immediate Code Cleanup Plan

This section defines future implementation work. It is not marked complete by
this documentation update.

### Phase 1: New Writes Only

- Introduce or reuse one checkpoint/artifact naming helper for new active
  mainline writes.
- Keep all old readers unchanged.
- Route new checkpoint payloads to `*_checkpoint.pt`.
- Route new output payloads to `*_outputs.pt`.
- Route new embedding payloads to `*_embeddings.pt`.
- Route new scalar score vectors to `*_scores.pt`.
- Add manifest fields for schema, selection metric, selection split, and source
  checkpoint path where the current code omits them.

### Phase 2: Highest-Risk Legacy Names

- Leave `best.pkl` readable, but stop using it as the only checkpoint name for
  new runs.
- For legacy distillation paths that remain active, write a manifest-backed
  explicit checkpoint name in addition to the legacy compatibility file.
- Replace future `router_checkpoint.pt` writes with method-scoped names such as
  `multiview_router_checkpoint.pt` or
  `residual_risk_selector_checkpoint.pt`.
- Ensure RGCN oracle diagnostic outputs do not introduce new `stage_*`,
  `frozen_*`, or `g0_*` output names, even when reading legacy `frozen/g0`
  inputs.

### Phase 3: Load Boundary Narrowing

- Prefer `safe_torch_load` for dict-like local checkpoint payloads.
- Convert text and fusion checkpoint reloads away from direct
  `torch.load(..., weights_only=False)` if their payload schemas remain simple.
- For dual-router explanation embeddings and oracle diagnostics, keep broad
  compatibility only when necessary and validate tensor/dict payload shapes.
- Record trusted-local-only assumptions in manifests or adjacent comments for
  any remaining broad load path.

## Validation Contract

Documentation updates should be checked by:

- Scanning active `LLMbot/` `.py` files for `torch.save`, `torch.load`,
  `safe_torch_load`, `checkpoint`, and `best.pkl`.
- Confirming every active save/load family is represented in the matrices above.
- Confirming `LLMbot/baseline/` and `LLMbot/code/` are not described as active
  mainline contracts.
- Confirming no research code, checkpoint file, saved artifact, dataset, or test
  file was modified during documentation-only updates.

Future code updates should be checked by:

- CLI/help or import/compile checks for touched code.
- Manifest inspection for new fields and naming.
- Static grep for prohibited new names: `best.pkl`, `stage_*`, `frozen_*`,
  `g0_*`, `new_*`, and `final_*`.
- No training run unless explicitly requested by an experiment-running task.
