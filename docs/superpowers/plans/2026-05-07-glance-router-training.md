# GLANCE Router Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a maintainable GLANCE paper-style advantage/cost-aware training path as an explicit ablation while keeping the default Stage 2 selector as OOF residual-risk screening.

**Architecture:** `LLMbot/baseline/core/estimators.py` will own the reusable GLANCE target builders and the selector training branch. The default `GlanceForContextResidualRiskSelector.fit()` path remains `training_objective="residual_error"` and trains on OOF residual errors only; the new `training_objective="glance_advantage"` path requires explicit GNN-vs-LLM/refiner counterfactual supervision and records that it is not the canonical Stage 2 residual-risk selector. CLI and StageRunner wiring expose the ablation without changing existing artifact compatibility.

**Tech Stack:** Python, NumPy, PyTorch tensors as inputs, scikit-learn `StandardScaler` and `LogisticRegression`, existing pytest suite under `LLMbot/baseline/core/tests/`.

---

## Evidence From GLANCE Paper

Loveland et al. 2025 define GLANCE as a lightweight router over cheap node features: GNN embedding, uncertainty, degree/node features, and soft estimated homophily. The router outputs `a_v = sigmoid(w^T f_v)` and routes top-K nodes under a fixed budget. Its training is not residual-error detection: it compares the loss of an LLM/refiner route against a GNN-only counterfactual, subtracts an LLM query cost beta, and optimizes a cost-aware routing objective.

For this repository, the safe adaptation is:

- Default Stage 2: train `P(Stage1 wrong | Stage1 outputs/features)` on OOF residual labels.
- Optional GLANCE ablation: train a route-benefit score from explicit GNN-vs-LLM/refiner counterfactual losses or correctness labels, only when those labels are supplied.
- Never silently infer paper-style GLANCE route labels from Stage 3 action or repair outcomes.

---

## File Structure

- Modify: `LLMbot/baseline/core/estimators.py`
  - Add small target-building helpers for residual and GLANCE advantage objectives.
  - Add a `GlanceTrainingTarget` container so the classifier fit path has one clean interface.
  - Update `GlanceForContextResidualRiskSelector` to support `training_objective="residual_error"` and `training_objective="glance_advantage"`.
  - Preserve legacy aliases such as `GlanceForContextResidualRiskRouter`.
- Modify: `LLMbot/baseline/core/parser_args.py`
  - Add `glance_for_context_residual_risk_selector` to `--estimator_mode`.
  - Add `glance_for_context` to `--risk_variant`.
  - Add `--glance_training_objective` and `--glance_llm_query_cost`.
- Modify: `LLMbot/baseline/core/trainer.py`
  - Instantiate the GLANCE selector when requested.
  - Pass the training objective and cost into the estimator.
  - Do not pass Stage 3 outcomes into Stage 2.
- Modify: `LLMbot/baseline/core/stage2_router_oof_smoke.py`
  - Include `glance_for_context` in the variant suite.
  - Add CLI arguments mirroring `parser_args.py`.
  - Pass the training objective and query cost into the estimator.
- Test: `LLMbot/baseline/core/tests/test_artifact_and_subgroup_helpers.py`
  - Add tests for advantage target construction.
  - Add tests that requesting GLANCE advantage training without explicit counterfactual supervision raises a clear error.
  - Add tests that the residual default remains OOF-only and screening-only.

---

### Task 1: Lock Current Residual vs Paper-Style Semantics With Tests

**Files:**
- Modify: `LLMbot/baseline/core/tests/test_artifact_and_subgroup_helpers.py`

- [ ] **Step 1: Add a failing test for GLANCE advantage target construction**

Add this test near the existing GLANCE tests:

```python
def test_glance_advantage_targets_from_losses_are_cost_aware():
    target = estimators.build_glance_advantage_training_target(
        gnn_loss=np.array([0.8, 0.2, 0.4, 1.2]),
        llm_loss=np.array([0.3, 0.4, 0.1, 1.1]),
        llm_query_cost=0.2,
        fit_idx=np.array([0, 1, 2, 3]),
    )

    assert target.objective == "glance_advantage"
    assert target.target_semantics == "1[loss_gnn - loss_llm - beta > 0]"
    assert target.fit_idx.tolist() == [0, 1, 2, 3]
    assert target.labels.tolist() == [1, 0, 1, 0]
    assert target.advantage.tolist() == pytest.approx([0.3, -0.4, 0.1, -0.1])
    assert target.metadata["llm_query_cost"] == pytest.approx(0.2)
```

- [ ] **Step 2: Add a failing test for correctness-label fallback**

Add this test after the loss-based target test:

```python
def test_glance_advantage_targets_from_correctness_labels_use_utility_margin():
    target = estimators.build_glance_advantage_training_target(
        gnn_correct=np.array([False, True, False, True]),
        llm_correct=np.array([True, True, False, False]),
        llm_query_cost=0.25,
        fit_idx=np.array([0, 1, 2, 3]),
    )

    assert target.labels.tolist() == [1, 0, 0, 0]
    assert target.advantage.tolist() == pytest.approx([0.75, -0.25, -0.25, -1.25])
    assert target.metadata["target_source"] == "correctness_advantage"
```

- [ ] **Step 3: Add a failing test for explicit-supervision requirement**

Add this test after the existing `test_glance_for_context_selector_is_oof_residual_risk_not_action_router`:

```python
def test_glance_advantage_training_requires_counterfactual_supervision():
    labels = torch.tensor([0, 1, 0, 1])
    probs = torch.tensor([[0.8, 0.2], [0.7, 0.3], [0.4, 0.6], [0.2, 0.8]], dtype=torch.float32)
    selector = estimators.GlanceForContextResidualRiskSelector(training_objective="glance_advantage")

    with pytest.raises(ValueError, match="requires explicit GNN-vs-LLM counterfactual supervision"):
        selector.fit(
            logits=torch.log(probs.clamp_min(1e-8)),
            probs=probs,
            labels=labels,
            train_idx=torch.tensor([0, 1]),
            val_idx=torch.tensor([2]),
            fit_idx=torch.tensor([0, 1, 2]),
        )
```

- [ ] **Step 4: Run the focused tests and confirm failure**

Run:

```bash
cd LLMbot/baseline/core
python -m pytest tests/test_artifact_and_subgroup_helpers.py -q
```

Expected: FAIL because `build_glance_advantage_training_target` and `training_objective` are not implemented yet.

---

### Task 2: Add GLANCE Training Target Abstractions

**Files:**
- Modify: `LLMbot/baseline/core/estimators.py`

- [ ] **Step 1: Add a small immutable target container**

Add near the constants/imports:

```python
from dataclasses import dataclass
```

Add after `_stage1_residual_errors`:

```python
@dataclass(frozen=True)
class GlanceTrainingTarget:
    objective: str
    labels: np.ndarray
    fit_idx: np.ndarray
    target_semantics: str
    metadata: dict
    advantage: np.ndarray | None = None
```

If the local Python version rejects `np.ndarray | None`, use `Optional[np.ndarray]` and import `Optional` from `typing`.

- [ ] **Step 2: Add a residual target builder**

Add after the dataclass:

```python
def build_residual_error_training_target(labels, probs, fit_idx):
    _, _, residual_errors, _ = _stage1_residual_errors(labels, probs)
    fit_idx_np = _valid_index_array(fit_idx, residual_errors.shape[0])
    return GlanceTrainingTarget(
        objective="residual_error",
        labels=residual_errors[fit_idx_np].astype(np.int32),
        fit_idx=fit_idx_np,
        target_semantics="1[Stage1 prediction != y]",
        metadata={
            "target_source": "oof_stage1_residual_error",
            "stage3_action_or_repair_outcome_used": False,
            "llm_counterfactual_outcome_used": False,
            "fit_count": int(fit_idx_np.size),
            "positive_count": int(residual_errors[fit_idx_np].sum()) if fit_idx_np.size else 0,
        },
        advantage=None,
    )
```

- [ ] **Step 3: Add a GLANCE advantage target builder**

Add after the residual target builder:

```python
def build_glance_advantage_training_target(
    *,
    fit_idx,
    llm_query_cost=0.2,
    gnn_loss=None,
    llm_loss=None,
    gnn_correct=None,
    llm_correct=None,
):
    # Normalize every provided vector to node-aligned NumPy arrays.
    # Loss route: advantage = loss_gnn - loss_llm - beta.
    # Correctness route: advantage = 1[llm correct] - 1[gnn correct] - beta.
```

Implementation requirements:

- Raise `ValueError("GLANCE advantage training requires explicit GNN-vs-LLM counterfactual supervision ...")` if neither `(gnn_loss, llm_loss)` nor `(gnn_correct, llm_correct)` is provided.
- Validate that paired arrays have the same node count.
- Use `_valid_index_array(fit_idx, node_count)`.
- Construct `advantage_fit`, then labels `advantage_fit > 0`.
- Metadata must include:
  - `target_source`: `loss_advantage` or `correctness_advantage`
  - `target_semantics`
  - `llm_query_cost`
  - `stage3_action_or_repair_outcome_used`: `False`
  - `llm_counterfactual_outcome_used`: `True`
  - `fit_count`
  - `positive_count`

- [ ] **Step 4: Run the two target tests**

Run:

```bash
cd LLMbot/baseline/core
python -m pytest tests/test_artifact_and_subgroup_helpers.py::test_glance_advantage_targets_from_losses_are_cost_aware tests/test_artifact_and_subgroup_helpers.py::test_glance_advantage_targets_from_correctness_labels_use_utility_margin -q
```

Expected: PASS.

---

### Task 3: Refactor GLANCE Selector Training

**Files:**
- Modify: `LLMbot/baseline/core/estimators.py`

- [ ] **Step 1: Extend constructor arguments**

Change `GlanceForContextResidualRiskSelector.__init__` signature to include:

```python
training_objective="residual_error",
```

Store:

```python
self.training_objective = str(training_objective)
```

- [ ] **Step 2: Rename logistic helper for semantics**

Rename `_fit_logistic_router(self, feature_matrix, residual_fit_labels)` to:

```python
def _fit_logistic_score_model(self, feature_matrix, target_labels):
```

Keep a legacy alias if needed:

```python
_fit_logistic_router = _fit_logistic_score_model
```

- [ ] **Step 3: Add `_build_training_target`**

Add a private method:

```python
def _build_training_target(self, labels, probs, fit_idx, **kwargs):
    objective = str(kwargs.get("training_objective", self.training_objective)).lower()
    if objective in {"residual", "residual_error", "stage2_residual_error"}:
        return build_residual_error_training_target(labels=labels, probs=probs, fit_idx=fit_idx)
    if objective in {"glance_advantage", "advantage", "paper_advantage"}:
        return build_glance_advantage_training_target(
            fit_idx=fit_idx,
            llm_query_cost=kwargs.get("llm_query_cost", self.llm_query_cost),
            gnn_loss=kwargs.get("gnn_loss"),
            llm_loss=kwargs.get("llm_loss"),
            gnn_correct=kwargs.get("gnn_correct"),
            llm_correct=kwargs.get("llm_correct"),
        )
    raise ValueError(f"Unknown GLANCE training objective: {objective}")
```

- [ ] **Step 4: Use the target object in `fit`**

Replace the direct `residual_errors[fit_idx_np]` training section with:

```python
training_target = self._build_training_target(labels, probs, fit_idx, **kwargs)
fit_idx_np = training_target.fit_idx
router_available = self._fit_logistic_score_model(feature_matrix[fit_idx_np], training_target.labels)
```

For `train_metrics`, use:

```python
training_labels_for_metrics = training_target.labels
```

Metadata must record:

```python
"training_objective": training_target.objective,
"target_semantics": training_target.target_semantics,
"target_metadata": training_target.metadata,
"llm_counterfactual_outcome": bool(training_target.metadata.get("llm_counterfactual_outcome_used", False)),
```

Keep residual-error metrics in `build_manifest` unchanged, because Stage 2 evaluation still measures residual-error ranking even when a GLANCE advantage ablation is trained.

- [ ] **Step 5: Update candidate metadata**

In `_metadata_for_candidates`, set `fit_scope` and family based on `self.training_objective`:

```python
"fit_scope": "explicit_glance_counterfactual_advantage_labels" if self.training_objective == "glance_advantage" else "train_split_oof_residual_labels_only"
```

- [ ] **Step 6: Run selector tests**

Run:

```bash
cd LLMbot/baseline/core
python -m pytest tests/test_artifact_and_subgroup_helpers.py::test_glance_for_context_selector_is_oof_residual_risk_not_action_router tests/test_artifact_and_subgroup_helpers.py::test_glance_advantage_training_requires_counterfactual_supervision -q
```

Expected: PASS.

---

### Task 4: Wire Optional GLANCE Variant Without Changing Canonical Stage 2

**Files:**
- Modify: `LLMbot/baseline/core/parser_args.py`
- Modify: `LLMbot/baseline/core/trainer.py`
- Modify: `LLMbot/baseline/core/stage2_router_oof_smoke.py`

- [ ] **Step 1: Add CLI choices**

In `parser_args.py`, add:

```python
"glance_for_context_residual_risk_selector",
```

to `--estimator_mode` choices.

Add:

```python
"glance_for_context",
```

to `--risk_variant` choices.

Add module controls:

```python
parser.add_argument(
    "--glance_training_objective",
    type=str,
    default="residual_error",
    choices=["residual_error", "glance_advantage"],
    help="GLANCE-inspired selector objective. glance_advantage requires explicit GNN-vs-LLM counterfactual supervision.",
)
parser.add_argument("--glance_llm_query_cost", type=float, default=0.2)
```

- [ ] **Step 2: Instantiate GLANCE selector in trainer**

In `_build_multiview_router_bundle`, before GETS branch, add:

```python
if str(risk_variant).lower() in {"glance_for_context", "glance_residual_risk_selector"} or estimator_mode == "glance_for_context_residual_risk_selector":
    estimator = GlanceForContextResidualRiskSelector(
        budgets=budgets,
        training_objective=getattr(self.args, "glance_training_objective", "residual_error"),
        llm_query_cost=getattr(self.args, "glance_llm_query_cost", 0.2),
    )
```

Import `GlanceForContextResidualRiskSelector` at the top of `trainer.py` if it is not already imported.

- [ ] **Step 3: Pass GLANCE objective kwargs in trainer fit**

Add to `estimator.fit(...)`:

```python
training_objective=getattr(self.args, "glance_training_objective", "residual_error"),
llm_query_cost=getattr(self.args, "glance_llm_query_cost", 0.2),
```

No Stage 3 outcome keys should be passed.

- [ ] **Step 4: Add smoke CLI variant**

In `stage2_router_oof_smoke.py`, add `glance_for_context` to parser risk choices and `variant_names`.

Instantiate:

```python
elif variant == "glance_for_context":
    estimator = GlanceForContextResidualRiskSelector(
        budgets=budgets,
        training_objective=cli.glance_training_objective,
        llm_query_cost=cli.glance_llm_query_cost,
    )
```

Add CLI args:

```python
parser.add_argument("--glance_training_objective", default="residual_error", choices=["residual_error", "glance_advantage"])
parser.add_argument("--glance_llm_query_cost", type=float, default=0.2)
```

Import `GlanceForContextResidualRiskSelector`.

- [ ] **Step 5: Run compile**

Run:

```bash
cd LLMbot/baseline/core
python -m py_compile estimators.py parser_args.py trainer.py stage2_router_oof_smoke.py stage2_residual_risk_oof_smoke.py
```

Expected: no output and exit code 0.

---

### Task 5: Verification and Reporting

**Files:**
- No new code files unless a test failure requires a small fix.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
cd LLMbot/baseline/core
python -m pytest tests/test_artifact_and_subgroup_helpers.py -q
```

Expected: PASS.

- [ ] **Step 2: Inspect changed files**

Run:

```bash
git diff -- LLMbot/baseline/core/estimators.py LLMbot/baseline/core/parser_args.py LLMbot/baseline/core/trainer.py LLMbot/baseline/core/stage2_router_oof_smoke.py LLMbot/baseline/core/tests/test_artifact_and_subgroup_helpers.py docs/superpowers/plans/2026-05-07-glance-router-training.md
```

Expected: Only scoped GLANCE target/training, CLI, and tests changes.

- [ ] **Step 3: Final evidence statement**

Report:

- Current GLANCE-style selector is trained today as OOF residual-error logistic ranking.
- It was not trained with GLANCE's original advantage/cost-aware LLM-query objective before this change.
- The new code adds that objective as an explicit ablation requiring counterfactual GNN-vs-LLM supervision.
- Default Stage 2 remains residual-risk hard-node selection and does not use Stage 3 action or repair outcomes.

---

## Self-Review

- Spec coverage: The plan answers the user's core question and adds a GLANCE paper-style objective while preserving Stage 2 semantics.
- Placeholder scan: No task contains open-ended placeholders; each code change has concrete signatures and expected behavior.
- Type consistency: The target builders, selector constructor, CLI names, and metadata keys are consistent across tasks.
