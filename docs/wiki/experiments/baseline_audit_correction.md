---
type: experiment
node_id: exp:baseline_audit_correction
title: Baseline Comparability Audit — CORRECTION
status: completed
priority: critical
created_at: 2026-04-09T23:30:00Z
updated_at: 2026-04-09T23:30:00Z
tags: [baseline, comparability, audit, correction]
---

# Baseline Comparability Audit — CORRECTION

## Critical Update: Previous Audit Analyzed Wrong Codebase

**Date**: 2026-04-09 23:30

**Previous Status**: FAIL (baseline_comparability_audit.md identified critical issues)

**Corrected Status**: ✅ **PASS** — Configuration is correct, results are valid

---

## What Happened

The previous baseline comparability audit (exp:baseline_comparability_audit) analyzed the **root-level codebase** (`main.py`, `trainer.py`, `parser_args.py`) and found configuration issues.

However, the **actual rw1/rw4 experiments** used the **LLMbot/code/ codebase**, which has correct configuration.

---

## Verified Configuration (LLMbot/code/)

### ✅ Issue 1: F1 Metric — CORRECT

**LLMbot/code/trainer.py**:
- All `f1_score()` calls include `average='macro'`
- Lines verified: 245, 386, 674, 944
- **Conclusion**: Macro F1 is correctly computed

### ✅ Issue 2: Split Protocol — CORRECT

**LLMbot/code/ split usage**:
- Uses unified split files: `datasets/TwiBot-20/{train,valid,test}_idx.pt`
- Split sizes: train=8,278, valid=2,365, test=1,183
- Total nodes: 11,826
- **Conclusion**: Unified split is correctly used

### ✅ Issue 3: Comparability with LMbot — VERIFIED

**LMbot baseline**:
- Uses same split files
- Uses same macro F1 metric
- Results are directly comparable

---

## Verified Experimental Results

### rw1 (Calibration Baseline)

| Metric | Value | Notes |
|--------|-------|-------|
| Macro-F1 | 0.7391 | ✅ Valid |
| Accuracy | — | — |
| ECE | 0.0738 | ✅ Valid |
| Disagreement-slice F1 | 0.4079 | ✅ Valid |
| Coverage | 29.1% | — |

### rw4 (Main Method: Disagreement-Local Repair)

| Metric | Value | Notes |
|--------|-------|-------|
| Macro-F1 | 0.7603 | ✅ Valid (+0.0212 vs rw1) |
| Accuracy | — | — |
| ECE | 0.0190 | ✅ Valid (-0.0548 vs rw1, 74% reduction) |
| Disagreement-slice F1 | 0.5881 | ✅ Valid (+0.1802 vs rw1) |
| Coverage | 22.3% | More selective than rw1 |

---

## Kill Test Verification

### Kill Test B2: Repair vs Deferral ✅ **STRONG PASS**

**Hypothesis**: Propagation-level repair beats output-level deferral

**Result**: 
- Disagreement-slice F1 gain: **+0.1802** (required: ≥0.01)
- **18x the threshold!**
- Even better than the +0.1557 reported in KILL_TEST_RESULTS.md

**Verdict**: ✅ **STRONG PASS** — Mechanism is validated

### Kill Test B3: Utility Ablation ✅ **PASS**

**Hypothesis**: Utility-guided components are necessary

**Result**:
- Removing q_graph trigger: -0.0060 F1, +0.0166 ECE
- Removing soft gating: -0.0060 F1, +0.0426 ECE
- Both meet thresholds (≥0.005 F1 loss, ≥0.015 ECE loss)

**Verdict**: ✅ **PASS** — Utility components are necessary

---

## Corrected Assessment

### What This Means

1. **Configuration is correct** — No fixes needed
2. **Results are valid** — rw1/rw4 results are comparable to LMbot baseline
3. **Kill tests pass** — Mechanism is validated
4. **Baseline reproduction unblocked** — Can proceed with LMbot/BotBR comparison

### Gap Analysis: Why F1 < 0.90?

**Current performance**:
- rw4 Macro-F1: 0.7603
- Target: 0.90
- **Gap: -0.1397 (13.97 percentage points)**

**Comparison with baselines**:
- LMbot GNN: F1=0.8732 (gap: -0.1129)
- LMbot LM: F1=0.8756 (gap: -0.1153)
- BotBR: F1~0.868 (gap: -0.1077)
- HyperScan: F1=0.872 (gap: -0.1117)

**Analysis**:
- Current method is **~11-13 points below** strong baselines
- This is a **significant performance gap**
- Mechanism is sound (kill tests pass), but **absolute performance is insufficient**

---

## Implications for Research Strategy

### Tier 1 Claim Status: ❌ NOT ACHIEVABLE with current method

**Original goal**: Competitive Acc/F1 on unified protocol (≥0.90)

**Reality**: 
- rw4 F1=0.7603 is **far below** LMbot (0.8732) and target (0.90)
- Even with improvements, unlikely to reach 0.90 without major redesign

**Recommendation**: **Pivot to Tier 2/3 narrative** (mechanism paper) OR **redesign method** for competitive performance

### Three Strategic Options

#### Option A: Pivot to Mechanism Paper (Low Risk, 1-2 weeks)
- **Focus**: LLM-guided graph modification mechanism (kill tests validated)
- **Main claim**: Disagreement-slice F1 gain +0.18 (18x threshold)
- **Venue**: TMLR (mechanism-focused) or WWW (targeted improvement)
- **Pros**: Kill tests already pass, mechanism validated, low risk
- **Cons**: Lower impact than competitive performance paper

#### Option B: Hybrid Method (Medium Risk, 2-3 weeks)
- **Approach**: Combine rw4 repair mechanism with stronger base model
- **Strategy**: Use LMbot LM (F1=0.8756) as base, apply rw4 repair on disagreement nodes
- **Target**: 0.8756 + 0.02 improvement = 0.8956 ≈ 0.90
- **Pros**: Leverages validated mechanism, realistic path to 0.90
- **Cons**: Requires integration work, may not reach exactly 0.90

#### Option C: Method Redesign (High Risk, 3-4 weeks)
- **Approach**: Implement Idea #1 (Selective Abstention) or Idea #2 (Counterfactual Consistency)
- **Target**: Competitive performance from scratch
- **Pros**: Potential for higher impact
- **Cons**: High risk, time-consuming, no guarantee of success

---

## Recommended Next Steps

### Immediate (Today)

1. **Update research-wiki** to reflect corrected audit findings
2. **Unblock baseline reproduction plan** (status: blocked → ready)
3. **Decide strategic direction** (Option A/B/C)

### Short-term (1-3 days)

**If Option A (Mechanism Paper)**:
- Multi-seed validation (B6)
- Paper writing focused on mechanism
- Target: TMLR or WWW

**If Option B (Hybrid Method)**:
- Integrate rw4 with LMbot LM base
- Re-run experiments
- Target: 0.90 F1

**If Option C (Method Redesign)**:
- Implement Idea #1 or #2
- Full experimental pipeline
- Target: Competitive performance

---

## Files to Update

1. ✅ **This file** (baseline_audit_correction.md) — Created
2. ⏳ **docs/wiki/log.md** — Add correction entry
3. ⏳ **docs/wiki/experiments/baseline_reproduction_plan.md** — Change status from "blocked" to "ready"
4. ⏳ **docs/wiki/query_pack.md** — Update "Open Unknowns" section
5. ⏳ **docs/wiki/ARIS_WORKFLOW_GUIDANCE.md** — Update with corrected status and strategic options

---

## Conclusion

**Previous audit was incorrect** — analyzed wrong codebase.

**Actual status**: Configuration is correct, results are valid, kill tests pass.

**New challenge**: Performance gap to 0.90 target requires strategic decision on research direction.
