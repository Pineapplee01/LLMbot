# Baseline Comparability Protocol

**Version:** 1.0  
**Date:** 2026-04-09  
**Status:** Active

## Purpose

This protocol ensures fair comparison between baselines and proposed methods by standardizing:
- Data splits
- Evaluation metrics
- Experimental procedures
- Reporting standards

## Requirements

### 1. Data Splits

**MUST:**
- ✅ Use canonical splits from `datasets/TwiBot-20/`
- ✅ Load `train_idx.pt`, `valid_idx.pt`, `test_idx.pt` exactly as provided
- ✅ Maintain node ordering from raw dataset

**MUST NOT:**
- ❌ Regenerate splits with different random seeds
- ❌ Modify split indices
- ❌ Use different train/val/test ratios
- ❌ Create custom splits

### 2. Evaluation Metrics

**Primary Metrics (REQUIRED):**
- Accuracy
- Macro-F1

**Auxiliary Metrics (OPTIONAL):**
- ECE (Expected Calibration Error)
- NLL (Negative Log-Likelihood)
- Brier Score
- AURC (Area Under Risk-Coverage Curve)

**Reporting:**
- Primary metrics in main comparison table
- Auxiliary metrics in separate diagnostic table
- Mean and standard deviation across multiple seeds (≥3)

### 3. Hyperparameter Tuning

**MUST:**
- ✅ Tune hyperparameters on validation set only
- ✅ Select best model based on validation performance
- ✅ Report final results on test set (single evaluation)

**MUST NOT:**
- ❌ Tune on test set
- ❌ Select models based on test performance
- ❌ Evaluate test set multiple times during development

### 4. Reproducibility

**MUST:**
- ✅ Set random seeds for reproducibility
- ✅ Document all hyperparameters
- ✅ Report library versions (PyTorch, transformers, etc.)
- ✅ Provide configuration files

**MUST NOT:**
- ❌ Cherry-pick best runs
- ❌ Report results without seeds
- ❌ Omit preprocessing steps

## Baseline Checklist

Before claiming improvement over a baseline, verify:

- [ ] Baseline uses canonical splits from `datasets/TwiBot-20/`
- [ ] Baseline reports Accuracy and Macro-F1
- [ ] Baseline was not tuned on test set
- [ ] Baseline uses same node ordering
- [ ] Baseline preprocessing is documented
- [ ] Your method follows same protocol

## Comparison Table Format

**Main Table (Primary Metrics):**

| Method | Accuracy | Macro-F1 | Seeds |
|--------|----------|----------|-------|
| BotBR | 0.XXX ± 0.XXX | 0.XXX ± 0.XXX | 3 |
| LMbot GNN | 0.8533 ± 0.XXX | 0.8732 ± 0.XXX | 5 |
| LMbot LM | 0.8554 ± 0.XXX | 0.8756 ± 0.XXX | 5 |
| HyperScan | - | 0.872* | - |
| Ours (RW4) | 0.XXX ± 0.XXX | 0.XXX ± 0.XXX | 3 |

*Reported from paper (reproduction in progress)

**Auxiliary Table (Diagnostic Metrics):**

| Method | ECE ↓ | NLL ↓ | Brier ↓ | AURC ↓ |
|--------|-------|-------|---------|--------|
| ... | ... | ... | ... | ... |

## Handling Non-Reproducible Baselines

If a baseline cannot be reproduced on the unified protocol:

1. **Include in main table** with annotation
2. **Mark as "reported"** with paper citation
3. **Document reproduction status** (in progress / attempted / not attempted)
4. **Do not exclude** from comparison

Example:
```
HyperScan: 87.2% F1 (reported, HyperScan paper on TwiBot-20)
```

## Verification

**Before submission:**
```bash
# Verify splits match canonical
python tools/verify_splits.py --dataset TwiBot-20

# Check protocol compliance
python tools/check_protocol.py --experiment rw4_disagreement_local

# Generate comparison table
python tools/compare_results.py --output experiments/results/comparison.csv
```

## References

- TwiBot-20 dataset: [paper citation]
- Canonical splits: `datasets/TwiBot-20/`
- Baseline implementations: `botbr/`, `HyperScan/`

## Updates

Protocol updates require:
1. Version increment
2. Changelog entry
3. Notification to all contributors
4. Re-evaluation of affected experiments

---

**Compliance:** All experiments in this project must follow this protocol. Non-compliant results will not be included in papers or reports.
