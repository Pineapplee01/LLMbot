# Phase 0: Environment Validation

**Date**: 2026-04-17
**Status**: PASS

## Compile Check
- baseline/core/ (10 files): ALL PASS
- baseline/analysis/ (8 files): ALL PASS
- baseline/baselines/ (5 files): ALL PASS

## Remote Server
- Host: 172.31.106.108:10011
- Seeds 1-5: GNN best.pkl + LM_pretrain best.pkl + embeddings_iter_-1.pt — ALL PRESENT
- Code synced: LLMbot/baseline/ uploaded and import-tested
- Import fix: added `sys.path.insert(0, core/)` to analysis/ and baselines/ scripts

## Path Convention
- Dataset: `/root/workspace/LMbot/datasets/TwiBot-20/`
- Experiments: `/root/workspace/LMbot/TwiBot-20_seed_{seed}/`
- Diagnostics output: `TwiBot-20_seed_{seed}/diagnostics/`
- exp_base template: `TwiBot-20_seed_{seed}`

## Version Constraints
- Phase 1: build_failure_regime_table.py **v2** (prior-only, no oracle contamination)
- Phase 4: action_outcome_table.py **v3** (train/val/test discipline, soft message scaling)
