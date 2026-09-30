# Routed-Node Qwen PEFT vs LMBot RoBERTa Finetune Comparison

Date: 2026-06-07
Status: approved-in-thread
Scope: `botsay_official/` plus `LLMbot/` active mainline

## Goal

Run a routed-node-specific comparison between:

1. `Qwen2.5-7B-Instruct + LoRA` under the existing `botsay_official` SFT shell
2. `LLMbot` semantic finetuning with `roberta_finetuned`

The comparison must answer two questions on the same routed-node target set:

- does node-specialized finetuning reduce `break` on routed hard nodes?
- how strongly does semantic strength correlate with final correction quality?

## Shared Experiment Contract

- Dataset: `TwiBot-20`
- Routed set source:
  `routed_nodes_highbase_preiter_budget020_seed1_20260529.json`
- Routed split contract:
  - `train`: first `split_counts.train` node indices
  - `valid`: next `split_counts.valid` node indices
  - `test`: final `split_counts.test` node indices
- Node index semantics:
  indices point into the canonical labeled order
  `train + dev + test`
- Evaluation target:
  routed-node metrics plus frozen-SimTeG override replay
  (`fix / break / net`)

## Qwen Line

Reuse the existing `botsay_official` pipeline:

- `approach-finetune.py` generates SFT corpus from routed subsets
- `sft.py` performs LoRA SFT
- `approach-finetune_eval.py` evaluates routed subsets

Required adaptations:

- accept `--routed_nodes_path`
- accept `--routed_split {train,valid,test,all}`
- exemplar retrieval pool may use routed-train ids
- retrieval must exclude the current target node to avoid self-retrieval leakage

This remains a BotSay-compatible shell, not an official BotSay reproduction.

## RoBERTa Line

Reuse `LLMbot` semantic training code:

- `--experiment_task semantic_encoder_finetune`
- `--semantic_encoder roberta_finetuned`

Required adaptations:

- semantic stages accept `--routed_nodes_path`
- routed split replaces canonical `train_idx / valid_idx / test_idx`
- `roberta_finetuned` must initialize from the existing SimTeG LM checkpoint
  (`LM_pretrain/best.pkl`) when available
- no fallback to raw `roberta-base`

Optional paired ablation:

- `semantic_embedding_classifier` may also use the same routed split for a
  cached-embedding MLP comparison

## Edit Zones

- `botsay_official/retrieve_utils.py`
- `botsay_official/approach-descandmeta.py`
- `botsay_official/approach-finetune.py`
- `botsay_official/approach-finetune_eval.py`
- `LLMbot/parser_args.py`
- `LLMbot/trainer_semantic.py`
- `LLMbot/README.md`
- `docs/code/parser.md`
- `docs/code/research.md`
- `docs/ARCHITECTURE.md`
- `LLMbot/experiments.md`

## Out Of Scope

- no backbone or router retraining changes
- no prompt-cache regeneration
- no new source files beyond required docs
- no claim that either line is a full official reproduction of external papers

## Validation

- local static check:
  `python -m py_compile ...`
- routed smoke on server for:
  - Qwen corpus generation
  - Qwen routed eval
  - LLMbot semantic finetune with routed split
- full server runs record artifact roots and metrics paths in
  `LLMbot/experiments.md`
