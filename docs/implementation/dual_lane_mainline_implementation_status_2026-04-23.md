# Dual-Lane Mainline Implementation Status

**Date**: 2026-04-23  
**Snapshot type**: dated implementation snapshot, not semantic versioning  
**Code surface**: `LLMbot/baseline/core`

## Summary

这份说明面向后续实现者和 agent，总结当前 `LLMbot/baseline/core` dual-lane mainline 改造后的代码状态：哪些接口已经变化，哪些能力已经实现，哪些事实已经验证，以及哪些部分仍处于 `partial / pending` 状态。

## Version definition

- 这里的“当前版本”是本轮 dual-lane harness 改造后的日期快照，不是 semver，也不是 release tag。
- 快照对象是当前默认执行面 `LLMbot/baseline/core`。
- 本文不回写 `LLMbot/code` 的历史方法线，也不把历史结论混入当前 mainline 状态。

## Interface and contract changes

### CLI / stage surface

- `main.py` 是 formal stages 的唯一正式入口。
- `parser_args.py` 已注册 supporting-comparison stages：
  - `semantic_source_matrix`
  - `positioning_matrix`
- `--reset_split` 默认值已改为 `-1`。
- formal stages 收到非 `-1` split override 时直接 `fail fast`。
- formal stage 不再静默改写用户传入的 `GNN_model`。

### Artifact contract

- 每个 formal stage 统一输出：
  - `metrics.json`
  - `budget_curve.csv`
  - `subgroup_report.json`
  - `notes.md`
- 同时保留内部工件，供跨 lane 读取和诊断：
  - `risk_manifest.json`
  - `action_table.json`
  - `selection_metrics.json`
  - `gain_cost_report.json`
  - `gain_explain.json`

### Subgroup contract

- subgroup 层显式拆成两套 manifest：
  - `subgroup_manifest_operational`
  - `subgroup_manifest_analysis`
- selector / trigger / policy 只允许读取 `operational` manifest。
- analysis 侧保留更强的 oracle / learned diagnostics，不作为 selector-safe 输入。

### Mode metadata

- 下列 mode 角色已经在 metadata 中显式编码：
  - `qwen_frozen`: `supporting_mainline`
  - `qwen_peft`: `supporting_mainline`
  - `ib_edl`: `supporting_mainline_calibration`
  - `glance_boundary`: `main_paper_boundary_baseline`
  - `cs`: `conditional_supporting_baseline`

## Implemented

### `main.py` 单入口治理

- `main.py` 现在承担 dual-lane mainline 的正式路由职责。
- formal split 约束在入口层显式执行，而不是依赖调用者自觉遵守。
- 之前会绕开 stage 语义的 shortcut / alias 已从正式路径中清理。

### `trainer.py` 的 dual-lane `StageRunner`

- `StageRunner` 已统一承载以下 stages：
  - `vertical_minimal`
  - `estimator_matrix`
  - `semantic_matrix`
  - `semantic_source_matrix`
  - `repair_matrix`
  - `selector_matrix`
  - `positioning_matrix`
  - `backbone_stress`
  - `appendix`
- 公共 helper 统一负责：
  - metrics payload
  - budget curve
  - subgroup report
  - paired bootstrap summary
  - gain/cost 与 gain explanation 输出

### semantic source 的真实执行路径

- `semantic_source_matrix` 不再只是 metadata 占位。
- 当前已实现的 source / calibration 路径包括：
  - `S0`: 复用最佳 semantic operator 结果上的 RoBERTa local source
  - `S1`: 使用 cached `qwen3_emb_last.pt` 的 Qwen frozen local source
  - `S2`: 基于 cached Qwen embedding 的 low-rank adapter-style PEFT proxy
  - `S3`: 基于历史 head 接入的 IB-EDL calibration-support 分支
- 当前边界仍需明确：
  - `S2` 是 proxy-style adapter，不是完整 Qwen fine-tuning
  - `S3` 是 calibration-support，不改写 semantic branch 的主身份

### repair / selector / backbone stress 关键修复

- repair 路径现在避免把无 `probs` 的 repair 结果直接送进 selector fusion。
- selector 的单动作基线改为依据 validation-side 证据选择，而不是更松的 shortcut。
- backbone stress 使用与目标 backbone 对齐的 base prediction，修正了先前 cross-backbone delta 的比较错误。

### 真实 smoke 暴露并已修复的运行路径问题

- `wandb` 顶层 import 已改为 optional，`--disable_wandb` 能走通。
- dataset 路径解析已兼容 `TwiBot-20` / `Twibot-20` 命名差异，并支持 repo-root `datasets/` lookup。
- `StageRunner.run` 命名冲突导致的 `NullRun object is not callable` 已修复。

## Verified

### 编译与测试

- `python -m py_compile` 已通过，覆盖本轮 touched 的 mainline 关键文件。
- `pytest LLMbot/baseline/core/tests -q` 当前结果为 `7 passed`。

### 当前测试明确覆盖到的内容

- parser / mainline surface 的 stage 注册
- `semantic_source_matrix` 的 formal split enforcement
- artifact helper 的 JSON / tensor round-trip
- `operational / analysis` subgroup schema 的基本边界

### formal smoke 已推进到哪一层

本轮 formal smoke 直接使用正式入口：

```bash
python main.py \
  --stage vertical_minimal \
  --dataset TwiBot-20 \
  --GNN_model rgcn \
  --seeds 1 \
  --disable_wandb \
  --artifact_root G:\Research\BotDetection\results\dual_lane_smoke \
  --max_iters 1 \
  --LM_pretrain_epochs 0 \
  --GNN_epochs_per_iter 1 \
  --LM_epochs_per_iter 1 \
  --LM_eval_patience 1 \
  --batch_size_LM 8
```

当前已确认的推进层级：

- dataset resolution 已落到 repo-root dataset tree
- formal stage routing 已进入 `StageRunner`
- `results/dual_lane_smoke/seed_1/` 下的 runtime 目录树已创建
- smoke 已进入真实模型执行，并写出至少一个 runtime checkpoint：
  - `runtime/rgcn/checkpoints/LM_pretrain/best.pkl`

## Partial / pending

### faithful literature anchors 仍未闭环

- `CaGCN`、`GATS`、`LA-GNN` 还没有全部作为完整 faithful confirmation lane 接入 formal harness。
- 因此当前实现不能写成 faithful-complete，只能写成 harness-ready but literature-anchor incomplete。

### semantic lane 仍有 proxy / support 边界

- `S2` 仍应被表述为 PEFT-style adapter proxy。
- `S3` 仍应被表述为 calibration-support。
- 两者都不应被写成已经完成端到端 faithful semantic-source reproduction。

### formal smoke 尚未完成 stage artifact 写出

- `vertical_minimal` smoke 已到达真实执行，但没有在当前 CPU / 时间窗内完成。
- 本轮尚未观察到 `stages/vertical_minimal/` 下完整写出的 `metrics.json`、`budget_curve.csv`、`subgroup_report.json`、`notes.md` 四件套。

## Risk and boundary notes

- 不把未完成的 faithful estimator / semantic / repair reproduction 写成“已验证”。
- 不把长跑超时写成“实现失败”；当前更像是 runtime budget 问题，而不是 harness 结构错误。
- 当前状态更适合这样理解：
  - `code-ready`: yes，dual-lane harness surface 已能编译并通过聚焦测试
  - `harness-ready`: partial，formal path 已进入真实执行，但 stage-level artifact completion 仍待验证
  - `paper-ready`: not yet，faithful anchor closure 与完整 formal-stage 验证尚未完成

## Current state shorthand

- `implemented`: dual-lane mainline harness、supporting semantic-source lane、核心 contract 变更
- `verified`: `py_compile`、`core/tests`、以及 formal-entrypoint smoke 进入真实执行
- `partial / pending`: faithful anchor closure 与受限时间窗下的完整 stage artifact 写出
