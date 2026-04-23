---
type: project
node_id: project:dual_lane_mainline_status_2026-04-23
title: Dual-Lane Mainline Status
created_at: 2026-04-23T00:00:00Z
updated_at: 2026-04-23T00:00:00Z
tags: [baseline, mainline, dual-lane, status, lmbot]
---

# Dual-Lane Mainline Status

## Purpose

这是一张面向 2026-04-23 的状态真值卡，用来记录当前 `LLMbot/baseline/core` dual-lane mainline 的实现边界、已经验证的事实，以及仍处于 `partial / pending` 状态的部分。

## 当前主线结论

- 正式实验唯一入口是 `LLMbot/baseline/core/main.py`。
- dual-lane mainline 的 core lane 与 supporting lane 已经接入同一套 `baseline/core` harness。
- 当前 mainline 指向 `LLMbot/baseline/core`，不是 `LLMbot/code`。
- 当前 stage surface 已覆盖：
  - core-claim lane: `vertical_minimal`、`estimator_matrix`、`semantic_matrix`、`repair_matrix`、`selector_matrix`、`backbone_stress`
  - supporting-comparison lane: `semantic_source_matrix`、`positioning_matrix`

## Implemented

### Harness 与 stage 治理

- formal stages 默认使用 `--reset_split=-1`。
- formal stages 如果收到非 `-1` 的 split override，会在入口层直接 `fail fast`。
- `main.py` 保持为唯一正式 CLI 入口，不再允许旁路 runner 承担正式实验职责。
- formal stage 不再静默把 `GNN_model` 覆写成 `botrgcn`。

### 统一 artifact contract

- 每个 formal stage 统一输出四件套：
  - `metrics.json`
  - `budget_curve.csv`
  - `subgroup_report.json`
  - `notes.md`
- harness 同时保留内部工件，用于后续 lane 串接和诊断，包括 `risk_manifest.json`、`action_table.json`、`selection_metrics.json`、`gain_cost_report.json`、`gain_explain.json` 等。

### Subgroup 双 manifest

- subgroup contract 已拆分为：
  - `subgroup_manifest_operational`
  - `subgroup_manifest_analysis`
- selector、trigger、policy 只允许读取 `operational` manifest。
- `camouflage_heavy` 与 `harmful_propagation` 留在 analysis 侧，不作为 selector-safe 信号进入决策路径。

### Supporting-mainline semantic source lane

- `semantic_source_matrix` 已进入 mainline stage surface，不再只是 appendix 叙事。
- 当前 semantic-source lane 已有可执行路径：
  - `S0`: RoBERTa local source
  - `S1`: Qwen frozen local source
  - `S2`: 基于 cached Qwen embedding 的 PEFT-style adapter proxy
  - `S3`: RoBERTa + IB-EDL calibration-support 分支
- 这意味着 semantic source / calibration 已进入研究主线的 supporting-comparison lane。

### 运行路径上的关键修复

- `wandb` 已改为 optional dependency，`--disable_wandb` 可以关闭依赖路径。
- dataset 路径解析已兼容 `TwiBot-20` / `Twibot-20` 命名差异，并支持回溯到 repo-root `datasets/`。
- `StageRunner.run` 命名冲突导致的 `NullRun object is not callable` 已修复。
- repair / selector / backbone stress 的关键串接逻辑已做主线修补，不再让无 `probs` 的 repair 结果污染 selector 路径。

## Verified

### 静态与测试验证

- `python -m py_compile` 已通过，覆盖本轮 touched 的 mainline 关键文件。
- `pytest LLMbot/baseline/core/tests -q` 当前结果为 `7 passed`。
- 当前测试已明确覆盖：
  - formal stage 的 stage 注册
  - `semantic_source_matrix` 的 formal split enforcement
  - artifact helper 的 round-trip
  - subgroup 的 `operational / analysis` 双 manifest schema

### main.py smoke 的真实推进状态

- smoke 是通过正式入口 `main.py` 触发的，不是旁路脚本。
- 当前 smoke 已推进到真实数据解析、依赖加载、模型执行准备与运行目录写出。
- 已在 `results/dual_lane_smoke/seed_1/` 下观察到 runtime 树创建，并写出至少一个 checkpoint：
  - `runtime/rgcn/checkpoints/LM_pretrain/best.pkl`
- 这说明 formal path 已进入真实模型执行链路，而不只是停留在 CLI 或 stage 路由层。

## Partial / Pending

### Faithful literature anchor 尚未闭环

- faithful `CaGCN` / `GATS` / `LA-GNN` 仍未完整接成正式确认 lane。
- repair side 的 literature anchor closure 也还没有写到可以称为 faithful-complete 的程度。

### semantic-source 当前仍有 proxy 边界

- `S2` 目前是 PEFT-style proxy，不是端到端 Qwen 微调。
- `S3` 是 calibration-support 分支，不改写 semantic branch 的主身份。

### Smoke 仍未 artifact-complete

- `vertical_minimal` 的 formal smoke 在当前 CPU / 时间窗内未完成。
- 因此，本轮尚未观察到 `stages/vertical_minimal/` 下完整写出的四件套 artifact。
- 当前可确认状态是：
  - formal path 已进入真实执行
  - 但 stage-level artifact completion 仍待后续验证

## 当前环境与运行阻塞

### 已暴露且已处理

- `wandb` 顶层 import 曾阻塞 `--disable_wandb` smoke，代码路径已修复。
- `Twibot-20` / `TwiBot-20` 的 dataset alias 问题已在解析层修复。
- `StageRunner.run` 命名冲突已修复。

### 已暴露但本质上属于环境或运行预算

- 当前环境最初缺失 `torch-geometric`，需要先补齐依赖后才能推进 smoke。
- `roberta-base` 的实际执行仍依赖 Hugging Face 模型访问与本地 cache。
- 当前剩余阻塞更偏向 runtime budget / cache 准备，而不是 harness 结构错误。

## 下一步建议

- 先做可复用缓存的更快 smoke，优先打通 `vertical_minimal` 与 `estimator_matrix` 的 stage-level artifact completion。
- 再补 faithful estimator / semantic / repair anchors，把 literature confirmation lane 关掉。
- 继续保持 `LLMbot/code` 的历史方法线与当前 `baseline/core` 主线状态分离，不混写成同一条版本叙事。

## 状态速记

- `implemented`: dual-lane harness surface、统一 artifact contract、双 manifest subgroup、supporting-mainline semantic source lane 已落地
- `verified`: `py_compile`、`core/tests`、以及通过正式入口推进到真实模型执行的 smoke
- `partial / pending`: faithful literature anchors 与 stage-level artifact completion 仍未闭环
