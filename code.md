# Code And Structure Risk Register

Current active code mainline: `LLMbot/`.
Current active NLPCC paper/task line: `NLPCC/`.

## Active Mainline Update 2026-06-23

- `LLMbot/docs/ARCHITECTURE.md` is now the detailed owner map for the active
  nested code mainline.
- Distillation gain/cost budget-curve, paired-bootstrap delta, and empty
  cost-report helpers are owned by `LLMbot/trainer_distillation_metrics.py`.
- Split-safe pseudo-label training-index helpers are owned by
  `LLMbot/trainer_indexing.py`.
- `LLMbot/trainer_distillation.py` keeps trainer classes and graph-seed
  execution; `LLMbot/trainer_legacy_impl.py` imports the shared metric owner
  and shared pseudo-label guard instead of carrying duplicate implementations.
- This update reduces duplication only. It does not change CLI behavior,
  artifact paths, metrics schema, or research-claim status.

The older `LLMbot/baseline/` audit below is historical risk context and does
not redefine the current default code mainline.

# LLMbot/baseline 代码风险审计

审计日期：2026-05-07

审计范围：`LLMbot/baseline/` 下的 `core/`、`analysis/`、`baselines/` 源码；排除 `saved_artifacts/`、`__pycache__/`、`.pytest_cache/`、`.tmp_*/`、`router_runs/` 等生成目录。

审计标准：按顶级会议/期刊研究代码的最低要求审计，包括可复现性、可追踪产物、公平比较、测试集隔离、安全反序列化、模块边界、测试覆盖和长期维护成本。

总体结论：`REQUEST CHANGES / WATCH`。当前 staged harness 已经有部分 protocol、manifest、split gate 和 survivor artifact 设计，但默认 legacy 路径、sidecar 分析脚本、checkpoint 加载和 provenance 仍不足以直接作为论文级 claim-grade 代码基础。

## 实现风险

| ID | 严重级别 | 证据 | 风险 | 建议 |
| --- | --- | --- | --- | --- |
| IR-01 | HIGH | 全局审计发现 `LLMbot/baseline/core`、`analysis`、`baselines` 内约 147 处 `torch.load(...)`，只有约 32 处显式 `weights_only=True`。典型无保护加载包括 `core/trainer.py:269`、`core/trainer.py:279`、`core/trainer.py:340`、`core/trainer.py:512`、`core/model_building.py:320`、`core/utils.py:111-115`、`baselines/run_phase5_matrix.py:47-72`。 | PyTorch pickle 反序列化可执行任意对象逻辑；如果 checkpoint、dataset artifact、diagnostic artifact 来自远程服务器或历史实验目录，存在安全和结果污染风险。不同脚本对 `weights_only` 的使用不一致，也会导致未来 PyTorch 版本迁移时行为不稳定。 | 建立唯一 `safe_torch_load(path, *, expected_type, map_location, trusted=False)`；默认 `weights_only=True`；对必须加载完整对象的历史 artifact 使用显式 allowlist、hash/schema 校验和 manifest 记录；CI 中禁止新增裸 `torch.load`。 |
| IR-02 | HIGH | `analysis/frmi_experiment.py:42` 和 `analysis/frmi_experiment.py:328` 显式使用 `torch.load(..., weights_only=False)` 读取 `all_node_outputs.pt`；同文件 `analysis/frmi_experiment.py:27-30` 硬编码 `DATASET`、`LLMBOT`、`RESULTS` 路径。 | 该脚本会绕过安全加载策略和主 harness 路径解析；如果用户把其输出用于论文实验，结果依赖当前工作目录和未验证 artifact，且安全边界最弱。 | 将 FRMI 实验脚本迁入统一 artifact loader；只允许从 manifest 指向的 artifact 读取；禁止默认 `weights_only=False`，除非 manifest 标记 artifact 来源、hash 和可信级别。 |
| IR-03 | HIGH | `core/trainer.py:276-284` 在 pretrain checkpoint 目录非空且已有 embedding 时，用 `os.listdir(...)[0]` 取第一个 checkpoint，随后立即调用 `self.eval('test')` 并写入 `pretrain accuracy/f1`。 | `os.listdir()[0]` 顺序不稳定，可能加载错误 checkpoint；更严重的是训练/恢复路径提前触碰 test split，若这些指标进入日志、W&B、筛选或调参，会构成测试集泄漏。 | checkpoint 恢复必须显式指定 `best.pkl` 或 manifest 中的 checkpoint；pretrain/validation 过程禁止调用 test；只有最终冻结配置后由独立 evaluation entrypoint 读取 test。 |
| IR-04 | HIGH | `core/trainer.py:253`、`core/trainer.py:3144`、`core/trainer.py:3421` 对未知 optimizer 执行 `return NotImplementedError`，而不是 `raise NotImplementedError(...)`。 | 调用方会收到一个异常类对象而非 optimizer，后续 scheduler、step 或 load_state_dict 处才出现间接错误，定位困难；若某些路径未立即触发，可能造成训练逻辑静默偏离。 | 改为 `raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")`，并对 LM/GNN/MLP optimizer factory 加参数化单元测试。 |
| IR-05 | HIGH | CLI 默认 stage 是 `legacy_distill`：`core/parser_args.py:11`；`main.py:481-486` 对 `legacy_distill` 直接调用 `run_legacy_graph_seed`，不进入 `StageRunner`；legacy runner 使用 `prepare_path(args.experiment_name + f'_seed_{seed}')`：`core/trainer.py:1094-1095`；而 staged artifact root 是 `seed_{seed}/stages/{stage}`：`core/artifacts.py:99-108`。 | 默认路径绕过 staged manifest、dependency manifest、split provenance 和 stage trace。顶会/期刊复现实验如果沿用 README 的 legacy 命令，产物证据弱于 formal stages。 | 将 `legacy_distill` 明确降级为 compatibility-only，或给 legacy 路径补齐同等 manifest、command、dependency、split hash、dataset hash 和 survivor artifact。 |
| IR-06 | HIGH | 项目比较协议要求 canonical splits：`docs/protocols/baseline_comparability.md:20-28`；但 `main.py:442-443` 只对 `FORMAL_STAGES` 和 `PREPARATION_STAGES` 启用 canonical split gate；`main.py:452-457` 仍允许 `args.reset_split != "-1"` 时重新随机划分；`main.py:498-500` 的报错不覆盖 `legacy_distill`。 | 只要用户用 legacy/default 路径跑 comparison，就可能使用自定义 split，却不触发 formal gate。这样会破坏 baseline comparability，导致论文结果不可比。 | 增加 `--claim_grade` 或 `--formal_comparison` 模式，对所有可报告 stage 强制 `--reset_split -1`；非 canonical split artifact 必须在 manifest 标为 exploratory/non-reportable。 |
| IR-07 | MEDIUM-HIGH | Qwen 默认路径硬编码为远程机器 cache snapshot：`core/LM.py:26-34`、`core/model_building.py:21-24`、`core/parser_args.py:104-107`；同时 `core/LM.py:34` 和 `core/model_building.py:242-243` 启用 `trust_remote_code=True`。 | 代码对 `/root/.cache/...` 环境强绑定，本地/审稿复现容易失败；`trust_remote_code=True` 扩大远程模型代码执行面，且 manifest 未强制记录模型 revision、hash 和信任开关。 | 要求用户显式传入 `--qwen_model_path` 或 HuggingFace revision；将 `trust_remote_code` 做成显式 CLI flag；manifest 记录模型 id、revision/hash、是否 remote code。 |
| IR-08 | MEDIUM-HIGH | CLI 默认 GNN 是 `botrgcn`：`core/parser_args.py:48-51`；checkpoint fallback 默认配置却是 `rgcn`：`core/model_building.py:304-316`；`load_gnn_checkpoint` 在 checkpoint 缺少 `model_config` 时使用 fallback：`core/model_building.py:319-330`；`LLMbot/baseline/AGENTS.md` 也说明旧 checkpoint 是 RGCN。 | 同一个实验命令、checkpoint 和 manifest 可能指向不同 GNN identity。旧 checkpoint 若缺 `model_config`，会被静默解释为 RGCN，论文表格中的模型名容易漂移。 | claim-grade checkpoint 必须包含 `model_config`、architecture name、input dim、relation count、checkpoint hash；缺失时 fail fast，不自动 fallback。 |
| IR-09 | MEDIUM-HIGH | `capture_code_commit` 只执行 `git rev-parse HEAD`，失败时捕获所有异常并返回 `"unknown"`：`core/artifacts.py:85-96`；当前沙箱下 `git -C LLMbot ...` 触发 safe.directory/dubious ownership 限制。 | 失败时 manifest 仍可继续写入 `"unknown"`，审计者无法知道代码版本、dirty state 或子仓库状态。顶会/期刊 artifact 需要能追溯到精确 commit 和环境。 | formal/claim-grade 模式下 commit 捕获失败应报错；manifest 增加 dirty diff hash、submodule state、Python/PyTorch/transformers/PyG/sklearn 版本、命令行和 dataset/split hash。 |
| IR-10 | MEDIUM | `core/estimators.py:458-463` 在 logistic regression 拟合失败时捕获所有异常并返回 `(nan, nan)`。 | 校准/风险估计失败会变成 NaN 指标继续流动，后续报告可能只看到退化结果而不知道根因。 | 捕获具体 sklearn 异常；manifest 写入失败原因、样本数和类别分布；claim-grade 路径遇到核心估计器失败应 fail fast。 |
| IR-11 | MEDIUM | `core/trainer.py:1447-1517` 的 `_build_or_load_lm_only_head` 在缺少 frozen LM-only head 时会现场训练 logistic regression 并写 `outputs.pt`；`dual_posterior_ranker` 在 `core/trainer.py:1969-1977` 调用它；dependency manifest 只在 `core/trainer.py:2789-2803` 标记 `recomputed_upstream`。 | Stage 2 ablation 运行时可能隐式生成上游 artifact，导致同一 stage 同时承担准备和评估职责；这降低了复现隔离性。 | 将 LM-only head 生成拆成显式 preparation stage；claim-grade Stage 2 只允许读取已冻结 dependency，缺失即失败。 |
| IR-12 | MEDIUM | `core/trainer.py:2538` 保存 `lm_pred_test.pt`，`core/trainer.py:2567-2568` 写 `regime_table_val.csv` 和 `regime_table_test.csv`；多处 analysis/baseline 脚本直接读取 test split 并输出测试侧诊断，例如 `analysis/run_frmi_v2_conditions.py:214-289`。 | 测试侧诊断对错误分析有价值，但如果这些产物反复驱动方法设计或阈值选择，会形成隐性 test-set overfitting。 | 将 test-side diagnostics 标为 post-hoc analysis；方法开发、selector、阈值、regime rule 只能使用 train/validation；文档中区分 development diagnostics 与 final held-out evaluation。 |

## 维护风险

| ID | 严重级别 | 证据 | 风险 | 建议 |
| --- | --- | --- | --- | --- |
| MR-01 | HIGH | `core/trainer.py` 约 168 KB，`core/estimators.py` 约 123 KB，`core/stage2_router_oof_smoke.py` 约 50 KB；`StageRunner` 从 `core/trainer.py:1219` 开始同时管理 split state、artifact root、dependency loading、stage dispatch 和报告生成；`StageRunner.run` 在 `core/trainer.py:3001-3024` 分发多个 stage family。 | 单文件承担过多职责，review、定位 bug、添加测试都很困难；新功能容易以局部补丁形式继续堆叠，扩大回归面。 | 按边界拆分：`legacy_trainers.py`、`stage_runner.py`、`stage2_runner.py`、`artifact_contracts.py`、`checkpoint_io.py`、`estimators/*`。拆分前先补 characterization tests。 |
| MR-02 | HIGH | sidecar 脚本绕过 harness：`analysis/run_frmi_v2_conditions.py:26-27`、`analysis/frmi_v2_probe.py:24-25`、`baselines/run_graph_preprocess_baselines.py:19-20` 修改 `sys.path`；`analysis/run_frmi_v2_conditions.py:305-306` 默认 TwiBot-22/GATv2；`analysis/run_frmi_v2_conditions.py:329-333` 写 `refine-logs/frmi_v2_results.json`；`analysis/frmi_v2_probe.py:334-337` 写 `probe_model.pkl`、`.npy`；`baselines/run_graph_preprocess_baselines.py:177` 写 CSV。 | 这些脚本难以被统一 manifest、split gate、dependency gate 和 CI 约束；论文结果可能来自多个互不兼容的入口。 | 把 sidecar 脚本分为 `exploratory/` 和 `claim_grade/`；claim-grade 脚本必须复用 `artifacts.py`、统一 args、统一 manifest 和 split/provenance 校验。 |
| MR-03 | HIGH | `LLMbot/.gitignore:3` 使用 `*` 忽略全部文件，只白名单少量历史根文件和 `code/*.py`；没有显式白名单 `LLMbot/baseline/**`。 | 新增 baseline 源码、测试或审计修复可能被 Git 默默忽略，造成“本地能跑、仓库不可复现”的风险；审稿/协作时也难以确认完整 diff。 | 调整 `LLMbot/.gitignore`，显式 track `baseline/**/*.py`、`baseline/**/*.md`、`baseline/core/tests/**/*.py` 和轻量 manifest；继续忽略大模型、数据和结果。 |
| MR-04 | MEDIUM-HIGH | 同类 safe loader 分散在 `core/utils.py:70-74`、`core/artifacts.py:78-82`、`core/frozen_g0.py:49-53`、`core/main.py:246-250`、`core/stage2_router_oof_smoke.py:41-45`，但大量调用没有复用这些 helper。 | 安全策略和兼容 fallback 被复制；未来 PyTorch 版本升级或 artifact schema 变化时，需要全仓多点修改，且容易遗漏。 | 合并为 `checkpoint_io.py` 或 `artifact_io.py`；所有加载 checkpoint/dataset/stage output 的函数必须走同一入口，并在测试中覆盖 old/new PyTorch 行为。 |
| MR-05 | MEDIUM-HIGH | 动态导入历史/相邻代码：`core/faithful_gates.py:34-38` 从 `LLMbot/code/graph_calibration.py` 加载；`core/trainer.py:1080-1086` 动态加载 `semantic_ib_edl_head.py`；多个 analysis/baseline 脚本使用 `sys.path.insert`。 | 模块边界不清，import 依赖运行目录和文件布局；类型检查、打包、测试发现和 IDE 索引都变弱。 | 把可复用模块放进明确 package；用正常 import 和 `python -m` 运行；历史实现若必须动态加载，应有 manifest、version 和 hash。 |
| MR-06 | MEDIUM | 测试集中在 `LLMbot/baseline/core/tests/`，当前仅发现 `test_stage_scaffolding.py`、`test_artifact_and_subgroup_helpers.py`、`test_model_building_and_rgt.py` 三个测试模块；`analysis/` 和 `baselines/` 未发现对应测试文件。 | 核心 staged path 有一定保护，但 checkpoint IO、legacy runner、analysis sidecar、baseline scripts、FRMI utilities 的回归风险仍高。 | 为高风险路径补最小测试：safe load policy、optimizer factory、legacy split guard、checkpoint model_config required、sidecar argument/path resolution、manifest schema。 |
| MR-07 | MEDIUM | `core/artifacts.py:85-96`、`core/estimators.py:458-463` 使用 broad `except Exception`；`core/frozen_g0.py:20` 对可选 `wandb` import 直接 `pass`。 | 宽泛异常会隐藏依赖、路径、权限、数值不稳定等问题；维护者需要从下游 NaN 或缺 artifact 反推根因。 | 限定异常类型；将 optional dependency 状态写入 runtime manifest；对于 claim-grade 路径，关键依赖缺失或关键 metric 失败应直接失败。 |
| MR-08 | MEDIUM | 模型构建逻辑分散：`core/LM.py` 直接创建 LM 和 Qwen LoRA；`core/model_building.py` 也有 tokenizer/model/checkpoint builder；`core/trainer.py` 保存和恢复 LM/GNN/MLP checkpoint；analysis/baseline 脚本重复加载 checkpoint 和 dataset。 | 模型配置、tokenizer、checkpoint schema、device/dtype 规则容易漂移；新增 backbone 时需要修改多个文件。 | 定义模型注册表和 checkpoint schema；训练、分析、baseline 只消费 registry API，不直接拼装模型和 checkpoint。 |
| MR-09 | MEDIUM | `core/parser_args.py` 参数面很宽，stage、backbone、estimator、repair、selector、semantic、appendix、Qwen/PEFT、artifact 等选项都在同一 parser 内；`main.py:490-517` 再根据 stage 改写 `args.stage`、`args.use_GNN` 和 split 行为。 | 隐式参数交互多，错误组合很难在 parse 阶段发现；维护者难以判断一个 flag 在哪个 stage 生效。 | 按 stage 建立 typed config/validation layer；parse 后先 normalize，再做 stage-specific validation，禁止 silent arg mutation。 |
| MR-10 | MEDIUM | 项目协议要求集中 runtime glue 和用 tests/manifests/stage traces 验证：`docs/protocols/harness-standard.md:17-19`；比较协议要求 seeds、library versions、configuration files：`docs/protocols/baseline_comparability.md:62-65`。当前 `StageRunner._base_artifact_bundle` 只记录 config、split size、node count 和 commit：`core/trainer.py:1247-1258`。 | artifact provenance 的字段粒度不足，后续维护者难以判定某个结果是否满足论文级复现要求。 | 把 split hash、label hash、dataset path/hash、command、environment versions、git dirty state、dependency artifact hashes 提升为所有 claim-grade stage 的公共 contract。 |
| MR-11 | LOW-MEDIUM | README 和代码默认之间存在多处需要人工记忆的约束：README 推荐 `--reset_split -1`、`--GNN_model rgcn`；代码默认 stage 为 `legacy_distill`、GNN 为 `botrgcn`；`AGENTS.md` 说明旧 checkpoint 是 RGCN。 | 文档、默认值和历史 checkpoint 之间的差异会持续制造新误用；新人或审稿复现者容易跑出非预期配置。 | 将 canonical defaults 写进代码 validator；README 只展示 validator 可接受的 claim-grade 命令；compatibility command 加显式警告。 |

## 优先整改顺序

1. 先封住 claim-grade 风险：所有可报告入口强制 canonical split、manifest、commit/env/dataset hash、禁止 test 参与训练/恢复/选择。
2. 统一 checkpoint 和 artifact IO：替换裸 `torch.load`，禁止默认 `weights_only=False`，为 checkpoint schema 加校验。
3. 将 `legacy_distill` 降级或纳入 staged contract，避免默认命令产生低证据 artifact。
4. 拆分 `trainer.py` 和 `estimators.py` 前先补 characterization tests，尤其覆盖 optimizer、checkpoint load、Stage 2 dependency、legacy split gate。
5. 清理 sidecar 脚本：探索脚本标记 exploratory，论文级脚本必须通过同一 harness。

## 本次审计未覆盖的事项

- 未运行完整训练或实验；本报告是静态代码审计加局部命令验证。
- 未下载外部模型、未访问远程 GPU、未验证数据文件内容 hash。
- 未对 `saved_artifacts/`、checkpoint、dataset 二进制文件做安全扫描。
