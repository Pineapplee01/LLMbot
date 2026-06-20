# NLPCC 写作规范

## 适用范围与依据

本规范面向 `paper/NLPCC/` 当前论文线，服务于团队内部写作、改稿与终稿压缩，不是 Springer LNCS 官方模板说明的逐条翻译。

本规范同时基于两类证据：

- `paper/NLPCC/` 当前本地约束：
  - `main.tex` 采用 `llncs` 单栏模板。
  - 当前默认骨架为 `Abstract -> Introduction -> Related Work -> Methodology -> Experiments -> Conclusion -> References`。
  - 当前审稿模式为双盲。
  - 当前页数硬约束来自 [PAPER_PLAN.md](G:\Research\BotDetection\paper\NLPCC\PAPER_PLAN.md): `12 pages`, including references。
- NLPCC 2025 四卷 Springer LNAI 录用论文集的实际共性：
  - 单栏排版。
  - 首页顺序稳定。
  - 正文章节编号稳定。
  - `References` 不编号且位于最后。
  - `Acknowledgments` 可选，通常位于 `Conclusion` 与 `References` 之间。
  - 正文普遍使用数字引用风格。

默认原则：

- 以当前本地稿件约束为硬约束。
- 以 NLPCC 2025 录用论文集共性作为 house style 参考。
- 当两者冲突时，以当前本地 `paper/NLPCC/` 论文线的约束为准。

## NLPCC / Springer LNCS 共性

### 版式

- 使用 Springer LNCS / LNAI 单栏论文风格，不采用双栏模板。
- 首页顺序固定为：
  - `Title`
  - `Authors`
  - `Affiliations/emails`
  - `Abstract`
  - `Keywords`
  - `1 Introduction`
- 图注放在图下方。
- 表注放在表上方。
- 展示公式使用 LaTeX 数学环境，编号放在右侧。

### 章节组织

- 正文主章节全部编号。
- 二级小节使用 `3.1`, `3.2` 等编号。
- `References` 不编号，始终放在最后。
- `Acknowledgments` 可选、不编号，若存在则放在 `Conclusion` 后、`References` 前。
- `Appendix` 不是默认结构，只在主线无法容纳补充材料时使用。

### 写作风格

- 语言简洁、常规、偏工程/NLP 会议论文风格。
- 引言通常以问题背景、研究缺口、方法概览、贡献总结收束。
- 引言末尾建议使用 contribution bullets，总结 2-4 个具体贡献。
- `Conclusion` 只回收正文已经证明的结论，不引入新 claim。

## 默认论文骨架

当前 `paper/NLPCC/` 默认采用以下固定骨架：

1. `Abstract`
2. `1 Introduction`
3. `2 Related Work`
4. `3 Methodology`
5. `4 Experiments`
6. `5 Conclusion`
7. optional `Acknowledgments`
8. `References`

补充约束：

- 默认不单独新增一级 `Problem Setup` 章节。
- 问题定义、符号表、任务设定放在 `3 Methodology` 开头的小节中。
- 若需要写 limitation 或 future work，优先并入 `5 Conclusion` 末段，不默认扩成新的一级章节。
- 若正文出现 appendix，必须明确它是补充内容，不得承载唯一主证据。

## 章节篇幅安排

页数判断以最终 PDF 编译结果为准，不以字数近似替代。

### 总预算

- 总页数必须 `<= 12`，且 `12 pages including references`。
- 主文正文不含参考文献建议控制在 `10.0–10.5` 页。

### 分章节预算

| 章节 | 预算 |
| --- | --- |
| `Abstract` | `150–200` 词 |
| `Keywords` | `3–5` 个 |
| `Introduction` | `1.0–1.25` 页 |
| `Related Work` | `0.75–1.0` 页 |
| `Methodology` | `2.75–3.25` 页 |
| `Experiments` | `3.75–4.25` 页 |
| `Conclusion` | `0.35–0.5` 页 |
| `Acknowledgments` | `0–0.15` 页 |
| `References` | `1.0–1.5` 页 |

### 篇幅纪律

- `Experiments` 必须是全文最长章节。
- `Methodology + Experiments` 应占正文主体的大部分篇幅。
- `Introduction + Related Work` 合计通常不应超过约 `2` 页。
- 若主文超页，优先压缩：
  - 背景铺垫
  - 冗长 related work 复述
  - 次要 case study
  - 过细 implementation prose
- 不允许靠删减核心实验、主结果表、关键消融来满足页数。
- 双盲阶段 `Acknowledgments` 默认一句简短占位，例如 `Omitted for double-blind review.`，不展开。

## 实验覆盖要求

正文 claim 必须被正文表格、图或明确数字支撑。支撑主 claim 的唯一关键证据不得移入 appendix。

### 1. Experimental Setup

必须包含以下四部分：

- `Datasets and Splits`
  - 明确数据集名称。
  - 明确训练/验证/测试划分。
  - 明确任何 sampled subset、过滤或重采样。
- `Baselines`
  - 给出主要对比方法。
  - 简要说明每类 baseline 的代表性。
- `Evaluation Metrics`
  - 说明主指标与辅助指标。
- `Implementation Details`
  - 给出关键超参、训练/验证策略、基础设置。

### 2. Main Results

必须覆盖：

- `TwiBot-20` 主结果表，作为主 benchmark。
- `TwiBot-22 sampled` 的补充结果或背景对照。

约束：

- `TwiBot-20` 主表必须承载主性能 claim。
- `TwiBot-22 sampled` 可使用紧凑表或简短结果段落，不要求扩展成第二个大表，但不能完全缺失。

### 3. Router / Risk Quality

若正文声称风险路由、校准或 hard-node localization 有效，则必须报告风险定位指标。默认必须覆盖：

- `AUROC-error`
- `AUPRC-error`
- `AURC`

如果未来改用同等级指标，可以替换，但必须说明其与当前 claim 的对应关系。

### 4. Ablation Study

当前论文线默认必须覆盖以下 ablation：

- full model
- w/o router
- w/o residual
- frozen SimTeG -> RGCN
- random routed residual

如果压缩版面，允许压缩文字解释，不允许删掉这组核心 ablation 对比本身。

### 5. Hyperparameter Sensitivity

默认必须覆盖：

- `K`
- `fanout`
- routed budget `beta`

原则：

- 只保留能解释方法行为的核心敏感性。
- 次要 sweep 可以移至 appendix，但正文必须保留代表性图或结果总结。

### 6. Case Study / Error Analysis

以下至少保留一个：

- 一个 routed hard account 的纠错案例。
- 一组具有代表性的 error analysis。

如果空间允许，优先保留 case study，并在结尾注明其角色是机制说明而非统计证据。

### 7. 当前论文线的实验主线

当前 `paper/NLPCC/` 论文建议按以下问题顺序组织实验：

1. frozen detector context
2. selective residual correction
3. router quality
4. residual sensitivity
5. evidence boundary / qualitative mechanism

## 引用、参考文献、脚注与致谢

### 正文引用

- 正文统一采用 Springer LNCS/LNAI 数字引用风格。
- 使用方括号数字，例如：
  - `[1]`
  - `[2, 3]`
  - `[1, 3, 22]`
- 不使用 author-year 正文引用风格。

### 参考文献标题与生成方式

- 参考文献标题固定为 `References`。
- 参考文献列表由 LNCS BibTeX 样式自动生成。
- 当前默认样式是 `splncs04`。
- 不手工混排，不手工混用“按字母序”和“按首次出现顺序”。
- 参考文献的最终排序以所选 LNCS 模板 / BibTeX 样式生成结果为准，全篇保持一致。

### 参考文献数量

- 目标总数：`28–34` 条唯一参考文献。
- 可接受范围：`24–36` 条。
- 当前本地 `references.bib` 约 `31` 条，属于推荐规模。

### 参考文献分布建议

- `2–3` 条数据集 / benchmark。
- `6–8` 条 social bot detection baseline。
- `4–6` 条 graph / hypergraph / higher-order 相关工作。
- `3–5` 条 reliability / calibration / selective / conformal 相关工作。
- `2–4` 条 LM / text encoder / multimodal 背景工作。
- `3–5` 条 survey / threat / platform 背景工作。

原则：

- 不为凑数量堆砌弱相关文献。
- 只引用与任务、方法、实验设置、主张直接相关的文献。

### 脚注

- 脚注只用于：
  - 代码链接
  - 数据链接
  - 模型链接
  - 网页资源 URL
  - 其他资源性补充说明
- 不允许把以下内容放进脚注：
  - 核心方法定义
  - 关键实验 caveat
  - 主要 claim
  - 影响结论解释的重要限制

默认 house style：

- 若代码/数据发布本身是卖点，可在摘要或引言末一句明确说明。
- 否则优先在首次提及时用脚注给 URL。

### 致谢

- 统一采用 `Acknowledgments` 这一拼写。
- 若存在，位置固定为 `Conclusion` 后、`References` 前。
- 默认写成一小段。
- 用途优先级：
  - 基金支持
  - 项目号
  - 匿名审稿人感谢
- 双盲阶段不写任何可识别作者身份的信息。

## must / should / may

### must

- 遵守默认骨架和 `<= 12` 页总预算。
- `Main Results`、`Router / Risk Quality`、`Ablation Study`、`Hyperparameter Sensitivity` 四类核心实验齐备。
- `TwiBot-20` 主结果表必须存在。
- `TwiBot-22 sampled` 的补充结果或背景对照必须存在。
- 正文每个性能或机制 claim 在正文表图中有对应证据。
- 明确标签来源、数据划分、防泄漏措施、类不平衡指标。
- `References` 不编号。
- 不把唯一关键证据移入 appendix。

### should

- 引言末尾使用 2–4 条 contribution bullets。
- 补充错误分析或 case study。
- 补充局限性、复现细节、随机种子、硬件或运行设置。
- 明确哪些结果是主证据，哪些是补充分析。
- 对资源链接给出可复现说明。

### may

- 使用 appendix 承载补充而非主证据。
- 增加额外可视化、校准曲线、更多案例。
- 增加伦理声明或数据使用声明。

## 社交机器人检测专项 checklist

- 明确检测粒度是账号级，不与帖子级、评论级或对话级任务混写。
- 明确标签来源：
  - 人工标注
  - 平台标记
  - 规则标签
  - 弱监督标签
- 明确标签可信度和潜在偏差。
- 明确是否存在跨时间、跨事件、跨平台或跨语言泛化 claim。
- 若有上述泛化 claim，必须有对应实验支撑。
- 主指标优先 `Macro-F1`，同时报告 `Accuracy`。
- 若涉及风险路由、可靠性或校准，补充 `AURC`、`AUROC-error`、`AUPRC-error` 或等价指标。
- 明确假阳性和假阴性的业务含义，避免只报总体分数。
- 明确数据泄漏防护：
  - 不混用同一账号跨 split 信息。
  - 不用测试标签构造训练参考集。
  - 不把 explanation-only artifacts 当成主证据。

## 提交前终检 checklist

- [ ] 最终 PDF 总页数是否 `<= 12`。
- [ ] `Abstract` 是否在 `150–200` 词范围内。
- [ ] 关键词是否为 `3–5` 个。
- [ ] 默认骨架是否仍为 `Introduction -> Related Work -> Methodology -> Experiments -> Conclusion`。
- [ ] `Experiments` 是否是全文最长章节。
- [ ] `TwiBot-20` 主结果表是否存在。
- [ ] `TwiBot-22 sampled` 的补充结果或背景对照是否存在。
- [ ] `AUROC-error`、`AUPRC-error`、`AURC` 或等价风险定位指标是否存在。
- [ ] 核心 ablation 是否覆盖 full / w/o router / w/o residual / frozen SimTeG -> RGCN / random routed residual。
- [ ] 敏感性分析是否覆盖 `K`、`fanout`、`beta`。
- [ ] case study 或 error analysis 是否至少保留一个。
- [ ] 图注是否在图下，表注是否在表上，公式编号是否在右。
- [ ] 正文引用是否统一为数字引用。
- [ ] `References` 是否不编号。
- [ ] `Acknowledgments` 若存在，是否位于 `Conclusion` 与 `References` 之间。
- [ ] 双盲阶段是否没有暴露作者身份、机构或致谢信息。
- [ ] 参考文献数量是否落在 `24–36` 条，最好接近 `28–34` 条。
- [ ] appendix 若存在，是否只承载补充而非唯一主证据。

## 备注

- 本规范是当前 `paper/NLPCC/` 论文线的执行文档，不自动覆盖未来其他 venue。
- 若页数限制、审稿模式、模板或主实验主张发生变化，应同步更新本文件。
