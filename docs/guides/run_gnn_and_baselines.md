# 在本项目上运行 GNN 与基线方法 — 调用顺序、数据依赖与调用方法

> 位置：`docs/run_gnn_and_baselines.md`

## 概览
项目通过命令行参数控制是否使用 GNN（`--use_GNN`）。总体流程为：解析参数 → 读取数据 → 构建 LM 与基线/GNN Trainer → LM 预训练 → 迭代知识蒸馏（LM ↔ GNN 或 LM ↔ MLP）→ 测试并保存结果。

本文档说明调用顺序、数据依赖、关键函数与文件，以及示例运行命令，便于在不同数据集上切换 GNN/基线方法。

---

## 主要入口
- 文件：`main.py`
- 入口调用：`if __name__ == '__main__': args = parser_args(); main(args)`
- 参数解析：`parser_args()` 定义在 `parser_args.py`

---

## 关键 CLI 参数（摘要，自 `parser_args.py`）
- 数据与实验：`--dataset`、`--experiment_name`、`--project_name`
- LM：`--LM_model`、`--max_length`、`--LM_pretrain_epochs`、`--batch_size_LM` 等
- 是否使用 GNN：`--use_GNN`（flag）
- GNN 模型选择：`--GNN_model`（可选值：`rgcn`, `rgt`, `simplehgn`, `hgt`）
- GNN 超参：`--n_layers`, `--hidden_dim`, `--n_relations`, `--att_heads`, `--RGT_semantic_heads`, `--SimpleHGN_att_res`
- 基线（MLP）相关：`--MLP_n_layers`, `--MLP_hidden_dim`
- 训练流程控制：`--max_iters`, `--seeds`, `--device` 等

---

## 数据依赖（必须存在于 `datasets/<dataset>/`）
- `norm_user_text.json` — 每个用户的拼接文本（字符串列表），作为 LM 的文本输入（该文件通常为预处理产物，仓库中未包含生成脚本）
- `labels.pt` — 标签张量（one-hot 或概率矩阵），与 `norm_user_text.json` 长度一致
- `train_idx.pt`, `valid_idx.pt`, `test_idx.pt` — 索引张量，用于构造训练/验证/测试集
- 若使用 GNN，还需：`edge_index.pt`（shape [2, E]）和 `edge_type.pt`（长度 E）

实验运行过程中，程序会在 `prepare_path(args.experiment_name + f'_seed_{seed}')` 创建实验目录，并将中间产物写入 `.../intermediate/{LM,GNN,MLP}`。

---

## 高层调用链（步骤与文件映射）
1. 解析命令行参数：`parser_args()`（`parser_args.py`）
2. 在 `main.py` 的 seed 循环中执行：
   - 初始化：`seed_setting(seed)`（`utils.py`）
   - 创建路径：`prepare_path(experiment_name)`（`utils.py`）
   - 加载数据：`data = load_raw_data(args.dataset, args.use_GNN)`（`utils.py`）
     - 读取 `train_idx`, `valid_idx`, `test_idx`, `norm_user_text.json`, `labels`，若 `use_GNN` 则额外读取 `edge_index`, `edge_type`。
   - 初始化 wandb：`setup_wandb(args, seed)`（`utils.py`）

3. 构建 Trainer
   - `LMTrainer = LM_Trainer(...)`（`trainer.py`），传入 `user_seq=data['user_text']`, `hard_labels=data['labels']` 等
   - `MLPTrainer = MLP_Trainer(...)`（`trainer.py`）
   - `LMTrainer.build_model()` → `model_building.build_LM_model()`：创建 `LM_Model` 和 tokenizer；在 tokenizer 中注册特殊 token `['DESCRIPTION:','METADATA:','TWEET:']` 并添加 `@USER`, `#HASHTAG`, `HTTPURL`, `EMOJI` 等占位 token
   - `LMTrainer.pretrain()`：LM 预训练并输出初始 `embeddings_iter_-1.pt` 与 `soft_labels_iter_-1.pt`

4. 若使用 GNN：
   - `GNNTrainer = GNN_Trainer(...)`（`trainer.py`），传入 `edge_index`, `edge_type` 等
   - `GNNTrainer.build_model()` → `model_building.build_GNN_model()`，根据 `--GNN_model` 实例化：
     - `rgcn` → `GNNs.RGCN`
     - `rgt` → `GNNs.RGT`（内部使用 `RGT.RGTLayer`）
     - `simplehgn` → `GNNs.SimpleHGN`（使用 `SimpleHGN.SimpleHGNConv`）
     - `hgt` → `GNNs.HGT`

5. 迭代知识蒸馏（在 `main.py`）
   - 每次迭代 `iter`：
     - 从 LM 加载蒸馏产物：`embeddings_LM, soft_labels_LM = load_distilled_knowledge('LM', LM_intermediate, iter-1)`（`utils.py`）
     - 若 `use_GNN`：
       - `GNNTrainer.train(embeddings_LM, soft_labels_LM)` → 使用 `build_GNN_dataloader()`（`dataloader.py`）构造 `NeighborLoader` 批次进行 GNN 训练；损失根据 pseudo-label (`is_pl`) 决定为 KD loss 或 CE loss
       - `GNNTrainer.infer(embeddings_LM)` → 对全量节点推理并保存 `soft_labels_iter_{iter}.pt`（GNN 中间结果）
       - `soft_labels_GNN = load_distilled_knowledge('GNN', GNN_intermediate, iter)`
       - `LMTrainer.train(soft_labels_GNN)` → LM 使用 GNN 产生的软标签继续训练（LM 使用 `build_LM_dataloader` 将字符串转成 tokenizer 输入）
       - `LMTrainer.infer()` → LM 对所有节点推理并写出 `embeddings_iter_{iter}.pt` 与 `soft_labels_iter_{iter}.pt`
     - 若不使用 GNN（使用 MLP 基线）：
       - `MLPTrainer.train(embeddings_LM, soft_labels_LM)` → 基于 LM embedding 训练 MLP
       - `MLPTrainer.infer(embeddings_LM)` → 保存 `soft_labels_iter_{iter}.pt`（MLP 中间结果）
       - `soft_labels_MLP = load_distilled_knowledge('MLP', MLP_intermediate, iter)`
       - `LMTrainer.train(soft_labels_MLP)`，随后 `LMTrainer.infer()`

6. 迭代结束后
   - `LMTrainer.test()` → 输出 LM 的测试结果
   - 若使用 GNN：`GNNTrainer.test(embeddings_LM)` → 输出 GNN 的测试结果
   - 若使用 MLP：`MLPTrainer.test(embeddings_LM)` → 输出 MLP 的测试结果
   - 保存结果文件：`results_LM.json`, `results_GNN.json` 或 `results_MLP.json`

---

## 关键实现点（细节与注意事项）
- LM 文本读取：`utils.load_raw_data()` 直接读取 `norm_user_text.json`；`dataloader.LM_dataset` 在 `__getitem__` 中返回字符串（不做 tokenize），最终由 `trainer.LM_Trainer.batch_to_tensor()` 调用 tokenizer（transformers）把字符串转为 ids（`add_special_tokens=False`, `truncation=True`, `max_length`）
- tokenizer special tokens：在 `model_building.build_LM_model()` 中通过 `LM_tokenizer.add_special_tokens({'additional_special_tokens': ['DESCRIPTION:','METADATA:','TWEET:']})` 注册，并通过 `LM_tokenizer.add_tokens([...])` 加入 `@USER` 等额外 tokens；随后模型 embedding 被 `resize_token_embeddings` 调整
- GNN 数据加载：`dataloader.build_GNN_dataloader()` 使用 `torch_geometric.data.Data(x=LM_embedding, edge_index, edge_type, labels)` 再借助 `NeighborLoader` 按 `input_nodes`（训练索引）采样子图批次
- 蒸馏机制：LM 与 GNN/MLP 通过 `soft_labels_iter_{iter}.pt`（soft probabilities）和 `embeddings_iter_{iter}.pt`（LM embedding）互相蒸馏；训练时对 pseudo-label（is_pl）节点使用 KD 损失，对真实标签节点使用交叉熵

---

## 模型与代码文件一览
- LM：`LM.py`, 构建与 token 管理在 `model_building.py:build_LM_model`
- GNN：`GNNs.py`（实现 `RGCN`, `RGT`, `SimpleHGN`, `HGT`）
- 低秩基线：MLP 使用 `torch_geometric.nn.models.MLP` / Trainer 中 `MLP_Trainer`
- 辅助实现：`SimpleHGN.py`, `RGT.py`
- 训练与蒸馏控制：`trainer.py`
- 数据加载：`dataloader.py`
- 实验/路径/IO：`utils.py`

---

## 常用示例命令（Windows PowerShell 语法）
- 使用 RGT（GNN）：
```powershell
python main.py --project_name demo --experiment_name exp_rgt --dataset TwiBot-20 --use_GNN --GNN_model rgt --LM_model roberta --max_iters 5 --seeds 1
```

- 使用 MLP 基线（不使用 GNN）：
```powershell
python main.py --project_name demo --experiment_name exp_mlp --dataset TwiBot-20 --LM_model roberta --max_iters 5 --seeds 1
```

- 快速调试（减小 epoch / batch size）：
```powershell
python main.py --project_name debug --experiment_name debug0 --dataset TwiBot-20 --use_GNN --GNN_model rgcn --LM_pretrain_epochs 1 --GNN_epochs_per_iter 2 --LM_epochs_per_iter 1 --batch_size_LM 8 --batch_size_GNN 1024 --max_iters 2 --seeds 1
```

---

## 建议与后续操作
- 若你需要我把这份文档放到仓库（已完成），或者做下列任一项，请回复选择：
  - 运行 tokenization demo（对 `norm_user_text.json` 前 N 条进行 tokenize 并输出 token/ids/截断统计），无需额外数据。
  - 生成可执行的预处理脚本 `scripts/preprocess_twb20.py`（需要你提供原始数据格式说明或允许我假设常见 JSON/CSV 格式）。
  - 将文档调整为更简短的 README 或添加流程图/依赖图。

---

*文档生成器：基于仓库源码自动分析，若代码在后续有改动，请重新生成以保持同步。*
