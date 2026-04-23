# 🚀 Qwen + Attention Fusion 快速开始指南

## 3分钟快速上手

### 第1步：验证环境 (30秒)

```powershell
# 检查Python版本（需要3.8+）
python --version

# 检查CUDA可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')" 
```

### 第2步：测试代码 (2分钟)

```powershell
# 运行快速测试（会自动下载Qwen模型，首次约2分钟）
python test_qwen_attention.py
```

**预期输出：**
```
==================================================
测试1: Qwen模型
==================================================
✅ Qwen模型测试通过!
   - Embedding shape: torch.Size([2, 896])
   - Logits shape: torch.Size([2, 2])

🎉 所有测试通过！代码已准备好运行完整实验。
```

### 第3步：运行实验 (3-8小时)

```powershell
# 快速实验（Qwen-0.5B，约3小时）
python main_qwen_attention.py `
    --project_name lmbot_quick_test `
    --experiment_name test_run `
    --dataset TwiBot-20 `
    --qwen_model small `
    --device 0 `
    --fusion_type cross `
    --epochs 10 `
    --seeds 1
```

---

## 📋 完整实验流程

### 实验1：Baseline对比

```powershell
# Step 1: 运行原始baseline
python main.py `
    --project_name comparison `
    --experiment_name baseline `
    --dataset TwiBot-20 `
    --use_GNN `
    --device 0 `
    --seeds 1,2,3,4,5

# Step 2: 运行Qwen + Attention
python main_qwen_attention.py `
    --project_name comparison `
    --experiment_name qwen_attn `
    --dataset TwiBot-20 `
    --qwen_model small `
    --fusion_type cross `
    --device 0 `
    --seeds 1,2,3,4,5

# Step 3: 在WandB查看对比结果
```

### 实验2：不同融合机制对比

```powershell
# 交叉注意力
python main_qwen_attention.py --experiment_name fusion_cross --fusion_type cross --seeds 1,2,3

# 双向注意力
python main_qwen_attention.py --experiment_name fusion_bidir --fusion_type bidirectional --seeds 1,2,3

# 自适应注意力
python main_qwen_attention.py --experiment_name fusion_adaptive --fusion_type adaptive --seeds 1,2,3
```

### 实验3：不同模型大小对比

```powershell
# Qwen-0.5B（快速）
python main_qwen_attention.py --qwen_model small --seeds 1,2,3,4,5

# Qwen-7B（性能最佳，需要40GB+显存）
python main_qwen_attention.py --qwen_model 7b --use_fp16 --batch_size_lm 4 --seeds 1,2,3,4,5
```

---

## 🎛️ 常用配置组合

### 配置1：快速调试（10分钟）
```powershell
python main_qwen_attention.py `
    --qwen_model small `
    --epochs 1 `
    --batch_size_lm 32 `
    --seeds 1
```

### 配置2：标准实验（3小时/seed）
```powershell
python main_qwen_attention.py `
    --qwen_model small `
    --fusion_type cross `
    --epochs 15 `
    --batch_size_lm 16 `
    --seeds 1,2,3,4,5
```

### 配置3：高性能实验（8小时/seed）
```powershell
python main_qwen_attention.py `
    --qwen_model 7b `
    --use_fp16 `
    --fusion_type adaptive `
    --epochs 10 `
    --batch_size_lm 8 `
    --fusion_hidden_dim 512 `
    --seeds 1,2,3,4,5
```

### 配置4：显存受限（12GB GPU）
```powershell
python main_qwen_attention.py `
    --qwen_model small `
    --batch_size_lm 8 `
    --batch_size_gnn 50000 `
    --fusion_hidden_dim 128 `
    --seeds 1,2,3
```

---

## 📊 查看结果

### WandB在线查看
```
访问: https://wandb.ai/<your-username>/<project-name>
```

### 本地查看
```powershell
# 查看实验目录
ls *_qwen_attn_seed_*/

# 查看最佳模型
ls *_qwen_attn_seed_*/checkpoints/fusion/best.pt
```

---

## ⚠️ 故障排查

### 问题1：显存不足
```powershell
# 解决方案
python main_qwen_attention.py `
    --qwen_model small `              # 使用小模型
    --batch_size_lm 4 `                # 减小batch size
    --use_fp16                         # 使用混合精度（仅7B）
```

### 问题2：模型下载失败
```powershell
# 设置国内镜像
$env:HF_ENDPOINT = "https://hf-mirror.com"
python main_qwen_attention.py ...
```

### 问题3：CUDA错误
```powershell
# 检查CUDA版本
nvidia-smi
python -c "import torch; print(torch.version.cuda)"

# 重装PyTorch（匹配CUDA版本）
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu118
```

---

## 📈 预期性能

| 模型配置 | Test Acc | Test F1 | 训练时间 | 显存 |
|---------|----------|---------|---------|------|
| RoBERTa + KD (baseline) | 85.4% | 87.8% | 2h | 12GB |
| Qwen-0.5B + Cross Attn | TBD | TBD | 3h | 18GB |
| Qwen-0.5B + Bidir Attn | TBD | TBD | 3.5h | 20GB |
| Qwen-7B + Cross Attn | TBD | TBD | 8h | 42GB |

---

## 🎯 下一步

### 1. 运行测试
```powershell
python test_qwen_attention.py
```

### 2. 快速实验
```powershell
python main_qwen_attention.py --qwen_model small --epochs 1 --seeds 1
```

### 3. 完整实验
```powershell
.\configs\run_comparison_experiments.ps1
```

### 4. 分析结果
- 查看WandB仪表盘
- 记录实验结果
- 撰写实验报告

---

## 📚 更多资源

- 详细指南: `docs/QWEN_ATTENTION_GUIDE.md`
- 对比实验: `docs/README_COMPARISON.md`
- 实现总结: `docs/IMPLEMENTATION_SUMMARY.md`
- 测试脚本: `test_qwen_attention.py`

---

**准备好了吗？开始你的第一个实验：**

```powershell
python test_qwen_attention.py && python main_qwen_attention.py --qwen_model small --epochs 1 --seeds 1
```
