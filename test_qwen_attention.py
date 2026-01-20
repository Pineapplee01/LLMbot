"""
快速测试脚本 - 验证Qwen + Attention代码是否正常工作
使用极小数据集和少量epoch进行sanity check
"""
import torch
from QwenModel import Qwen_Model_Small, build_qwen_tokenizer
from GNNs import RGCN
from AttentionFusion import CrossAttentionFusion
import warnings
warnings.filterwarnings('ignore')

def test_qwen_model():
    """测试Qwen模型加载和前向传播"""
    print("\n" + "="*50)
    print("测试1: Qwen模型")
    print("="*50)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 配置
    model_config = {
        'qwen_model_path': 'Qwen/Qwen2.5-0.5B-Instruct',
        'use_fp16': False,
        'classifier_n_layers': 2,
        'classifier_hidden_dim': 128,
        'dropout': 0.4,
        'activation': 'leakyrelu',
        'lm_dropout': 0.1,
        'att_dropout': 0.1
    }
    
    try:
        # 构建模型
        print("正在加载Qwen模型...")
        model = Qwen_Model_Small(model_config).to(device)
        tokenizer = build_qwen_tokenizer(model_config)
        model.LM.resize_token_embeddings(len(tokenizer))
        
        # 测试输入
        texts = [
            "METADATA: John Doe DESCRIPTION: A normal user TWEET: Hello world!",
            "METADATA: Bot123 DESCRIPTION: Automated account TWEET: Buy now! @USER"
        ]
        
        # Tokenize
        print("正在tokenize文本...")
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        # Forward pass
        print("正在进行前向传播...")
        model.eval()
        with torch.no_grad():
            embeddings, logits = model(tokenized)
        
        print(f"✅ Qwen模型测试通过!")
        print(f"   - Embedding shape: {embeddings.shape}")
        print(f"   - Logits shape: {logits.shape}")
        print(f"   - Hidden size: {model.hidden_size}")
        return True, model.hidden_size
        
    except Exception as e:
        print(f"❌ Qwen模型测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def test_gnn_model(lm_hidden_size):
    """测试GNN模型"""
    print("\n" + "="*50)
    print("测试2: GNN模型")
    print("="*50)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    gnn_config = {
        'in_dim': lm_hidden_size,
        'hidden_dim': 128,
        'out_dim': 2,
        'n_layers': 2,
        'n_relations': 2,
        'dropout': 0.4,
        'activation': 'leakyrelu',
        'device': device,
        'GNN_model': 'rgcn'
    }
    
    try:
        # 构建GNN
        print("正在构建RGCN模型...")
        gnn = RGCN(gnn_config).to(device)
        
        # 测试输入
        num_nodes = 10
        x = torch.randn(num_nodes, lm_hidden_size).to(device)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        ]).to(device)
        edge_type = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0]).to(device)
        
        # Forward pass
        print("正在进行前向传播...")
        gnn.eval()
        with torch.no_grad():
            out = gnn(x, edge_index, edge_type)
        
        print(f"✅ GNN模型测试通过!")
        print(f"   - Output shape: {out.shape}")
        return True
        
    except Exception as e:
        print(f"❌ GNN模型测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_attention_fusion(lm_hidden_size):
    """测试注意力融合模块"""
    print("\n" + "="*50)
    print("测试3: 注意力融合模块")
    print("="*50)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gnn_dim = 128
    
    try:
        # 构建融合模块
        print("正在构建CrossAttentionFusion...")
        fusion = CrossAttentionFusion(
            lm_dim=lm_hidden_size,
            gnn_dim=gnn_dim,
            hidden_dim=256,
            num_heads=8,
            dropout=0.1
        ).to(device)
        
        # 测试输入
        batch_size = 4
        lm_features = torch.randn(batch_size, lm_hidden_size).to(device)
        gnn_features = torch.randn(batch_size, gnn_dim).to(device)
        
        # Forward pass
        print("正在进行前向传播...")
        fusion.eval()
        with torch.no_grad():
            logits, fused = fusion(lm_features, gnn_features)
        
        print(f"✅ 注意力融合测试通过!")
        print(f"   - Logits shape: {logits.shape}")
        print(f"   - Fused features shape: {fused.shape}")
        return True
        
    except Exception as e:
        print(f"❌ 注意力融合测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end():
    """端到端测试"""
    print("\n" + "="*50)
    print("测试4: 端到端集成测试")
    print("="*50)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 1. 构建Qwen
        print("步骤1: 构建Qwen模型...")
        model_config = {
            'qwen_model_path': 'Qwen/Qwen2.5-0.5B-Instruct',
            'use_fp16': False,
            'classifier_n_layers': 2,
            'classifier_hidden_dim': 128,
            'dropout': 0.4,
            'activation': 'leakyrelu',
            'lm_dropout': 0.1,
            'att_dropout': 0.1
        }
        qwen_model = Qwen_Model_Small(model_config).to(device)
        tokenizer = build_qwen_tokenizer(model_config)
        qwen_model.LM.resize_token_embeddings(len(tokenizer))
        lm_hidden_size = qwen_model.hidden_size
        
        # 2. 构建GNN
        print("步骤2: 构建GNN模型...")
        gnn_config = {
            'in_dim': lm_hidden_size,
            'hidden_dim': 128,
            'out_dim': 2,
            'n_layers': 2,
            'n_relations': 2,
            'dropout': 0.4,
            'activation': 'leakyrelu',
            'device': device,
            'GNN_model': 'rgcn'
        }
        gnn_model = RGCN(gnn_config).to(device)
        
        # 3. 构建Fusion
        print("步骤3: 构建Attention Fusion...")
        fusion_model = CrossAttentionFusion(
            lm_dim=lm_hidden_size,
            gnn_dim=128,
            hidden_dim=256,
            num_heads=8,
            dropout=0.1
        ).to(device)
        
        # 4. 端到端前向传播
        print("步骤4: 端到端前向传播...")
        texts = ["METADATA: test TWEET: test message"] * 4
        tokenized = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        qwen_model.eval()
        gnn_model.eval()
        fusion_model.eval()
        
        with torch.no_grad():
            # LM forward
            lm_embeddings, _ = qwen_model(tokenized)
            
            # GNN forward
            num_nodes = 10
            x = torch.randn(num_nodes, lm_hidden_size).to(device)
            edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]).to(device)
            edge_type = torch.tensor([0, 1, 0, 1]).to(device)
            gnn_out = gnn_model(x, edge_index, edge_type)
            gnn_features = gnn_out[:4]
            
            # Fusion
            final_logits, fused_features = fusion_model(lm_embeddings, gnn_features)
        
        print(f"✅ 端到端测试通过!")
        print(f"   - Final logits shape: {final_logits.shape}")
        print(f"   - Predictions: {torch.argmax(final_logits, dim=1)}")
        return True
        
    except Exception as e:
        print(f"❌ 端到端测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("Qwen + Attention Fusion 代码验证测试")
    print("="*70)
    print("\n此脚本将快速测试所有核心组件是否正常工作")
    print("注意：首次运行会自动下载Qwen模型（约2GB），请耐心等待\n")
    
    results = []
    
    # 测试1: Qwen模型
    success, lm_hidden_size = test_qwen_model()
    results.append(("Qwen模型", success))
    
    if not success:
        print("\n⚠️ Qwen模型测试失败，跳过后续测试")
        return
    
    # 测试2: GNN模型
    success = test_gnn_model(lm_hidden_size)
    results.append(("GNN模型", success))
    
    # 测试3: 注意力融合
    success = test_attention_fusion(lm_hidden_size)
    results.append(("注意力融合", success))
    
    # 测试4: 端到端
    success = test_end_to_end()
    results.append(("端到端集成", success))
    
    # 汇总结果
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{name:20s} {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 所有测试通过！代码已准备好运行完整实验。")
        print("\n下一步:")
        print("  1. 使用 configs/qwen_small_config.ps1 运行快速实验")
        print("  2. 或运行: python main_qwen_attention.py --qwen_model small --epochs 1 --seeds 1")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息并修复问题。")


if __name__ == '__main__':
    main()
