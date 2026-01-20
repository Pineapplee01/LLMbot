"""
Main entry for Qwen + Attention Fusion experiments
与原main.py形成对照实验
"""
import argparse
import torch
from utils import seed_setting, setup_wandb, load_raw_data, prepare_path
from QwenModel import Qwen_Model, Qwen_Model_Small, build_qwen_tokenizer
from QwenAttentionTrainer import QwenAttentionTrainer
from GNNs import RGCN, RGT, SimpleHGN, HGT
from AttentionFusion import (
    CrossAttentionFusion,
    BiDirectionalAttentionFusion,
    SimpleAttentionFusion,
    AdaptiveAttentionFusion
)


def parse_args():
    parser = argparse.ArgumentParser(description='Qwen + Attention Fusion for Bot Detection')
    
    # 实验设置
    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='TwiBot-20')
    parser.add_argument('--seeds', type=str, default='1,2,3,4,5')
    parser.add_argument('--device', type=int, default=0)
    
    # Qwen模型配置
    parser.add_argument('--qwen_model', type=str, default='small',
                       choices=['7b', 'small'], help='Qwen模型大小')
    parser.add_argument('--qwen_model_path', type=str, default=None,
                       help='Qwen模型路径，默认自动选择')
    parser.add_argument('--use_fp16', action='store_true',
                       help='使用FP16精度以节省显存')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--lm_dropout', type=float, default=0.1)
    parser.add_argument('--att_dropout', type=float, default=0.1)
    parser.add_argument('--classifier_n_layers', type=int, default=2)
    parser.add_argument('--classifier_hidden_dim', type=int, default=128)
    
    # GNN配置
    parser.add_argument('--gnn_model', type=str, default='rgcn',
                       choices=['rgcn', 'rgt', 'simplehgn', 'hgt'])
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_relations', type=int, default=2)
    parser.add_argument('--gnn_dropout', type=float, default=0.4)
    parser.add_argument('--att_heads', type=int, default=8)
    
    # Attention Fusion配置
    parser.add_argument('--fusion_type', type=str, default='cross',
                       choices=['cross', 'bidirectional', 'simple', 'adaptive'])
    parser.add_argument('--fusion_hidden_dim', type=int, default=256)
    parser.add_argument('--fusion_heads', type=int, default=8)
    parser.add_argument('--fusion_dropout', type=float, default=0.1)
    
    # 训练配置
    parser.add_argument('--train_mode', type=str, default='joint',
                       choices=['joint', 'sequential'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size_lm', type=int, default=16)
    parser.add_argument('--batch_size_gnn', type=int, default=100000)
    parser.add_argument('--lr_lm', type=float, default=1e-5)
    parser.add_argument('--lr_gnn', type=float, default=5e-4)
    parser.add_argument('--lr_fusion', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--eval_patience', type=int, default=5)
    
    return parser.parse_args()


def build_models(args, device):
    """构建Qwen、GNN和Fusion模型"""
    
    # 1. 构建Qwen模型
    print("\n===== Building Qwen Model =====")
    if args.qwen_model_path is None:
        if args.qwen_model == '7b':
            args.qwen_model_path = 'Qwen/Qwen2.5-Coder-7B-Instruct'
        else:  # small
            args.qwen_model_path = 'Qwen/Qwen2.5-0.5B-Instruct'
    
    model_config = {
        'qwen_model_path': args.qwen_model_path,
        'use_fp16': args.use_fp16,
        'classifier_n_layers': args.classifier_n_layers,
        'classifier_hidden_dim': args.classifier_hidden_dim,
        'dropout': 0.4,
        'activation': 'leakyrelu',
        'lm_dropout': args.lm_dropout,
        'att_dropout': args.att_dropout
    }
    
    if args.qwen_model == '7b':
        qwen_model = Qwen_Model(model_config).to(device)
    else:
        qwen_model = Qwen_Model_Small(model_config).to(device)
    
    tokenizer = build_qwen_tokenizer(model_config)
    
    # 调整tokenizer对应的embedding层大小
    qwen_model.LM.resize_token_embeddings(len(tokenizer))
    
    lm_hidden_size = qwen_model.hidden_size
    
    # 2. 构建GNN模型
    print("\n===== Building GNN Model =====")
    gnn_config = {
        'in_dim': lm_hidden_size,  # 使用Qwen的hidden size
        'hidden_dim': args.hidden_dim,
        'out_dim': 2,
        'n_layers': args.n_layers,
        'n_relations': args.n_relations,
        'dropout': args.gnn_dropout,
        'activation': 'leakyrelu',
        'att_heads': args.att_heads,
        'device': device,
        'GNN_model': args.gnn_model
    }
    
    if args.gnn_model == 'rgcn':
        gnn_model = RGCN(gnn_config).to(device)
    elif args.gnn_model == 'rgt':
        gnn_model = RGT(gnn_config).to(device)
    elif args.gnn_model == 'simplehgn':
        gnn_model = SimpleHGN(gnn_config).to(device)
    elif args.gnn_model == 'hgt':
        gnn_model = HGT(gnn_config).to(device)
    
    print(f"GNN model: {args.gnn_model}")
    print(f"Total params: {sum(p.numel() for p in gnn_model.parameters()):,}")
    
    # 3. 构建Attention Fusion模型
    print("\n===== Building Attention Fusion Model =====")
    if args.fusion_type == 'cross':
        fusion_model = CrossAttentionFusion(
            lm_dim=lm_hidden_size,
            gnn_dim=args.hidden_dim,
            hidden_dim=args.fusion_hidden_dim,
            num_heads=args.fusion_heads,
            dropout=args.fusion_dropout
        ).to(device)
    elif args.fusion_type == 'bidirectional':
        fusion_model = BiDirectionalAttentionFusion(
            lm_dim=lm_hidden_size,
            gnn_dim=args.hidden_dim,
            hidden_dim=args.fusion_hidden_dim,
            num_heads=args.fusion_heads,
            dropout=args.fusion_dropout
        ).to(device)
    elif args.fusion_type == 'simple':
        fusion_model = SimpleAttentionFusion(
            lm_dim=lm_hidden_size,
            gnn_dim=args.hidden_dim,
            hidden_dim=args.fusion_hidden_dim,
            dropout=args.fusion_dropout
        ).to(device)
    elif args.fusion_type == 'adaptive':
        fusion_model = AdaptiveAttentionFusion(
            lm_dim=lm_hidden_size,
            gnn_dim=args.hidden_dim,
            hidden_dim=args.fusion_hidden_dim,
            num_heads=args.fusion_heads,
            dropout=args.fusion_dropout
        ).to(device)
    
    print(f"Fusion type: {args.fusion_type}")
    print(f"Total params: {sum(p.numel() for p in fusion_model.parameters()):,}")
    
    return qwen_model, gnn_model, fusion_model, tokenizer


def main():
    args = parse_args()
    
    # 设置设备
    if args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # 多种子实验
    seeds = list(map(int, args.seeds.strip().split(',')))
    all_results = []
    
    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"Running experiment with seed: {seed}")
        print(f"{'='*80}\n")
        
        # 设置随机种子
        seed_setting(seed)
        
        # 准备路径
        experiment_name = f"{args.experiment_name}_qwen_attn_seed_{seed}"
        _, _, _, _, _, _, _, _, _ = prepare_path(experiment_name)
        ckpt_filepath = f"{experiment_name}/checkpoints/fusion"
        
        # 加载数据
        print("Loading data...")
        data = load_raw_data(args.dataset, use_GNN=True)
        
        # 设置wandb
        run = setup_wandb(args, seed)
        
        # 构建模型
        qwen_model, gnn_model, fusion_model, tokenizer = build_models(args, device)
        
        # 构建训练器
        trainer = QwenAttentionTrainer(
            qwen_model=qwen_model,
            gnn_model=gnn_model,
            fusion_model=fusion_model,
            tokenizer=tokenizer,
            device=device,
            epochs=args.epochs,
            lr_lm=args.lr_lm,
            lr_gnn=args.lr_gnn,
            lr_fusion=args.lr_fusion,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            max_length=args.max_length,
            batch_size_lm=args.batch_size_lm,
            batch_size_gnn=args.batch_size_gnn,
            train_idx=data['train_idx'],
            valid_idx=data['valid_idx'],
            test_idx=data['test_idx'],
            hard_labels=data['labels'],
            user_seq=data['user_text'],
            edge_index=data['edge_index'],
            edge_type=data['edge_type'],
            ckpt_filepath=ckpt_filepath,
            run=run,
            train_mode=args.train_mode,
            eval_patience=args.eval_patience
        )
        
        # 训练
        test_acc, test_f1 = trainer.train()
        
        all_results.append({
            'seed': seed,
            'test_accuracy': test_acc,
            'test_f1': test_f1
        })
        
        print(f"\nSeed {seed} Results:")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Test F1: {test_f1:.4f}")
        
        # 结束wandb run
        if run is not None:
            run.finish()
    
    # 汇总结果
    print(f"\n{'='*80}")
    print("Summary of All Seeds:")
    print(f"{'='*80}")
    
    test_accs = [r['test_accuracy'] for r in all_results]
    test_f1s = [r['test_f1'] for r in all_results]
    
    print(f"Test Accuracy: {sum(test_accs)/len(test_accs):.4f} ± {torch.tensor(test_accs).std().item():.4f}")
    print(f"Test F1: {sum(test_f1s)/len(test_f1s):.4f} ± {torch.tensor(test_f1s).std().item():.4f}")
    
    print("\nDetailed Results:")
    for r in all_results:
        print(f"  Seed {r['seed']}: Acc={r['test_accuracy']:.4f}, F1={r['test_f1']:.4f}")


if __name__ == '__main__':
    main()
