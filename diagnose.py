"""
Diagnostic Tools for Analyzing Fusion Behavior

Usage:
    python diagnostic_tools.py --checkpoint path/to/checkpoint.pt --split test
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, linregress
from pathlib import Path


def analyze_gate_behavior(df, save_path='gate_analysis.png'):
    """
    Analyze how gate responds to different scenarios.
    
    Args:
        df: DataFrame from trainer.diagnose_errors()
        save_path: Where to save the visualization
    
    Expected Patterns (GOOD model):
    - High homophily + both correct → Low gate (trust graph)
    - Low homophily + text correct → High gate (trust text)
    - Strong negative correlation between homophily and gate
    """
    
    # Create 2x2 grid of correctness scenarios
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    scenarios = [
        ('Both Correct', (df['is_correct_text']) & (df['is_correct_graph'])),
        ('Text Correct, Graph Wrong', (df['is_correct_text']) & (~df['is_correct_graph'])),
        ('Graph Correct, Text Wrong', (~df['is_correct_text']) & (df['is_correct_graph'])),
        ('Both Wrong', (~df['is_correct_text']) & (~df['is_correct_graph']))
    ]
    
    for ax, (title, mask) in zip(axes.flat, scenarios):
        data = df[mask]
        
        if len(data) > 0:
            # Scatter: Homophily vs Gate
            scatter = ax.scatter(
                data['homophily'], 
                data['gate'],
                c=data['is_correct_full'],
                cmap='RdYlGn',
                alpha=0.6,
                s=30,
                edgecolors='black',
                linewidths=0.5
            )
            
            ax.set_xlabel('Homophily (Graph Reliability)', fontsize=11)
            ax.set_ylabel('Gate (Text Weight)', fontsize=11)
            ax.set_title(f'{title} (n={len(data)})', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            
            # Add trend line
            if len(data) > 10:
                slope, intercept, r_value, p_value, _ = linregress(
                    data['homophily'], data['gate']
                )
                x_line = np.linspace(data['homophily'].min(), data['homophily'].max(), 100)
                y_line = slope * x_line + intercept
                
                color = 'green' if slope < -0.1 else 'red'  # Good if negative slope
                ax.plot(x_line, y_line, '--', color=color, alpha=0.7, linewidth=2,
                       label=f'Slope={slope:.3f}, R²={r_value**2:.3f}')
                
                # Add text annotation
                ax.text(0.05, 0.95, f'p={p_value:.4f}', 
                       transform=ax.transAxes,
                       verticalalignment='top',
                       fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.legend(loc='lower right', fontsize=9)
    
    plt.colorbar(scatter, ax=axes, label='Fusion Correct', shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gate analysis visualization saved to: {save_path}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*70)
    print("GATE BEHAVIOR ANALYSIS")
    print("="*70)
    
    for title, mask in scenarios:
        data = df[mask]
        if len(data) > 0:
            print(f"\n📌 {title} (n={len(data)}):")
            print(f"  Gate Mean: {data['gate'].mean():.4f} ± {data['gate'].std():.4f}")
            print(f"  Gate Range: [{data['gate'].min():.4f}, {data['gate'].max():.4f}]")
            print(f"  Homophily Mean: {data['homophily'].mean():.4f}")
            print(f"  Fusion Accuracy: {data['is_correct_full'].mean():.2%}")
            
            # Correlation
            if len(data) > 10:
                corr, pval = pearsonr(data['homophily'], data['gate'])
                status = "✅ GOOD" if corr < -0.3 else "⚠️ WEAK" if corr < 0 else "❌ WRONG"
                print(f"  Correlation (Homophily↔Gate): {corr:.3f} (p={pval:.4f}) {status}")
    
    print("="*70)
    
    # Overall diagnosis
    both_correct = df[(df['is_correct_text']) & (df['is_correct_graph'])]
    if len(both_correct) > 10:
        overall_corr, _ = pearsonr(both_correct['homophily'], both_correct['gate'])
        
        print("\n🎯 OVERALL DIAGNOSIS:")
        if overall_corr < -0.3:
            print("  ✅ Gate is working correctly!")
            print("     → Negative correlation means high homophily → low gate (trust graph)")
        elif -0.3 <= overall_corr < 0:
            print("  ⚠️ Gate shows weak pattern")
            print("     → Some correlation, but not strong enough")
        else:
            print("  ❌ Gate is broken!")
            print("     → Positive/zero correlation means gate ignores homophily")


def check_uncertainty_calibration(df, save_path='uncertainty_calibration.png'):
    """
    Verify that uncertainty actually predicts errors.
    
    Good calibration:
    - Low uncertainty → High accuracy
    - High uncertainty → Low accuracy
    - Strong positive correlation between uncertainty and error rate
    """
    
    # Check if uncertainty_text exists (might be sigma_text in old version)
    unc_col = 'uncertainty_text' if 'uncertainty_text' in df.columns else 'sigma_text'
    
    # Bin by uncertainty
    df['unc_bin'] = pd.cut(df[unc_col], bins=5, 
                           labels=['Very Low', 'Low', 'Med', 'High', 'Very High'])
    
    # Calculate metrics per bin
    calibration_stats = df.groupby('unc_bin').agg({
        'is_correct_full': ['count', 'mean'],
        unc_col: 'mean'
    }).round(4)
    
    print("\n" + "="*70)
    print("UNCERTAINTY CALIBRATION CHECK")
    print("="*70)
    print("\nAccuracy by Uncertainty Bin:")
    print(calibration_stats)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy vs Uncertainty Bin
    bin_stats = df.groupby('unc_bin')['is_correct_full'].agg(['mean', 'count'])
    bin_stats['mean'].plot(kind='bar', ax=ax1, color='steelblue', edgecolor='black')
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_xlabel('Uncertainty Bin', fontsize=11)
    ax1.set_title('Calibration: Accuracy vs Uncertainty', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    
    # Add count labels
    for i, (idx, row) in enumerate(bin_stats.iterrows()):
        ax1.text(i, row['mean'] + 0.02, f"n={int(row['count'])}", 
                ha='center', fontsize=9)
    
    # Plot 2: Scatter - Uncertainty vs Correctness
    sample_size = min(2000, len(df))
    df_sample = df.sample(sample_size) if len(df) > sample_size else df
    
    jitter_x = np.random.normal(0, 0.02, len(df_sample))
    jitter_y = np.random.normal(0, 0.02, len(df_sample))
    
    ax2.scatter(df_sample[unc_col] + jitter_x, 
               df_sample['is_correct_full'].astype(float) + jitter_y,
               alpha=0.3, s=20, c='steelblue', edgecolors='none')
    ax2.set_xlabel('Uncertainty', fontsize=11)
    ax2.set_ylabel('Correct (1) / Wrong (0)', fontsize=11)
    ax2.set_title('Scatter: Uncertainty vs Correctness', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    # Add trend line
    if len(df_sample) > 10:
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, _ = linregress(
            df_sample[unc_col], df_sample['is_correct_full'].astype(float)
        )
        x_line = np.linspace(df_sample[unc_col].min(), df_sample[unc_col].max(), 100)
        y_line = slope * x_line + intercept
        
        color = 'green' if slope < -0.1 else 'red'
        ax2.plot(x_line, y_line, '--', color=color, linewidth=2, alpha=0.7,
                label=f'Slope={slope:.3f}, R²={r_value**2:.3f}')
        ax2.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Uncertainty calibration plot saved to: {save_path}")
    plt.close()
    
    # Diagnosis
    print("\n🎯 CALIBRATION DIAGNOSIS:")
    accuracy_trend = bin_stats['mean'].values
    
    if len(accuracy_trend) >= 3:
        # Check if accuracy decreases with uncertainty
        is_decreasing = all(accuracy_trend[i] >= accuracy_trend[i+1] - 0.05 
                           for i in range(len(accuracy_trend)-1))
        
        gap = accuracy_trend[0] - accuracy_trend[-1]
        
        if is_decreasing and gap > 0.15:
            print("  ✅ Excellent calibration!")
            print(f"     → Accuracy drops {gap:.2%} from low to high uncertainty")
        elif is_decreasing and gap > 0.05:
            print("  ⚠️ Moderate calibration")
            print(f"     → Accuracy drops {gap:.2%}, but could be stronger")
        else:
            print("  ❌ Poor calibration!")
            print("     → Uncertainty does not predict errors reliably")


def compare_modalities(df, save_path='modality_comparison.png'):
    """
    Compare performance of text-only, graph-only, and fusion.
    """
    
    # Calculate accuracies
    acc_text = df['is_correct_text'].mean()
    acc_graph = df['is_correct_graph'].mean()
    acc_fusion = df['is_correct_full'].mean()
    
    print("\n" + "="*70)
    print("MODALITY COMPARISON")
    print("="*70)
    print(f"\n📊 Overall Accuracies:")
    print(f"  Text Only:  {acc_text:.4f} ({acc_text:.2%})")
    print(f"  Graph Only: {acc_graph:.4f} ({acc_graph:.2%})")
    print(f"  Fusion:     {acc_fusion:.4f} ({acc_fusion:.2%})")
    
    best_single = max(acc_text, acc_graph)
    fusion_gain = acc_fusion - best_single
    
    print(f"\n🎯 Fusion Value-Add: {fusion_gain:+.4f} ({fusion_gain:+.2%})")
    
    if fusion_gain > 0.02:
        print("  ✅ Fusion is significantly better than best single modality!")
    elif fusion_gain > 0:
        print("  ⚠️ Fusion helps marginally")
    else:
        print("  ❌ Fusion is worse than best single modality (BUG!)")
    
    # Breakdown by case type
    print("\n📋 Accuracy Breakdown by Case Type:")
    case_analysis = df.groupby('case_type')['is_correct_full'].agg(['count', 'mean'])
    print(case_analysis)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Overall comparison
    modalities = ['Text Only', 'Graph Only', 'Fusion']
    accuracies = [acc_text, acc_graph, acc_fusion]
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    bars = ax1.bar(modalities, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Modality Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.4f}', ha='center', fontsize=10, fontweight='bold')
    
    # Plot 2: Case type breakdown
    case_analysis['mean'].plot(kind='barh', ax=ax2, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Fusion Accuracy', fontsize=11)
    ax2.set_ylabel('Case Type', fontsize=11)
    ax2.set_title('Accuracy by Case Type', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add count labels
    for i, (idx, row) in enumerate(case_analysis.iterrows()):
        ax2.text(row['mean'] + 0.02, i, f"n={int(row['count'])}", 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Modality comparison plot saved to: {save_path}")
    plt.close()


def run_full_diagnosis(csv_path, output_dir='./diagnosis_results'):
    """
    Run all diagnostic tools on a diagnosis CSV file.
    
    Usage:
        run_full_diagnosis('diagnosis_test_seed42.csv')
    """
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"\n📁 Loaded diagnosis file: {csv_path}")
    print(f"   Shape: {df.shape}")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Run analyses
    print("\n" + "="*70)
    print("RUNNING FULL DIAGNOSTIC SUITE")
    print("="*70)
    
    analyze_gate_behavior(df, save_path=f"{output_dir}/gate_analysis.png")
    check_uncertainty_calibration(df, save_path=f"{output_dir}/uncertainty_calibration.png")
    compare_modalities(df, save_path=f"{output_dir}/modality_comparison.png")
    
    print("\n" + "="*70)
    print("✅ DIAGNOSIS COMPLETE")
    print("="*70)
    print(f"\n📂 Results saved to: {output_dir}/")
    print("   - gate_analysis.png")
    print("   - uncertainty_calibration.png")
    print("   - modality_comparison.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnostic Tools for Fusion Analysis')
    parser.add_argument('--csv', type=str, required=True, 
                       help='Path to diagnosis CSV file')
    parser.add_argument('--output_dir', type=str, default='./diagnosis_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    run_full_diagnosis(args.csv, args.output_dir)