# -*- coding: utf-8 -*-
"""
Visualization tool module
Generate training curves, metric comparison charts, etc.
"""

import os
import warnings
import matplotlib
matplotlib.use('Agg')  # Use when server has no GUI
import matplotlib.pyplot as plt

# Filter font warnings
warnings.filterwarnings('ignore', message='.*font.*')
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves
    
    Args:
        train_losses:  list of training losses
        val_losses: list of validation losses
        save_path:  save path
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    plt.close()


def plot_metrics_comparison(metrics_dict, save_path=None):
    """
    Plot metric comparison bar chart for different models
    
    Args: 
        metrics_dict: {model_name: {metric_name: value}}
        save_path: save path
    """
    models = list(metrics_dict. keys())
    if not models:
        return
    
    metric_names = list(metrics_dict[models[0]].keys())
    x = range(len(metric_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    for i, model in enumerate(models):
        values = [metrics_dict[model].get(m, 0) for m in metric_names]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax. bar([xi + offset for xi in x], values, width, label=model, color=colors[i % len(colors)])
        
        # Display values above bars
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    plt.close()


def plot_rouge_curves(rouge_history, save_path=None):
    """
    Plot ROUGE metric curves over epochs
    
    Args:
        rouge_history: {metric_name: [values per epoch]}
        save_path: save path
    """
    plt. figure(figsize=(10, 6))
    
    colors = {'rouge1_f': '#1f77b4', 'rouge2_f': '#ff7f0e', 'rougeL_f': '#2ca02c'}
    labels = {'rouge1_f': 'ROUGE-1', 'rouge2_f': 'ROUGE-2', 'rougeL_f': 'ROUGE-L'}
    
    for metric, values in rouge_history.items():
        epochs = range(1, len(values) + 1)
        color = colors.get(metric, '#333333')
        label = labels.get(metric, metric)
        plt.plot(epochs, values, '-o', label=label, color=color, linewidth=2, markersize=5)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('ROUGE Score During Training', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存:  {save_path}")
    plt.close()


def plot_attention_weights(attention_weights, src_tokens, tgt_tokens, save_path=None):
    """
    Visualize attention weight heatmap
    
    Args:
        attention_weights: attention matrix of shape (tgt_len, src_len)
        src_tokens: list of source sequence tokens
        tgt_tokens:  list of target sequence tokens
        save_path: save path
    """
    fig, ax = plt. subplots(figsize=(10, 8))
    
    im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
    
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, fontsize=10, rotation=45, ha='right')
    ax.set_yticklabels(tgt_tokens, fontsize=10)
    
    ax.set_xlabel('Source', fontsize=12)
    ax.set_ylabel('Target', fontsize=12)
    ax.set_title('Attention Weights', fontsize=14)
    
    plt.colorbar(im, ax=ax)
    
    if save_path: 
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    plt.close()


if __name__ == '__main__': 
    # Test plotting
    train_losses = [2.5, 2.0, 1.7, 1.5, 1.3, 1.2, 1.1, 1.0]
    val_losses = [2.6, 2.1, 1.9, 1.7, 1.5, 1.4, 1.35, 1.3]
    plot_training_curves(train_losses, val_losses, 'test_loss.png')
    
    metrics_dict = {
        'Baseline': {'ROUGE-1': 0.35, 'ROUGE-2': 0.18, 'ROUGE-L': 0.32, 'BLEU-4': 0.15},
        'Transformer': {'ROUGE-1':  0.42, 'ROUGE-2': 0.25, 'ROUGE-L':  0.39, 'BLEU-4': 0.22}
    }
    plot_metrics_comparison(metrics_dict, 'test_comparison.png')
    
    print("可视化测试完成")