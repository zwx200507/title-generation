# -*- coding: utf-8 -*-
"""
Model evaluation script
Compare Baseline and Transformer model performance
"""

import os
import sys
import pickle
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(os. path.dirname(os.path. dirname(os.path.abspath(__file__))))
from model.config import Seq2SeqConfig, TransformerConfig
from model.seq2seq import build_seq2seq_model
from model. transformer import build_transformer_model
from utils.vocab import Vocabulary
from utils.metrics import compute_all_metrics
from utils.visualize import plot_metrics_comparison


class TitleDataset(Dataset):
    def __init__(self, data, vocab, max_src_len, max_tgt_len):
        self.data = data
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_ids = self.vocab.encode(src, add_bos=False, add_eos=False, max_len=self.max_src_len)
        tgt_ids = self.vocab.encode(tgt, add_bos=True, add_eos=True, max_len=self.max_tgt_len)
        return {
            'src': src_ids,
            'tgt': tgt_ids,
            'src_text': src,
            'tgt_text': tgt
        }


def collate_fn(batch, pad_idx):
    src_list = [torch.LongTensor(item['src']) for item in batch]
    tgt_list = [torch.LongTensor(item['tgt']) for item in batch]
    src_texts = [item['src_text'] for item in batch]
    tgt_texts = [item['tgt_text'] for item in batch]
    
    src_lens = torch.LongTensor([len(s) for s in src_list])
    src_padded = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)
    
    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_len': src_lens,
        'src_text': src_texts,
        'tgt_text': tgt_texts
    }


def evaluate_model(model, model_type, dataloader, vocab, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_refs = []
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt_texts = batch['tgt_text']
            
            if model_type == 'seq2seq':
                src_len = batch['src_len']
                generated, _ = model. generate(
                    src, src_len, max_len=30,
                    bos_idx=vocab.bos_idx, eos_idx=vocab. eos_idx
                )
            else:
                generated = model. generate(
                    src, max_len=30,
                    bos_idx=vocab.bos_idx, eos_idx=vocab.eos_idx
                )
            
            for i in range(src.size(0)):
                pred = vocab.decode(generated[i].tolist())
                all_preds.append(pred)
                all_refs.append(tgt_texts[i])
    
    # Calculate metrics
    metrics = compute_all_metrics(all_preds, all_refs)
    
    return metrics, all_preds, all_refs


def main(args):
    device = torch.device('cuda' if torch.cuda. is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'processed')
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    with open(os.path.join(data_dir, 'test.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    vocab = Vocabulary.load(os.path.join(data_dir, 'vocab.json'))
    
    if args.num_samples:
        test_data = test_data[:args.num_samples]
    
    print(f"测试集大小:  {len(test_data)}")
    
    # Data loader
    seq2seq_config = Seq2SeqConfig()
    test_dataset = TitleDataset(test_data, vocab, seq2seq_config.max_src_len, seq2seq_config.max_tgt_len)
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        collate_fn=lambda b: collate_fn(b, vocab.pad_idx)
    )
    
    all_metrics = {}
    
    # Evaluate Baseline
    baseline_path = os.path.join(checkpoint_dir, 'baseline_best.pt')
    if os.path.exists(baseline_path):
        print("\n评估 Baseline (Seq2Seq + LSTM + Attention)...")
        model = build_seq2seq_model(len(vocab), seq2seq_config, vocab.pad_idx, device)
        model.load_state_dict(torch.load(baseline_path, map_location=device))
        
        metrics, preds, refs = evaluate_model(model, 'seq2seq', test_loader, vocab, device)
        all_metrics['Baseline'] = metrics
        
        print("Baseline 指标:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # Save samples
        with open(os.path.join(results_dir, 'baseline_samples.txt'), 'w', encoding='utf-8') as f:
            for i in range(min(20, len(preds))):
                f.write(f"[{i+1}]\n")
                f.write(f"参考: {refs[i]}\n")
                f.write(f"预测: {preds[i]}\n\n")
    else:
        print(f"Baseline模型不存在: {baseline_path}")
    
    # Evaluate Transformer
    transformer_path = os.path.join(checkpoint_dir, 'transformer_best.pt')
    if os.path.exists(transformer_path):
        print("\n评估 Transformer...")
        trans_config = TransformerConfig()
        model = build_transformer_model(len(vocab), trans_config, vocab.pad_idx, device)
        model.load_state_dict(torch. load(transformer_path, map_location=device))
        
        metrics, preds, refs = evaluate_model(model, 'transformer', test_loader, vocab, device)
        all_metrics['Transformer'] = metrics
        
        print("Transformer 指标:")
        for k, v in metrics.items():
            print(f"  {k}: {v:. 4f}")
        
        # Save samples
        with open(os.path.join(results_dir, 'transformer_samples.txt'), 'w', encoding='utf-8') as f:
            for i in range(min(20, len(preds))):
                f.write(f"[{i+1}]\n")
                f.write(f"参考: {refs[i]}\n")
                f.write(f"预测: {preds[i]}\n\n")
    else:
        print(f"Transformer模型不存在: {transformer_path}")
    
    # Save metrics comparison
    if len(all_metrics) > 0:
        with open(os.path.join(results_dir, 'metrics. json'), 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        
        # Plot comparison chart
        if len(all_metrics) > 1:
            # Convert to plotting format
            plot_data = {}
            for model_name, metrics in all_metrics. items():
                plot_data[model_name] = {
                    'ROUGE-1': metrics['rouge1_f'],
                    'ROUGE-2': metrics['rouge2_f'],
                    'ROUGE-L': metrics['rougeL_f'],
                    'BLEU-4': metrics['bleu4']
                }
            plot_metrics_comparison(plot_data, os.path.join(results_dir, 'model_comparison.png'))
        
        print(f"\n结果已保存到: {results_dir}")


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=None, help='评估样本数（默认全部）')
    args = parser.parse_args()
    
    main(args)