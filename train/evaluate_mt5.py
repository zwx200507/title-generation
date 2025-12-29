# -*- coding: utf-8 -*-
"""
Evaluate pre-trained model (mT5) performance
Compare with Baseline
"""

import os
import sys
import pickle
import argparse
import json
import torch
from torch. utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import Seq2SeqConfig
from model.seq2seq import build_seq2seq_model
from utils.vocab import Vocabulary
from utils.metrics import compute_all_metrics
from utils.visualize import plot_metrics_comparison


class TitleDatasetMT5(Dataset):
    def __init__(self, data, tokenizer, max_src_len=256, max_tgt_len=32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        inputs = self.tokenizer(
            src, max_length=self.max_src_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask':  inputs['attention_mask'].squeeze(),
            'src_text': src,
            'tgt_text': tgt
        }


class TitleDatasetBaseline(Dataset):
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
        return {
            'src': src_ids,
            'src_text': src,
            'tgt_text': tgt
        }


def collate_fn_baseline(batch, pad_idx):
    import torch.nn as nn
    src_list = [torch. LongTensor(item['src']) for item in batch]
    src_texts = [item['src_text'] for item in batch]
    tgt_texts = [item['tgt_text'] for item in batch]
    src_lens = torch.LongTensor([len(s) for s in src_list])
    src_padded = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    return {
        'src': src_padded,
        'src_len': src_lens,
        'src_text':  src_texts,
        'tgt_text': tgt_texts
    }


def evaluate_mt5(model, tokenizer, dataloader, device):
    """Evaluate mT5 model"""
    model.eval()
    all_preds = []
    all_refs = []
    
    with torch.no_grad():
        for batch in dataloader: 
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=32,
                num_beams=4,
                early_stopping=True
            )
            
            for i in range(input_ids.size(0)):
                pred = tokenizer. decode(generated[i], skip_special_tokens=True)
                all_preds.append(pred)
                all_refs.append(batch['tgt_text'][i])
    
    metrics = compute_all_metrics(all_preds, all_refs)
    return metrics, all_preds, all_refs


def evaluate_baseline(model, vocab, dataloader, device):
    """Evaluate Baseline model"""
    model. eval()
    all_preds = []
    all_refs = []
    
    with torch. no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            src_len = batch['src_len']
            tgt_texts = batch['tgt_text']
            
            generated, _ = model.generate(
                src, src_len, max_len=30,
                bos_idx=vocab.bos_idx, eos_idx=vocab.eos_idx
            )
            
            for i in range(src.size(0)):
                pred = vocab.decode(generated[i].tolist())
                all_preds.append(pred)
                all_refs.append(tgt_texts[i])
    
    metrics = compute_all_metrics(all_preds, all_refs)
    return metrics, all_preds, all_refs


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    base_dir = os. path.dirname(os.path. dirname(os.path.abspath(__file__)))
    
    if args.dataset == 'thucnews':
        data_dir = os.path.join(base_dir, 'data', 'processed_thucnews')
        print("使用 THUCNews 数据集")
    else:
        data_dir = os.path.join(base_dir, 'data', 'processed')
        print("使用 LCSTS 数据集")
    
    checkpoint_dir = os.path. join(base_dir, 'checkpoints')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load test data
    with open(os.path.join(data_dir, 'test.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    
    if args.num_samples:
        test_data = test_data[:args.num_samples]
    
    print(f"测试集大小: {len(test_data)}")
    
    all_metrics = {}
    
    # ========== Evaluate Baseline ==========
    baseline_path = os.path.join(checkpoint_dir, 'baseline_best. pt')
    if os.path.exists(baseline_path):
        print("\n评估 Baseline (Seq2Seq + LSTM + Attention)...")
        vocab = Vocabulary.load(os.path.join(data_dir, 'vocab.json'))
        config = Seq2SeqConfig()
        
        test_dataset = TitleDatasetBaseline(test_data, vocab, config.max_src_len, config.max_tgt_len)
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False,
            collate_fn=lambda b: collate_fn_baseline(b, vocab.pad_idx)
        )
        
        model = build_seq2seq_model(len(vocab), config, vocab.pad_idx, device)
        model.load_state_dict(torch.load(baseline_path, map_location=device))
        
        metrics, preds, refs = evaluate_baseline(model, vocab, test_loader, device)
        all_metrics['Baseline'] = metrics
        
        print("Baseline 指标:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    
    # ========== Evaluate mT5 ==========
    mt5_path = os.path. join(checkpoint_dir, 'mt5_best')
    if os.path.exists(mt5_path):
        print("\n评估 mT5 (预训练模型微调)...")
        from transformers import MT5ForConditionalGeneration, MT5Tokenizer
        
        tokenizer = MT5Tokenizer.from_pretrained(mt5_path)
        model = MT5ForConditionalGeneration.from_pretrained(mt5_path)
        model = model.to(device)
        
        test_dataset = TitleDatasetMT5(test_data, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        metrics, preds, refs = evaluate_mt5(model, tokenizer, test_loader, device)
        all_metrics['mT5'] = metrics
        
        print("mT5 指标:")
        for k, v in metrics. items():
            print(f"  {k}: {v:.4f}")
        
        # Save samples
        with open(os. path.join(results_dir, 'mt5_samples.txt'), 'w', encoding='utf-8') as f:
            for i in range(min(20, len(preds))):
                f.write(f"[{i+1}]\n")
                f.write(f"参考: {refs[i]}\n")
                f.write(f"预测: {preds[i]}\n\n")
    else:
        print(f"mT5 模型不存在: {mt5_path}")
    
    # Save results
    if len(all_metrics) > 0:
        with open(os.path. join(results_dir, 'metrics_mt5.json'), 'w', encoding='utf-8') as f:
            json. dump(all_metrics, f, ensure_ascii=False, indent=2)
        
        if len(all_metrics) > 1:
            plot_data = {}
            for model_name, metrics in all_metrics.items():
                plot_data[model_name] = {
                    'ROUGE-1': metrics['rouge1_f'],
                    'ROUGE-2': metrics['rouge2_f'],
                    'ROUGE-L': metrics['rougeL_f'],
                    'BLEU-4': metrics['bleu4']
                }
            plot_metrics_comparison(plot_data, os.path. join(results_dir, 'mt5_comparison.png'))
        
        print(f"\n结果已保存到: {results_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='lcsts', choices=['lcsts', 'thucnews'])
    args = parser.parse_args()
    
    main(args)