# -*- coding: utf-8 -*-
"""
Baseline Seq2Seq model training script
"""

import os
import sys
import time
import pickle
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append(os. path.dirname(os.path. dirname(os.path.abspath(__file__))))
from model.config import Seq2SeqConfig
from model.seq2seq import build_seq2seq_model
from utils.vocab import Vocabulary
from utils.metrics import compute_all_metrics
from utils.visualize import plot_training_curves, plot_rouge_curves


class TitleDataset(Dataset):
    """Title generation dataset"""
    
    def __init__(self, data, vocab, max_src_len, max_tgt_len):
        self.data = data
        self.vocab = vocab
        self. max_src_len = max_src_len
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
    """Custom batch processing"""
    src_list = [torch.LongTensor(item['src']) for item in batch]
    tgt_list = [torch.LongTensor(item['tgt']) for item in batch]
    
    src_lens = torch.LongTensor([len(s) for s in src_list])
    
    # padding
    src_padded = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    tgt_padded = nn.utils. rnn.pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)
    
    return {
        'src': src_padded,
        'tgt':  tgt_padded,
        'src_len': src_lens
    }


def train_epoch(model, dataloader, optimizer, criterion, clip, device, teacher_forcing):
    """Train one epoch"""
    model.train()
    epoch_loss = 0
    
    for batch in dataloader:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_len = batch['src_len']
        
        optimizer.zero_grad()
        
        output = model(src, src_len, tgt, teacher_forcing)
        
        # Calculate loss, ignore first token (BOS)
        output = output[:, 1:].contiguous().view(-1, output.size(-1))
        tgt = tgt[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, tgt)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in dataloader: 
            src = batch['src']. to(device)
            tgt = batch['tgt'].to(device)
            src_len = batch['src_len']
            
            output = model(src, src_len, tgt, teacher_forcing_ratio=0)
            
            output = output[:, 1:].contiguous().view(-1, output.size(-1))
            tgt = tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def generate_samples(model, dataloader, vocab, device, num_samples=5):
    """Generate samples"""
    model.eval()
    samples = []
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            src_len = batch['src_len']
            
            generated, _ = model.generate(
                src, src_len, 
                max_len=30, 
                bos_idx=vocab.bos_idx, 
                eos_idx=vocab.eos_idx
            )
            
            for i in range(min(num_samples, src.size(0))):
                pred = vocab.decode(generated[i].tolist())
                samples.append(pred)
                if len(samples) >= num_samples:
                    break
            
            if len(samples) >= num_samples:
                break
    
    return samples


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # Config
    config = Seq2SeqConfig()
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args. lr
    
    # Load data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.dataset == 'thucnews':
        data_dir = os.path.join(base_dir, 'data', 'processed_thucnews')
        print("使用 THUCNews 数据集")
    else:
        data_dir = os.path.join(base_dir, 'data', 'processed')
        print("使用 LCSTS 数据集")
    
    with open(os.path.join(data_dir, 'train. pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(data_dir, 'val. pkl'), 'rb') as f:
        val_data = pickle.load(f)
    
    vocab = Vocabulary.load(os.path.join(data_dir, 'vocab.json'))
    
    # Debug mode uses small data
    if args.debug:
        train_data = train_data[:500]
        val_data = val_data[:100]
    
    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}")
    
    # Data loaders
    train_dataset = TitleDataset(train_data, vocab, config.max_src_len, config.max_tgt_len)
    val_dataset = TitleDataset(val_data, vocab, config.max_src_len, config.max_tgt_len)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, vocab.pad_idx), num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config. batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, vocab.pad_idx), num_workers=0
    )
    
    # Model
    model = build_seq2seq_model(len(vocab), config, vocab.pad_idx, device)
    print(f"模型参数量: {sum(p. numel() for p in model.parameters()):,}")
    
    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab. pad_idx)
    
    # Training
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0
    
    save_dir = os.path. join(base_dir, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n开始训练...")
    for epoch in range(config.epochs):
        start_time = time.time()
        
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, 
            config.clip_grad, device, config.teacher_forcing_ratio
        )
        val_loss = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses. append(val_loss)
        
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1:02d} | 耗时: {elapsed:.1f}s | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model. state_dict(), os.path.join(save_dir, 'baseline_best.pt'))
            patience_counter = 0
            print("  => 保存最佳模型")
        # else:
        #     patience_counter += 1
        # 
        # # Early stopping
        # if patience_counter >= config.patience:
        #     print(f"验证集loss连续{config.patience}个epoch未改善，早停")
        #     break
        
        # Print samples
        if (epoch + 1) % 5 == 0 or epoch == 0:
            samples = generate_samples(model, val_loader, vocab, device, num_samples=3)
            print("  生成样例:")
            for i, s in enumerate(samples):
                print(f"    [{i+1}] {s}")
    
    # Save training curves
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    plot_training_curves(train_losses, val_losses, os.path.join(results_dir, 'baseline_loss.png'))
    
    print(f"\n训练完成! 最佳验证Loss: {best_val_loss:. 4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dataset', type=str, default='lcsts', choices=['lcsts', 'thucnews'],
                        help='选择数据集:  lcsts 或 thucnews')
    parser.add_argument('--debug', action='store_true', help='调试模式，使用小数据集')
    args = parser.parse
