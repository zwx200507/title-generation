# -*- coding: utf-8 -*-
"""
Title generation based on pre-trained model (mT5)
Fine-tuning using HuggingFace transformers
"""

import os
import sys
import time
import pickle
import argparse
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os. path.dirname(os.path. dirname(os.path.abspath(__file__))))


class TitleDataset(Dataset):
    """Title generation dataset"""
    
    def __init__(self, data, tokenizer, max_src_len=256, max_tgt_len=32):
        self.data = data
        self.tokenizer = tokenizer
        self. max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        
        # Encode input
        inputs = self.tokenizer(
            src,
            max_length=self.max_src_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Encode target
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt,
                max_length=self.max_tgt_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask']. squeeze(),
            'labels': labels['input_ids'].squeeze(),
            'tgt_text': tgt
        }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train one epoch"""
    model. train()
    total_loss = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids']. to(device)
        attention_mask = batch['attention_mask']. to(device)
        labels = batch['labels'].to(device)
        
        # Set padding token labels to -100
        labels[labels == model.config.pad_token_id] = -100
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss. item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            labels[labels == model.config.pad_token_id] = -100
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
    
    return total_loss / len(dataloader)


def generate_samples(model, tokenizer, dataloader, device, num_samples=5):
    """Generate samples"""
    model.eval()
    samples = []
    refs = []
    
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
            
            for i in range(min(num_samples - len(samples), input_ids.size(0))):
                pred = tokenizer.decode(generated[i], skip_special_tokens=True)
                samples.append(pred)
                refs.append(batch['tgt_text'][i])
            
            if len(samples) >= num_samples:
                break
    
    return samples, refs


def main(args):
    from transformers import MT5ForConditionalGeneration, MT5Tokenizer
    from transformers import get_linear_schedule_with_warmup
    
    device = torch.device('cuda' if torch.cuda. is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # Load pre-trained model
    print(f"加载预训练模型: {args.model_name}")
    tokenizer = MT5Tokenizer.from_pretrained(args. model_name)
    model = MT5ForConditionalGeneration.from_pretrained(args. model_name)
    model = model.to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.dataset == 'thucnews': 
        data_dir = os.path.join(base_dir, 'data', 'processed_thucnews')
        print("使用 THUCNews 数据集")
    else:
        data_dir = os. path.join(base_dir, 'data', 'processed')
        print("使用 LCSTS 数据集")
    
    with open(os. path.join(data_dir, 'train.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(data_dir, 'val.pkl'), 'rb') as f:
        val_data = pickle.load(f)
    
    if args.debug:
        train_data = train_data[:500]
        val_data = val_data[: 100]
    
    if args.max_samples:
        train_data = train_data[:args.max_samples]
    
    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}")
    
    # Data loaders
    train_dataset = TitleDataset(train_data, tokenizer)
    val_dataset = TitleDataset(val_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model. parameters(), lr=args.lr)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training
    best_val_loss = float('inf')
    save_dir = os.path.join(base_dir, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n开始训练 {args.model_name}...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = evaluate(model, val_loader, device)
        
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1:02d} | 耗时: {elapsed:.1f}s | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(os.path.join(save_dir, 'mt5_best'))
            tokenizer.save_pretrained(os.path.join(save_dir, 'mt5_best'))
            print("  => 保存最佳模型")
        
        # Print samples
        if (epoch + 1) % 2 == 0 or epoch == 0:
            samples, refs = generate_samples(model, tokenizer, val_loader, device, num_samples=3)
            print("  生成样例:")
            for i, (pred, ref) in enumerate(zip(samples, refs)):
                print(f"    [{i+1}] 参考:  {ref}")
                print(f"        预测: {pred}")
    
    print(f"\n训练完成! 最佳验证Loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='google/mt5-small',
                        help='预训练模型名称')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--max_samples', type=int, default=None, help='最大训练样本数')
    parser.add_argument('--dataset', type=str, default='lcsts', choices=['lcsts', 'thucnews'])
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    main(args)