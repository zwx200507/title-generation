# -*- coding: utf-8 -*-
"""
Data preprocessing module
Load LCSTS dataset, clean text, build vocabulary, split dataset
"""

import os
import re
import json
import pickle
import random
import pandas as pd
from collections import Counter

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.vocab import Vocabulary


def clean_text(text):
    """Clean text"""
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove extra whitespace
    text = re. sub(r'\s+', ' ', text)
    # Remove special characters but keep Chinese punctuation
    text = re.sub(r'[^\u4e00-\u9fa5\u3000-\u303f\uff00-\uffefa-zA-Z0-9，。！？、；：""''（）]', '', text)
    return text.strip()


def load_lcsts_data(data_path):
    """Load LCSTS dataset"""
    print(f"加载数据: {data_path}")
    df = pd.read_csv(data_path)
    
    # Clean data
    sources = []
    targets = []
    
    for _, row in df.iterrows():
        src = clean_text(row['text1'])
        tgt = clean_text(row['text2'])
        
        # Filter empty data and too short data
        if len(src) >= 10 and len(tgt) >= 2:
            sources.append(src)
            targets.append(tgt)
    
    print(f"有效样本数: {len(sources)}")
    return sources, targets


def analyze_data(sources, targets):
    """Analyze data statistics"""
    src_lens = [len(s) for s in sources]
    tgt_lens = [len(t) for t in targets]
    
    print("\n=== 数据统计 ===")
    print(f"样本总数: {len(sources)}")
    print(f"源文本长度: 平均{sum(src_lens)/len(src_lens):.1f}, 最大{max(src_lens)}, 最小{min(src_lens)}")
    print(f"目标长度: 平均{sum(tgt_lens)/len(tgt_lens):.1f}, 最大{max(tgt_lens)}, 最小{min(tgt_lens)}")
    
    # Character statistics
    all_chars = Counter()
    for s in sources + targets: 
        all_chars.update(list(s))
    print(f"不同字符数: {len(all_chars)}")
    print(f"最常见的10个字符: {all_chars.most_common(10)}")


def split_data(sources, targets, train_ratio=0.9, val_ratio=0.05):
    """Split train/validation/test sets"""
    data = list(zip(sources, targets))
    random.shuffle(data)
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"\n=== 数据划分 ===")
    print(f"训练集: {len(train_data)}")
    print(f"验证集: {len(val_data)}")
    print(f"测试集: {len(test_data)}")
    
    return train_data, val_data, test_data


def build_vocab_from_data(train_data, min_freq=2):
    """Build vocabulary from training data"""
    texts = [src for src, tgt in train_data] + [tgt for src, tgt in train_data]
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build_vocab(texts, is_char_level=True)
    return vocab


def save_processed_data(train_data, val_data, test_data, vocab, output_dir):
    """Save processed data"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data
    with open(os.path.join(output_dir, 'train.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(output_dir, 'val.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
    with open(os.path.join(output_dir, 'test.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    
    # Save vocabulary
    vocab.save(os.path.join(output_dir, 'vocab.json'))
    
    # Save config
    config = {
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'vocab_size': len(vocab)
    }
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\n数据已保存到: {output_dir}")


def main():
    # Path configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'lcsts_test_set', 'train.csv')
    output_dir = os. path.join(base_dir, 'data', 'processed')
    
    # Load data
    sources, targets = load_lcsts_data(data_path)
    
    # Analyze data
    analyze_data(sources, targets)
    
    # Split data
    train_data, val_data, test_data = split_data(sources, targets)
    
    # Build vocabulary
    vocab = build_vocab_from_data(train_data, min_freq=2)
    
    # Save data
    save_processed_data(train_data, val_data, test_data, vocab, output_dir)
    
    # Show examples
    print("\n=== 数据样例 ===")
    for i in range(3):
        src, tgt = train_data[i]
        print(f"[{i+1}] 文章: {src[:80]}...")
        print(f"    标题: {tgt}")
        print()


if __name__ == '__main__':
    main()