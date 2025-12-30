# -*- coding: utf-8 -*-
"""
THUCNews data preprocessing module
Download data to project directory, support loading from local file or HuggingFace
"""

import os
import re
import json
import pickle
import random
from collections import Counter

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.vocab import Vocabulary


def clean_text(text):
    """Clean text"""
    if not text:
        return ""
    text = str(text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\u4e00-\u9fa5\u3000-\u303f\uff00-\uffefa-zA-Z0-9，。！？、；：""''（）]', '', text)
    return text.strip()


def load_from_local_json(json_path, max_samples=None):
    """Load data from local JSON/JSONL file"""
    print(f"从本地文件加载: {json_path}")
    
    sources = []
    targets = []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f: 
            if line.strip():
                try:
                    item = json.loads(line)
                    # Try different field names
                    title = item.get('title', item.get('label', ''))
                    content = item.get('content', item.get('text', ''))
                    
                    title = clean_text(title)
                    content = clean_text(content)
                    
                    if 2 <= len(title) <= 50 and 50 <= len(content) <= 500:
                        sources.append(content)
                        targets.append(title)
                        
                        if max_samples and len(sources) >= max_samples:
                            break
                except:
                    continue
    
    print(f"有效样本数: {len(sources)}")
    return sources, targets


def load_from_huggingface(cache_dir, max_samples=None):
    """Load data from HuggingFace, cache to specified directory"""
    print("从 HuggingFace 加载 THUCNews 数据集...")
    print(f"缓存目录: {cache_dir}")
    
    # Set domestic mirror
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # Set cache directory to project directory
    os.environ['HF_HOME'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = cache_dir
    
    try:
        from datasets import load_dataset
        ds = load_dataset("SirlyDreamer/THUCNews", cache_dir=cache_dir)
    except Exception as e:
        print(f"加载失败:  {e}")
        print("\n请手动下载数据后使用 --local 参数指定路径")
        return [], []
    
    sources = []
    targets = []
    
    # Check dataset fields first
    sample = ds['train'][0]
    print(f"数据集字段: {list(sample.keys())}")
    print(f"样例数据: {sample}")
    
    # Get field names
    title_key = 'title' if 'title' in sample else 'label'
    content_key = 'content' if 'content' in sample else 'text'
    
    for item in ds['train']:
        title = clean_text(str(item.get(title_key, '')))
        content = clean_text(str(item.get(content_key, '')))
        
        if 2 <= len(title) <= 50 and 50 <= len(content) <= 500:
            sources.append(content)
            targets.append(title)
            
            if max_samples and len(sources) >= max_samples:
                break
    
    print(f"有效样本数: {len(sources)}")
    return sources, targets


def analyze_data(sources, targets, name="数据集"):
    """Analyze data statistics"""
    if len(sources) == 0:
        print(f"{name} 没有数据")
        return
    
    src_lens = [len(s) for s in sources]
    tgt_lens = [len(t) for t in targets]
    
    print(f"\n=== {name} 统计 ===")
    print(f"样本总数: {len(sources)}")
    print(f"源文本长度: 平均{sum(src_lens)/len(src_lens):.1f}, 最大{max(src_lens)}, 最小{min(src_lens)}")
    print(f"目标长度: 平均{sum(tgt_lens)/len(tgt_lens):.1f}, 最大{max(tgt_lens)}, 最小{min(tgt_lens)}")


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
    print(f"训练集:  {len(train_data)}")
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
    
    with open(os.path.join(output_dir, 'train.pkl'), 'wb') as f:
        pickle. dump(train_data, f)
    with open(os.path.join(output_dir, 'val.pkl'), 'wb') as f:
        pickle. dump(val_data, f)
    with open(os.path.join(output_dir, 'test.pkl'), 'wb') as f:
        pickle. dump(test_data, f)
    
    vocab.save(os.path.join(output_dir, 'vocab.json'))
    
    config = {
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size':  len(test_data),
        'vocab_size': len(vocab),
        'dataset': 'thucnews'
    }
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\n数据已保存到: {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_samples', type=int, default=None, help='最大样本数')
    parser.add_argument('--local', type=str, default=None, help='本地 JSON/JSONL 文件路径')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os. path.dirname(os.path. abspath(__file__)))
    output_dir = os.path. join(base_dir, 'data', 'processed_thucnews')
    cache_dir = os.path.join(base_dir, 'data', 'thucnews_cache')  # Cache to project directory
    
    # Prioritize using local file
    if args.local and os.path.exists(args.local):
        sources, targets = load_from_local_json(args.local, args.max_samples)
    else:
        # Check default local path
        default_local = os.path. join(base_dir, 'data', 'thucnews', 'train.jsonl')
        if os.path.exists(default_local):
            sources, targets = load_from_local_json(default_local, args.max_samples)
        else:
            sources, targets = load_from_huggingface(cache_dir, args.max_samples)
    
    if len(sources) == 0:
        print("没有加载到有效数据！")
        return
    
    analyze_data(sources, targets, "THUCNews")
    train_data, val_data, test_data = split_data(sources, targets)
    vocab = build_vocab_from_data(train_data, min_freq=2)
    save_processed_data(train_data, val_data, test_data, vocab, output_dir)
    
    print("\n=== 数据样例 ===")
    for i in range(min(3, len(train_data))):
        src, tgt = train_data[i]
        print(f"[{i+1}] 内容: {src[:80]}...")
        print(f"    标题: {tgt}\n")


if __name__ == '__main__':
    main()