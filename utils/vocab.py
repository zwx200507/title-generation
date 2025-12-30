# -*- coding: utf-8 -*-
"""
Vocabulary management module
Handle mapping between Chinese characters and indices
"""

import json
import os
from collections import Counter


class Vocabulary: 
    """Vocabulary class, supports character-level and word-level"""
    
    # Special tokens
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    BOS_TOKEN = '<BOS>'
    EOS_TOKEN = '<EOS>'
    
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Initialize special tokens"""
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        for idx, token in enumerate(special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token
    
    @property
    def pad_idx(self):
        return self.word2idx[self.PAD_TOKEN]
    
    @property
    def unk_idx(self):
        return self.word2idx[self.UNK_TOKEN]
    
    @property
    def bos_idx(self):
        return self. word2idx[self.BOS_TOKEN]
    
    @property
    def eos_idx(self):
        return self. word2idx[self.EOS_TOKEN]
    
    def __len__(self):
        return len(self.word2idx)
    
    def build_vocab(self, texts, is_char_level=True):
        """
        Build vocabulary from text list
        
        Args: 
            texts: list of texts
            is_char_level: whether character-level, default True
        """
        for text in texts:
            if is_char_level:
                tokens = list(text)
            else:
                tokens = text.split()
            self.word_freq.update(tokens)
        
        # Filter by frequency and add to vocabulary
        idx = len(self.word2idx)
        for word, freq in self.word_freq.most_common():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        print(f"词表大小: {len(self)}")
        print(f"过滤掉的低频词: {sum(1 for f in self.word_freq. values() if f < self.min_freq)}")
    
    def encode(self, text, is_char_level=True, add_bos=True, add_eos=True, max_len=None):
        """
        Encode text to index sequence
        
        Args: 
            text: input text
            is_char_level: whether character-level
            add_bos:  whether to add start token
            add_eos:  whether to add end token
            max_len: maximum length (excluding special tokens)
        
        Returns:
            list of indices
        """
        if is_char_level:
            tokens = list(text)
        else:
            tokens = text. split()
        
        if max_len:
            tokens = tokens[:max_len]
        
        indices = [self.word2idx.get(t, self. unk_idx) for t in tokens]
        
        if add_bos:
            indices = [self.bos_idx] + indices
        if add_eos:
            indices = indices + [self.eos_idx]
        
        return indices
    
    def decode(self, indices, remove_special=True):
        """
        Decode index sequence to text
        
        Args:
            indices: list of indices
            remove_special: whether to remove special tokens
        
        Returns:
            decoded text
        """
        special_indices = {self.pad_idx, self.bos_idx, self.eos_idx}
        tokens = []
        for idx in indices: 
            if remove_special and idx in special_indices:
                continue
            if idx == self.eos_idx:
                break
            tokens.append(self.idx2word.get(idx, self. UNK_TOKEN))
        return ''.join(tokens)
    
    def save(self, path):
        """Save vocabulary to file"""
        data = {
            'word2idx': self. word2idx,
            'min_freq': self.min_freq
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"词表已保存到:  {path}")
    
    @classmethod
    def load(cls, path):
        """Load vocabulary from file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = cls(min_freq=data['min_freq'])
        vocab.word2idx = data['word2idx']
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        print(f"词表加载完成，大小: {len(vocab)}")
        return vocab


if __name__ == '__main__': 
    # Simple test
    vocab = Vocabulary(min_freq=1)
    texts = ['今天天气很好', '明天可能下雨', '今天出门记得带伞']
    vocab.build_vocab(texts, is_char_level=True)
    
    encoded = vocab.encode('今天天气', add_bos=True, add_eos=True)
    print(f"编码: {encoded}")
    
    decoded = vocab.decode(encoded)
    print(f"解码: {decoded}")