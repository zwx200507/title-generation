# -*- coding: utf-8 -*-
"""
Evaluation metrics module
Includes text generation evaluation metrics such as ROUGE and BLEU
"""

import numpy as np
from collections import Counter


def get_ngrams(text, n):
    """Get n-grams"""
    tokens = list(text)
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def rouge_n(pred, ref, n=1):
    """
    Calculate ROUGE-N score
    
    Args:
        pred: predicted text
        ref: reference text
        n: n value for n-gram
    
    Returns:
        dict containing precision, recall, f1
    """
    pred_ngrams = Counter(get_ngrams(pred, n))
    ref_ngrams = Counter(get_ngrams(ref, n))
    
    # Calculate overlap
    overlap = sum((pred_ngrams & ref_ngrams).values())
    
    pred_count = sum(pred_ngrams.values())
    ref_count = sum(ref_ngrams.values())
    
    if pred_count == 0:
        precision = 0
    else:
        precision = overlap / pred_count
    
    if ref_count == 0:
        recall = 0
    else: 
        recall = overlap / ref_count
    
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def lcs_length(s1, s2):
    """Calculate longest common subsequence length"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def rouge_l(pred, ref):
    """
    Calculate ROUGE-L score (based on longest common subsequence)
    
    Args:
        pred:  predicted text
        ref: reference text
    
    Returns:
        dict containing precision, recall, f1
    """
    lcs = lcs_length(list(pred), list(ref))
    
    pred_len = len(pred)
    ref_len = len(ref)
    
    if pred_len == 0:
        precision = 0
    else:
        precision = lcs / pred_len
    
    if ref_len == 0:
        recall = 0
    else:
        recall = lcs / ref_len
    
    if precision + recall == 0:
        f1 = 0
    else:
        # ROUGE-L typically uses beta=1.2
        beta = 1.2
        f1 = (1 + beta**2) * precision * recall / (recall + beta**2 * precision)
    
    return {'precision': precision, 'recall': recall, 'f1':  f1}


def bleu_score(pred, ref, max_n=4, weights=None):
    """
    Calculate BLEU score
    
    Args:
        pred:  predicted text
        ref: reference text
        max_n: maximum n-gram
        weights: weights for each n-gram, default uniform distribution
    
    Returns:
        BLEU score
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n
    
    pred_tokens = list(pred)
    ref_tokens = list(ref)
    
    if len(pred_tokens) == 0:
        return 0.0
    
    # Calculate precision for each n-gram
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(get_ngrams(pred, n))
        ref_ngrams = Counter(get_ngrams(ref, n))
        
        overlap = sum((pred_ngrams & ref_ngrams).values())
        total = sum(pred_ngrams.values())
        
        if total == 0:
            precisions.append(0)
        else:
            precisions.append(overlap / total)
    
    # Geometric mean
    log_precision = 0
    for w, p in zip(weights, precisions):
        if p > 0:
            log_precision += w * np.log(p)
        else:
            return 0.0
    
    # Brevity penalty
    bp = 1.0
    if len(pred_tokens) < len(ref_tokens):
        bp = np.exp(1 - len(ref_tokens) / len(pred_tokens))
    
    return bp * np.exp(log_precision)


def char_accuracy(pred, ref):
    """
    Calculate character-level accuracy
    
    Args:
        pred: predicted text
        ref: reference text
    
    Returns:
        accuracy
    """
    if len(ref) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    
    correct = sum(1 for p, r in zip(pred, ref) if p == r)
    return correct / max(len(pred), len(ref))


def compute_all_metrics(preds, refs):
    """
    Batch compute all metrics
    
    Args:
        preds: list of predicted texts
        refs: list of reference texts
    
    Returns: 
        average values of all metrics
    """
    metrics = {
        'rouge1_f': [], 'rouge2_f': [], 'rougeL_f': [],
        'bleu1': [], 'bleu2': [], 'bleu4': [],
        'char_acc': []
    }
    
    for pred, ref in zip(preds, refs):
        r1 = rouge_n(pred, ref, n=1)
        r2 = rouge_n(pred, ref, n=2)
        rl = rouge_l(pred, ref)
        
        metrics['rouge1_f'].append(r1['f1'])
        metrics['rouge2_f'].append(r2['f1'])
        metrics['rougeL_f'].append(rl['f1'])
        
        metrics['bleu1']. append(bleu_score(pred, ref, max_n=1))
        metrics['bleu2'].append(bleu_score(pred, ref, max_n=2))
        metrics['bleu4'].append(bleu_score(pred, ref, max_n=4))
        
        metrics['char_acc'].append(char_accuracy(pred, ref))
    
    # Calculate average
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    return avg_metrics


if __name__ == '__main__':
    # Test
    pred = '今天天气很好'
    ref = '今天的天气非常好'
    
    print("ROUGE-1:", rouge_n(pred, ref, n=1))
    print("ROUGE-2:", rouge_n(pred, ref, n=2))
    print("ROUGE-L:", rouge_l(pred, ref))
    print("BLEU-4:", bleu_score(pred, ref, max_n=4))
    print("Char Acc:", char_accuracy(pred, ref))