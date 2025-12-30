# -*- coding: utf-8 -*-
"""
Complete Transformer model
For Chinese title generation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch. float).unsqueeze(1)
        div_term = torch. exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[: , 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self. d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformation and split heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None: 
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted sum
        context = torch.matmul(attn, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self. d_model)
        output = self.W_o(context)
        
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self. fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    """Encoder layer"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn. LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        # Self-Attention + Add & Norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self. dropout(attn_output))
        
        # FFN + Add & Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self. dropout(ffn_output))
        
        return x


class DecoderLayer(nn.Module):
    """Decoder layer"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Masked Self-Attention
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-Attention
        attn_output, _ = self. cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self. dropout(attn_output))
        
        # FFN
        ffn_output = self. ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x


class TransformerEncoder(nn. Module):
    """Transformer encoder"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.scale = math. sqrt(d_model)
    
    def forward(self, src, src_mask):
        x = self.embedding(src) * self.scale
        x = self.pos_encoding(x)
        
        for layer in self. layers:
            x = layer(x, src_mask)
        
        return x


class TransformerDecoder(nn.Module):
    """Transformer decoder"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.scale = math.sqrt(d_model)
    
    def forward(self, tgt, enc_output, src_mask, tgt_mask):
        x = self.embedding(tgt) * self.scale
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        output = self.fc_out(x)
        return output


class TransformerModel(nn.Module):
    """Complete Transformer model"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_encoder_layers, 
                 num_decoder_layers, d_ff, dropout, pad_idx, device):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size, d_model, num_heads, num_encoder_layers, d_ff, dropout, pad_idx
        )
        self.decoder = TransformerDecoder(
            vocab_size, d_model, num_heads, num_decoder_layers, d_ff, dropout, pad_idx
        )
        self.pad_idx = pad_idx
        self.device = device
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def make_src_mask(self, src):
        # (batch, 1, 1, src_len)
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask. to(self.device)
    
    def make_tgt_mask(self, tgt):
        batch_size, tgt_len = tgt.size()
        
        # padding mask
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        
        # Lower triangular mask to prevent seeing future information
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device)).bool()
        
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        enc_output = self.encoder(src, src_mask)
        output = self. decoder(tgt, enc_output, src_mask, tgt_mask)
        
        return output
    
    def generate(self, src, max_len, bos_idx, eos_idx):
        """Autoregressive generation"""
        self. eval()
        with torch.no_grad():
            batch_size = src.size(0)
            src_mask = self.make_src_mask(src)
            enc_output = self.encoder(src, src_mask)
            
            # Initialize target sequence
            tgt = torch. LongTensor([[bos_idx]] * batch_size).to(self.device)
            
            for _ in range(max_len):
                tgt_mask = self.make_tgt_mask(tgt)
                output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
                
                # Get prediction of the last time step
                next_token = output[: , -1, : ].argmax(dim=-1, keepdim=True)
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # Stop if all have generated EOS
                if (next_token == eos_idx).all():
                    break
            
        return tgt[: , 1:]  # Remove BOS


def build_transformer_model(vocab_size, config, pad_idx, device):
    """Build Transformer model"""
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        pad_idx=pad_idx,
        device=device
    )
    return model.to(device)


class LabelSmoothingLoss(nn. Module):
    """Label smoothing loss"""
    
    def __init__(self, vocab_size, pad_idx, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        # pred: (batch * seq_len, vocab_size)
        # target: (batch * seq_len,)
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[: , self.pad_idx] = 0
            mask = (target == self.pad_idx)
            true_dist[mask] = 0
        
        return -(true_dist * pred).sum(dim=-1).mean()


if __name__ == '__main__':
    # Simple test
    from config import TransformerConfig
    
    config = TransformerConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vocab_size = 5000
    pad_idx = 0
    
    model = build_transformer_model(vocab_size, config, pad_idx, device)
    
    # Test forward propagation
    batch_size = 4
    src_len = 50
    tgt_len = 15
    
    src = torch. randint(1, vocab_size, (batch_size, src_len)).to(device)
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len)).to(device)
    
    outputs = model(src, tgt)
    print(f"输出形状: {outputs.shape}")  # (batch, tgt_len, vocab_size)
    
    # Test generation
    generated = model.generate(src, max_len=20, bos_idx=2, eos_idx=3)
    print(f"生成形状: {generated.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")