# -*- coding: utf-8 -*-
"""
Seq2Seq + LSTM + Bahdanau Attention model
Baseline model for Chinese title generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    """Bidirectional LSTM encoder"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn. Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_len):
        # src: (batch, src_len)
        embedded = self. dropout(self.embedding(src))
        
        # Pack variable-length sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # Concatenate bidirectional hidden state
        # hidden: (num_layers * 2, batch, hidden)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        cell = torch.cat([cell[-2], cell[-1]], dim=1)
        
        hidden = torch.tanh(self.fc_hidden(hidden))
        cell = torch.tanh(self.fc_cell(cell))
        
        # outputs: (batch, src_len, hidden*2)
        # hidden/cell: (batch, hidden)
        return outputs, hidden, cell


class Attention(nn.Module):
    """Bahdanau Attention"""
    
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hidden_dim * 2 + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs, mask):
        # hidden: (batch, dec_hidden)
        # encoder_outputs: (batch, src_len, enc_hidden*2)
        batch_size = encoder_outputs. size(0)
        src_len = encoder_outputs.size(1)
        
        # Expand hidden
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Calculate attention energy
        energy = torch.tanh(self.attn(torch. cat([hidden, encoder_outputs], dim=2)))
        attention = self.v(energy).squeeze(2)
        
        # Apply mask
        attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    """LSTM decoder with Attention"""
    
    def __init__(self, vocab_size, embed_dim, enc_hidden_dim, dec_hidden_dim, dropout, pad_idx):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = Attention(enc_hidden_dim, dec_hidden_dim)
        self.lstm = nn.LSTMCell(embed_dim + enc_hidden_dim * 2, dec_hidden_dim)
        self.fc = nn.Linear(enc_hidden_dim * 2 + dec_hidden_dim + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_token, hidden, cell, encoder_outputs, mask):
        # input_token: (batch,)
        embedded = self. dropout(self.embedding(input_token))
        
        # Calculate attention
        attn_weights = self.attention(hidden, encoder_outputs, mask)
        
        # Weighted sum to get context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        # LSTM input
        lstm_input = torch.cat([embedded, context], dim=1)
        hidden, cell = self.lstm(lstm_input, (hidden, cell))
        
        # Prediction
        output = torch.cat([embedded, hidden, context], dim=1)
        prediction = self.fc(output)
        
        return prediction, hidden, cell, attn_weights


class Seq2Seq(nn. Module):
    """Complete Seq2Seq model"""
    
    def __init__(self, encoder, decoder, pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device
    
    def create_mask(self, src):
        return (src != self.pad_idx)
    
    def forward(self, src, src_len, tgt, teacher_forcing_ratio=0.5):
        # src: (batch, src_len)
        # tgt: (batch, tgt_len)
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self. decoder.vocab_size
        
        # Store outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # Encoding
        encoder_outputs, hidden, cell = self.encoder(src, src_len)
        mask = self.create_mask(src)
        
        # First input is BOS
        input_token = tgt[:, 0]
        
        for t in range(1, tgt_len):
            prediction, hidden, cell, _ = self.decoder(
                input_token, hidden, cell, encoder_outputs, mask
            )
            outputs[:, t] = prediction
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input_token = tgt[:, t] if teacher_force else top1
        
        return outputs
    
    def generate(self, src, src_len, max_len, bos_idx, eos_idx):
        """Generation mode"""
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            
            encoder_outputs, hidden, cell = self.encoder(src, src_len)
            mask = self.create_mask(src)
            
            input_token = torch.LongTensor([bos_idx] * batch_size).to(self.device)
            
            generated = []
            attentions = []
            
            for _ in range(max_len):
                prediction, hidden, cell, attn = self.decoder(
                    input_token, hidden, cell, encoder_outputs, mask
                )
                top1 = prediction.argmax(1)
                generated.append(top1.cpu())
                attentions.append(attn.cpu())
                
                input_token = top1
                
                # Stop if all samples have generated EOS
                if (top1 == eos_idx).all():
                    break
            
            generated = torch.stack(generated, dim=1)
            attentions = torch.stack(attentions, dim=1)
            
        return generated, attentions


def build_seq2seq_model(vocab_size, config, pad_idx, device):
    """Build Seq2Seq model"""
    encoder = Encoder(
        vocab_size, config. embed_dim, config.hidden_dim,
        config.num_layers, config.dropout, pad_idx
    )
    decoder = Decoder(
        vocab_size, config.embed_dim, config.hidden_dim,
        config.hidden_dim, config.dropout, pad_idx
    )
    model = Seq2Seq(encoder, decoder, pad_idx, device)
    return model.to(device)


if __name__ == '__main__':
    # Simple test
    from config import Seq2SeqConfig
    
    config = Seq2SeqConfig()
    device = torch.device('cuda' if torch.cuda. is_available() else 'cpu')
    
    vocab_size = 5000
    pad_idx = 0
    
    model = build_seq2seq_model(vocab_size, config, pad_idx, device)
    
    # Test forward propagation
    batch_size = 4
    src_len = 50
    tgt_len = 15
    
    src = torch.randint(1, vocab_size, (batch_size, src_len)).to(device)
    src_lens = torch. LongTensor([src_len] * batch_size)
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len)).to(device)
    
    outputs = model(src, src_lens, tgt, teacher_forcing_ratio=0.5)
    print(f"输出形状: {outputs.shape}")  # (batch, tgt_len, vocab_size)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")