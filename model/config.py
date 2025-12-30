# -*- coding: utf-8 -*-
"""
Model configuration module
"""

class BaseConfig: 
    """Base configuration"""
    # Data related
    max_src_len = 150       # Maximum source text length
    max_tgt_len = 30        # Maximum target text length
    
    # Training related
    batch_size = 64
    epochs = 30
    learning_rate = 1e-3
    clip_grad = 5.0         # Gradient clipping
    teacher_forcing_ratio = 0.5
    
    # Early stopping
    patience = 5
    
    # Path
    save_dir = 'checkpoints'


class Seq2SeqConfig(BaseConfig):
    """Seq2Seq + LSTM + Attention model configuration"""
    model_name = 'seq2seq'
    
    # Model structure
    embed_dim = 256
    hidden_dim = 512
    num_layers = 2
    dropout = 0.3
    bidirectional = True


class TransformerConfig(BaseConfig):
    """Transformer model configuration"""
    model_name = 'transformer'
    
    # Model structure (adapted for 100k scale dataset)
    d_model = 256           # Model dimension
    num_heads = 8           # Number of attention heads
    num_encoder_layers = 3  # Number of encoder layers (reduced from 6 to 3)
    num_decoder_layers = 3  # Number of decoder layers (reduced from 6 to 3)
    d_ff = 512              # Feed-forward network dimension (reduced from 1024 to 512)
    dropout = 0.2           # Increase dropout to prevent overfitting
    
    # Training related
    warmup_steps = 1000     # Learning rate warmup steps (reduced from 4000 to 1000)
    label_smoothing = 0.05  # Label smoothing (reduced from 0.1 to 0.05)


def get_config(model_type='seq2seq'):
    """Get configuration"""
    if model_type == 'seq2seq': 
        return Seq2SeqConfig()
    elif model_type == 'transformer':
        return TransformerConfig()
    else:
        raise ValueError(f"未知模型类型: {model_type}")