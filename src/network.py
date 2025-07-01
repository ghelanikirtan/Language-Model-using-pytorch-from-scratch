# My Languague Model's network:
import torch
import torch.nn as nn 
import math
from constants import DEVICE

#! Positional Encoding:
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
    
    
#! Scaled Dot-Product Attention:
class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
        
    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e4) 
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn    
    
#! Multi-Head Attention:
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0 
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(self.d_k)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        output, attn = self.attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        output = self.W_o(output)
        return output, attn
            
            
#! Feed Forward Mechanism
class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ff_layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        return self.ff_layer(x)
    
#! Transformer Block
class TransformerBlock(nn.Module):
    """Decoder Layer"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.MHA = MultiHeadAttention(d_model, n_heads)
        self.FFN = FeedForward(d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_output, attn = self.MHA(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        ffn_output = self.FFN(x)
        x = self.layer_norm2(x + self.dropout(ffn_output))
        return x, attn

#! Transformer Decoder:
class TransformerDecoder(nn.Module):
    """"""
    def __init__(self, 
                 vocab_size, 
                 d_model, 
                 n_layers, 
                 n_heads, 
                 d_ff,
                 max_len=5000,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.linear = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.linear.weight = self.embedding.weight

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        attentions = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attentions.append(attn)
        output = self.linear(x)
        return output, attentions
    
    def generate_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask==0


#! Language Model
class LanguageModel(nn.Module):
    """"""
    def __init__(self, 
                 vocab_size, 
                 d_model=256, 
                 n_layers=4, 
                 n_heads=4, 
                 d_ff=1024,
                 dropout=0.1,
                 max_len=5000):
        super().__init__()
        self.decoder = TransformerDecoder(
            vocab_size,
            d_model,
            n_layers,
            n_heads,
            d_ff,
            max_len,
            dropout)
        
    def forward(self, x):
        mask = self.decoder.generate_mask(x.size(1)).to(x.device)
        output, attentions = self.decoder(x, mask)
        return output, attentions

















