import torch
import torch.nn as nn
import math

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.base = base

    def forward(self, x, seq_len=None):
        # x: (batch_size, seq_len, num_heads, head_dim)
        if seq_len is None:
            seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        # Different dimensions have different frequencies
        emb = torch.cat((freqs, freqs), dim=-1) # (seq_len, dim)
        return emb

    def rotate_half(self, x):
        # Rotates the last dimension of x by half its size
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, freqs):
        # q: (batch_size, seq_len, num_heads, head_dim)
        # k: (batch_size, seq_len, num_heads, head_dim)
        # freqs: (seq_len, head_dim)

        q_rotated = (q * freqs.cos() + self.rotate_half(q) * freqs.sin()).float()
        k_rotated = (k * freqs.cos() + self.rotate_half(k) * freqs.sin()).float()

        return q_rotated, k_rotated

class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, rotary_dim=None, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.rotary_dim = rotary_dim if rotary_dim is not None else self.head_dim
        self.rotary_emb = RotaryEmbedding(self.rotary_dim)

    def _split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.transpose(1, 2) # (batch_size, num_heads, seq_len, head_dim)

    def _merge_heads(self, x, batch_size):
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.embed_dim)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query_heads = self._split_heads(query, batch_size)
        key_heads = self._split_heads(key, batch_size)

        # Apply RoPE to query and key
        seq_len_q = query_heads.shape[-2]
        seq_len_k = key_heads.shape[-2]
        freqs_q = self.rotary_emb(query_heads, seq_len=seq_len_q)[..., :self.rotary_dim]
        freqs_k = self.rotary_emb(key_heads, seq_len=seq_len_k)[..., :self.rotary_dim]

        query_rotated, key_rotated = self.rotary_emb.apply_rotary_pos_emb(query_heads, key_heads, freqs_q)

        # Standard multi-head attention calculation
        attn_output = torch.matmul(query_rotated, key_rotated.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_output = attn_output.masked_fill(attention_mask.unsqueeze(1), float("-inf"))

        attn_weights = F.softmax(attn_output, dim=-1)
        attn_output = torch.matmul(attn_weights, value_heads)

        attn_output = self._merge_heads(attn_output, batch_size)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        return attn_output, attn_weights