import torch
import torch.nn as nn
import torch.nn.functional as F

class RaffelAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model

        # 内容投影矩阵
        self.content_q_proj = nn.Linear(d_model, d_model)
        self.content_k_proj = nn.Linear(d_model, d_model)
        self.content_v_proj = nn.Linear(d_model, d_model)

        # 位置投影矩阵
        self.position_q_proj = nn.Linear(d_model, d_model)
        self.position_k_proj = nn.Linear(d_model, d_model)

        # 可训练的相对位置偏置
        self.relative_bias = nn.Parameter(torch.Tensor(1, num_heads, 1, 1)) # 简化版本，实际可能更复杂
        nn.init.zeros_(self.relative_bias)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3) # (batch_size, num_heads, seq_length, head_dim)

    def _merge_heads(self, x, batch_size):
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, -1, self.d_model)

    def forward(self, hidden_states, position_embeddings, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()
        
        # 内容交互
        content_query=self.content_q_proj(hidden_states)
        content_key=self.content_k_proj(hidden_states)
        content_value=self.content_v_proj(hidden_states)
        
        content_query_heads=self._split_heads(content_query,batch_size)
        content_key_heads=self._split_heads(content_key,batch_size)
        content_value_heads=self._split_heads(content_key,batch_size)
        
        content_attn_scores=torch.matmul(content_query_heads,content_key_heads.transpose(-1,-2))
        
        # 位置交互
        position_query=self.position_q_proj(position_embeddings)
        position_key = self.position_k_proj(position_embeddings)

        position_query_heads = self._split_heads(position_query, batch_size)
        position_key_heads = self._split_heads(position_key, batch_size)

        position_attn_scores = torch.matmul(position_query_heads, position_key_heads.transpose(-1, -2))
        
        # 整体注意力分数
        attn_scores=content_attn_scores+position_attn_scores+self.relative_bias
        
        if attention_mask is not None:
            attn_scores=attn_scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2),float("-inf"))
        
        attn_weights=F.softmax(attn_scores/(self.head_dim**0.5),dim=-1)
        attn_weights=self.dropout(attn_weights)
        
        context_heads=torch.matmul(attn_weights,content_value_heads)
        context=self._merge_heads(context_heads,batch_size)
        output=self.out_proj(context)
        output=self.dropout(output)
        
        return output,attn_weights