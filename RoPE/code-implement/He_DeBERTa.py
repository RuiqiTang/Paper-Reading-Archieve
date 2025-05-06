
import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativePositionEmbedding(nn.Module):
    def __init__(self, num_embeddings,embedding_dim):
        super().__init__()
        self.embedding=nn.Embedding(num_embeddings,embedding_dim)
        self.num_embeddings=num_embeddings
        self.embedding_dim=embedding_dim
    
    def forward(self,relative_positions):
        '''
            Need to shift and clamp the relative positions to be within the embedding range
            relative_positions: size:[batch_size,seq_len,seq_len]
        '''
        shifted_relative_positions=relative_positions+self.num_embeddings//2
        shifted_relative_positions=torch.clamp(shifted_relative_positions,0,self.num_embeddings-1)
        return self.embedding(shifted_relative_positions)

class HeAttention(nn.Module):
    def __init__(self,d_model, num_heads, num_relative_embeddings, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model
        self.num_relative_embeddings = num_relative_embeddings
        
        # 内容投影矩阵
        self.content_q_proj = nn.Linear(d_model, d_model)
        self.content_k_proj = nn.Linear(d_model, d_model)
        self.content_v_proj = nn.Linear(d_model, d_model)

        # 相对位置投影矩阵
        self.relative_k_proj = nn.Linear(d_model, d_model)
        self.relative_q_proj = nn.Linear(d_model, d_model)

        self.relative_embeddings = RelativePositionEmbedding(num_relative_embeddings, d_model)
        
        self.out_proj=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)
        
    def _split_heads(self,x,batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3) # (batch_size, num_heads, seq_length, head_dim)

    def _merge_heads(self, x, batch_size):
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, -1, self.d_model)

    def forward(self,hidden_states,attention_masks=None):
        batch_size, seq_length, _ = hidden_states.size()

        # 计算相对位置
        relative_positions = torch.arange(seq_length).unsqueeze(0) - torch.arange(seq_length).unsqueeze(1) # (seq_length, seq_length)
        relative_positions = relative_positions.unsqueeze(0).repeat(batch_size, 1, 1) # (batch_size, seq_length, seq_length)
        relative_embeddings = self.relative_embeddings(relative_positions) # (batch_size, seq_length, seq_length, d_model)

        # 内容交互
        content_query = self.content_q_proj(hidden_states)
        content_key = self.content_k_proj(hidden_states)
        content_value = self.content_v_proj(hidden_states)

        content_query_heads = self._split_heads(content_query, batch_size)
        content_key_heads = self._split_heads(content_key, batch_size)
        content_value_heads=self._split_heads(content_value,batch_size)

        content_attn_scores = torch.matmul(content_query_heads, content_key_heads.transpose(-1, -2))
        
        # 内容-相对位置交互
        relative_key=self.relative_k_proj(relative_embeddings)  #(batch_size,seq_len,seq_len,d_model)
        relative_key_heads=relative_key.view(batch_size,seq_length,seq_length,self.num_heads,self.head_dim).permute(0,3,1,4,2)
        content_relative_attn_scores=torch.matmul(
            content_query_heads.unsqueeze(-1),
            relative_key_heads.transpose(-2,-1)
        ).squeeze(-1)    #(batch_size,num_heads,seq_len,seq_len)
        
        # 相对位置-内容交互
        relative_query=self.relative_q_proj(relative_embeddings)    #(batch_size,seq_len,seq_len,d_model)
        relative_query_heads=relative_query.view(batch_size, seq_length, seq_length, self.num_heads, self.head_dim).permute(0, 3, 1, 4, 2)
        relative_content_attn_scores = torch.matmul(
            relative_query_heads, 
            content_key_heads.unsqueeze(2).transpose(-2, -1) 
        ).squeeze(-1) # (batch_size, num_heads, seq_length, seq_length)

        # 总的注意力分数
        attn_scores=content_attn_scores+content_relative_attn_scores+relative_content_attn_scores
        
        if attention_masks is not None:
            attn_scores = attn_scores.masked_fill(attention_masks.unsqueeze(1).unsqueeze(2), float("-inf"))
        
        attn_weights = F.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_heads = torch.matmul(attn_weights, content_value)
        context = self._merge_heads(context_heads, batch_size)
        output = self.out_proj(context)
        output = self.dropout(output)

        return output, attn_weights
