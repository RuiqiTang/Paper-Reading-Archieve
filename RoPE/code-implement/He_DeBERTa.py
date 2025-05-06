
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
   