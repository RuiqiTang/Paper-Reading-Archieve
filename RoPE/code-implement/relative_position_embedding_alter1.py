'''
Refer to:https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py
'''
import torch
from torch import nn

class RelativePosition(nn.Module):
    def __init__(self,num_units,max_relative_position):
        super().__init__()
        self.num_units=num_units
        self.max_relative_position=max_relative_position
        self.embeddings_table=nn.Parameter(
            torch.tensor(max_relative_position*2+1,num_units)
        )
        nn.init.xavier_uniform_(self.embeddings_table)
    
    def forward(self,length_q,length_k):
        range_vec_q=torch.arange(length_q)
        range_vec_k=torch.arange(length_k)
        
        distance_mat=range_vec_k[None,:]-range_vec_q[:,None]
        distance_mat_clipped=torch.clamp(distance_mat,-self.max_relative_position,self.max_relative_position)
        final_mat=distance_mat_clipped+self.max_relative_position
        final_mat=torch.LongTensor(final_mat).cuda()
        embeddings=self.embeddings_table[final_mat].cuda()
        return embeddings

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self,hid_dim,n_heads,dropout,device):
        super().__init__()
        
        assert hid_dim%n_heads==0 # hid_dim可以被n_heads整除
        
        self.hid_dim=hid_dim
        self.n_heads=n_heads
        self.head_dim=self.hid_dim//self.n_heads
        self.max_relative_position=2
        
        self.relative_position_k=RelativePosition(self.head_dim,self.max_relative_position)
        self.relative_position_v=RelativePosition(self.head_dim,self.max_relative_position)
        
        self.fc_q=nn.Linear(self.hid_dim,self.hid_dim)
        self.fc_k=nn.Linear(self.hid_dim,self.hid_dim)
        self.fc_v=nn.Linear(self.hid_dim,self.hid_dim)
        
        self.fc_o=nn.Linear(self.hid_dim,self.hid_dim)
        self.dropout=nn.Dropout(dropout)
        self.scale=torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self,query,key,value,mask=None):
        '''
            Q: [batch_size,query_len,hid_dim]
            K: [batch_size,key_len,hid_dim]
            V: [batch_size,value_len,hid_dim]
        '''
        batch_size=query.shape[0]
        len_q=query.shape[1]
        len_k=key.shape[1]
        len_v=value.shape[1]
        
        query=self.fc_q(query)
        key=self.fc_k(key)
        value=self.fc_v(value)
        
        # Normal Attention Layer
        # #reshape Q,K
        r_q1=query.view(batch_size,-1,self.n_heads,self.head_dim).permute(0,2,1,3) #[batch_size,n_heads,seq_len,head_dim]
        r_k1=key.view(batch_size,-1,self.n_heads,self.head_dim).permute(0,2,1,3)
        attn1=torch.matmul(r_q1,r_k1.permute(0,1,3,2)) # permute->[batch_size,n_heads,head_dim,seq_len]
        # attn1.shape->[batch_size,n_heads,seq_len,seq_len](inner product)
        
        # Calculate relative position attention score
        r_q2=query.permute(1,0,2).contiguous().view(len_q,batch_size*self.n_heads,self.head_dim)
        #   size->[seq_len,batch_size,hidden_size] -> guarantee continuous  -> reshape
        r_k2=self.relative_position_k(len_q,len_k)
        attn2=torch.matmul(r_q2,r_k2.transpose(1,2)).transpose(0,1)
        attn2=attn2.contiguous().view(batch_size,self.n_heads,len_q,len_k)
        attn=(attn1+attn2)/2
        
        if mask is not None:
            attn=attn.masked_fill(mask==0,-1e10)
        # apply dropout layer
        attn=self.dropout(torch.softmax(attn,dim=-1))
        
        r_v1=value.view(batch_size,-1,self.n_heads,self.head_dim).permute(0,2,1,3)
        weight1=torch.matmul(attn,r_v1)
        r_v2=self.relative_position_v(len_q,len_v)
        weight2=attn.permute(2,0,1,3).contigous().view(len_q,batch_size*self.n_heads,len_k)
        weight2=torch.matmul(weight2,r_v2)
        weight2=weight2.transpose(0,1).contiguous().view(batch_size,self.n_heads,len_q,self.head_dim)
        
        x=weight1+weight2 # size(x) -> [batch_size,n_heads,query_len,head_dim]
        x=x.permute(0,2,1,3).contiguous()
        x=x.view(batch_size,-1,self.hid_dim) # size(x) -> [batch_size,query_len,hid_dim]
        x=self.fc_o(x) # size(x) -> [batch_size,query_len,hid_dim]
        
        return x
        
        
        
        
        
        