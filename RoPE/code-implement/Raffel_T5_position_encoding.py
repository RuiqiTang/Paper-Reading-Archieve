import torch 
from torch import nn 
import torch.nn.functional as F 

class T5RelativePositionBias(nn.Module):
    def __init__(self, num_heads,bidirectional=True,num_buckets=32,max_distanct=128):
        super.__init__()
        self.num_buckets=num_buckets
        self.max_distance=max_distanct
        self.relative_attention_bias=nn.Embedding(num_buckets,num_heads)
        self.bidirectional=bidirectional
    
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Translate relative position to a bucket number for relative attention biases.
        """
        ret = 0
        n=-relative_position
        if bidirectional:
            num_buckets//=2
            ret+=(n<0)*num_buckets
            n=torch.abs(n)
        else:
            n=torch.max(n,torch.zeros_like(n))
        
        max_exact=num_buckets//2
        is_small=n<max_exact
        
        val_if_large=max_exact+(
            torch.log(n.float()/max_exact)/torch.log(torch.tensor(max_distance/max_exact))*(num_buckets-max_exact)
        ).long() 
        val_if_large=torch.min(val_if_large,torch.full_like(val_if_large,num_buckets-1))
        ret+=torch.where(is_small,n,val_if_large)
        return ret 

    def forward(self,query_length,key_length):
        '''
            Compute Relative postion bias
        '''
        q_pos=torch.arange(query_length,dtype=torch.long,device=self.relative_attention_bias.weight.device)
        k_pos=torch.arange(key_length,dtype=torch.long,device=self.relative_attention_bias.weight.device)
        relative_position=k_pos[None,:]-q_pos[:,None] # shape:[query_len,key_len]
        relative_bucket=self._relative_position_bucket(
            relative_position,self.bidirectional,self.num_buckets,self.max_distance
        )
        return self.relative_attention_bias(relative_bucket).permute(2,0,1).unsqueeze(0) #shape:[1,num_heads,q_len,k_len]

class T5Attention(nn.Module):
    def __init__(self, d_model,num_heads,dropout_rate=0.1,relative_attention_num_buckets=32, relative_attention_max_distance=128):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.head_dim=d_model//num_heads
        self.dropout=nn.Dropout(dropout_rate)

        self.q=nn.Linear(d_model,d_model)
        self.k=nn.Linear(d_model,d_model)
        self.v=nn.Linear(d_model,d_model)
        self.o=nn.Linear(d_model,d_model)
    
        self.relative_attention_bias=T5RelativePositionBias(
            num_heads=num_heads,
            bidirectional=True,
            num_buckets=relative_attention_num_buckets,
            max_distanct=relative_attention_max_distance
        )
        self.pruned_heads=set()
        
    def _split_heads(self,x,batch_size):
        x=x.view(batch_size,-1,self.num_heads,self.head_dim)
        return x.permute(0,2,1,3)   # shape:[batch_size,num_heads,seq_length,head_dim]

    def _merge_heads(self,x,batch_size):
        x=x.permute(0,2,1,3).contiguous()
        return x.view(batch_size,-1,self.d_model)

    def forward(self,hidden_states,attention_mask=None,relative_pos=None):
        batch_size,seq_length,_=hidden_states.size()
        
        # Linear proj for q,k,v
        query=self.q(hidden_states)
        key=self.k(hidden_states)
        value=self.v(hidden_states)

        # Split heads
        query=self._split_heads(query,batch_size)
        key=self._split_heads(key,batch_size)
        value=self._split_heads(value,batch_size)

        # Calculate attention score
        attention_scores=torch.matmul(query,key.transpose(-1,-2))
        
        # Apply relative position bias
        if self.relative_attention_bias is not None:
            relative_bias=self.relative_attention_bias(seq_length,seq_length)
            attention_scores=attention_scores+relative_bias
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores=attention_scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        
        # Normalize attn scores to get prob
        attention_probs=F.softmax(attention_scores,dim=-1)
        attention_probs=self.dropout(attention_probs)
    
        # Calculate context vector
        context=torch.matmul(attention_probs,value)
        # Merge heads
        context=self._merge_heads(context,batch_size)
        
        # output
        output=self.o(context)
        
        return output,attention_probs
        
                