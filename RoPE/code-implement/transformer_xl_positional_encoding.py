import torch
import torch.nn as nn
import torch.nn.functional as F

class RelPosEmb(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

        # 创建一个位置嵌入表，大小为 (2 * max_len - 1, d_model)
        # 索引 i 对应于相对距离 i - max_len + 1
        self.rel_pos_emb = nn.Parameter(torch.Tensor(2 * max_len - 1, d_model))
        nn.init.xavier_normal_(self.rel_pos_emb)

    def forward(self, seq_len):
        """
        生成给定序列长度的相对位置嵌入。

        Args:
            seq_len: 当前序列的长度。

        Returns:
            rel_pos_encoding: 形状为 (seq_len, seq_len, d_model) 的相对位置编码。
        """
        # 创建一个索引张量，表示序列中每个位置的索引
        pos_ids = torch.arange(seq_len).unsqueeze(0)  # (1, seq_len)

        # 创建一个相对距离矩阵
        # 对于序列中的每个位置 i 和 j，计算它们的相对距离 j - i
        rel_dist = pos_ids - pos_ids.transpose(0, 1)  # (seq_len, seq_len)

        # 将相对距离平移到 [0, 2 * max_len - 2] 的范围内，以便索引到 self.rel_pos_emb
        # 相对距离为 k 时，对应的索引为 k + max_len - 1
        rel_pos_idx = rel_dist + self.max_len - 1  # (seq_len, seq_len)
        rel_pos_idx = torch.clamp(rel_pos_idx, 0, 2 * self.max_len - 2) # 确保索引在有效范围内

        # 根据相对位置索引获取对应的嵌入向量
        rel_pos_encoding = self.rel_pos_emb[rel_pos_idx]  # (seq_len, seq_len, d_model)

        return self.dropout(rel_pos_encoding)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # Transformer-XL 中用于相对位置编码的额外线性层
        self.r_proj = nn.Linear(d_model, d_model) # 对应公式中的 W_kr

        # 可学习的位置偏置向量
        self.u = nn.Parameter(torch.Tensor(self.n_head, self.d_head)) # 对应公式中的 u
        self.v = nn.Parameter(torch.Tensor(self.n_head, self.d_head)) # 对应公式中的 v
        nn.init.xavier_normal_(self.u)
        nn.init.xavier_normal_(self.v)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, r):
        """
        带相对位置编码的多头自注意力机制。

        Args:
            q: 查询张量，形状为 (batch_size, seq_len, d_model)
            k: 键张量，形状为 (batch_size, seq_len, d_model)
            v: 值张量，形状为 (batch_size, seq_len, d_model)
            r: 相对位置编码，形状为 (seq_len, seq_len, d_model)

        Returns:
            attn_output: 注意力输出，形状为 (batch_size, seq_len, d_model)
            attn_weights: 注意力权重，形状为 (batch_size, n_head, seq_len, seq_len)
        """
        batch_size, seq_len, _ = q.size()

        # 线性变换并分头
        q_h = self.q_proj(q).view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)  # (batch_size, n_head, seq_len, d_head)
        k_h = self.k_proj(k).view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)  # (batch_size, n_head, seq_len, d_head)
        v_h = self.v_proj(v).view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)  # (batch_size, n_head, seq_len, d_head)

        # 相对位置编码的线性变换
        r_h = self.r_proj(r).view(seq_len, seq_len, self.n_head, self.d_head).permute(2, 0, 1, 3) # (n_head, seq_len, seq_len, d_head)

        # 1. 内容-内容交互 (ac)
        # (batch_size, n_head, seq_len, d_head) @ (batch_size, n_head, d_head, seq_len) -> (batch_size, n_head, seq_len, seq_len)
        ac = torch.matmul(q_h, k_h.transpose(2, 3))

        # 2. 内容-位置交互 (bd)
        # (batch_size, n_head, seq_len, d_head) @ (n_head, d_head, seq_len, seq_len) -> (batch_size, n_head, seq_len, seq_len)
        bd = torch.matmul(q_h, r_h.permute(0, 2, 1, 3))

        # 3. 位置-内容交互 (ce)
        # (batch_size, n_head, seq_len, d_head) 与 (n_head, d_head, seq_len) 的交互
        # 需要将 u 扩展到 batch_size 和 seq_len 维度
        u_for_q = self.u.unsqueeze(0).unsqueeze(2) # (1, n_head, 1, d_head)
        ce = torch.matmul(u_for_q, k_h.transpose(2, 3)).squeeze(2) # (batch_size, n_head, seq_len, seq_len)

        # 4. 位置-位置交互 (de)
        # (batch_size, n_head, seq_len, d_head) 与 (n_head, d_head, seq_len, seq_len) 的交互
        # 需要将 v 扩展到 batch_size 和 seq_len 维度
        v_for_q = self.v.unsqueeze(0).unsqueeze(2) # (1, n_head, 1, d_head)
        de = torch.matmul(v_for_q, r_h.permute(0, 2, 1, 3)).squeeze(2) # (batch_size, n_head, seq_len, seq_len)

        # 总的注意力分数
        attn_scores = (ac + bd + ce + de) / (self.d_head ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, v_h)  # (batch_size, n_head, seq_len, d_head)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # (batch_size, seq_len, d_model)
        attn_output = self.o_proj(attn_output)
        attn_output = self.dropout(attn_output)

        return attn_output, attn_weights