Ref:
1. https://zhuanlan.zhihu.com/p/669379369
2. 代码文件：[He_DeBERTa](code-implement/He_DeBERTa.py)
3. 论文地址：https://arxiv.org/pdf/2006.03654

## 主要解决的问题

0. 全称：Decoding-enhanced BERT with disentangled attention

1. 在DeBERTa之前，transformer架构就是将content信息的向量与position信息的向量简单的加在一起

2. 论文认为这种处理方式不好，将代表word/token的content信息和position信息用两个向量分别表示（即disentangled）

## Disentangled Attention 的计算过程

对于序列中的第i个词（作为query）和第j个词（作为key），DeBERTa的注意力分数 
$A_{ij}$
的计算方式如下：

$$A_{ij}=(C_i+P_{i|j})^T(C_j+P_{i|j})$$

其中，

- C_i 第i个词的内容向量

- C_j 第j个词的内容向量

- P_{i|j} 当第j个词作为key时，第i个词的相对位置向量，编码了从j到i的相对位置信息

- P_{j|i} ...

展开这个公式，可以得到

$$A_{ij}=C_i^T C_j+C_i^T P_{j|i} + P_{i|J}^T C_j+ P_{i|J}^T P_{j|i}$$

这个公式包含了四项：

- 内容-内容注意力：$C_i^T C_j$，这是仅基于词的内容计算的注意力

- 内容-相对位置注意力：$C_i^T P_{j|i}$，查询词的内容与目标词的相对位置之间的注意力

- 相对位置-内容注意力：$P_{i|J}^T C_j$，查询词的显贵位置与目标词的内容之间的注意力

- 显贵位置-相对位置注意力：$P_{i|J}^TP_{j|i}$，查询词的相对位置与目标词的相对我诶之之间的注意力

## DeBERTa如何获得这些向量？

- 内容向量$C$：每个词的输入嵌入$x_i$经过一个线性变换得到内容向量，对于不同的注意力头，会使用不同的变换矩阵

$$C_i=W_c x_i$$

- 相对位置向量$P$：DeBERTa使用一个独立的嵌入矩阵来编码相对位置

    - 对于相对距离$d=i-j$，会有一个对应的相对位置嵌入 $p_d$
    - 然后，这个相对位置嵌入会通过两个不通过的线性变换，分别在q视角下的相对位置向量和k视角下的相对位置向量

    $$P_{i|j}=W_{pq} p_{i-j}$$

    $$P_{j|i}=W_{pk} p_{j-i}$$

## DeBERTa的优势

1. 更好的建模能力: 通过解耦内容和位置信息，模型可以更灵活地学习它们各自对注意力权重的贡献

2. 更强的泛化能力: 解耦的表示可能有助于模型更好地泛化到不同长度的序列

3. 在下游任务中通常取得更好的性能