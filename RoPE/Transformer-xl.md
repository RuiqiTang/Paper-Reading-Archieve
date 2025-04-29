Paper from:
```
@article{dai2019transformer,
  title={Transformer-xl: Attentive language models beyond a fixed-length context},
  author={Dai, Zihang and Yang, Zhilin and Yang, Yiming and Carbonell, Jaime and Le, Quoc V and Salakhutdinov, Ruslan},
  journal={arXiv preprint arXiv:1901.02860},
  year={2019}
}
```
Ref: https://zhuanlan.zhihu.com/p/271984518

Exercise Link: [transformer_xl_positional_encoding.py](code-implement/transformer_xl_positional_encoding.py)

# Transformer-Xl position Encoding

- 核心问题:Transformer-xl 解决的问题是，在重用隐藏层信息时，如何保持位置编码的一致性？

## 符号定义与Idea

- Tokens $x=(x_1,\cdots,x_T)$

- 在标准的Transformer中，序列order的信息由一组position encoding来提供，记做 
$U\in \mathcal{R}^{L_{\max}\times d}$，其中第i行
$U_i$
对应段内的第i个绝对位置，L_max规定了建模的最大长度

- 与Transformer不同的是，Transformer-xl的上一个片段的状态会被缓存下来，然后在计算当前段的时候再重复使用上个时间片的hidden states
    - 因为上一个片段的特征在当前片段中进行了重复使用，这也就赋予了Transformer-xl建模更长期的依赖的能力

- 引入位置编码，Transformer-xl的hidden state sequence可以被建模为

$$h_{\tau+1}=f(h_\tau,E_{s_{\tau+1}}+U_{1:L})$$

$$h_\tau=f(h_{\tau+1},E_{s_\tau}+U_{1:L})$$
- 其中，
$E_{s_\tau}$
是对于sequence
$s_\tau$
的word embedding

- 上式出现的一个问题在于，
$h_\tau,E_{s_\tau}$
都与序列的相对位置有关，但是
$U_{1:L}$
与序列的相对位置无关，由此Transformer-xl提出在hidden states 编码相对位置


## Vaswani 对于attention score的分解

对于query矩阵
$q_i$
和key矩阵
$k_i$
之间的attention score，有

$$\begin{aligned}
A_{i,j}^{abs}&=E_{x_i}^T W_q^T W_k E_{x_j}+E_{x_i}^T W_q^T W_k U_j\\
&=U_i^T W_q^T W_k E_{x_j}+U_i^T W_q^T W_k U_j
\end{aligned}$$

基于仅依赖于相对位置编码的思想，可以**重参数化**四个项

$$\begin{aligned}
A_{i,j}^{rel}&=
\underset{(a)}{E_{x_i}^T W_q^T W_k E_{x_j}}+
\underset{(b)}{E_{x_i}^T W_q^T W_k R_{i-j}}\\
&=u^T W_k E_{x_j}+v^T W_k R_{i-j}
\end{aligned}$$

以下是对此的解释：

- 变化1: 将所有绝对位置编码矩阵
$U$
替换成为相对矩阵 
$R_{i-j}$
，反映了**只有相对距离才有影响**的先验假设

- 变化2: 引入一个可被训练的参数
$u$
来替换query 
$U_i^T W_q^T$
，并且使用
$v$
来替换 
$U_i^T W_q^T$
，表明，对不同单词的注意偏差应该保持不变，而不管查询的位置如何

- 变化3: 将两个权重矩阵
$W_{k,E}$
和
$W_{k,R}$
分开，分别用于生成<u>基于内容的密钥向量</u>和<u>基于位置的密钥向量</u>


如何直观的进行理解？

- Term (a): 没有考虑位置编码的原始分数，只是基于内容的寻址
- Term (b): 相对于当前内容的位置偏差
- Term (c): 从内容层面衡量键的重要性，全局内容bias
- Term (d): 从相对层面衡量键的重要性，全局位置bias

由此，Transformer-xl可以变为，可以建模的长期依赖的长度为
$O(NL)$

$$
\begin{aligned}
    \tilde{\mathbf{h}}_\tau^{n - 1} &= [\text{SG}(\mathbf{m}_\tau^{n - 1}) \odot \mathbf{h}_\tau^{n - 1}] \\
    \mathbf{q}_\tau^n, \mathbf{k}_\tau^n, \mathbf{v}_\tau^n &= \tilde{\mathbf{h}}_\tau^{n - 1} \mathbf{W}_q^{n\top}, \tilde{\mathbf{h}}_\tau^{n - 1} \mathbf{W}_{k, E}^n, \tilde{\mathbf{h}}_\tau^{n - 1} \mathbf{W}_v^{n\top} \\
    \mathbf{A}_{i, j}^n &= \mathbf{q}_{\tau, i}^n\top \mathbf{k}_{\tau, j}^n + \mathbf{q}_{\tau, i}^n\top \mathbf{W}_{k, R}^n \mathbf{R}_{i - j} \\
                     &\quad + \mathbf{u}^\top \mathbf{k}_{\tau, j}^n + \mathbf{v}^\top \mathbf{W}_{k, R}^n \mathbf{R}_{i - j} \\
    \mathbf{a}_\tau^n &= \text{Masked-Softmax}(\mathbf{A}_\tau^n) \mathbf{v}_\tau^n \\
    \mathbf{o}_\tau^n &= \text{LayerNorm}(\text{Linear}(\mathbf{a}_\tau^n) + \mathbf{h}_\tau^{n - 1}) \\
    \mathbf{h}_\tau^n &= \text{Positionwise-Feed-Forward}(\mathbf{o}_\tau^n)
\end{aligned}
$$