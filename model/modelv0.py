import sys
from pathlib import Path

# 添加 frontend 目录的父目录到 sys.path
project_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(project_dir))

import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
import re

import os
from tokenizer.chatglm3.tokenization_chatglm import ChatGLMTokenizer
from tokenizer.babyllemma.tokenizer_babyllemma import BALTokenizer
from tokenizer.qwen72B.tokenization_qwen import QWenTokenizer

from tqdm import tqdm
import json
from openai import OpenAI
import dotenv

dotenv.load_dotenv()
# openai = OpenAI(
#     base_url = "https://agent-api.blankenschool.com/v1",
#     api_key = os.getenv("OPENAI_API_KEY"),
# )

@dataclass
class ModelArgs:
    # Llama 7B 模型的默认超参数

    # Transformer模型的维度（dimension）
    dim: int = 4096

    # Transformer层的数量
    n_layers: int = 32

    # 注意力头的数量
    n_heads: int = 32

    # 注意力机制中键值（key-value）头的数量，如果为None，则使用n_heads
    n_kv_heads: Optional[int] = None

    # 词汇表的大小
    vocab_size: int = 32000

    # 隐藏层的维度，如果为None，则使用默认值
    hidden_dim: Optional[int] = None

    # MLP隐藏层大小的倍数，用于确定隐藏层的大小
    multiple_of: int = 256

    # Layer Normalization 中 epsilon 的值
    norm_eps: float = 1e-5

    # 最大序列长度
    max_seq_len: int = 2048

    # Dropout概率
    dropout: float = 0.0

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        # 初始化RMSNorm模块，dim是输入张量的维度，eps是Layer Normalization中的epsilon值

        # 设置epsilon值
        self.eps = eps

        # 创建一个可学习的参数张量，用于缩放Layer Normalization的输出
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 定义一个辅助函数_norm，用于对输入张量进行Layer Normalization操作
        # x: 输入张量

        # 计算逐元素平方，然后在指定维度上取平均，再开方的倒数, eps在这里防止x.pow(2).mean(-1, keepdim=True)为0
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 定义前向传播函数，对输入张量进行Layer Normalization并乘以学习到的缩放因子

        # 将输入张量转换为float类型，进行Layer Normalization，然后将结果转回原始数据类型
        output = self._norm(x.float()).type_as(x)

        # 乘以学习到的缩放因子
        return output * self.weight

# 个人理解：这里相当于预先生成好了对应每一个位置和最大dim的位置编码矩阵，然后在forward的时候直接取出来用, 所以这里的dim应该传入的是transformers的最大dim, end应该传入的是transformers的最大序列长度
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # 预计算RoPE相对位置嵌入所需的频率

    # 计算频率，其中dim是嵌入维度，theta是一个缩放因子
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # 生成一个序列t，表示相对位置的范围
    t = torch.arange(end, device=freqs.device)  # type: ignore

    # 计算相对位置嵌入的频率矩阵
    freqs = torch.outer(t, freqs).float()  # type: ignore

    # 计算频率矩阵的实部和虚部，分别表示RoPE的余弦和正弦部分
    freqs_cos = torch.cos(freqs)  # 实部
    freqs_sin = torch.sin(freqs)  # 虚部

    # 返回计算得到的余弦和正弦部分
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # 获取输入张量x的维数
    ndim = x.ndim
    
    # 确保x的维数至少为2，因为我们要处理的是至少二维的数据（例如，批次大小和特征维度）
    # assert 0 <= 1 < ndim 这里修改一个错误
    assert 1 < ndim

    # 确保传入的频率张量freqs_cis的形状与x的第二个维度和最后一个维度相匹配
    # freqs_cis的形状应该是(x的第二个维度的大小, x的最后一个维度的大小)
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    ##########################################################################
    # enumerate(x.shape) 会遍历 x 的形状（shape），同时提供每个维度的索引和大小。这里的 i 是维度的索引（从0开始），d 是对应维度的大小。

    # 举个例子，假设 x 是一个形状为 (2, 3, 4, 5) 的四维张量，那么 enumerate(x.shape) 会产生以下的索引和大小：

    # 当 i 为 0 时，d 为 2（第一个维度的大小）
    # 当 i 为 1 时，d 为 3（第二个维度的大小）
    # 当 i 为 2 时，d 为 4（第三个维度的大小）
    # 当 i 为 3 时，d 为 5（第四个维度的大小）
    # 列表推导式中的条件 if i == 1 or i == ndim - 1 检查当前维度是否是第二个维度（索引为1）或最后一个维度（索引为 ndim - 1）。如果是，它就使用 x 的相应维度大小 d；如果不是，它就使用1。

    # 对于上面的例子，这个列表推导式将生成新的形状列表 [1, 3, 1, 5]，这意味着第二个维度和最后一个维度保持不变，而其他维度被设置为1。这个新的形状可以用于调整 freqs_cis 的形状，以便在不改变其在第二个维度和最后一个维度上的数据的情况下，将其广播到 x 的形状。
    ###########################################################################

    # 将freqs_cis张量的形状调整为新的形状，以便可以广播到x的形状
    # view函数仅改变张量的形状，不改变数据本身
    return freqs_cis.view(shape)

# 定义一个函数，用于将旋转位置嵌入应用于查询（xq）和键（xk）张量。
def apply_rotary_emb(
    xq: torch.Tensor,  # 查询张量
    xk: torch.Tensor,  # 键张量
    freqs_cos: torch.Tensor,  # 用于旋转的余弦频率
    freqs_sin: torch.Tensor   # 用于旋转的正弦频率
) -> Tuple[torch.Tensor, torch.Tensor]:

    # 将查询张量重塑为复数表示形式，将最后一个维度分成实部和虚部。
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    
    # 将键张量重塑为复数表示形式，将最后一个维度分成实部和虚部。
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # 重塑余弦频率以便广播，以匹配查询张量实部的形状。
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    
    # 重塑正弦频率以便广播，以匹配查询张量实部的形状。
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 使用余弦和正弦频率对查询张量的实部和虚部应用旋转。
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    
    # 使用余弦和正弦频率对键张量的实部和虚部应用旋转。
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 将旋转后的查询张量的最后两个维度压平成一个维度。
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    
    # 将旋转后的键张量的最后两个维度压平成一个维度。
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    # 返回旋转后的查询和键张量，并确保它们与输入张量的类型相同。
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 定义一个函数 repeat_kv，它接受一个张量 x 和一个整数 n_rep 作为输入，并返回一个张量。

    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    # 这个函数是想要模拟 torch.repeat_interleave 函数的行为，换言之，可以用torch.repeat_interleave(x, dim=2, repeats=n_rep)直接代替


    bs, slen, n_kv_heads, head_dim = x.shape
    # 解包张量 x 的形状，分别得到批次大小（bs）、序列长度（slen）、键值对头的数量（n_kv_heads）和每个头的维度（head_dim）。

    if n_rep == 1:
        # 如果 n_rep（重复次数）为1，则不需要重复，直接返回原始张量 x。
        return x

    # 如果 n_rep 不为1，以下操作将张量 x 的特定维度进行重复。
    return (
        x[:, :, :, None, :]
        # 使用 None（或者说是 np.newaxis）在第四个维度（从0开始计数）增加一个新的维度，形状变为(bs, slen, n_kv_heads, 1, head_dim)。
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        # 使用 .expand 方法将新加的维度扩展 n_rep 次。
        # 这不会复制数据，但会使得每个元素在新维度上重复 n_rep 次。
        # 形状变为(bs, slen, n_kv_heads, n_rep, head_dim)。
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        # 使用 .reshape 方法将扩展后的张量重新塑形。
        # 这里将 n_kv_heads 和 n_rep 这两个维度合并，新的形状是(bs, slen, n_kv_heads * n_rep, head_dim)。
        # 这一步实际上是在模拟 repeat_interleave 的行为，
        # 在 n_kv_heads 维度上重复每个元素 n_rep 次。
    )

class Attention(nn.Module):

    def __init__(self, args: ModelArgs):
        # 类的构造函数，接收一个名为 args 的参数，这个参数是一个包含模型配置的对象。

        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 设置键值对头的数量。如果 args 中 n_kv_heads 没有设置，则使用 n_heads 的值。

        assert args.n_heads % self.n_kv_heads == 0
        # 断言总头数是键值对头数的整数倍，这是为了确保可以平均分配头到不同的键值对。

        model_parallel_size = 1
        # 设置模型并行的大小，默认为1，表示不进行模型并行。

        self.n_local_heads = args.n_heads // model_parallel_size
        # 计算本地头的数量，即在单个模型副本中的头的数量。

        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # 计算本地键值对头的数量。

        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 计算重复的次数，即每个键值对头需要重复的头的数量。

        self.head_dim = args.dim // args.n_heads
        # 计算每个头的维度大小。

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        # 创建一个线性层用于生成查询（query）的权重。

        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 创建一个线性层用于生成键（key）的权重。

        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 创建一个线性层用于生成值（value）的权重。

        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        # 创建一个线性层用于输出的权重。

        self.attn_dropout = nn.Dropout(args.dropout)
        # 创建一个 Dropout 层用于注意力权重。

        self.resid_dropout = nn.Dropout(args.dropout)
        # 创建一个 Dropout 层用于残差连接。

        self.dropout = args.dropout
        # 保存 dropout 的比例。

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # 检查 PyTorch 是否有 'scaled_dot_product_attention' 函数，以确定是否使用 Flash Attention。

        if not self.flash:
            # 如果不使用 Flash Attention，则进行如下操作：
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 打印警告信息。

            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            # 创建一个形状为 (1, 1, max_seq_len, max_seq_len) 的张量，用负无穷填充，
            # 这通常用于在自注意力中屏蔽未来的位置。

            mask = torch.triu(mask, diagonal=1)
            # 使用上三角函数，将 mask 张量转换为上三角矩阵，对角线上方的元素保持为负无穷，
            # 这样在自注意力计算时，可以避免位置 i 关注到位置 j (j > i)。

            self.register_buffer("mask", mask)
            # 将 mask 注册为模块的缓冲区，这样它就不会被视为模型参数。
        else:
            print("using fast attention —— scaled_dot_product_attention")


    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        # 定义 Attention 类的前向传播函数，接收输入张量 x 以及用于旋转位置编码（RoPE）的频率张量 freqs_cos 和 freqs_sin。
        bsz, seqlen, _ = x.shape
        # 从输入张量 x 中获取批次大小（bsz）、序列长度（seqlen）和特征维度（这里未使用，因此用 _ 忽略）。

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # 对输入 x 应用三个不同的线性变换得到查询（Q）、键（K）和值（V）。

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        # 重塑 Q、K、V，为多头注意力分割头的维度。

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        # 应用旋转位置编码（RoPE）到查询和键上，这是一种相对位置编码方法。

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        # 通过 repeat_kv 函数重复键和值，以匹配查询的数量。

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        # 交换头的维度和序列长度的维度，以便于执行批量矩阵乘法。

        # flash implementation
        if self.flash:
            # 如果使用 Flash Attention 的优化实现：
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # manual implementation
            # 否则使用手动实现的注意力计算：
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            # 计算查询和键的点积，并通过头维度的平方根进行缩放。

            assert hasattr(self, 'mask')
            # 断言类中确实存在 mask 属性。

            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            # 将 mask 添加到分数上，以屏蔽未来的信息。

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            # 应用 softmax 函数来获得注意力权重，并确保数据类型与查询相同。

            scores = self.attn_dropout(scores)
            # 应用 dropout 到注意力权重上。

            output = torch.matmul(scores, xv)
            # 使用注意力权重对值进行加权求和。

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        # 交换头和序列长度的维度，并重塑输出以合并头维度。

        # final projection into the residual stream
        output = self.wo(output)
        # 使用输出权重矩阵对合并后的输出进行线性变换。

        output = self.resid_dropout(output)
        # 应用 dropout 到最终的输出上。

        return output
        # 返回最终的输出。

class FeedForward(nn.Module):
    # 定义一个前馈神经网络模块，继承自PyTorch的nn.Module基类。
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        # 初始化函数，接收输入维度dim、隐藏层维度hidden_dim、hidden_dim的倍数multiple_of和dropout率。
        super().__init__()  # 调用父类的初始化函数。
        if hidden_dim is None:
            # 如果没有指定隐藏层维度，则进行以下计算以确定其大小。
            hidden_dim = 4 * dim  # 默认情况下，隐藏层维度是输入维度的四倍。
            # hidden_dim = int(2 * hidden_dim / 3)  # 将隐藏层维度调整为其2/3。
            # 确保hidden_dim是multiple_of的倍数，若不是，则向上取整到最近的multiple_of倍数。
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        # 定义第一个线性变换层，从输入维度到隐藏层维度，不使用偏置项。
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二个线性变换层，从隐藏层维度回到输入维度，不使用偏置项。
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义第三个线性变换层，从输入维度到隐藏层维度，不使用偏置项。
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义一个dropout层，用于在训练期间随机丢弃一些神经元以减少过拟合。
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播函数定义了模型如何处理输入数据x。
        # self.w1(x)应用第一个线性变换。
        # F.silu(self.w1(x))应用SiLU激活函数（也称为Swish）。
        # self.w3(x)应用第三个线性变换。
        # 将激活后的结果与第三个线性变换的结果相乘。
        # self.w2(...)应用第二个线性变换。
        # self.dropout(...)应用dropout。
        # 返回最终的结果。
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    # 定义一个Transformer模块的类，继承自PyTorch的nn.Module。
    def __init__(self, layer_id: int, args: ModelArgs):
        # 初始化函数，接收层的ID（layer_id）和一个包含模型参数的对象（args）。
        super().__init__()  # 调用父类的初始化函数。
        self.n_heads = args.n_heads  # 设置多头注意力的头数。
        self.dim = args.dim  # 设置模型的维度。
        self.head_dim = args.dim // args.n_heads  # 计算每个头的维度。
        
        # 创建一个Attention对象，用于计算多头自注意力。
        self.attention = Attention(args)
        
        # 创建一个FeedForward对象，用于定义前馈网络部分。
        self.feed_forward = FeedForward(
            dim=args.dim,  # 输入维度。
            hidden_dim=args.hidden_dim,  # 隐藏层维度。
            multiple_of=args.multiple_of,  # 确保隐藏层维度是这个值的倍数。
            dropout=args.dropout,  # Dropout比率。
        )
        
        self.layer_id = layer_id  # 存储层的ID。
        
        # 创建两个RMSNorm层，用于对注意力和前馈网络层的输出进行规范化。
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        # 定义前向传播函数，它指定了数据如何通过这个模块。
        # 对输入x应用规范化和注意力机制，然后将结果与原始输入相加实现残差连接。
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        # 对上一步的输出h应用规范化和前馈网络，然后再次与h相加实现第二个残差连接。
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        
        # 返回最终的输出。
        return out

class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]  # 定义一个类属性，用于存储最后一次前向传播的损失值，它可能是None或者torch.Tensor。

    def __init__(self, params: ModelArgs):
        super().__init__()  # 调用nn.Module的初始化函数。
        self.params = params  # 存储传入的模型参数。
        self.vocab_size = params.vocab_size  # 从参数中获取词汇表的大小。
        self.n_layers = params.n_layers  # 从参数中获取Transformer层的数量。

        # 创建一个词嵌入层，将词汇表中的每个词映射到一个高维空间。
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)  # 创建一个dropout层，用于正则化。
        self.layers = torch.nn.ModuleList()  # 创建一个ModuleList，用于存储所有的Transformer层。
        for layer_id in range(params.n_layers):  # 循环创建每一个Transformer层。
            self.layers.append(TransformerBlock(layer_id, params))  # 将每个Transformer层添加到ModuleList中。
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)  # 创建一个RMSNorm规范化层。
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)  # 创建输出层。

        # 将输出层的权重与词嵌入层的权重绑定，这是权重共享的一种方法。
        self.tok_embeddings.weight = self.output.weight

        # 预计算用于RoPE相对位置编码的频率项。
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        # 将这些预计算的值注册为模型的缓冲区，这样它们就可以在模型保存和加载时保持不变。
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # 初始化模型的所有权重。
        self.apply(self._init_weights)
        # 根据GPT-2论文中的建议，对残差连接的线性投影层进行特殊的初始化。
        for pn, p in self.named_parameters():  # 遍历所有的参数。
            p.requires_grad = True  # 设置参数的requires_grad属性为True。
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):  # 如果参数名以特定的后缀结束。
                # 根据GPT-2的建议进行初始化。
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

        # 初始化last_loss属性，用于存储最后一次调用forward时的损失。
        self.last_loss = None

    def _init_weights(self, module):
        # 这是一个私有方法，用于初始化传入模块的权重。
        if isinstance(module, nn.Linear):
            # 如果模块是线性层（全连接层）。
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # 使用均值为0，标准差为0.02的正态分布初始化线性层的权重矩阵。
            if module.bias is not None:
                # 如果线性层有偏置项。
                torch.nn.init.zeros_(module.bias)
                # 使用0初始化偏置项。
        elif isinstance(module, nn.Embedding):
            # 如果模块是嵌入层。
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # 使用均值为0，标准差为0.02的正态分布初始化嵌入层的权重矩阵

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 定义模型的前向传播函数，接收输入的tokens和可选的targets作为目标标签。
        _bsz, seqlen = tokens.shape  # 从输入tokens的形状中获取批次大小和序列长度。
        h = self.tok_embeddings(tokens)  # 将输入的tokens通过词嵌入层得到嵌入表示。
        h = self.dropout(h)  # 对嵌入表示应用dropout进行正则化。
        # 根据序列长度截取预计算的相对位置编码的余弦和正弦部分。
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            # 遍历Transformer模型的每一层。
            h = layer(h, freqs_cos, freqs_sin)  # 将嵌入表示和位置频率信息传递给每一层，并更新h。
        h = self.norm(h)  # 对最后一层的输出应用规范化层。

        if targets is not None:
            # 如果提供了目标标签，计算模型的输出与目标之间的交叉熵损失。
            logits = self.output(h)  # 将最后一层的输出通过线性层得到logits。
            # 计算logits和targets之间的交叉熵损失，并存储在last_loss属性中。
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 如果没有提供目标标签，我们处于推理模式。
            # 为了优化性能，只计算序列最后一个位置的输出。
            logits = self.output(h[:, [-1], :])  # 使用列表[-1]来保持时间维度。
            self.last_loss = None  # 没有目标，所以last_loss设置为None。

        return logits  # 返回logits，无论是整个序列的还是最后位置的。

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 此函数用于配置优化器，设置权重衰减、学习率、beta参数和设备类型。

        # 从模型中获取所有参数，创建一个参数名到参数对象的映射。
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 过滤掉不需要梯度的参数。
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # 创建优化器参数组。对于维度大于等于2的参数（如矩阵），将应用权重衰减。
        # 通常这包括了线性层的权重和嵌入层的权重，而偏置和层归一化的参数不会应用权重衰减。
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        # 将这些参数分为两组，一组应用权重衰减，另一组不应用。
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # 打印出应用和不应用权重衰减的参数数量。
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # 检查是否可以使用融合版本的AdamW优化器，这通常在CUDA设备上可用，可以提高性能。
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        # 创建AdamW优化器实例，并根据设备类型决定是否使用融合版本。
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer  # 返回配置好的优化器实例。

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # 此函数用于估算模型的浮点运算效率（Model Flops Utilization, MFU），
        # 单位是A100 GPU的bfloat16浮点数峰值运算能力（FLOPS）。

        # 首先估算每次迭代中的浮点运算次数（FLOPs）。
        # 参考资料是PaLM论文的附录B: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())  # 计算模型所有参数的总元素数。
        cfg = self.params  # 获取模型的配置参数。
        # 从模型配置中提取层数(L)，头数(H)，每个头的维度(Q)，最大序列长度(T)。
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        # 每个token需要的浮点运算次数，根据PaLM论文的计算方法得出。
        flops_per_token = 6*N + 12*L*H*Q*T
        # 每次前向和后向传播所需的浮点运算次数，等于每个token的FLOPs乘以序列长度T。
        flops_per_fwdbwd = flops_per_token * T
        # 每次迭代所需的浮点运算次数，等于每次前向后向传播的FLOPs乘以迭代次数。
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # 将模型的浮点运算吞吐量表示为A100 GPU bfloat16峰值FLOPS的比例。
        flops_achieved = flops_per_iter * (1.0/dt) # 每秒的浮点运算次数。
        flops_promised = 312e12 # A100 GPU的bfloat16浮点数峰值运算能力是312 TFLOPS。
        mfu = flops_achieved / flops_promised  # 计算MFU，即实际FLOPS与峰值FLOPS的比值。

        return mfu  # 返回模型的浮点运算效率（MFU）。

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        last_token = None
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            logits = logits[:, -1, :] # crop to just the final time step
            if temperature == 0.0:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
                if idx_next == 151643:
                    break
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    # 标记超过top_p的部分
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # 保证至少有一个token是可选的
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # 创建一个原始logits大小的mask，默认全部为False（即保留所有token）
                    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                    # 将需要移除的sorted indices映射回原始indices
                    indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

                    # 使用mask来更新logits
                    logits[indices_to_remove] = -float('Inf')

                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                
                # if last_token is not None:
                #     probs[0, last_token] = 0.0

                idx_next = torch.multinomial(probs, num_samples=1)
                #chatglm3
                # if idx_next == 2:
                #     break
                #qwen
                if idx_next == 151643 or idx_next == 151645:
                    break
                
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            last_token = idx_next


        return idx
    @torch.inference_mode()
    def generate_messages(self, tokenizer:QWenTokenizer, input_messages:list, max_new_tokens = 2048, temperature=1.0, top_k=None, top_p=None, device = "cuda"):
        """
        Take a list of input messages and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        last_role = None
        input_str = ""
        for id, input_message in enumerate(input_messages):
            if input_message["role"] == "user":
                if last_role == "user":
                    raise ValueError("连续两个用户消息")
                if id == len(input_messages) - 1:
                    input_str += "<|im_start|>user\n" + input_message["content"] + "\n <|im_end|><|im_start|>assistant\n"
                else:
                    input_str += "<|im_start|>user\n" + input_message["content"] + "\n <|im_end|>"
                
                last_role = "user"

            elif input_message["role"] == "assistant":
                if last_role == "assistant":
                    raise ValueError("连续两个助手消息")
                input_str += "<|im_start|>assistant\n" + input_message["content"] + "\n <|im_end|>"
                last_role = "assistant"
        print(input_str)
        input_tokens = tokenizer(input_str, add_special_tokens=False, return_attention_mask=False)

        input_length = len(input_str)

        input_tokens = torch.tensor(input_tokens["input_ids"], dtype=torch.long, device=device).unsqueeze(0)

        output_tokens = self.generate(input_tokens, max_new_tokens, temperature, top_k, top_p).to("cpu").numpy()

        output_text = tokenizer.decode(output_tokens[0].tolist())[input_length:]

        return output_text