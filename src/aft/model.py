"""End-to-end models that stack AFT blocks."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from aft.blocks import AFTBlock

class AFTLanguageModel(nn.Module):
    """由多层 AFTBlock 堆叠成的小型自回归语言模型。"""

    def __init__(
            self,
            vocab_size,
            d_model,
            hidden_dim,
            n_layers,
            max_seq_len,
            dropout,
            aft_type="simple",
            local_window_size=None,
            kernel_size=None,
            causal=False,
            use_low_rank_bias=False,
            bias_rank=64,
            use_checkpoint=False,
    ):
        super().__init__()

        # 把 token id [B, T] 转成连续向量 [B, T, D]。
        self.token_emb = nn.Embedding(vocab_size, d_model)#查表层，词表里有vocab_size个token，每个token用d_model维向量表示，并且输出结果相较于输入结果多了一维，形状是向量的维数

        # 为每个可能的位置学习一个位置向量。
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # 堆叠 n_layers 个结构相同但参数独立的 block。
        self.blocks = nn.ModuleList([
            AFTBlock(d_model,
                     hidden_dim,
                     dropout,
                     aft_type=aft_type,
                     max_seq_len=max_seq_len,
                     local_window_size=local_window_size,
                     kernel_size=kernel_size,
                     causal=causal,
                     use_low_rank_bias=use_low_rank_bias,
                     bias_rank=bias_rank
                     )
            for _ in range(n_layers)
        ])

        # 投影到词表分数之前做最后一次归一化。
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)#把隐藏向量变成词表预测分数
        self.dropout = nn.Dropout(dropout)
        # Activation checkpointing，中文常叫“激活检查点”：
        # 训练时少保存中间激活，反向传播时重新计算一部分前向，以时间换显存。
        self.use_checkpoint = use_checkpoint
        self.use_low_rank_bias = use_low_rank_bias
        self.bias_rank = bias_rank

    def forward(self, input_ids):
        # input_ids: [B, T]，其中每个值都是整数 token id。
        B, T = input_ids.shape

        # positions: [T]，放在和 input_ids 相同的设备上。
        positions = torch.arange(T, device=input_ids.device)#生成一串连续整数

        # token embedding 和 position embedding 形状可相加：
        # x 是 [B, T, D]，pos 是 [T, D]，pos 会沿 batch 维广播。
        x = self.token_emb(input_ids)
        pos = self.pos_emb(positions)
        x = x + pos
        x = self.dropout(x)

        # 依次通过多层 AFTBlock。
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                # checkpoint 只在训练阶段使用；验证/生成阶段不需要节省反向传播激活。
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # logits[b, t] 是位置 t 上对所有下一个 token 类别的预测分数。
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
