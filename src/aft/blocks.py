"""Transformer-style blocks built from AFT layers."""

import torch
import torch.nn as nn
from aft.layers import FeedForward, AFTSimple, AFTFull, AFTLocal, AFTConv

class AFTBlock(nn.Module):
    """使用可切换 AFT 变体的 Pre-LN Transformer 风格 block。"""

    def __init__(
            self,
            d_model,
            hidden_dim,
            dropout,
            aft_type="simple",
            max_seq_len=None,
            local_window_size=None,
            kernel_size=None,
    ):
        super().__init__()

        # Pre-LN：先归一化再进入序列混合层，训练通常更稳定。
        self.norm1 = nn.LayerNorm(d_model)

        # 根据 aft_type 选择具体的 token 混合子层。
        if aft_type == "simple":
            self.aft = AFTSimple(d_model, dropout)
        elif aft_type == "full":
            if max_seq_len is None:
                raise ValueError("max_seq_len must be provided when aft_type='full'")
            self.aft = AFTFull(d_model, max_seq_len, dropout)
        elif aft_type == "local":
            if max_seq_len is None:
                raise ValueError("max_seq_len must be provided when aft_type='local'")
            if local_window_size is None:
                raise ValueError("local_window_size must be provided when aft_type='local'")
            self.aft = AFTLocal(d_model, max_seq_len, local_window_size, dropout)
        elif aft_type == "conv":
            if kernel_size is None:
                raise ValueError("kernel_size must be provided when aft_type='conv'")
            self.aft = AFTConv(d_model, kernel_size, dropout)
        else:
            raise ValueError(f"Unsupported aft_type: {aft_type}. Expected 'simple', 'full', 'local', or 'conv'.")

        # 第二个 Pre-LN 分支：逐位置前馈网络。
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_dim, dropout)

    def forward(self, x):
        # 残差分支 1：混合不同 token 位置的信息。
        x = x + self.aft(self.norm1(x))

        # 残差分支 2：对每个 token 的特征做非线性变换。
        x = x + self.ffn(self.norm2(x))

        return x
