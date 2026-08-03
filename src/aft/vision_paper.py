"""Paper-style AFT-conv image classifier.

This module makes the ImageNet AFT-conv head parameterization explicit and
configurable. ``vision.py`` is the valid special case ``n_heads=d_model``:

- ``d_model`` is the model width.
- ``n_heads`` is an independent hyperparameter.
- ``head_dim = d_model // n_heads``.
- Q and V are reshaped to ``[B, N, n_heads, head_dim]``.
- K has shape ``[B, N, n_heads]``.
- Each head has one shared spatial position-bias kernel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from aft.layers import FeedForward


class PaperPatchEmbedding(nn.Module):
    """Convert an image to non-overlapping patch tokens."""

    def __init__(self, image_size, patch_size, in_channels, d_model):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: [B, C, H, W] -> [B, D, Gh, Gw] -> [B, N, D]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PaperAFTConv2D(nn.Module):
    """AFT-conv with paper-style grouped heads.

    For each head, K is one scalar per patch and the same spatial bias kernel
    is shared across all value dimensions in that head.
    """

    def __init__(
            self,
            d_model,
            image_size,
            patch_size,
            kernel_size,
            n_heads,
            dropout,
            bias_limit=3.0,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd to preserve grid size")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.bias_limit = bias_limit

        # Paper-style shapes:
        # q/v: [B, N, D] -> [B, N, H, C]
        # k:   [B, N, D] -> [B, N, H]
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, n_heads)
        self.to_v = nn.Linear(d_model, d_model)

        # One KxK position-bias kernel per head. When n_heads=d_model this
        # becomes the channel-wise special case implemented in vision.py.
        self.position_bias = nn.Parameter(
            torch.empty(n_heads, 1, kernel_size, kernel_size)
        )
        nn.init.normal_(self.position_bias, mean=0.0, std=0.02)

        # Same normalized-bias parameterization as the existing implementation,
        # but now at head level rather than feature-channel level.
        self.position_gain = nn.Parameter(torch.zeros(n_heads, 1, 1, 1))
        self.position_offset = nn.Parameter(torch.zeros(n_heads, 1, 1, 1))

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _conv_weight(self, dtype):
        # raw position_bias: [H, 1, K, K]
        raw_bias = self.position_bias

        bias_mean = raw_bias.mean(dim=(2, 3), keepdim=True)
        bias_std = raw_bias.std(dim=(2, 3), keepdim=True, unbiased=False)
        # normalized/gated position_bias: [H, 1, K, K]
        normalized_bias = (raw_bias - bias_mean) / (bias_std + 1e-6)

        position_bias = self.position_gain * normalized_bias + self.position_offset

        position_bias = self.bias_limit * torch.tanh(position_bias / self.bias_limit)

        conv_weight = torch.exp(position_bias.float()) - 1.0
        conv_weight = conv_weight.to(dtype)
        return conv_weight

    def forward(self, x):
        # x: [B, N, D], where N = grid_size * grid_size.是 patch token 数量
        B, N, D = x.shape
        if N != self.num_patches:
            raise ValueError(
                f"expected {self.num_patches} patches, got {N}"
            )

        H = self.n_heads #H = head 数量
        C = self.head_dim #每个 head 里的 value 维度，也就是 head_dim
        G = self.grid_size #单行patch数量

        # 1. q: [B, N, D] -> [B, N, H, C]
        q = self.to_q(x)
        q = q.reshape(B, N, H, C)
        # 2. k: [B, N, D] -> [B, N, H]
        k = self.to_k(x)
        # 3. v: [B, N, D] -> [B, N, H, C]
        v = self.to_v(x)
        v = v.reshape(B, N, H, C)
        # 4. compute exp(k) stably in float32
        k_float = k.float()
        v_float = v.float()
        k_float = k_float - k_float.amax(dim=1, keepdim=True)
        exp_k = torch.exp(k_float)
        # 5. compute global numerator/denominator
        #[B, N, H, C]
        kv = exp_k.unsqueeze(-1) * v_float
        global_numerator = kv.sum(dim=1, keepdim=True)
        global_denominator = exp_k.sum(dim=1, keepdim=True).unsqueeze(-1)
        # 6. reshape patch tokens back to a 2D grid
        kv_2d = kv.permute(0, 2, 3, 1).reshape(B, H * C, G, G) #[B, H*C, G, G]
        normalizer_2d = exp_k.permute(0, 2, 1).reshape(B, H, G, G) #[B, H, G, G]
        # 7. apply head-level position-bias convolution
        head_weight = self._conv_weight(kv_2d.dtype) #[H, 1, K, K]
        value_weight = head_weight.repeat_interleave(C, dim=0) #[H*C, 1, K, K]

        #F.conv2d只能处理[B, channel, height, width]格式
        local_numerator = F.conv2d(
            kv_2d,
            value_weight,
            padding=self.kernel_size // 2,
            groups=H * C,
        ) #[B, H*C, G, G]

        local_denominator = F.conv2d(
            normalizer_2d,
            head_weight,
            padding=self.kernel_size // 2,
            groups=H
        ) #[B, H, G, G]
        # 8. combine global and local terms
        local_numerator = local_numerator.reshape(B, H, C, N).permute(0, 3, 1, 2)
        local_denominator = (
            local_denominator.reshape(B, H, N)
            .permute(0, 2, 1)
            .unsqueeze(-1)
        )
        numerator = global_numerator + local_numerator #[B, N, H, C]
        denominator = global_denominator + local_denominator #[B, N, H, 1]
        # 9. apply sigmoid(q), out_proj, and dropout
        y = torch.sigmoid(q.float()) * numerator / (denominator + 1e-6)

        y = y.reshape(B, N, D)
        y = y.to(x.dtype)

        y = self.out_proj(y)
        y = self.dropout(y) #[B, N, D]

        return y


class PaperVisionBlock(nn.Module):
    """Pre-LN vision block with paper-style AFT-conv."""

    def __init__(
            self,
            d_model,
            hidden_dim,
            image_size,
            patch_size,
            kernel_size,
            n_heads,
            dropout,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.aft = PaperAFTConv2D(
            d_model=d_model,
            image_size=image_size,
            patch_size=patch_size,
            kernel_size=kernel_size,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(
            d_model=d_model,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x):
        x = x + self.aft(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PaperAFTImageClassifier(nn.Module):
    """ImageNet classifier using the paper-style AFT-conv block."""

    def __init__(
            self,
            image_size,
            patch_size,
            in_channels,
            num_classes,
            d_model,
            hidden_dim,
            n_layers,
            kernel_size,
            n_heads,
            dropout,
            use_position_embedding=False,
    ):
        super().__init__()

        self.patch_embed = PaperPatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=d_model,
        )
        self.use_position_embedding = use_position_embedding

        if use_position_embedding:
            self.position_embedding = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches, d_model)
            )
        else:
            self.position_embedding = None

        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            PaperVisionBlock(
                d_model=d_model,
                hidden_dim=hidden_dim,
                image_size=image_size,
                patch_size=patch_size,
                kernel_size=kernel_size,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, N, D]
        x = self.patch_embed(x)

        if self.position_embedding is not None:
            x = x + self.position_embedding

        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        logits = self.head(x)
        return logits
