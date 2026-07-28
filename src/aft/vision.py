import torch
import torch.nn as nn
import torch.nn.functional as F

from aft.layers import FeedForward

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, d_model):
        """把图片切成 patch：[B, 3, 224, 224] -> [B, N, D]，H和W都是224，N = patch数量"""
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2 #每个小块相当于一个token
        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        ) #把每个patch压成一个d_model维向量，所以这里d_model自然而然也变成了通道数

    def forward(self, x):
        x = self.proj(x) #[B, D, 14, 14]
        x = x.flatten(2) #[B, D, 196]
        x = x.transpose(1, 2) #[B, 196, D]
        return x

class AFTConv2D(nn.Module):
    def __init__(self, d_model, image_size, patch_size, kernel_size, dropout):
        super().__init__()

        self.d_model = d_model
        self.image_size = image_size
        self.patch_size = patch_size
        self.kernel_size = kernel_size

        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)

        self.position_bias = nn.Parameter(
            torch.zeros(d_model, 1, kernel_size, kernel_size)
        ) #1是每个输出通道只看1个输入通道，是论文中的depthwise convolution深度可分离

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        kv = torch.exp(k) * v
        normalizer = torch.exp(k)

        kv = kv.transpose(1, 2).reshape(B, D, self.grid_size, self.grid_size)
        normalizer = normalizer.transpose(1, 2).reshape(B, D, self.grid_size, self.grid_size)

        conv_weight = torch.exp(self.position_bias)

        numerator = F.conv2d(
            kv,
            conv_weight,
            padding=self.kernel_size // 2,
            groups=D,
        ) #只是一个函数，自己不保存参数，需要手动传参数

        denominator = F.conv2d(
            normalizer,
            conv_weight,
            padding=self.kernel_size // 2,
            groups=D,
        )

        numerator = numerator.reshape(B, D, N).transpose(1, 2)
        denominator = denominator.reshape(B, D, N).transpose(1, 2)

        y = torch.sigmoid(q) * numerator / (denominator + 1e-6)
        y = self.out_proj(y)
        y = self.dropout(y)
        return y

class VisionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            hidden_dim,
            image_size,
            patch_size,
            kernel_size,
            dropout,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.aft = AFTConv2D(
            d_model=d_model,
            image_size=image_size,
            patch_size=patch_size,
            kernel_size=kernel_size,
            dropout=dropout
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(
            d_model=d_model,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, x):
        x = x + self.aft(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class AFTImageClassifier(nn.Module):
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
            dropout,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=d_model,
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, d_model)
        )
        self.dropout = nn.Dropout(dropout)

        self.blocks =nn.ModuleList([
            VisionBlock(
                d_model=d_model,
                hidden_dim=hidden_dim,
                image_size=image_size,
                patch_size=patch_size,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        #x: [B, 3, 224, 224]
        x = self.patch_embed(x) #[B, T, D] T其实就是patch的数量，为196

        x = x + self.position_embedding #position_embedding:  [1, T, D]
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        x = x.mean(dim = 1) #[2, 64]

        logits = self.head(x) #[B, num_classes]

        return logits
