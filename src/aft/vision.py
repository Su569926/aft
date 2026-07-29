import torch
import torch.nn as nn
import torch.nn.functional as F

from aft.layers import FeedForward

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, d_model):
        """把图片切成 patch：[B, 3, 224, 224] -> [B, N, D]，H和W都是224，N = patch数量"""
        super().__init__()

        # image_size 和 patch_size 决定二维 patch 网格大小。
        # 例如 224 / 16 = 14，所以一张图片会变成 14 x 14 = 196 个 patch token。
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
        # 输入 x: [B, C, H, W]，例如 ImageNet 是 [B, 3, 224, 224]。
        # Conv2d 的 kernel_size=stride=patch_size，所以卷积窗口不会重叠；
        # 每个窗口正好对应一个 patch，并被投影成 d_model 维。
        x = self.proj(x) #[B, D, 14, 14]
        # flatten(2) 只把 H 和 W 两个空间维度压平，通道维 D 不动。
        x = x.flatten(2) #[B, D, 196]
        # Transformer/AFT 后续习惯使用 [B, token数量, 特征维度]。
        x = x.transpose(1, 2) #[B, 196, D]
        return x

class AFTConv2D(nn.Module):
    def __init__(self, d_model, image_size, patch_size, kernel_size, dropout):
        super().__init__()

        self.d_model = d_model
        self.image_size = image_size
        self.patch_size = patch_size
        self.kernel_size = kernel_size

        # patch token 会被还原成二维网格：[B, N, D] -> [B, D, grid_size, grid_size]。
        # 例如 image_size=224, patch_size=16 时，grid_size=14，num_patches=196。
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        # q/k/v 的输入输出形状都保持 [B, N, D]。
        # q 负责门控当前位置，k 决定来源位置权重，v 携带被聚合的信息。
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)

        # 这是论文 AFT-conv 里的局部相对位置参数 w。
        # 形状 [D, 1, K, K] 配合 groups=D 做 depthwise convolution：
        # 每个特征通道都有自己的一张 K x K 局部位置权重表，不混合不同通道。
        self.position_bias = nn.Parameter(
            torch.zeros(d_model, 1, kernel_size, kernel_size)
        ) #1是每个输出通道只看1个输入通道，是论文中的depthwise convolution深度可分离

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, D]，N 必须等于 grid_size * grid_size。
        # B 是 batch size，N 是 patch token 数量，D 是 d_model。
        B, N, D = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # AFT 的核心分子/分母形式：
        # numerator   = sum(exp(k) * v)
        # denominator = sum(exp(k))
        # 这里先保留逐位置的 exp(k)*v，后面分别做全局求和和局部卷积修正。
        exp_k = torch.exp(k)
        kv = exp_k * v
        normalizer = exp_k

        # 全局项：沿 N 这个 patch/token 维度求和。
        # 形状 [B, 1, D] 后面会广播到 [B, N, D]，表示每个位置都能看到全局汇总。
        global_numerator = kv.sum(dim=1, keepdim=True)
        global_denominator = normalizer.sum(dim=1, keepdim=True)

        # 卷积需要 [B, C, H, W]，所以把 token 序列还原成二维 patch 网格。
        # [B, N, D] -> [B, D, N] -> [B, D, grid_size, grid_size]。
        kv_2d = kv.transpose(1, 2).reshape(B, D, self.grid_size, self.grid_size)
        normalizer_2d = normalizer.transpose(1, 2).reshape(B, D, self.grid_size, self.grid_size)

        # exp(w)-1 是论文 AFT-conv 的局部修正形式。
        # 当 position_bias 初始化为 0 时，局部项为 0，模型先退化成全局 AFT-simple。
        conv_weight = torch.exp(self.position_bias) - 1.0

        # 局部分子项：在二维 patch 网格上，用 K x K 邻域补充局部位置偏置。
        # groups=D 表示每个通道单独卷积，输入输出仍是 [B, D, grid, grid]。
        local_numerator = F.conv2d(
            kv_2d,
            conv_weight,
            padding=self.kernel_size // 2,
            groups=D,
        ) #只是一个函数，自己不保存参数，需要手动传参数

        # 局部分母项：和分子使用同一套 exp(w)-1，保证仍然是 AFT 的加权平均结构。
        local_denominator = F.conv2d(
            normalizer_2d,
            conv_weight,
            padding=self.kernel_size // 2,
            groups=D,
        )

        # 把卷积结果从 [B, D, grid, grid] 变回 [B, N, D]，继续按 token 序列处理。
        local_numerator = local_numerator.reshape(B, D, N).transpose(1, 2)
        local_denominator = local_denominator.reshape(B, D, N).transpose(1, 2)

        # 最终使用“全局 AFT 项 + 局部卷积修正项”。
        # global_* 的 [B, 1, D] 会广播到每个 patch 位置。
        numerator = local_numerator + global_numerator
        denominator = local_denominator + global_denominator

        # sigmoid(q) 是当前位置门控；out_proj 再混合特征维度；dropout 用于正则化。
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

        # 图像版 block 的主分支：先把图片切成 patch token。
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
        # x: [B, N, D]。
        # Pre-LN + 残差：先归一化，再做 AFTConv2D，最后加回原输入。
        x = x + self.aft(self.norm1(x))
        # FFN 逐 token 处理特征维度，不改变 token 数量，输出仍是 [B, N, D]。
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
            use_position_embedding=False,
    ):
        super().__init__()

        # PatchEmbedding 把图片 [B, C, H, W] 转成 patch 序列 [B, N, D]。
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=d_model,
        )
        self.use_position_embedding = use_position_embedding

        # ImageNet 的 AFT-conv 默认不使用绝对位置编码；
        # 如果做消融实验，可以打开 use_position_embedding。
        if use_position_embedding:
            self.position_embedding = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches, d_model)
            )
        else:
            self.position_embedding = None
        self.dropout = nn.Dropout(dropout)

        # 堆叠 n_layers 个图像 AFT block，所有 block 的输入输出形状都是 [B, N, D]。
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

        # 分类前最后归一化，然后用线性层把图像表示映射成类别 logits。
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        #x: [B, 3, 224, 224]
        x = self.patch_embed(x) #[B, T, D] T其实就是patch的数量，为196

        if self.position_embedding is not None:
            # position_embedding: [1, T, D]，会沿 batch 维广播到 [B, T, D]。
            x = x + self.position_embedding #position_embedding:  [1, T, D]

        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # global average pooling：把所有 patch token 平均成一张图的整体表示。
        # [B, T, D] -> [B, D]。
        x = x.mean(dim = 1) #[2, 64]

        # 分类头输出每个类别的未归一化分数，不需要手动 softmax；
        # CrossEntropyLoss 会在内部处理 softmax/log-softmax。
        logits = self.head(x) #[B, num_classes]

        return logits
