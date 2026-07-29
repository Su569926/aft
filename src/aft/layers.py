"""Low-level layers such as AFT variants and feed-forward networks."""

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """AFT 混合层后面的逐位置前馈网络。

    输入和输出形状都是 [B, T, D]。nn.Linear 只作用在最后一维，
    所以每个 token 位置会被独立处理，不会混合不同位置。
    """

    def __init__(self, d_model, hidden_dim, dropout):
        super().__init__()

        # 先把特征维 D 扩大到 hidden_dim，经过非线性后再投影回 D。
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)

        return x

class AFTSimple(nn.Module):
    """AFT-simple：所有位置共享同一个全局 key/value 聚合结果。"""

    def __init__(self, d_model, dropout):
        super().__init__()

        # q 用来门控每个目标位置；k 决定来源位置权重；v 携带被聚合的内容。
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, D]
        # AFT = Attention Free Transformer，中文可以理解为“无注意力 Transformer”。
        # 这里不显式计算 attention matrix，而是用 k 的 softmax 做全局加权平均。
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # 沿序列维 T 做归一化；每个 batch、每个特征维都会得到一组位置权重。
        weights = torch.softmax(k, dim=1)

        # weights 和 v 逐元素相乘，然后沿所有来源位置求和。
        weighted = weights * v
        weighted = weighted.sum(dim=1, keepdim=True)

        # [B, 1, D] 会广播到所有位置，再被每个位置自己的 q 门控。
        y_t = torch.sigmoid(q) * weighted
        y = self.out_proj(y_t) #重新混合特征维度，让独立计算出来的特征维度再次组合；增加模型表达能力，适合下一层使用的表示；保持transformer习惯。
        y = self.dropout(y)

        return y

class AFTFull(nn.Module):
    """AFT-full：每一对目标/来源位置都有一个可学习的位置偏置。"""

    def __init__(self, d_model, max_seq_len, dropout):
        super().__init__()

        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # position_bias[t, s] 表示“目标位置 t 看来源位置 s”时额外加上的位置偏置。
        # 这是 AFT-full 对论文位置权重 w_{t,s} 的直接参数化。
        self.position_bias = nn.Parameter(torch.zeros(max_seq_len, max_seq_len)) #位置偏置矩阵的形状是[max_seq_len, max_seq_len]

    def forward(self, x):
        # B 是 batch 大小，T 是当前序列长度，D 是特征维度。
        B, T, D = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # 当前输入只包含 0..T-1 这些位置，所以只取左上角 T x T 的偏置矩阵。
        bias = self.position_bias[:T, :T]

        # 调整形状以便广播，给所有 [目标位置 t, 来源位置 s] 同时计算分数。
        k = k.unsqueeze(1) # [B, 1, T, D]
        v = v.unsqueeze(1) # [B, 1, T, D]
        bias = bias.unsqueeze(0).unsqueeze(-1) # [1, T, T, 1]

        scores = k + bias  # [B, T, T, D]，第一个T来源于目标位置t，第二个T来源于来源位置s
        # 数值稳定处理：先减去来源维度上的最大值，再 exp，避免 exp 输入太大溢出。
        scores = scores - scores.amax(dim=2, keepdim=True)
        scores = torch.exp(scores)

        # 沿来源位置 s 求和，为每个目标位置 t 留下一个上下文向量。
        numerator = (scores * v).sum(dim=2)
        denominator = scores.sum(dim=2)

        context = numerator / denominator #[B, T, D]
        y = torch.sigmoid(q) * context
        y = self.out_proj(y)
        y = self.dropout(y)
        return y

class AFTLocal(nn.Module):
    """AFT-local：在 AFT-full 的基础上屏蔽局部窗口外的位置。"""

    def __init__(
            self,
            d_model,
            max_seq_len,
            local_window_size,
            dropout,
            causal=False,
            use_low_rank_bias=False,
            bias_rank=64,
    ):
        super().__init__()

        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # AFT-local 仍然保留 [max_seq_len, max_seq_len] 的位置偏置表；
        # forward 时只让局部窗口内的位置使用额外偏置。
        self.use_low_rank_bias = use_low_rank_bias
        self.bias_rank = bias_rank

        # 公式 7：w_{t,s} = u_t^T v_s。
        # position_u: [max_seq_len, R]，给目标位置 t 用。
        # position_v: [max_seq_len, R]，给来源位置 s 用。
        # 不能全 0 初始化，否则 u 和 v 的梯度会互相卡住，所以用小随机数。
        if use_low_rank_bias:
            self.position_u = nn.Parameter(torch.empty(max_seq_len, bias_rank))
            self.position_v = nn.Parameter(torch.empty(max_seq_len, bias_rank))
            nn.init.normal_(self.position_u, mean=0.0, std=0.02)
            nn.init.normal_(self.position_v, mean=0.0, std=0.02)
        else:
            self.position_bias = nn.Parameter(torch.zeros(max_seq_len, max_seq_len))
        self.local_window_size = local_window_size
        self.causal = causal

    def forward(self, x):
        B, T, D = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        if self.use_low_rank_bias:
            # [T, R] @ [R, T] -> [T, T]
            # bias[t, s] 就是 u_t 和 v_s 的点积，对应论文里的 u_t^T v_s。
            bias = self.position_u[:T] @ self.position_v[:T].transpose(0, 1)
        else:
            # 先取完整的 [T, T] 偏置矩阵，再只保留附近窗口内的来源位置。
            bias = self.position_bias[:T, :T]
        positions = torch.arange(T, device=x.device)
        target_positions = positions.unsqueeze(1)
        source_positions = positions.unsqueeze(0)
        distance = torch.abs(target_positions - source_positions) #[T, T]
        local_mask = distance <= self.local_window_size
        # 窗口外位置不使用额外局部偏置，但仍通过 k/v 保持全局连接。
        bias = bias.masked_fill(~local_mask, 0.0)

        if self.causal:
            # causal mask 用在自回归任务中，保证位置 t 不能看未来位置 s > t。
            causal_mask = source_positions <= target_positions
            bias = bias.masked_fill(~causal_mask, -1e9)

        k = k.unsqueeze(1)  # [B, 1, T, D]
        v = v.unsqueeze(1)  # [B, 1, T, D]
        bias = bias.unsqueeze(0).unsqueeze(-1)  # [1, T, T, 1]

        scores = k + bias # [B, T, T, D]，第一个T来源于目标位置t，第二个T来源于来源位置s
        # 和 AFT-full 一样，exp 前先减最大值做数值稳定。
        scores = scores - scores.amax(dim = 2, keepdim=True)
        scores = torch.exp(scores)

        numerator = (scores * v).sum(dim=2)
        denominator = scores.sum(dim=2)

        context = numerator / denominator  # [B, T, D]
        y = torch.sigmoid(q) * context
        y = self.out_proj(y)
        y = self.dropout(y)
        return y
