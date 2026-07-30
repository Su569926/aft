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
        # x: [B, T, D]
        # B = batch size，T = 当前序列长度，D = d_model 特征维度。
        B, T, D = x.shape
        # q/k/v 都是从 x 线性投影出来的，形状不变：
        # q: [B, T, D]，每个目标位置 t 的门控向量。
        # k: [B, T, D]，每个来源位置 s 的权重向量。
        # v: [B, T, D]，每个来源位置 s 携带的内容向量。
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        if self.causal:
            # causal AFT-local 用在自回归任务里，位置 t 只能看 s <= t 的历史。
            # 旧写法会构造 scores: [B, T, T, D]。CIFAR10 展平成 byte 序列后 T=3071，
            # 这个四维张量太大，所以这里改成不显式展开 [T, T, D] 的等价写法。
            # 这里转成 float32 是为了让 exp 在 AMP 混合精度下更稳定。
            # k_float/v_float: [B, T, D]
            k_float = k.float()
            v_float = v.float()

            # exp_k / kv: [B, T, D]
            # 先计算没有位置偏置时的全局历史聚合，也就是每个 t 看所有 s<=t。
            exp_k = torch.exp(k_float)
            kv = exp_k * v_float
            # cumsum(dim=1) 沿序列维做前缀和：
            # global_numerator[:, t, :] = sum_{s=0..t} exp(k_s) * v_s，形状 [B, T, D]。
            # global_denominator[:, t, :] = sum_{s=0..t} exp(k_s)，形状 [B, T, D]。
            global_numerator = kv.cumsum(dim=1)
            global_denominator = exp_k.cumsum(dim=1)

            # local_* 只保存局部窗口带来的“修正量”，形状仍然是 [B, T, D]。
            local_numerator = torch.zeros_like(global_numerator)
            local_denominator = torch.zeros_like(global_denominator)

            # offset = t - s。causal 情况下只需要 offset>=0 的历史位置。
            max_offset = min(self.local_window_size, T - 1)
            for offset in range(max_offset + 1):
                # 当前这条对角线的长度。
                # 例：T=6, offset=2 时，合法配对是 (t,s)=(2,0),(3,1),(4,2),(5,3)，长度是 4。
                source_length = T - offset

                if self.use_low_rank_bias:
                    # 论文公式 7: w_{t,s} = u_t^T v'_s。
                    # target_bias: [source_length, R]，对应 t=offset..T-1。
                    # source_bias: [source_length, R]，对应 s=0..T-offset-1。
                    target_bias = self.position_u[offset:T].float()
                    source_bias = self.position_v[:source_length].float()
                    # 逐元素乘后仍是 [source_length, R]，sum(dim=-1) 后得到当前对角线上的 bias:
                    # bias[i] = w_{t=i+offset, s=i}，形状 [source_length]。
                    bias = (target_bias * source_bias).sum(dim=-1)
                else:
                    # 从完整位置参数表里取 w_{t,t-offset} 这一条对角线。
                    target_positions = torch.arange(offset, T, device=x.device)
                    source_positions = target_positions - offset
                    # target_positions/source_positions: [source_length]
                    # bias: [source_length]，对应这一条对角线上的 w_{t,s}。
                    bias = self.position_bias[target_positions, source_positions].float()

                # 全局项已经有 exp(k_s)。局部窗口内应该从 exp(k_s) 变成 exp(k_s + w_{t,s})，
                # 所以只需要额外加 exp(k_s) * (exp(w_{t,s}) - 1)。
                # correction: [1, source_length, 1]，可以广播到 [B, source_length, D]。
                correction = torch.exp(bias).view(1, source_length, 1) - 1.0
                # 左边 local_numerator[:, offset:, :]: [B, source_length, D]，对应目标位置 t=offset..T-1。
                # 右边 kv[:, :source_length, :]: [B, source_length, D]，对应来源位置 s=0..T-offset-1。
                # 因为这条对角线满足 t=s+offset，所以两边第二维一一对齐。
                local_numerator[:, offset:, :] = (
                    local_numerator[:, offset:, :]
                    + correction * kv[:, :source_length, :]
                )
                # local_denominator 的形状变化同上，只是这里累加的是权重 exp(k_s)，不乘 v_s。
                local_denominator[:, offset:, :] = (
                    local_denominator[:, offset:, :]
                    + correction * exp_k[:, :source_length, :]
                )

            # numerator/denominator: [B, T, D]
            # 这两行把“全局历史基础项”和“局部位置偏置修正项”加起来。
            numerator = global_numerator + local_numerator
            denominator = global_denominator + local_denominator
            # context: [B, T, D]，对应每个目标位置 t 聚合后的上下文向量。
            context = numerator / denominator.clamp_min(1e-6)

            # sigmoid(q): [B, T, D]，对 context 做逐元素门控。
            # out_proj 输出仍是 [B, T, D]，用于重新混合特征维。
            y = torch.sigmoid(q.float()) * context
            y = y.to(q.dtype)
            y = self.out_proj(y)
            y = self.dropout(y)
            return y

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
            # AMP = Automatic Mixed Precision，自动混合精度；autocast 下 bias 可能是 float16。
            # float16 不能表示 -1e9，所以这里用 -1e4：它足够让 exp 权重接近 0，同时不会溢出。
            bias = bias.masked_fill(~causal_mask, -1e4)

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
