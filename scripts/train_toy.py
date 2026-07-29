"""Toy training entrypoint for checking whether the model can learn."""

import torch
import torch.nn as nn

from aft.model import AFTLanguageModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 极小的合成任务：学习“下一个 token = 当前 token + 1 后对 vocab_size 取余”。
vocab_size = 10
seq_len = 8
batch_size = 16
num_steps = 200

def make_batch(batch_size, seq_len, vocab_size, device):
    # 每一行从随机 token 开始，让模型学习规律，而不是记住一个固定 batch。
    starts = torch.randint(0, vocab_size, (batch_size, 1), device=device)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    input_ids = (starts + positions) % vocab_size
    targets = (input_ids + 1) % vocab_size
    return input_ids, targets

# 推理加载 checkpoint 时，模型超参数必须和这里保持一致。
# 这里用 AFT-local 做 toy sanity check；1D AFTConv 已删除，图像卷积版本在 vision.py 里。
model = AFTLanguageModel(
    vocab_size=vocab_size,
    d_model=32,
    hidden_dim=128,
    n_layers=2,
    max_seq_len=seq_len,
    dropout=0.0,
    aft_type="local",
    local_window_size=2,
    causal=True,
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(num_steps):
    input_ids, targets = make_batch(batch_size, seq_len, vocab_size, device)

    # logits: [B, T, vocab_size]，targets: [B, T]。
    logits = model(input_ids)

    # CrossEntropyLoss 需要 [N, C] 的预测和 [N] 的目标，所以把 B 和 T 展平。
    loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))

    # 标准训练步骤：清空旧梯度，计算新梯度，更新参数。
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print("step:", step, "loss:", loss.item())

# 只保存学到的参数；推理脚本会重新创建同结构模型，再加载这个 state_dict。
torch.save(model.state_dict(), "outputs/aft_local_toy.pt")
