import torch
from aft.model import AFTLanguageModel

vocab_size = 10
seq_len = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型配置必须和 scripts/train_toy.py 里训练 checkpoint 时完全一致。
model = AFTLanguageModel(
    vocab_size=10,
    d_model=32,
    hidden_dim=128,
    n_layers=2,
    max_seq_len=8,
    dropout=0.0,
    aft_type="conv",
    kernel_size=3,
).to(device)

# map_location 让 checkpoint 可以在 CPU 或 GPU 上加载。
state_dict = torch.load("outputs/aft_conv_toy.pt", map_location=device)
model.load_state_dict(state_dict)

# eval() 会关闭 dropout 等只在训练时启用的行为。
model.eval()

# 从一个短前缀开始，每次追加模型预测出的下一个 token。
tokens = [3, 4, 5]
while len(tokens) < seq_len:
    # 模型输入必须保留 batch 维：[1, 当前序列长度]。
    input_ids = torch.tensor([tokens], device=device)
    with torch.no_grad():
        logits = model(input_ids)

    # 使用最后一个位置的 logits，贪心选择分数最高的 token。
    next_logits = logits[0, -1]
    next_token = torch.argmax(next_logits).item()
    tokens.append(next_token)

print(tokens)
