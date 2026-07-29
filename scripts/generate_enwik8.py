from pathlib import Path
import argparse

import torch

from aft.model import AFTLanguageModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/aft_enwik8.pt")
    parser.add_argument("--prompt", type=str, default="The ")
    parser.add_argument("--num-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()

args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = Path(args.checkpoint)

checkpoint = torch.load(checkpoint_path, map_location=device)
config = checkpoint["config"]

# 生成时必须用 checkpoint 里保存的 config 重新创建同结构模型。
model = AFTLanguageModel(
    vocab_size=config["vocab_size"],
    d_model=config["d_model"],
    hidden_dim=config["hidden_dim"],
    n_layers=config["n_layers"],
    max_seq_len=config["seq_len"],
    dropout=config["dropout"],
    aft_type=config["aft_type"],
    local_window_size=config["local_window_size"],
    kernel_size=config.get("kernel_size"),
    causal=config.get("causal", False),
    use_checkpoint=False,
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

prompt = args.prompt
num_new_tokens = args.num_new_tokens
temperature = args.temperature
top_k = args.top_k

if top_k <= 0:
    top_k = None

tokens = list(prompt.encode("utf-8"))
# enwik8 是 byte-level language modeling；
# 每个 token 是 0..255 的 byte，所以 prompt 要先 encode 成字节列表。

print("device:", device)
print("checkpoint:", checkpoint_path)
print("prompt:", repr(prompt))
print("num_new_tokens:", num_new_tokens)
print("temperature:", temperature)
print("top_k:", top_k)

for _ in range(num_new_tokens):
    context = tokens[-config["seq_len"]:]
    input_ids = torch.tensor([context], dtype=torch.long, device=device) #[1, 当前上下文长度]

    with torch.no_grad():
        logits = model(input_ids) #[B, T, vocab_size]，B是1，T是当前上下文长度

    next_logits = logits[0, -1] / temperature #temperature负责调节随机性，小于1，概率更集中更保守；大于1，概率更平更随机

    if top_k is not None:
        # top-k sampling：只允许概率最高的 k 个 token 参与抽样，减少低概率噪声。
        values, indices = torch.topk(next_logits, top_k)
        filtered_logits = torch.full_like(next_logits, -float("inf"))
        filtered_logits[indices] = values
        next_logits = filtered_logits

    probs = torch.softmax(next_logits, dim = -1) #把原始分数变成概率
    next_token = torch.multinomial(probs, num_samples=1).item() #按概率抽样

    tokens.append(next_token)

# 生成结果仍然是一串 byte token，最后再解码成人类可读文本。
text = bytes(tokens).decode("utf-8", errors="replace")
print(text)
