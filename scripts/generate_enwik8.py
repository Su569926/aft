from pathlib import Path

import torch

from aft.model import AFTLanguageModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = Path("outputs/aft_enwik8.pt")

checkpoint = torch.load(checkpoint_path, map_location=device)
config = checkpoint["config"]

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
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

prompt = "The "
num_new_tokens = 300
temperature = 1.0

tokens = list(prompt.encode("utf-8"))

for _ in range(num_new_tokens):
    context = tokens[-config["seq_len"]:]
    input_ids = torch.tensor([context], dtype=torch.long, device=device) #[1, 当前上下文长度]

    with torch.no_grad():
        logits = model(input_ids) #[B, T, vocab_size]，B是1，T是当前上下文长度

    next_logits = logits[0, -1] / temperature #temperature负责调节随机性，小于1，概率更集中更保守；大于1，概率更平更随机
    probs = torch.softmax(next_logits, dim = -1) #把原始分数变成概率
    next_token = torch.multinomial(probs, num_samples=1).item() #按概率抽样

    tokens.append(next_token)

text = bytes(tokens).decode("utf-8", errors="replace")
print(text)