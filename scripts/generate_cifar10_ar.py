from pathlib import Path
import argparse

import torch
from PIL import Image

from aft.model import AFTLanguageModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/aft_cifar10_ar.pt")
    parser.add_argument("--output-image", type=str, default="outputs/cifar10_generated.png")
    parser.add_argument("--num-pixels", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--start-token", type=int, default=0)
    return parser.parse_args()

args = parse_args()
if args.num_pixels != 3072:
    raise ValueError("num_pixels must be 3072 for CIFAR10 images")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# checkpoint 里保存了训练好的模型参数，以及创建模型所需的结构配置。
checkpoint_path = Path(args.checkpoint)
checkpoint = torch.load(checkpoint_path, map_location=device)
config = checkpoint["config"]

# 推理时必须用和训练时相同的模型结构，然后再加载参数。
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
    causal=config.get("causal", True),
    use_checkpoint=False,
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

def sample_next_token(logits, temperature, top_k):
    # temperature 控制采样随机性；小于 1 更保守，大于 1 更随机。
    logits = logits / temperature

    # top-k 只保留分数最高的 k 个候选，避免采到很低概率的噪声像素值。
    if top_k is not None and top_k > 0:
        values, indices = torch.topk(logits, top_k)
        filtered_logits = torch.full_like(logits, -float("inf"))
        filtered_logits[indices] = values
        logits = filtered_logits

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()

# CIFAR10 图像是 3*32*32，因此完整图片需要 3072 个像素通道值。
tokens = [args.start_token]

while len(tokens) < args.num_pixels:
    # 模型最多只能看训练时的 seq_len 个上下文 token。
    context = tokens[-config["seq_len"]:]
    input_ids = torch.tensor([context], dtype=torch.long, device=device) #[1, 上下文长度]

    with torch.no_grad():
        logits = model(input_ids) #[B, T, vocab_size]

    # 只取最后一个位置，因为它对应“下一个像素值”的预测。
    next_logits = logits[0, -1] #[256]
    next_token = sample_next_token(next_logits, args.temperature, args.top_k)
    tokens.append(next_token)

# 自回归生成得到的是一维序列，先还原成 [C, H, W]。
image_tensor = torch.tensor(tokens, dtype=torch.uint8)
image_tensor = image_tensor.view(3, 32, 32)

# PIL 保存 RGB 图片需要 [H, W, C]。
image_tensor = image_tensor.permute(1, 2, 0) #[32, 32, 3]

image = Image.fromarray(image_tensor.cpu().numpy(), mode="RGB")

output_image = Path(args.output_image)
output_image.parent.mkdir(parents=True, exist_ok=True)
image.save(output_image)

print("saved image:", output_image)
