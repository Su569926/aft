from pathlib import Path
import torch

# 从 Matt Mahoney 基准数据下载得到的原始 enwik8 文件。
data_path = Path("data/enwik8/enwik8")
output_dir = Path("data/enwik8")

# 使用简单的顺序切分：前 90% 作为训练集，后 10% 作为验证集。
train_ratio = 0.9

# read_bytes() 得到的每个值都在 0..255，因此每个 byte 可以直接作为一个 token id。
raw = data_path.read_bytes()
tokens = torch.tensor(list(raw), dtype=torch.long)

# 保持原始顺序不打乱，因为语言建模依赖序列顺序。
n_train = int(len(tokens) * train_ratio)
train_tokens = tokens[:n_train]
val_tokens = tokens[n_train:]

output_dir.mkdir(parents=True, exist_ok=True)

# 保存成张量文件，训练时可以快速加载。
torch.save(train_tokens, output_dir / "train.pt")
torch.save(val_tokens, output_dir / "val.pt")

print("total tokens:", len(tokens))
print("train tokens:", len(train_tokens))
print("val tokens:", len(val_tokens))
print("vocab size:", 256)
