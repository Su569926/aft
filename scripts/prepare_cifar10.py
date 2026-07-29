from pathlib import Path

import torch
from torchvision.datasets import CIFAR10

data_dir = Path("data/cifar10")
raw_dir = data_dir / "raw"
output_dir = data_dir / "processed"

def image_to_sequence(image):
    # torchvision 读出的 CIFAR10 image 是 PIL 图片，原始排列可以理解为 [H, W, C]。
    # 我们做自回归建模时，把一张图片当成长度 3072 的 byte 序列。
    tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    tensor = tensor.view(32, 32, 3) #[H, W, C]
    # 改成 [C, H, W] 后再展平，和常规深度学习图像通道顺序保持一致。
    tensor = tensor.permute(2, 0, 1) #[C, H, W]
    sequence = tensor.reshape(-1).long() #展平成 [3072],.long() 是为了后面送进 embedding 和 loss
    return sequence

def convert_dataset(dataset):
    # 把整个 CIFAR10 数据集从图片集合转换成序列张量集合。
    sequences = []

    for image, _label in dataset:
        sequences.append(image_to_sequence(image))

    return torch.stack(sequences) #train: [50000, 3072] test:  [10000, 3072]

output_dir.mkdir(parents=True, exist_ok=True)

# CIFAR10 会下载/读取原始 Python 版数据；如果本地已有完整压缩包，torchvision 会直接校验使用。
train_dataset = CIFAR10(root=raw_dir, train=True, download=True)
test_dataset = CIFAR10(root=raw_dir, train=False, download=True)

train_sequences = convert_dataset(train_dataset)
test_sequences = convert_dataset(test_dataset)

# 保存成 .pt 后，训练脚本可以直接 torch.load，不需要每次重新读取图片并转换。
torch.save(train_sequences, output_dir / "train.pt")
torch.save(test_sequences, output_dir / "test.pt")

print("train shape:", train_sequences.shape)
print("test shape:", test_sequences.shape)
print("vocab size:", 256)
