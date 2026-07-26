from pathlib import Path

import torch
from torchvision.datasets import CIFAR10

data_dir = Path("data/cifar10")
raw_dir = data_dir / "raw"
output_dir = data_dir / "processed"

def image_to_sequence(image):
    tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    tensor = tensor.view(32, 32, 3) #[H, W, C]
    tensor = tensor.permute(2, 0, 1) #[C, H, W]
    sequence = tensor.reshape(-1).long() #展平成 [3072],.long() 是为了后面送进 embedding 和 loss
    return sequence

def convert_dataset(dataset):
    sequences = []

    for image, _label in dataset:
        sequences.append(image_to_sequence(image))

    return torch.stack(sequences) #train: [50000, 3072] test:  [10000, 3072]

output_dir.mkdir(parents=True, exist_ok=True)

train_dataset = CIFAR10(root=raw_dir, train=True, download=True)
test_dataset = CIFAR10(root=raw_dir, train=False, download=True)

train_sequences = convert_dataset(train_dataset)
test_sequences = convert_dataset(test_dataset)

torch.save(train_sequences, output_dir / "train.pt")
torch.save(test_sequences, output_dir / "test.pt")

print("train shape:", train_sequences.shape)
print("test shape:", test_sequences.shape)
print("vocab size:", 256)
