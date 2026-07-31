# CIFAR10 实现版本与训练记录

本文档记录 CIFAR10 自回归实验中 AFT-local 实现方式、优化器版本、训练命令和云端实测现象。目的不是复述最终代码，而是保留“为什么这么改”的过程，避免后续忘记哪些配置已经试过。

## 1. CIFAR10 自回归任务

CIFAR10 图片形状是：

```text
[3, 32, 32]
```

我们把图片展平成 byte 序列：

```text
3 * 32 * 32 = 3072
```

训练时使用 next-byte prediction：

```text
sequences: [B, 3072]
x = sequences[:, :seq_len]
y = sequences[:, 1:seq_len + 1]
```

完整图片建模时：

```text
seq_len = 3071
x: [B, 3071]
y: [B, 3071]
```

指标使用 BPD：

```text
BPD = CrossEntropyLoss / ln(2)
```

## 2. AFT-local 版本演进

### 2.1 dense 直接计算版

最早版本直接按照公式构造完整位置关系：

```python
k = k.unsqueeze(1)                  # [B, 1, T, D]
v = v.unsqueeze(1)                  # [B, 1, T, D]
bias = bias.unsqueeze(0).unsqueeze(-1)  # [1, T, T, 1]

scores = k + bias                   # [B, T, T, D]
scores = scores - scores.amax(dim=2, keepdim=True)
scores = torch.exp(scores)

numerator = (scores * v).sum(dim=2) # [B, T, D]
denominator = scores.sum(dim=2)     # [B, T, D]
```

优点：

```text
代码最接近公式，GPU 并行度高，短序列可能更快。
```

缺点：

```text
显存开销是 [B, T, T, D]。
当 T=3071, D=256 时，单个 scores 就约 24 亿个元素。
2 * RTX 5090 32GB 上运行完整 CIFAR10 配置时，在 backward 阶段 OOM。
```

实测现象：

```text
命令: seq_len=3071, n_layers=24, local_window_size=256, grad_accum_steps=64, amp
结果: CUDA out of memory，backward 时尝试额外分配约 9GB。
结论: dense 版不适合完整 CIFAR10 长序列训练。
```

### 2.2 offset 省显存版

为避免构造 `[B, T, T, D]`，使用公式分解：

```text
exp(k_s + w_{t,s})
= exp(k_s) * exp(w_{t,s})
= exp(k_s) + exp(k_s) * (exp(w_{t,s}) - 1)
```

先用前缀和计算全局历史基础项：

```python
exp_k = torch.exp(k.float())        # [B, T, D]
kv = exp_k * v.float()              # [B, T, D]

global_numerator = kv.cumsum(dim=1)
global_denominator = exp_k.cumsum(dim=1)
```

再沿着 offset 对角线补局部窗口修正：

```text
offset = t - s
offset = 0: 自己看自己
offset = 1: 看前一个位置
offset = 2: 看前两个位置
```

优点：

```text
避免完整 [B, T, T, D]，显存明显下降。
数学上和 dense causal local 公式等价。
```

缺点：

```text
Python for offset 循环太多。
local_window_size=256 时，每层要循环 257 次。
24 层时每个 micro batch 约 257 * 24 = 6168 次 offset 循环。
```

结论：

```text
省显存成功，但训练速度太慢，不适合正式长训。
```

### 2.3 chunked gather 版本

当前使用的是 chunked gather 版本。它保留：

```text
全局历史基础项: cumsum
局部位置偏置修正: exp(w_{t,s}) - 1
```

但不再按 offset 一条对角线一条对角线算，而是按目标位置分块：

```python
chunk_size = 256
offsets = torch.arange(max_offset + 1, device=x.device)  # [K]

for start in range(0, T, chunk_size):
    end = min(start + chunk_size, T)
    target_positions = torch.arange(start, end, device=x.device)  # [C]
    source_positions = target_positions.unsqueeze(1) - offsets.unsqueeze(0)  # [C, K]
```

核心张量形状：

```text
C = 当前 chunk 的目标位置数
K = local_window_size + 1

source_positions: [C, K]
kv_source:        [B, C, K, D]
expk_source:      [B, C, K, D]
bias:             [C, K]
correction:       [1, C, K, 1]
```

局部修正：

```python
local_numerator[:, start:end, :] = (correction * kv_source).sum(dim=2)
local_denominator[:, start:end, :] = (correction * expk_source).sum(dim=2)
```

优点：

```text
显存比 dense 低。
Python 循环次数从 local_window_size + 1 降到 ceil(T / chunk_size)。
例如 T=3071, chunk_size=256 时，每层约 13 次循环。
```

实测：

```text
2 * RTX 5090 上，seq_len=3071, n_layers=24, local_window_size=256, grad_accum_steps=64 时：
GPU 利用率可达 100%，显存约 3.5GB/GPU。
但 12 小时仍未到 step 500，原因是 grad_accum_steps=64 + n_layers=24 + checkpoint 重算导致单个 optimizer step 太重。
```

结论：

```text
chunked gather 是当前长序列必须使用的实现，但仍需要配合较轻训练配置。
```

## 3. AdamW 版本

当前 CIFAR10 训练脚本使用 AdamW 参数分组：

```text
普通权重矩阵: 使用 weight_decay
bias / LayerNorm / position 参数: weight_decay = 0
```

原因：

```text
bias、归一化参数和位置参数通常不做 weight decay，避免破坏尺度和位置建模。
```

代码结构：

```python
param_groups = build_weight_decay_param_groups(model, weight_decay)
optimizer = torch.optim.AdamW(param_groups, lr=learning_rate)
```

曾短暂回退到所有参数统一 AdamW：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)
```

但该回退只是为了排查速度/显存问题。最终恢复参数分组，因为 OOM 与 AdamW 参数分组无关，主要来自 AFT-local 的 dense 中间张量或训练配置过重。

## 4. 已尝试命令与结果

### 4.1 完整 24 层配置

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=2 scripts/train_cifar10_ar_ddp.py \
    --seq-len 3071 \
    --batch-size 1 \
    --grad-accum-steps 64 \
    --d-model 256 \
    --hidden-dim 1024 \
    --n-layers 24 \
    --dropout 0.1 \
    --num-steps 50000 \
    --eval-interval 500 \
    --save-interval 1000 \
    --learning-rate 3e-4 \
    --min-learning-rate 1e-5 \
    --warmup-steps 1000 \
    --weight-decay 0.01 \
    --aft-type local \
    --local-window-size 256 \
    --use-low-rank-bias \
    --bias-rank 64 \
    --amp \
    --use-checkpoint \
    --output-path outputs/aft_cifar10_ar_local256_l24_d256_lowrank64_chunked_cosine.pt \
    --log-path outputs/aft_cifar10_ar_local256_l24_d256_lowrank64_chunked_cosine_log.csv
```

实测现象：

```text
GPU: 2 * RTX 5090 32GB
显存: chunked 版约 3.5GB/GPU
GPU 利用率: 100%
运行时间: 约 12 小时仍未到 step 500
结论: 不是卡死，而是单个 optimizer step 计算量太大。
```

原因：

```text
step 500 = 500 * grad_accum_steps = 500 * 64 = 32000 个 micro batch
每个 micro batch 还要经过 24 层、T=3071、local_window_size=256。
use_checkpoint=True 会在 backward 时重算 forward，进一步拖慢。
```

### 4.2 dense 直接计算回退尝试

同样的完整命令在 dense 版 AFT-local 下运行：

```text
结果: CUDA out of memory
位置: backward 阶段
现象: Tried to allocate 9.00 GiB
结论: dense 版不能用于完整 CIFAR10 长序列配置。
```

### 4.3 当前可跑配置

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=2 scripts/train_cifar10_ar_ddp.py \
    --seq-len 3071 \
    --batch-size 1 \
    --grad-accum-steps 8 \
    --d-model 256 \
    --hidden-dim 1024 \
    --n-layers 12 \
    --dropout 0.1 \
    --num-steps 5000 \
    --eval-interval 100 \
    --save-interval 500 \
    --learning-rate 3e-4 \
    --min-learning-rate 1e-5 \
    --warmup-steps 200 \
    --weight-decay 0.01 \
    --aft-type local \
    --local-window-size 128 \
    --use-low-rank-bias \
    --bias-rank 64 \
    --amp \
    --output-path outputs/aft_cifar10_ar_t3071_w128_l12_d256_fast.pt \
    --log-path outputs/aft_cifar10_ar_t3071_w128_l12_d256_fast_log.csv
```

实测状态：

```text
GPU: 2 * RTX 5090 32GB
显存占用: 约 12.4GB/GPU
GPU 利用率: 约 83% 和 91%
现象: 5 分钟以内出现 step 100 日志
```

截至 step 2900 的日志摘要：

```text
step 0    lr=1.50e-6    val_loss=5.7114    val_bpd=8.2399
step 100  lr=1.515e-4   val_loss=4.6763    val_bpd=6.7464
step 200  lr=3.00e-4    val_loss=4.2923    val_bpd=6.1925
step 500  lr=2.972e-4   val_loss=4.1341    val_bpd=5.9642    saved checkpoint
step 1000 lr=2.806e-4   val_loss=3.7175    val_bpd=5.3632    saved checkpoint
step 1500 lr=2.506e-4   val_loss=3.8444    val_bpd=5.5464    saved checkpoint
step 2000 lr=2.105e-4   val_loss=3.7258    val_bpd=5.3751    saved checkpoint
step 2300 lr=1.833e-4   val_loss=3.6051    val_bpd=5.2011
step 2500 lr=1.645e-4   val_loss=3.7249    val_bpd=5.3740    saved checkpoint
step 2800 lr=1.361e-4   val_loss=3.5818    val_bpd=5.1674
step 2900 lr=1.267e-4   val_loss=3.6723    val_bpd=5.2980
```

观察：

```text
前 1000 step 下降很快。
step 2000 之后下降变慢，并在 5.17 到 5.37 BPD 附近震荡。
当前配置能验证模型、数据、DDP、checkpoint、BPD 指标都正常，但离论文级 CIFAR10 结果仍有明显差距。
```

该配置的关键变化：

```text
n_layers: 24 -> 12
local_window_size: 256 -> 128
grad_accum_steps: 64 -> 8
use_checkpoint: True -> False
eval_interval: 500 -> 100
```

为什么去掉 `--use-checkpoint`：

```text
activation checkpointing 会在 backward 时重跑部分 forward。
它适合显存不够时用时间换显存。
当前配置显存还有余量，所以去掉 checkpoint 可以换取速度。
```

### 4.4 24 层、window256、accum8、无 checkpoint 尝试

为了确认瓶颈是否主要来自 `grad_accum_steps=64` 和 activation checkpointing，尝试保留更接近论文的模型大小：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=2 scripts/train_cifar10_ar_ddp.py \
    --seq-len 3071 \
    --batch-size 1 \
    --grad-accum-steps 8 \
    --d-model 256 \
    --hidden-dim 1024 \
    --n-layers 24 \
    --dropout 0.1 \
    --num-steps 5000 \
    --eval-interval 500 \
    --save-interval 500 \
    --learning-rate 3e-4 \
    --min-learning-rate 1e-5 \
    --warmup-steps 200 \
    --weight-decay 0.01 \
    --aft-type local \
    --local-window-size 256 \
    --use-low-rank-bias \
    --bias-rank 64 \
    --amp \
    --output-path outputs/aft_cifar10_ar_t3071_w256_l24_d256_accum8_no_ckpt.pt \
    --log-path outputs/aft_cifar10_ar_t3071_w256_l24_d256_accum8_no_ckpt_log.csv
```

结果：

```text
use_checkpoint: False
n_layers: 24
local_window_size: 256
grad_accum_steps: 8
结果: forward 阶段 OOM
报错位置: src/aft/layers.py 中 expk_source = exp_k[:, source_positions, :]
现象: 每张 32GB 5090 几乎占满，尝试再分配 66MB 时失败。
```

结论：

```text
24 层 + window256 在无 activation checkpointing 时显存不足。
如果要保留 24 层和 window256，必须开启 --use-checkpoint，但速度会显著下降。
当前 2 * RTX 5090 环境下更现实的选择仍是 12 层 + window128 + accum8。
```

## 5. 当前结论

当前 CIFAR10 复现实验应使用：

```text
chunked AFT-local
AdamW 参数分组
seq_len=3071
n_layers=12
local_window_size=128
grad_accum_steps=8
不启用 use_checkpoint
```

完整论文近似配置仍保留为目标，但在当前纯 PyTorch 实现下训练成本过高。后续若要逼近论文表格配置，需要继续优化 AFT-local 实现，例如更强向量化、Triton/CUDA kernel，或者减少评估频率并使用更高算力。

## 6. 后续逼近论文配置的尝试顺序

当前已经确认 `12 层 + window128 + accum8 + no checkpoint` 可以跑。后续如果想逐步逼近论文配置，不要一次跳到 `24 层 + window256 + accum64`，而是按下面顺序试。

### 6.1 第一档：12 层，window256，accum8，不开 checkpoint

目的：

```text
只把 local_window_size 从 128 改回论文 CIFAR10 的 256。
保持 n_layers=12 和 grad_accum_steps=8，先确认 window256 是否能在 32GB 显存下跑。
```

命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=2 scripts/train_cifar10_ar_ddp.py \
    --seq-len 3071 \
    --batch-size 1 \
    --grad-accum-steps 8 \
    --d-model 256 \
    --hidden-dim 1024 \
    --n-layers 12 \
    --dropout 0.1 \
    --num-steps 5000 \
    --eval-interval 100 \
    --save-interval 500 \
    --learning-rate 3e-4 \
    --min-learning-rate 1e-5 \
    --warmup-steps 200 \
    --weight-decay 0.01 \
    --aft-type local \
    --local-window-size 256 \
    --use-low-rank-bias \
    --bias-rank 64 \
    --amp \
    --output-path outputs/aft_cifar10_ar_t3071_w256_l12_d256_accum8_no_ckpt.pt \
    --log-path outputs/aft_cifar10_ar_t3071_w256_l12_d256_accum8_no_ckpt_log.csv
```

判断：

```text
如果能跑，并且速度可接受，优先跑完这版。
如果 OOM，进入 6.2。
```

### 6.2 第二档：12 层，window256，accum8，开启 checkpoint

目的：

```text
如果第一档 OOM，就开启 activation checkpointing。
checkpoint 会降低显存，但 backward 会重算 forward，速度会变慢。
```

命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=2 scripts/train_cifar10_ar_ddp.py \
    --seq-len 3071 \
    --batch-size 1 \
    --grad-accum-steps 8 \
    --d-model 256 \
    --hidden-dim 1024 \
    --n-layers 12 \
    --dropout 0.1 \
    --num-steps 5000 \
    --eval-interval 100 \
    --save-interval 500 \
    --learning-rate 3e-4 \
    --min-learning-rate 1e-5 \
    --warmup-steps 200 \
    --weight-decay 0.01 \
    --aft-type local \
    --local-window-size 256 \
    --use-low-rank-bias \
    --bias-rank 64 \
    --amp \
    --use-checkpoint \
    --output-path outputs/aft_cifar10_ar_t3071_w256_l12_d256_accum8_ckpt.pt \
    --log-path outputs/aft_cifar10_ar_t3071_w256_l12_d256_accum8_ckpt_log.csv
```

判断：

```text
如果能跑但太慢，退回 12 层 window128 配置。
如果能跑且速度可接受，再尝试 6.3。
```

### 6.3 第三档：12 层，window256，增大 grad_accum_steps

目的：

```text
在模型规模不变的前提下，把 effective_batch_size 从 16 提高到 32 或 64。
grad_accum_steps 不明显增加单次 forward 显存，但会让一个 optimizer step 需要更多 micro batch，因此日志出现更慢。
```

accum16 命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=2 scripts/train_cifar10_ar_ddp.py \
    --seq-len 3071 \
    --batch-size 1 \
    --grad-accum-steps 16 \
    --d-model 256 \
    --hidden-dim 1024 \
    --n-layers 12 \
    --dropout 0.1 \
    --num-steps 5000 \
    --eval-interval 100 \
    --save-interval 500 \
    --learning-rate 3e-4 \
    --min-learning-rate 1e-5 \
    --warmup-steps 200 \
    --weight-decay 0.01 \
    --aft-type local \
    --local-window-size 256 \
    --use-low-rank-bias \
    --bias-rank 64 \
    --amp \
    --output-path outputs/aft_cifar10_ar_t3071_w256_l12_d256_accum16_no_ckpt.pt \
    --log-path outputs/aft_cifar10_ar_t3071_w256_l12_d256_accum16_no_ckpt_log.csv
```

accum32 命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=2 scripts/train_cifar10_ar_ddp.py \
    --seq-len 3071 \
    --batch-size 1 \
    --grad-accum-steps 32 \
    --d-model 256 \
    --hidden-dim 1024 \
    --n-layers 12 \
    --dropout 0.1 \
    --num-steps 5000 \
    --eval-interval 100 \
    --save-interval 500 \
    --learning-rate 3e-4 \
    --min-learning-rate 1e-5 \
    --warmup-steps 200 \
    --weight-decay 0.01 \
    --aft-type local \
    --local-window-size 256 \
    --use-low-rank-bias \
    --bias-rank 64 \
    --amp \
    --output-path outputs/aft_cifar10_ar_t3071_w256_l12_d256_accum32_no_ckpt.pt \
    --log-path outputs/aft_cifar10_ar_t3071_w256_l12_d256_accum32_no_ckpt_log.csv
```

判断：

```text
accum16 的 effective_batch_size = 1 * 2 * 16 = 32。
accum32 的 effective_batch_size = 1 * 2 * 32 = 64。
如果日志等待时间过长，不要继续加到 accum64。
```

### 6.4 第四档：增加层数到 16

目的：

```text
在 12 层 window256 已经确认可跑后，再尝试增加层数。
不要直接跳到 24 层；先试 16 层。
```

16 层命令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=2 scripts/train_cifar10_ar_ddp.py \
    --seq-len 3071 \
    --batch-size 1 \
    --grad-accum-steps 8 \
    --d-model 256 \
    --hidden-dim 1024 \
    --n-layers 16 \
    --dropout 0.1 \
    --num-steps 5000 \
    --eval-interval 100 \
    --save-interval 500 \
    --learning-rate 3e-4 \
    --min-learning-rate 1e-5 \
    --warmup-steps 200 \
    --weight-decay 0.01 \
    --aft-type local \
    --local-window-size 256 \
    --use-low-rank-bias \
    --bias-rank 64 \
    --amp \
    --output-path outputs/aft_cifar10_ar_t3071_w256_l16_d256_accum8_no_ckpt.pt \
    --log-path outputs/aft_cifar10_ar_t3071_w256_l16_d256_accum8_no_ckpt_log.csv
```

如果 16 层 OOM，再开 checkpoint：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=2 scripts/train_cifar10_ar_ddp.py \
    --seq-len 3071 \
    --batch-size 1 \
    --grad-accum-steps 8 \
    --d-model 256 \
    --hidden-dim 1024 \
    --n-layers 16 \
    --dropout 0.1 \
    --num-steps 5000 \
    --eval-interval 100 \
    --save-interval 500 \
    --learning-rate 3e-4 \
    --min-learning-rate 1e-5 \
    --warmup-steps 200 \
    --weight-decay 0.01 \
    --aft-type local \
    --local-window-size 256 \
    --use-low-rank-bias \
    --bias-rank 64 \
    --amp \
    --use-checkpoint \
    --output-path outputs/aft_cifar10_ar_t3071_w256_l16_d256_accum8_ckpt.pt \
    --log-path outputs/aft_cifar10_ar_t3071_w256_l16_d256_accum8_ckpt_log.csv
```

判断：

```text
如果 16 层稳定且速度可接受，再考虑 24 层。
当前已经确认 24 层 + window256 + no checkpoint 会 OOM。
24 层 + window256 + checkpoint 能跑但极慢，不适合作为当前主要实验。
```
