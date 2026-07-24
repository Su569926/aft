# 云 GPU 租用与使用入门

本文档记录本项目后续训练阶段使用云 GPU 的基本流程。当前项目早期写代码、跑 shape test、toy 训练可以用本地 CPU；真正训练较大模型或跑论文级实验时，再租云 GPU。

## 1. 先回答核心问题：用 CPU 版还是 GPU 版？

在云 4090 / 5090 上训练时，应该使用 **GPU 版 PyTorch**。

原因：

- CPU 版 PyTorch 只能在 CPU 上算，即使服务器有 4090 / 5090，也不会自动用上显卡。
- GPU 版 PyTorch 内置 CUDA 运行时，才能把张量和模型放到 `cuda` 上训练。
- AFT / Transformer 类模型主要计算在矩阵乘法、线性层、归一化和激活函数上，这些都适合 GPU 加速。

本项目的建议：

- 本地电脑：先装 CPU 版，适合学习 PyTorch、写模块、跑小测试。
- 云服务器：租 GPU 后使用带 CUDA 的 PyTorch 镜像，或安装 GPU 版 PyTorch。

判断是否真的用上 GPU：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
PY
```

如果输出里 `cuda available: True`，才说明 PyTorch 能看到 GPU。

## 2. 推荐先用哪个平台？

对初学者，建议优先用 **AutoDL**。

理由：

- 有网页控制台，适合第一次租机器。
- 能按 GPU 型号筛选，例如 4090 / 5090。
- 有现成 PyTorch 镜像，省去很多 CUDA 配置问题。
- 能直接开 JupyterLab 终端，也能之后用 VSCode SSH 连接。

备选平台：

- 恒源云：也适合国内深度学习租卡，支持实例、镜像、tmux、JupyterLab 等常见流程。
- 矩池云：国内也常见，但第一次使用时建议先选文档更顺手的平台。
- 阿里云 / 腾讯云 / 火山引擎：更偏企业云，稳定但新手配置成本和价格通常更高。

## 3. 租 AutoDL 的基本流程

1. 打开 AutoDL 官网并注册登录。
2. 进入控制台。
3. 点击“租用新实例”。
4. 选择计费方式，一般先选按量计费。
5. 选择地区，优先选延迟低、价格合适、有空闲机器的地区。
6. GPU 型号选择：
   - 初期调试：3090 / 4090 都可以。
   - 较大训练：4090 优先，性价比通常较好。
   - 想要更大显存和更强算力：5090 可以考虑。
7. GPU 数量先选 1 张。
8. 镜像选择 PyTorch + CUDA 镜像。
   - 4090：选较新的 PyTorch 2.x + CUDA 12.x 镜像。
   - 5090：优先选平台提供的较新 PyTorch 2.7/2.8 + CUDA 12.8 镜像。
9. 硬盘大小按需求选择。代码很小，但数据集和 checkpoint 可能很大。
10. 创建实例，等待状态变成运行中。
11. 不用时立刻关机，运行中通常会持续计费。

## 4. 第一次上云后怎么用？

最简单路线：先用平台自带 JupyterLab 的终端。

进入实例后，在终端执行：

```bash
nvidia-smi
```

这个命令能看到显卡型号、显存占用、驱动版本。

再检查 PyTorch：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
PY
```

如果平台镜像已经带了合适的 PyTorch，并且 `cuda available: True`，不要重复安装 PyTorch。

## 5. 怎么把本项目放到云服务器？

有三种常见方式。

### 方式 A：用 git

如果项目已经推到 GitHub / Gitee：

```bash
git clone <你的仓库地址>
cd "an attention free transformer"
```

这是最推荐的长期方式。

### 方式 B：网页上传

适合第一次临时测试。把项目压缩成 zip，通过 JupyterLab 上传，然后解压：

```bash
unzip project.zip
cd "an attention free transformer"
```

### 方式 C：scp / rsync

适合熟悉 SSH 后使用。第一次不必急着学。

## 6. 云服务器上的环境安装建议

如果镜像已经有 PyTorch + CUDA：

```bash
cd "an attention free transformer"
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e .
```

如果镜像没有合适的 PyTorch，先确认平台推荐的 CUDA 版本，再装对应 GPU 版 PyTorch。不要在云 GPU 上安装 CPU 版 `torch`。

安装后验证：

```bash
python - <<'PY'
import torch
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
print(y.shape)
print(torch.cuda.get_device_name(0))
PY
```

## 7. 训练时不要直接裸跑长任务

云服务器训练可能跑很久。推荐用 `tmux`：

```bash
tmux new -s aft
```

在 tmux 里运行训练：

```bash
python scripts/train_toy.py
```

临时退出但保持训练继续：

```text
Ctrl+B，然后按 D
```

重新进入：

```bash
tmux attach -t aft
```

## 8. 租卡选择建议

当前阶段：

- 写代码、跑单元测试：本地 CPU 足够。
- toy 训练：本地 CPU 或便宜云卡都可以。
- 跑较大 batch、较长序列、较多层 AFT：租 4090。
- 显存不足或希望更快：租 5090 或更大显存卡。

优先级建议：

1. 本地 CPU 把代码写通。
2. 云 4090 跑正式训练。
3. 只有确认 4090 显存或速度不够时，再考虑 5090。

## 9. 费用和数据安全注意事项

- 实例“运行中”通常会持续计费。
- 不训练时要关机。
- 重要代码用 git 保存，不要只放在云实例里。
- 重要 checkpoint 下载或同步到网盘/对象存储。
- 不要把密码、token、私钥写进代码仓库。

