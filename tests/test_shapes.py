"""Shape and gradient tests will be added as modules are implemented."""

import torch

from aft.layers import FeedForward, AFTSimple, AFTFull, AFTLocal
from aft.blocks import AFTBlock
from aft.model import AFTLanguageModel
from aft.vision import PatchEmbedding, AFTConv2D, VisionBlock, AFTImageClassifier

# 这些测试故意写得很小，用来快速发现 shape 或 API 调用错误。

def test_feed_forward_shape():
    ffn = FeedForward(8, 32, 0.0)
    x = torch.randn(2, 4, 8)
    y = ffn(x)
    assert y.shape == x.shape

def test_feed_forward_backward():
    # 一个标量假 loss 就足够检查参数是否能收到梯度。
    ffn = FeedForward(8, 32, 0.0)
    x = torch.randn(2, 4, 8)
    y = ffn(x)
    loss = y.mean()
    loss.backward()
    assert ffn.fc1.weight.grad is not None
    assert ffn.fc2.weight.grad is not None

def test_aft_simple_shape():
    aft = AFTSimple(8, 0.0)
    x = torch.randn(2, 4, 8)
    y = aft(x)
    assert y.shape == x.shape

def test_aft_simple_backward():
    aft = AFTSimple(8, 0.0)
    x = torch.randn(2, 4, 8)
    y = aft(x)
    loss = y.mean()
    loss.backward()
    assert aft.to_q.weight.grad is not None
    assert aft.to_k.weight.grad is not None
    assert aft.to_v.weight.grad is not None
    assert aft.out_proj.weight.grad is not None

def test_aft_block_shape():
    block = AFTBlock(8, 32, 0.0)
    x = torch.randn(2, 4, 8)
    y = block(x)
    assert y.shape == x.shape

def test_aft_block_backward():
    block = AFTBlock(8, 32, 0.0)
    x = torch.randn(2, 4, 8)
    y = block(x)
    loss = y.mean()
    loss.backward()
    assert block.aft.to_q.weight.grad is not None
    assert block.ffn.fc1.weight.grad is not None

def test_aft_language_model_shape():
    # 语言模型把 token id [B, T] 映射成 logits [B, T, vocab_size]。
    model = AFTLanguageModel(
        20,
        8,
        32,
        2,
        16,
        0.0
    )

    input_ids = torch.randint(0, 20, (2, 4))
    logits = model(input_ids)
    assert logits.shape == (2, 4, 20)

def test_aft_language_model_backward():
    model = AFTLanguageModel(
        20,
        8,
        32,
        2,
        16,
        0.0
    )

    input_ids = torch.randint(0, 20, (2, 4))
    logits = model(input_ids)
    loss = logits.mean()
    loss.backward()
    assert model.token_emb.weight.grad is not None
    assert model.blocks[0].aft.to_q.weight.grad is not None
    assert model.lm_head.weight.grad is not None

def test_aft_full_shape():
    aft = AFTFull(8, 16, 0.0)
    x = torch.randn(2, 4, 8)
    y = aft(x)
    assert y.shape == x.shape

def test_aft_language_model_full_shape():
    model = AFTLanguageModel(
        vocab_size=20,
        d_model=8,
        hidden_dim=32,
        n_layers=2,
        max_seq_len=16,
        dropout=0.0,
        aft_type="full",
    )

    input_ids = torch.randint(0, 20, (2, 4))
    logits = model(input_ids)
    assert logits.shape == (2, 4, 20)

def test_aft_local_shape():
    aft = AFTLocal(8, 16, 1, 0.0)
    x = torch.randn(2, 4, 8)
    y = aft(x)
    assert y.shape == x.shape

def test_aft_language_model_local_shape():
    model = AFTLanguageModel(
        vocab_size=20,
        d_model=8,
        hidden_dim=32,
        n_layers=2,
        max_seq_len=16,
        dropout=0.0,
        aft_type="local",
        local_window_size=1
    )

    input_ids = torch.randint(0, 20, (2, 4))
    logits = model(input_ids)
    assert logits.shape == (2, 4, 20)

def test_aft_local_causal_does_not_use_future_tokens():
    aft = AFTLocal(
        d_model=8,
        max_seq_len=16,
        local_window_size=2,
        dropout=0.0,
        causal=True,
    )

    aft.eval()

    x1 = torch.randn(1, 4, 8)
    x2 = x1.clone()

    x2[:, 3, :] = torch.randn(1, 8)

    y1 = aft(x1)
    y2 = aft(x2)
    assert torch.allclose(y1[:, :3, :], y2[:, :3, :], atol=1e-5)


def test_aft_block_causal_local_shape():
    block = AFTBlock(
        d_model=8,
        hidden_dim=32,
        dropout=0.0,
        aft_type="local",
        max_seq_len=16,
        local_window_size=2,
        causal=True,
    )

    x = torch.randn(2, 4, 8)
    y = block(x)
    assert y.shape == x.shape

def test_aft_language_model_causal_local_shape():
    model = AFTLanguageModel(
        vocab_size=20,
        d_model=8,
        hidden_dim=32,
        n_layers=2,
        max_seq_len=16,
        dropout=0.0,
        aft_type="local",
        local_window_size=2,
        causal=True,
    )

    input_ids = torch.randint(0, 20, (2, 4))
    logits = model(input_ids)

    assert logits.shape == (2, 4, 20)

def test_patch_embedding_shape():
    patch = PatchEmbedding(
        image_size=224,
        patch_size=16,
        in_channels=3,
        d_model=64,
    )

    x = torch.randn(2, 3, 224, 224)
    y = patch(x)

    assert y.shape == (2, 196, 64)


def test_aft_conv2d_shape_and_backward():
    aft = AFTConv2D(
        d_model=64,
        image_size=224,
        patch_size=16,
        kernel_size=3,
        dropout=0.0,
    )

    x = torch.randn(2, 196, 64)
    y = aft(x)

    assert y.shape == x.shape

    loss = y.mean()
    loss.backward()

    assert aft.to_q.weight.grad is not None
    assert aft.to_k.weight.grad is not None
    assert aft.to_v.weight.grad is not None
    assert aft.position_bias.grad is not None
    assert aft.out_proj.weight.grad is not None
    assert aft.position_gain.grad is not None
    assert aft.position_offset.grad is not None

def test_vision_block_shape_and_backward():
    block = VisionBlock(
        d_model=64,
        hidden_dim=256,
        image_size=224,
        patch_size=16,
        kernel_size=3,
        dropout=0.0,
    )

    x = torch.randn(2, 196, 64)
    y = block(x)

    assert y.shape == x.shape

    loss = y.mean()
    loss.backward()

    assert block.aft.to_q.weight.grad is not None
    assert block.aft.position_bias.grad is not None
    assert block.ffn.fc1.weight.grad is not None
    assert block.ffn.fc2.weight.grad is not None


def test_aft_image_classifier_shape_and_backward():
    model = AFTImageClassifier(
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        d_model=64,
        hidden_dim=256,
        n_layers=2,
        kernel_size=3,
        dropout=0.0,
        use_position_embedding=True,
    )

    x = torch.randn(2, 3, 224, 224)
    logits = model(x)

    assert logits.shape == (2, 1000)

    loss = logits.mean()
    loss.backward()

    assert model.patch_embed.proj.weight.grad is not None
    assert model.position_embedding.grad is not None
    assert model.blocks[0].aft.position_bias.grad is not None
    assert model.head.weight.grad is not None


def test_aft_image_classifier_without_position_embedding_shape():
    model = AFTImageClassifier(
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        d_model=64,
        hidden_dim=256,
        n_layers=2,
        kernel_size=3,
        dropout=0.0,
    )

    x = torch.randn(2, 3, 224, 224)
    logits = model(x)

    assert logits.shape == (2, 1000)
    assert model.position_embedding is None


def test_aft_local_low_rank_shape_and_backward():
    aft = AFTLocal(
        d_model=8,
        max_seq_len=16,
        local_window_size=2,
        dropout=0.0,
        causal=True,
        use_low_rank_bias=True,
        bias_rank=4,
    )

    x = torch.randn(2, 4, 8)
    y = aft(x)

    assert y.shape == x.shape

    loss = y.mean()
    loss.backward()

    assert aft.position_u.grad is not None
    assert aft.position_v.grad is not None


def test_aft_block_low_rank_local_shape_and_backward():
    block = AFTBlock(
        d_model=8,
        hidden_dim=32,
        dropout=0.0,
        aft_type="local",
        max_seq_len=16,
        local_window_size=2,
        causal=True,
        use_low_rank_bias=True,
        bias_rank=4,
    )

    x = torch.randn(2, 4, 8)
    y = block(x)

    assert y.shape == x.shape

    loss = y.mean()
    loss.backward()

    assert block.aft.position_u.grad is not None
    assert block.aft.position_v.grad is not None


def test_aft_language_model_low_rank_local_shape_and_backward():
    model = AFTLanguageModel(
        vocab_size=20,
        d_model=8,
        hidden_dim=32,
        n_layers=2,
        max_seq_len=16,
        dropout=0.0,
        aft_type="local",
        local_window_size=2,
        causal=True,
        use_low_rank_bias=True,
        bias_rank=4,
    )

    input_ids = torch.randint(0, 20, (2, 4))
    logits = model(input_ids)

    assert logits.shape == (2, 4, 20)

    loss = logits.mean()
    loss.backward()

    assert model.blocks[0].aft.position_u.grad is not None
    assert model.blocks[0].aft.position_v.grad is not None