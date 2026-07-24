"""Shape and gradient tests will be added as modules are implemented."""

import torch

from aft.layers import FeedForward, AFTSimple, AFTFull, AFTLocal, AFTConv
from aft.blocks import AFTBlock
from aft.model import AFTLanguageModel

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

def test_aft_conv_shape():
    aft = AFTConv(8, 3, 0.0)
    x = torch.randn(2, 4, 8)
    y = aft(x)
    assert y.shape == x.shape

def test_aft_language_model_conv_shape():
    model = AFTLanguageModel(
        vocab_size=20,
        d_model=8,
        hidden_dim=32,
        n_layers=2,
        max_seq_len=16,
        dropout=0.0,
        aft_type="conv",
        kernel_size=3,
    )

    input_ids = torch.randint(0, 20, (2, 4))
    logits = model(input_ids)
    assert logits.shape == (2, 4, 20)
