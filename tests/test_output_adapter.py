import pytest
import torch

from redis.hooks.output_adapter import extract_hidden, replace_hidden


def test_dual_stream_round_trip():
    text = torch.randn(2, 3, 4)
    image = torch.randn(2, 5, 4)
    out = (text, image)
    replacement = torch.zeros_like(image)
    assert extract_hidden(out, "transformer_blocks") is image
    updated = replace_hidden(out, replacement, "transformer_blocks")
    assert updated[0] is text
    assert updated[1] is replacement


def test_single_stream_round_trip():
    hidden = torch.randn(2, 8, 4)
    assert extract_hidden(hidden, "single_transformer_blocks") is hidden
    assert replace_hidden(hidden, hidden, "single_transformer_blocks") is hidden


def test_invalid_dual_stream_output_rejected():
    with pytest.raises(TypeError):
        extract_hidden(torch.randn(2, 3, 4), "transformer_blocks")

