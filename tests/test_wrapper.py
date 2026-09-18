import torch

from redis.hooks import (
    CaptureController,
    IdentityController,
    InterventionController,
    patch_flux2_blocks,
    restore_flux2_blocks,
)


class DummyDualBlock(torch.nn.Module):
    def forward(self, hidden_states, encoder_hidden_states):
        return encoder_hidden_states + 2, hidden_states + 1


class DummyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([DummyDualBlock(), DummyDualBlock()])
        self.single_transformer_blocks = torch.nn.ModuleList([])


def test_identity_wrapper_is_exact_and_restorable():
    model = DummyTransformer()
    original = model.transformer_blocks[0]
    hidden = torch.randn(3, 5, 4)
    text = torch.randn(3, 2, 4)
    expected = original(hidden, text)
    assert patch_flux2_blocks(model, IdentityController(), layer_ids=[0]) == [0]
    actual = model.transformer_blocks[0](hidden, text)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert restore_flux2_blocks(model) == [("transformer_blocks", 0)]
    assert model.transformer_blocks[0] is original


def test_capture_wrapper_records_hidden_without_modifying_it():
    model = DummyTransformer()
    controller = CaptureController()
    patch_flux2_blocks(model, controller, layer_ids=[1])
    hidden = torch.randn(3, 5, 4)
    text = torch.randn(3, 2, 4)
    out = model.transformer_blocks[1](hidden, text)
    captured = controller.activations[("transformer_blocks", 1)][0]
    assert torch.equal(captured, out[1].float().cpu())


def test_intervention_only_runs_on_selected_timestep():
    model = DummyTransformer()
    controller = InterventionController(
        method="residual_amplification",
        timestep_ids=[1],
        gamma=0.2,
        match_rms=False,
    )
    patch_flux2_blocks(model, controller, layer_ids=[0])
    hidden = torch.randn(3, 5, 4)
    text = torch.randn(3, 2, 4)
    first = model.transformer_blocks[0](hidden, text)[1]
    second = model.transformer_blocks[0](hidden, text)[1]
    vanilla = hidden + 1
    assert torch.equal(first, vanilla)
    assert not torch.equal(second, vanilla)
    assert len(controller.logs) == 1
    assert controller.logs[0]["timestep_id"] == 1
