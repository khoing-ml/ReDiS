import pytest
import torch

from redis.sampling import (
    batched_dot,
    cap_relative_norm,
    match_relative_norm,
    non_increasing_project,
    orthonormalize,
    project_onto_span,
    tangent_project,
)


def test_orthonormalize_handles_dependent_directions_per_sample():
    first = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    second = torch.tensor([[[2.0, 0.0]], [[1.0, 0.0]]])
    basis = orthonormalize([first, second])
    assert torch.allclose(batched_dot(basis[0], basis[0]), torch.ones(2))
    assert torch.allclose(basis[1][0], torch.zeros_like(basis[1][0]))
    assert float(batched_dot(basis[1], basis[1])[1]) == pytest.approx(1.0)


def test_projection_lands_in_span():
    value = torch.tensor([[[2.0, 3.0, 4.0]]])
    x = torch.tensor([[[1.0, 0.0, 0.0]]])
    y = torch.tensor([[[0.0, 1.0, 0.0]]])
    projected, _ = project_onto_span(value, [x, y])
    assert torch.allclose(projected, torch.tensor([[[2.0, 3.0, 0.0]]]))


def test_tangent_projection_is_orthogonal_to_projected_normal():
    proposal = torch.tensor([[[1.0, 2.0, 3.0]]])
    normal = torch.tensor([[[1.0, 1.0, 9.0]]])
    x = torch.tensor([[[1.0, 0.0, 0.0]]])
    y = torch.tensor([[[0.0, 1.0, 0.0]]])
    safe, diagnostics = tangent_project(proposal, normal, [x, y])
    assert float(batched_dot(safe, diagnostics["normal_subspace"])[0]) == pytest.approx(
        0.0, abs=1e-6
    )
    assert float(safe[0, 0, 2]) == 0.0


def test_partial_tangent_projection_retains_parallel_component():
    proposal = torch.tensor([[[1.0, 1.0]]])
    normal = torch.tensor([[[1.0, 0.0]]])
    safe, _ = tangent_project(proposal, normal, [proposal, normal], tangent_strength=0.5)
    assert torch.allclose(safe, torch.tensor([[[0.5, 1.0]]]), atol=1e-6)


def test_non_increasing_projection_removes_only_ascent_component():
    normal = torch.tensor([[[1.0, 0.0]]])
    x = torch.tensor([[[1.0, 0.0]]])
    y = torch.tensor([[[0.0, 1.0]]])
    ascent = torch.tensor([[[1.0, 1.0]]])
    safe, diagnostics = non_increasing_project(ascent, normal, [x, y])
    assert torch.allclose(safe, torch.tensor([[[0.0, 1.0]]]), atol=1e-6)
    assert diagnostics["constraint_active"].tolist() == [True]
    assert float(diagnostics["directional_derivative_post"][0]) == pytest.approx(
        0.0, abs=1e-6
    )


def test_non_increasing_projection_preserves_descent_component():
    normal = torch.tensor([[[1.0, 0.0]]])
    proposal = torch.tensor([[[-1.0, 1.0]]])
    safe, diagnostics = non_increasing_project(
        proposal,
        normal,
        [torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[0.0, 1.0]]])],
    )
    assert torch.allclose(safe, proposal, atol=1e-6)
    assert diagnostics["constraint_active"].tolist() == [False]


def test_cap_relative_norm_is_per_sample():
    reference = torch.ones(2, 1, 2)
    correction = torch.tensor([[[10.0, 0.0]], [[0.01, 0.0]]])
    capped, scales = cap_relative_norm(correction, reference, 0.1)
    achieved = capped.flatten(1).norm(dim=1) / reference.flatten(1).norm(dim=1)
    assert float(achieved[0]) == pytest.approx(0.1)
    assert torch.equal(capped[1], correction[1])
    assert float(scales[0]) < 1 and float(scales[1]) == 1


def test_match_relative_norm_equalizes_nonzero_corrections():
    reference = torch.ones(2, 1, 2)
    correction = torch.tensor([[[10.0, 0.0]], [[0.01, 0.0]]])
    matched, _ = match_relative_norm(correction, reference, 0.05)
    achieved = matched.flatten(1).norm(dim=1) / reference.flatten(1).norm(dim=1)
    assert torch.allclose(achieved, torch.full_like(achieved, 0.05))
