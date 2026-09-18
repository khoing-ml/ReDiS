import pytest
import torch

from redis.analysis import seed_spectrum
from redis.methods import (
    gaussian_noise,
    gram_isotropize,
    make_random_basis,
    projected_amplify,
    residual_amplify,
)
from redis.methods.transforms import cap_relative_correction, rms_match


def test_random_basis_is_orthonormal_and_seeded():
    q1 = make_random_basis(16, 4, device=torch.device("cpu"), seed=7)
    q2 = make_random_basis(16, 4, device=torch.device("cpu"), seed=7)
    assert torch.equal(q1, q2)
    assert torch.allclose(q1.T @ q1, torch.eye(4), atol=1e-6)


def test_full_projection_matches_raw_amplification():
    hidden = torch.randn(4, 5, 8)
    q = make_random_basis(8, 8, device=torch.device("cpu"), seed=3)
    raw = residual_amplify(hidden, 0.2)
    projected = projected_amplify(hidden, q, 0.2)
    assert torch.allclose(raw, projected, atol=1e-5)


def test_gaussian_noise_is_deterministic_and_scaled():
    hidden = torch.randn(4, 5, 8)
    first = gaussian_noise(hidden, 0.1, generator=torch.Generator().manual_seed(9))
    second = gaussian_noise(hidden, 0.1, generator=torch.Generator().manual_seed(9))
    assert torch.equal(first, second)
    assert not torch.equal(first, hidden)


def test_gram_beta_or_gamma_zero_is_identity():
    hidden = torch.randn(4, 5, 8)
    q = make_random_basis(8, 4, device=torch.device("cpu"), seed=0)
    assert gram_isotropize(hidden, q, beta=0.0, gamma=1.0) is hidden
    assert gram_isotropize(hidden, q, beta=1.0, gamma=0.0) is hidden


def test_gram_whitening_flattens_anisotropic_spectrum():
    base = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]
    ).reshape(4, 1, 3)
    q = torch.eye(3)
    before = seed_spectrum(base)["top1_ratio"]
    after_tensor = gram_isotropize(base, q, beta=1.0, gamma=1.0)
    after = seed_spectrum(after_tensor)["top1_ratio"]
    assert after < before


def test_rms_match_and_correction_cap():
    reference = torch.randn(4, 5, 8)
    modified = reference * 3
    matched = rms_match(modified, reference)
    assert torch.allclose(
        matched.square().mean(dim=(1, 2)).sqrt(),
        reference.square().mean(dim=(1, 2)).sqrt(),
        atol=1e-5,
    )
    capped = cap_relative_correction(reference + 10, reference, 0.1)
    ratio = (capped - reference).norm() / reference.norm()
    assert float(ratio) == pytest.approx(0.1, rel=1e-5)

