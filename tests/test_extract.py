"""Tests for extract.logistic_regression (Phase 3 spec §3), on synthetic data. No model needed."""

import torch

from trait_vectors.extract import logistic_regression


def synthetic(n=4000, d=10, seed=0):
    """X ~ N(0, I), y ~ Bernoulli(σ(X w_true + b_true)) with ||w_true|| = 2, b_true = -0.5."""
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(d, generator=g)
    w_true = 2 * u / u.norm()
    X = torch.randn(n, d, generator=g)
    y = (torch.rand(n, generator=g) < torch.sigmoid(X @ w_true - 0.5)).float()
    return X, y, w_true


def test_recovers_known_direction():
    """With little regularisation and many samples, (w, b) is close to the true (w_true, -0.5).

    The standard error per coordinate is about 1 / sqrt(n · E[σ(1 - σ)]) ≈ 0.04, so the bounds are loose.
    """
    X, y, w_true = synthetic()
    w, b = logistic_regression(X, y, lam=1e-4)
    cos = (w @ w_true.double() / (w.norm() * w_true.norm())).item()
    assert cos > 0.98
    assert abs(w.norm().item() - 2) < 0.2
    assert abs(b + 0.5) < 0.15


def test_first_order_condition():
    """At the minimum the gradient is zero: X^T (σ(Xw + b) - y) / n + lam w = 0, and mean(σ(Xw + b) - y) = 0."""
    X, y, _ = synthetic(n=500)
    X, y = X.double(), y.double()
    lam = 0.1
    w, b = logistic_regression(X, y, lam)
    residual = torch.sigmoid(X @ w + b) - y
    assert (X.T @ residual / len(y) + lam * w).abs().max() < 1e-6
    assert abs(residual.mean().item()) < 1e-6


def test_more_regularisation_shrinks_w():
    X, y, _ = synthetic(n=500)
    norms = [logistic_regression(X, y, lam)[0].norm().item() for lam in (0.001, 0.1, 10)]
    assert norms[0] > norms[1] > norms[2]
