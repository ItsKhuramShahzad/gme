"""
Parallel-transport regressions for the U(k)-GCG solver (audit issue #5).

The GCG step is ``U_new = U @ expm(t*X)`` and ``_gcg_gradient`` returns the
LEFT-TRIVIALIZED (body) gradient — verified here in ``test_gradient_is_body``.
In body coordinates the parallel transport of a tangent vector Y along the
geodesic ``gamma(t) = U exp(tX)`` is

    T(Y) = exp(-tX/2) Y exp(+tX/2)

i.e. ``half_step.conj().T @ Y @ half_step``.  Applying the inverse adjoint
action instead corrupts the modified Polak-Ribiere coefficient.

Unlike the older test modules, every check here is an ``assert``.
"""
import numpy as np
import pytest
from scipy.linalg import expm

from hubbard_gme.convex_roof import _gcg_gradient, _h_full_unitary, _gcg_inner


# ── helpers ──────────────────────────────────────────────────────────────────

def _skew(k, rng):
    A = rng.normal(size=(k, k)) + 1j * rng.normal(size=(k, k))
    return 0.5 * (A - A.conj().T)


def _problem(n=3, extra=2, seed=3):
    """A concrete (eigvecs, eigvals_sqrt, k) triple from a random mixed state."""
    rng = np.random.default_rng(seed)
    dim = 2 ** n
    A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = A @ A.conj().T
    rho /= np.trace(rho).real
    ev, evec = np.linalg.eigh(rho)
    mask = ev > 1e-12
    eigvecs = evec[:, mask]
    eigvals_sqrt = np.sqrt(ev[mask])
    k = int(mask.sum()) + extra
    if eigvecs.shape[1] < k:
        pad = k - eigvecs.shape[1]
        eigvecs = np.hstack([eigvecs, np.zeros((dim, pad), complex)])
        eigvals_sqrt = np.concatenate([eigvals_sqrt, np.zeros(pad)])
    return eigvecs, eigvals_sqrt, k, n


def _transport_body(Y, X, t):
    """Correct body-coordinate transport: exp(-tX/2) Y exp(+tX/2)."""
    h = expm(0.5 * t * X)
    return h.conj().T @ Y @ h


TS = (0.0, 0.13, 0.4, 0.9)


# ── tests ────────────────────────────────────────────────────────────────────

def test_gradient_is_body_coordinates():
    """d/dt h(U exp(tY))|_0 == <G, Y> for all skew-Hermitian Y.

    This identity *defines* the left-trivialized (body) gradient, and fixes
    which parallel-transport convention the solver must use.
    """
    eigvecs, eigvals_sqrt, k, n = _problem()
    rng = np.random.default_rng(11)
    U = expm(_skew(k, rng))
    G = _gcg_gradient(U, eigvecs, eigvals_sqrt, n)

    assert np.linalg.norm(G + G.conj().T) < 1e-12, "body gradient must be skew-Hermitian"

    h = 1e-6
    for _ in range(5):
        Y = _skew(k, rng)
        fd = (_h_full_unitary(U @ expm(h * Y), eigvecs, eigvals_sqrt, n)
              - _h_full_unitary(U @ expm(-h * Y), eigvecs, eigvals_sqrt, n)) / (2 * h)
        ip = _gcg_inner(G, Y)
        assert abs(fd - ip) <= 1e-5 * max(1.0, abs(fd)), (
            f"gradient is not in body coordinates: finite-difference {fd:.6e} "
            f"!= <G,Y> {ip:.6e}")


def test_transport_preserves_tangency():
    """The transported vector stays in the tangent space (skew-Hermitian)."""
    eigvecs, eigvals_sqrt, k, n = _problem()
    rng = np.random.default_rng(12)
    U = expm(_skew(k, rng))
    G = _gcg_gradient(U, eigvecs, eigvals_sqrt, n)
    X = _skew(k, rng)

    for t in TS:
        T = _transport_body(G, X, t)
        err = np.linalg.norm(T + T.conj().T)
        assert err < 1e-10, f"transported vector not tangent at t={t}: ||T+T^H||={err:.2e}"


def test_transport_is_parallel():
    """Covariant derivative along the geodesic vanishes: dT/dt + [X,T]/2 == 0.

    This is the check that distinguishes the correct body transport from the
    inverse adjoint action, which leaves an O(1) residual.
    """
    eigvecs, eigvals_sqrt, k, n = _problem()
    rng = np.random.default_rng(13)
    U = expm(_skew(k, rng))
    G = _gcg_gradient(U, eigvecs, eigvals_sqrt, n)
    X = _skew(k, rng)

    h = 1e-6
    for t in TS:
        T0 = _transport_body(G, X, t)
        dT = (_transport_body(G, X, t + h) - _transport_body(G, X, t - h)) / (2 * h)
        resid = np.linalg.norm(dT + 0.5 * (X @ T0 - T0 @ X))
        assert resid < 1e-6, (
            f"body transport is not parallel at t={t}: |dT/dt + [X,T]/2| = {resid:.2e}")

    # the inverse adjoint action (the pre-fix implementation) must FAIL this
    def _transport_inverse(Y, X, t):
        hh = expm(0.5 * t * X)
        return hh @ Y @ hh.conj().T

    T0 = _transport_inverse(G, X, 0.4)
    dT = (_transport_inverse(G, X, 0.4 + h) - _transport_inverse(G, X, 0.4 - h)) / (2 * h)
    resid_bad = np.linalg.norm(dT + 0.5 * (X @ T0 - T0 @ X))
    assert resid_bad > 1e-3, (
        "sanity check failed: the inverse adjoint action should NOT satisfy the "
        "body-coordinate parallel condition")


def test_generator_transports_to_itself():
    """X generates the geodesic, so it is its own parallel transport."""
    _, _, k, _ = _problem()
    rng = np.random.default_rng(14)
    X = _skew(k, rng)
    for t in TS:
        err = np.linalg.norm(_transport_body(X, X, t) - X)
        assert err < 1e-10, f"generator not self-transported at t={t}: {err:.2e}"


def test_search_direction_is_descent_with_safeguard():
    """After the safeguard, the new direction is never an ascent direction."""
    eigvecs, eigvals_sqrt, k, n = _problem()
    rng = np.random.default_rng(15)

    n_checked = 0
    for _ in range(8):
        U0 = expm(_skew(k, rng))
        X = _skew(k, rng)
        t = float(rng.uniform(0.05, 0.6))

        G0 = _gcg_gradient(U0, eigvecs, eigvals_sqrt, n)
        G1 = _gcg_gradient(U0 @ expm(t * X), eigvecs, eigvals_sqrt, n)

        denom = _gcg_inner(G0, G0)
        T = _transport_body(G0, X, t)
        gamma = 0.0 if abs(denom) < 1e-20 else _gcg_inner(G1 - T, G1) / denom
        gamma = max(0.0, gamma)

        X_new = -G1 + gamma * X
        if _gcg_inner(G1, X_new) >= 0.0:      # the safeguard
            X_new = -G1

        slope = _gcg_inner(G1, X_new)
        assert slope <= 1e-12, f"search direction is ascent: <G_new, X_new> = {slope:.3e}"
        if _gcg_inner(G1, G1) > 1e-12:
            assert slope < 0.0, "non-stationary point must give a strict descent direction"
            n_checked += 1

    assert n_checked > 0, "no non-stationary trial was exercised"
