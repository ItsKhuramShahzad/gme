"""
Behaviour of the CGME gradient at the separable boundary (audit issue #6).

``_cgme_pure_with_grad`` divides by ``C_safe = max(C, 1e-10)`` because
``dC/dpsi* = -(1/C) dP_max/dpsi*``.  The raw pure-state gradient therefore
diverges like 1/C, reaching ~2e10 at C = 0.

The audit asked whether that divergence reaches the optimizer.  It does not:
the pure-state gradient is multiplied by the component probabilities, summed
over the k components and contracted with U inside ``_gcg_gradient``, and the
1/C factor cancels.  Measured with |psi(eps)> = normalise(|000> + eps|111>):

    C = 1.98e-01  ->  ||grad_conj|| = 9.95e+00   ||G|| = 2.18e-03
    C = 2.00e-05  ->  ||grad_conj|| = 1.00e+05   ||G|| = 1.30e-02
    C = 0         ->  ||grad_conj|| = 2.00e+10   ||G|| = 1.50e-02

These tests lock in both halves of that statement, so a future refactor cannot
silently break the cancellation.
"""
import numpy as np
import pytest
from scipy.linalg import expm

from hubbard_gme.convex_roof import _cgme_pure_with_grad, _gcg_gradient


N_SPINS = 3
DIM = 2 ** N_SPINS


def _near_separable(eps):
    """|000> + eps|111>, normalised. C -> 0 as eps -> 0."""
    psi = np.zeros(DIM, complex)
    psi[0] = 1.0
    psi[-1] = eps
    return psi / np.linalg.norm(psi)


def _skew(k, rng):
    A = rng.normal(size=(k, k)) + 1j * rng.normal(size=(k, k))
    return 0.5 * (A - A.conj().T)


def test_raw_pure_gradient_scales_as_one_over_C():
    """Documents the known 1/C behaviour: C * ||grad|| is constant."""
    products = []
    for eps in (1e-2, 1e-3, 1e-4, 1e-5):
        psi = _near_separable(eps)
        C, grad = _cgme_pure_with_grad(psi, N_SPINS)
        assert C > 0, "test family should stay above the C_safe floor here"
        products.append(C * float(np.linalg.norm(grad)))

    ref = products[0]
    for value in products[1:]:
        assert abs(value - ref) < 1e-3 * ref, (
            f"C*||grad|| should be constant (1/C scaling); got {products}")


def test_pure_gradient_is_finite_at_the_boundary():
    """The C_safe floor must keep the raw gradient finite even at C = 0."""
    C, grad = _cgme_pure_with_grad(_near_separable(0.0), N_SPINS)
    assert C == pytest.approx(0.0, abs=1e-12)
    norm = float(np.linalg.norm(grad))
    assert np.isfinite(norm), "gradient must not be inf/nan at C = 0"
    # floor is 1e-10, so the bound is ~2/1e-10
    assert norm <= 2.1e10, f"unexpected magnitude at the boundary: {norm:.3e}"


@pytest.mark.parametrize("eps", [1e-1, 1e-3, 1e-5, 1e-8, 0.0])
def test_final_tangent_gradient_stays_bounded(eps):
    """The 1/C divergence must NOT reach the optimizer's tangent gradient.

    This is the property the audit asked us to measure: it is what makes the
    C_safe floor safe in practice.
    """
    psi = _near_separable(eps)
    noise = 1e-3
    rho = (1 - noise) * np.outer(psi, psi.conj()) + noise * np.eye(DIM) / DIM

    ev, evec = np.linalg.eigh(rho)
    mask = ev > 1e-12
    eigvecs = evec[:, mask]
    eigvals_sqrt = np.sqrt(ev[mask])
    k = int(mask.sum())

    U = expm(_skew(k, np.random.default_rng(1)))
    G = _gcg_gradient(U, eigvecs, eigvals_sqrt, N_SPINS)
    norm = float(np.linalg.norm(G))

    assert np.isfinite(norm), f"tangent gradient not finite at eps={eps}"
    # measured values are ~1e-2; 1.0 is a generous bound that would still catch
    # a genuine 1/C blow-up (which would be >= 1e4 here)
    assert norm < 1.0, (
        f"tangent gradient blew up near the separable boundary "
        f"(eps={eps}, ||G||={norm:.3e}) — the 1/C cancellation has been lost")
