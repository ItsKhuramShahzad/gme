"""
Standalone gradient verification for the GCG implementation.

Run this BEFORE relying on convex_roof_gcg() in production.

It compares the analytical gradient _gcg_gradient(U) against numerical
finite differences. If they agree to ~1e-5, the analytical formula is
correct. If they disagree by more, there is a bug in the derivation
or implementation.

The test is run on a small synthetic mixed RDM (random rank-3 ρ on 3 qubits)
because:
  - Small enough to be fast
  - Mixed enough that all the gradient machinery is exercised
  - 3 qubits = 7 bipartitions, so all the bipartition logic runs
"""

import numpy as np
from numpy.linalg import qr

# Replace this with: 
from hubbard_gme.convex_roof import (
     _cgme_pure_with_grad, _gcg_gradient, _h_full_unitary, )
# once you've pasted the GCG block into convex_roof.py.


def make_random_mixed_rho(n: int, rank: int, seed: int = 42) -> np.ndarray:
    """Build a random mixed density matrix on n qubits with given rank."""
    np.random.seed(seed)
    dim = 2 ** n
    # Random Hermitian
    A = np.random.randn(dim, rank) + 1j * np.random.randn(dim, rank)
    rho = A @ A.conj().T
    rho = rho / np.trace(rho).real
    return rho


def check_gradient_at_U(U, eigvecs, eigvals_sqrt, n, eps=1e-5):
    """
    Compare analytical _gcg_gradient(U) against numerical centred differences
    on each (real and imaginary) matrix element of U.

    Returns the max absolute difference between analytical and numerical
    gradient entries.
    """
    from convex_roof import _gcg_gradient, _h_full_unitary
    # ↑ adjust import path once integrated

    G_analytic = _gcg_gradient(U, eigvecs, eigvals_sqrt, n)

    k = U.shape[0]
    G_numerical = np.zeros((k, k), dtype=complex)

    for kk in range(k):
        for ll in range(k):
            # Centered difference on Re(U_kl)
            U_plus = U.copy()
            U_minus = U.copy()
            U_plus[kk, ll]  += eps
            U_minus[kk, ll] -= eps
            dRe = (_h_full_unitary(U_plus, eigvecs, eigvals_sqrt, n)
                 - _h_full_unitary(U_minus, eigvecs, eigvals_sqrt, n)) / (2 * eps)

            U_plus = U.copy()
            U_minus = U.copy()
            U_plus[kk, ll]  += 1j * eps
            U_minus[kk, ll] -= 1j * eps
            dIm = (_h_full_unitary(U_plus, eigvecs, eigvals_sqrt, n)
                 - _h_full_unitary(U_minus, eigvecs, eigvals_sqrt, n)) / (2 * eps)

            # Note: numerical derivatives are NOT projected onto the tangent
            # space of U(k), so they include the trivial direction. To compare
            # to G_analytic (which IS the Riemannian gradient), we project.
            # For a fair check at this stage, we instead reconstruct what the
            # analytical formula computes BEFORE projection — i.e., the raw
            # ∂h/∂Re U_{kl} and ∂h/∂Im U_{kl} from Eqs. 10-11.
            G_numerical[kk, ll] = dRe + 1j * dIm

    # The analytical gradient is the SKEW-HERMITIAN PROJECTION of the raw
    # gradient. The numerical gradient computed above is the raw gradient
    # in the Euclidean sense. To compare, project the numerical gradient
    # onto the skew-Hermitian Lie algebra by computing its matrix product
    # with U as in Eqs. 22-23, 26.
    # Specifically: A_kl = Σ_i [Re(numerical[i,l]) Re(U[i,k])
    #                         + Im(numerical[i,l]) Im(U[i,k])]
    #               S_kl = Σ_i [Im(numerical[i,l]) Re(U[i,k])
    #                         − Re(numerical[i,l]) Im(U[i,k])]
    Re_n = np.real(G_numerical)
    Im_n = np.imag(G_numerical)
    V_re = np.real(U)
    V_im = np.imag(U)
    A_num = Re_n.T @ V_re + Im_n.T @ V_im
    S_num = Im_n.T @ V_re - Re_n.T @ V_im
    G_proj = 0.5 * (A_num - A_num.T) + 0.5j * (S_num + S_num.T)

    diff = np.max(np.abs(G_analytic - G_proj))
    rel = diff / max(np.max(np.abs(G_proj)), 1e-12)
    return float(diff), float(rel)


def main():
    n = 3
    rho = make_random_mixed_rho(n, rank=3, seed=42)
    eigvals, eigvecs = np.linalg.eigh(rho)
    mask = eigvals > 1e-10
    eigvals_r = eigvals[mask]
    eigvecs_r = eigvecs[:, mask]
    eigvals_sqrt = np.sqrt(eigvals_r)
    r = int(np.sum(mask))
    k = r + 2

    # Test at 5 random points on U(k)
    for trial in range(5):
        np.random.seed(100 + trial)
        H = np.random.randn(k, k) + 1j * np.random.randn(k, k)
        H = 0.5 * (H + H.conj().T)
        from scipy.linalg import expm
        U = expm(1j * H)

        diff, rel = check_gradient_at_U(U, eigvecs_r, eigvals_sqrt, n)
        status = "✓" if rel < 1e-4 else "✗"
        print(f"  trial {trial+1}:  abs diff = {diff:.3e}   rel = {rel:.3e}   {status}")


if __name__ == "__main__":
    main()
