"""
test_meyerwallach.py
--------------------
Verification of the Meyer-Wallach measure and its analytical gradient.

Two tests:

  TEST A:  γ value agreement
           Compare _meyerwallach_pure_vector against an independent
           reference implementation (the "naive" form using one-qubit RDMs).
           Must agree to ~1e-12 on random pure states.

  TEST B:  γ gradient against finite differences
           Compare analytical gradient (∂γ/∂Re ψ, ∂γ/∂Im ψ from Wirtinger)
           to central differences. Must agree to ~1e-6.

Run:
    python test_meyerwallach.py
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hubbard_gme.convex_roof import (
    _meyerwallach_pure_vector,
    _meyerwallach_pure_with_grad,
)


# ─────────────────────────────────────────────────────────────────────────
# Independent reference implementation
# ─────────────────────────────────────────────────────────────────────────

def _meyerwallach_reference(psi, n):
    """
    Naive implementation of Eq. (45), used as ground truth for testing.
    Computes each ρ_k explicitly and sums Tr(ρ_k²).
    """
    psi_tensor = psi.reshape([2] * n)
    purity_sum = 0.0

    for k in range(n):
        # Trace out everything except qubit k
        axes = [k] + [i for i in range(n) if i != k]
        psi_reshaped = np.transpose(psi_tensor, axes).reshape(2, -1)
        rho_k = psi_reshaped @ psi_reshaped.conj().T
        purity_sum += float(np.real(np.trace(rho_k @ rho_k)))

    return 2.0 * (1.0 - purity_sum / n)


# ─────────────────────────────────────────────────────────────────────────
# TEST A: value agreement
# ─────────────────────────────────────────────────────────────────────────

def test_A_value(n=3, n_trials=10):
    print(f"\n  TEST A: γ value vs reference implementation")
    print(f"  " + "-" * 70)

    dim = 2 ** n
    n_pass = 0

    for trial in range(n_trials):
        np.random.seed(600 + trial)
        psi = np.random.randn(dim) + 1j * np.random.randn(dim)
        psi = psi / np.linalg.norm(psi)

        gamma_impl = _meyerwallach_pure_vector(psi, n)
        gamma_ref  = _meyerwallach_reference(psi, n)

        diff = abs(gamma_impl - gamma_ref)
        ok = diff < 1e-12
        n_pass += int(ok)
        print(f"    trial {trial+1}:  impl = {gamma_impl:.10f}   "
              f"ref = {gamma_ref:.10f}   "
              f"diff = {diff:.2e}   {'✓' if ok else '✗'}")

    print(f"  Passed: {n_pass}/{n_trials}")
    return n_pass == n_trials


# ─────────────────────────────────────────────────────────────────────────
# TEST B: gradient vs finite differences
# ─────────────────────────────────────────────────────────────────────────

def test_B_gradient(n=3, n_trials=5, eps=1e-6):
    print(f"\n  TEST B: gradient vs finite differences")
    print(f"  " + "-" * 70)

    dim = 2 ** n
    n_pass = 0

    for trial in range(n_trials):
        np.random.seed(700 + trial)
        psi = np.random.randn(dim) + 1j * np.random.randn(dim)
        psi = psi / np.linalg.norm(psi)

        gamma, grad_conj = _meyerwallach_pure_with_grad(psi, n)
        # Convert Wirtinger gradient to real-valued partials:
        dG_dRe_a = 2.0 * np.real(grad_conj)
        dG_dIm_a = 2.0 * np.imag(grad_conj)

        # Numerical central differences
        dG_dRe_n = np.zeros(dim)
        dG_dIm_n = np.zeros(dim)
        for i in range(dim):
            psi_p = psi.copy(); psi_p[i] += eps
            psi_m = psi.copy(); psi_m[i] -= eps
            dG_dRe_n[i] = (_meyerwallach_pure_vector(psi_p, n)
                         - _meyerwallach_pure_vector(psi_m, n)) / (2 * eps)
            psi_p = psi.copy(); psi_p[i] += 1j * eps
            psi_m = psi.copy(); psi_m[i] -= 1j * eps
            dG_dIm_n[i] = (_meyerwallach_pure_vector(psi_p, n)
                         - _meyerwallach_pure_vector(psi_m, n)) / (2 * eps)

        dRe = np.max(np.abs(dG_dRe_a - dG_dRe_n))
        dIm = np.max(np.abs(dG_dIm_a - dG_dIm_n))
        scale = max(np.max(np.abs(dG_dRe_n)), np.max(np.abs(dG_dIm_n)), 1e-12)
        rel = max(dRe, dIm) / scale
        ok = rel < 1e-4
        n_pass += int(ok)
        print(f"    trial {trial+1}:  γ = {gamma:.4f}  "
              f"|Re diff| = {dRe:.2e}   |Im diff| = {dIm:.2e}   "
              f"rel = {rel:.2e}   {'✓' if ok else '✗'}")

    print(f"  Passed: {n_pass}/{n_trials}")
    return n_pass == n_trials


# ─────────────────────────────────────────────────────────────────────────
# Sanity check: γ on known states
# ─────────────────────────────────────────────────────────────────────────

def test_C_known_states():
    print(f"\n  TEST C: γ on analytically known states")
    print(f"  " + "-" * 70)

    # GHZ_3: γ should be 1
    n = 3
    ghz = np.zeros(2 ** n, dtype=complex)
    ghz[0] = 1 / np.sqrt(2)     # |000>
    ghz[-1] = 1 / np.sqrt(2)    # |111>
    gamma_ghz, _ = _meyerwallach_pure_with_grad(ghz, n)
    print(f"    |GHZ_3>:        γ = {gamma_ghz:.6f}   (expected 1.000000)   "
          f"{'✓' if abs(gamma_ghz - 1.0) < 1e-10 else '✗'}")

    # W_3 = (|001> + |010> + |100>) / √3: γ should be 8/9 ≈ 0.8889
    w = np.zeros(2 ** n, dtype=complex)
    w[1] = w[2] = w[4] = 1 / np.sqrt(3)
    gamma_w, _ = _meyerwallach_pure_with_grad(w, n)
    expected_w = 8 / 9
    print(f"    |W_3>:          γ = {gamma_w:.6f}   (expected {expected_w:.6f})   "
          f"{'✓' if abs(gamma_w - expected_w) < 1e-10 else '✗'}")

    # Product state |000>: γ should be 0
    prod = np.zeros(2 ** n, dtype=complex)
    prod[0] = 1.0
    gamma_prod, _ = _meyerwallach_pure_with_grad(prod, n)
    print(f"    |000>:          γ = {gamma_prod:.6f}   (expected 0.000000)   "
          f"{'✓' if abs(gamma_prod) < 1e-10 else '✗'}")

    # Product Bell pair ⊗ |0>: γ should be 2/3 (qubits 0,1 fully entangled, qubit 2 separable)
    bell = np.zeros(2 ** n, dtype=complex)
    # |Bell>_01 ⊗ |0>_2 = (|00>+|11>)/√2 ⊗ |0>
    # In big-endian: |000> + |110> → indices 0 and 6
    bell[0] = bell[6] = 1 / np.sqrt(2)
    gamma_bell, _ = _meyerwallach_pure_with_grad(bell, n)
    expected_bell = 2 / 3
    print(f"    Bell ⊗ |0>:    γ = {gamma_bell:.6f}   (expected {expected_bell:.6f})   "
          f"{'✓' if abs(gamma_bell - expected_bell) < 1e-10 else '✗'}")


# ─────────────────────────────────────────────────────────────────────────

def main():
    pA = test_A_value()
    pB = test_B_gradient()
    test_C_known_states()

    print(f"\n  " + "=" * 70)
    print(f"  Summary:")
    print(f"    Test A (value vs reference):  {'PASS' if pA else 'FAIL'}")
    print(f"    Test B (gradient vs FD):       {'PASS' if pB else 'FAIL'}")
    print(f"  " + "=" * 70)


if __name__ == "__main__":
    main()
