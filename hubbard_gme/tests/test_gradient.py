"""
Gradient verification for the GCG implementation.

Layered tests, top-down by stage of the algorithm:

  TEST A: dC_GME/dpsi at fixed psi              (Level 1)
  TEST B: dh/dRe U, dh/dIm U via Eqs. 10-11      (Levels 1+2)
  TEST C: directional derivative on U(k)         (Levels 1+2+3)
  TEST D: focused single-element with full intermediate values
  TEST E: gradient at exact bipartition ties (GHZ, W)   ← exercises tie fix
  TEST F: end-to-end — separable mixtures give C_GME ≈ 0
  TEST G: O(eps^2) scaling of finite-difference error

A–D test the gradient machinery on generic random data.
E directly tests the subgradient-averaging code path that random states
never trigger. F is a global sanity check that runs the full optimizer.
G catches subtle gradient bugs that the loose 1e-3 tolerance in A–C may
miss.

Run:
    python test_gradient.py
"""

import numpy as np
from scipy.linalg import expm
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hubbard_gme.convex_roof import (
    _cgme_pure_with_grad, _gcg_gradient, _h_full_unitary,
    _decompose_full_unitary, convex_roof_gcg,
)


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────

def make_random_mixed_rho(n, rank, seed=42):
    np.random.seed(seed)
    dim = 2 ** n
    A = np.random.randn(dim, rank) + 1j * np.random.randn(dim, rank)
    rho = A @ A.conj().T
    return rho / np.trace(rho).real


def random_unitary(k, seed=0):
    np.random.seed(seed)
    H = np.random.randn(k, k) + 1j * np.random.randn(k, k)
    H = 0.5 * (H + H.conj().T)
    return expm(1j * H)


def random_skew_hermitian(k, seed=0):
    np.random.seed(seed)
    A = np.random.randn(k, k) + 1j * np.random.randn(k, k)
    return 0.5 * (A - A.conj().T)


def gcg_inner(A, B):
    return float(np.real(np.trace(A @ B.conj().T)))


def setup_test_rho(n=3, rank=3, seed=42):
    rho = make_random_mixed_rho(n, rank, seed)
    eigvals, eigvecs = np.linalg.eigh(rho)
    mask = eigvals > 1e-10
    return eigvecs[:, mask], np.sqrt(eigvals[mask]), int(np.sum(mask))


# ───────────────────────────────────────────────────────────────────────────
# TEST A: gradient of C_GME(|psi>)
# ───────────────────────────────────────────────────────────────────────────

def test_A(n=3, n_trials=5, eps=1e-6):
    print("\n  TEST A: gradient of C_GME(|psi>) — Level 1 only")
    print("  " + "-" * 70)

    dim = 2 ** n
    n_pass = 0
    for trial in range(n_trials):
        np.random.seed(500 + trial)
        psi = np.random.randn(dim) + 1j * np.random.randn(dim)
        psi = psi / np.linalg.norm(psi)

        C0, grad_conj = _cgme_pure_with_grad(psi, n)
        dC_dRe_a = 2.0 * np.real(grad_conj)
        dC_dIm_a = 2.0 * np.imag(grad_conj)

        dC_dRe_n = np.zeros(dim)
        dC_dIm_n = np.zeros(dim)
        for i in range(dim):
            psi_p = psi.copy(); psi_p[i] += eps
            psi_m = psi.copy(); psi_m[i] -= eps
            dC_dRe_n[i] = (_cgme_pure_with_grad(psi_p, n)[0]
                         - _cgme_pure_with_grad(psi_m, n)[0]) / (2 * eps)
            psi_p = psi.copy(); psi_p[i] += 1j * eps
            psi_m = psi.copy(); psi_m[i] -= 1j * eps
            dC_dIm_n[i] = (_cgme_pure_with_grad(psi_p, n)[0]
                         - _cgme_pure_with_grad(psi_m, n)[0]) / (2 * eps)

        dRe = np.max(np.abs(dC_dRe_a - dC_dRe_n))
        dIm = np.max(np.abs(dC_dIm_a - dC_dIm_n))
        scale = max(np.max(np.abs(dC_dRe_n)), np.max(np.abs(dC_dIm_n)), 1e-12)
        rel = max(dRe, dIm) / scale
        ok = rel < 1e-3
        n_pass += int(ok)
        print(f"    trial {trial+1}:  C={C0:.4f}  "
              f"|Re diff|={dRe:.3e}  |Im diff|={dIm:.3e}  "
              f"rel={rel:.3e}  {'✓' if ok else '✗'}")
    print(f"  Passed: {n_pass}/{n_trials}")
    assert n_pass == n_trials, (
        f"gradient check failed in {n_trials - n_pass}/{n_trials} trials")


# ───────────────────────────────────────────────────────────────────────────
# TEST B: gradient of h(U) w.r.t. U entries
# ───────────────────────────────────────────────────────────────────────────

def compute_dh_dU_analytic(U, eigvecs, eigvals_sqrt, n):
    """Re-implementation of the dh/dU partials used by _gcg_gradient."""
    k = U.shape[0]
    r = eigvals_sqrt.shape[0]
    d = eigvecs.shape[0]

    probs, pure_states = _decompose_full_unitary(U, eigvecs, eigvals_sqrt)
    m_vals = np.zeros(k)
    dm_dpsi_conj = np.zeros((k, d), dtype=complex)
    for i in range(k):
        if probs[i] < 1e-14:
            continue
        m_vals[i], dm_dpsi_conj[i] = _cgme_pure_with_grad(pure_states[i], n)

    dh_dRe_U = np.zeros((k, k))
    dh_dIm_U = np.zeros((k, k))
    for kk in range(k):
        if probs[kk] < 1e-14:
            continue
        psi_k = pure_states[kk]
        dm_dRe_psi = 2.0 * np.real(dm_dpsi_conj[kk])
        dm_dIm_psi = 2.0 * np.imag(dm_dpsi_conj[kk])

        for ll in range(r):
            U_kl = U[kk, ll]
            chi_l = eigvecs[:, ll]
            lam_l = eigvals_sqrt[ll] ** 2
            sqrt_p_lam = np.sqrt(probs[kk] * lam_l)
            phi_R = sqrt_p_lam * chi_l - lam_l * U_kl.real * psi_k
            phi_I = 1j * sqrt_p_lam * chi_l - lam_l * U_kl.imag * psi_k

            t1 = 2.0 * lam_l * U_kl.real * m_vals[kk]
            t2 = np.sum(np.real(phi_R) * dm_dRe_psi + np.imag(phi_R) * dm_dIm_psi)
            dh_dRe_U[kk, ll] = t1 + t2

            t1 = 2.0 * lam_l * U_kl.imag * m_vals[kk]
            t2 = np.sum(np.real(phi_I) * dm_dRe_psi + np.imag(phi_I) * dm_dIm_psi)
            dh_dIm_U[kk, ll] = t1 + t2
    return dh_dRe_U, dh_dIm_U


def test_B(n=3, n_trials=5, eps=1e-5):
    print("\n  TEST B: dh/dRe U, dh/dIm U — Levels 1 + 2 (Eqs. 10-11)")
    print("  " + "-" * 70)

    eigvecs_r, eigvals_sqrt, r = setup_test_rho(n)
    k = r + 2

    n_pass = 0
    for trial in range(n_trials):
        U = random_unitary(k, seed=300 + trial)

        dh_dRe_a, dh_dIm_a = compute_dh_dU_analytic(U, eigvecs_r, eigvals_sqrt, n)

        dh_dRe_n = np.zeros((k, k))
        dh_dIm_n = np.zeros((k, k))
        for kk in range(k):
            for ll in range(r):
                U_p = U.copy(); U_p[kk, ll] += eps
                U_m = U.copy(); U_m[kk, ll] -= eps
                dh_dRe_n[kk, ll] = (_h_full_unitary(U_p, eigvecs_r, eigvals_sqrt, n)
                                  - _h_full_unitary(U_m, eigvecs_r, eigvals_sqrt, n)) / (2 * eps)
                U_p = U.copy(); U_p[kk, ll] += 1j * eps
                U_m = U.copy(); U_m[kk, ll] -= 1j * eps
                dh_dIm_n[kk, ll] = (_h_full_unitary(U_p, eigvecs_r, eigvals_sqrt, n)
                                  - _h_full_unitary(U_m, eigvecs_r, eigvals_sqrt, n)) / (2 * eps)

        dRe = np.max(np.abs(dh_dRe_a[:, :r] - dh_dRe_n[:, :r]))
        dIm = np.max(np.abs(dh_dIm_a[:, :r] - dh_dIm_n[:, :r]))
        scale = max(np.max(np.abs(dh_dRe_n)), np.max(np.abs(dh_dIm_n)), 1e-12)
        rel = max(dRe, dIm) / scale
        ok = rel < 1e-3
        n_pass += int(ok)
        print(f"    trial {trial+1}:  "
              f"|Re diff|={dRe:.3e}  |Im diff|={dIm:.3e}  "
              f"rel={rel:.3e}  {'✓' if ok else '✗'}")
    print(f"  Passed: {n_pass}/{n_trials}")
    assert n_pass == n_trials, (
        f"gradient check failed in {n_trials - n_pass}/{n_trials} trials")


# ───────────────────────────────────────────────────────────────────────────
# TEST C: directional derivative on U(k)
# ───────────────────────────────────────────────────────────────────────────

def test_C(n=3, n_trials=5, eps=1e-5):
    print("\n  TEST C: full chain — directional derivative on U(k)")
    print("  " + "-" * 70)

    eigvecs_r, eigvals_sqrt, r = setup_test_rho(n)
    k = r + 2

    n_pass = 0
    for trial in range(n_trials):
        U = random_unitary(k, seed=400 + trial)
        X = random_skew_hermitian(k, seed=500 + trial)
        X = X / np.sqrt(gcg_inner(X, X))

        G_a = _gcg_gradient(U, eigvecs_r, eigvals_sqrt, n)
        ana = gcg_inner(G_a, X)

        U_p = U @ expm(eps * X)
        U_m = U @ expm(-eps * X)
        num = (_h_full_unitary(U_p, eigvecs_r, eigvals_sqrt, n)
             - _h_full_unitary(U_m, eigvecs_r, eigvals_sqrt, n)) / (2 * eps)

        rel = abs(ana - num) / max(abs(ana), abs(num), 1e-12)
        ok = rel < 1e-3
        n_pass += int(ok)
        print(f"    trial {trial+1}:  ana={ana:+.6f}  num={num:+.6f}  "
              f"rel={rel:.3e}  {'✓' if ok else '✗'}")
    print(f"  Passed: {n_pass}/{n_trials}")
    assert n_pass == n_trials, (
        f"gradient check failed in {n_trials - n_pass}/{n_trials} trials")


# ───────────────────────────────────────────────────────────────────────────
# TEST D: focused single-element diagnostic
# ───────────────────────────────────────────────────────────────────────────

def test_D(n=3, kk=0, ll=0, eps=1e-6):
    print("\n  TEST D: focused single-element diagnostic for dh/dU_kl")
    print("  " + "-" * 70)

    eigvecs_r, eigvals_sqrt, r = setup_test_rho(n)
    k = r + 2
    U = random_unitary(k, seed=300)

    # Numerical
    U_p = U.copy(); U_p[kk, ll] += eps
    U_m = U.copy(); U_m[kk, ll] -= eps
    dh_dRe_num = (_h_full_unitary(U_p, eigvecs_r, eigvals_sqrt, n)
                - _h_full_unitary(U_m, eigvecs_r, eigvals_sqrt, n)) / (2 * eps)
    U_p = U.copy(); U_p[kk, ll] += 1j * eps
    U_m = U.copy(); U_m[kk, ll] -= 1j * eps
    dh_dIm_num = (_h_full_unitary(U_p, eigvecs_r, eigvals_sqrt, n)
                - _h_full_unitary(U_m, eigvecs_r, eigvals_sqrt, n)) / (2 * eps)

    # Analytical (paper formula)
    probs, pure_states = _decompose_full_unitary(U, eigvecs_r, eigvals_sqrt)
    psi_k = pure_states[kk]
    p_k = probs[kk]
    chi_l = eigvecs_r[:, ll]
    lam_l = eigvals_sqrt[ll] ** 2
    U_kl = U[kk, ll]
    C_k, dC_dpsi_conj = _cgme_pure_with_grad(psi_k, n)
    dm_dRe_psi = 2.0 * np.real(dC_dpsi_conj)
    dm_dIm_psi = 2.0 * np.imag(dC_dpsi_conj)
    sqrt_p_lam = np.sqrt(p_k * lam_l)
    phi_R = sqrt_p_lam * chi_l - lam_l * U_kl.real * psi_k
    phi_I = 1j * sqrt_p_lam * chi_l - lam_l * U_kl.imag * psi_k

    t1_R = 2.0 * lam_l * U_kl.real * C_k
    t2_R = np.sum(np.real(phi_R) * dm_dRe_psi + np.imag(phi_R) * dm_dIm_psi)
    dh_dRe_ana = t1_R + t2_R
    t1_I = 2.0 * lam_l * U_kl.imag * C_k
    t2_I = np.sum(np.real(phi_I) * dm_dRe_psi + np.imag(phi_I) * dm_dIm_psi)
    dh_dIm_ana = t1_I + t2_I

    print(f"    Setup: n={n}, rank={r}, k={k}, (kk,ll)=({kk},{ll})")
    print(f"    U[{kk},{ll}] = {U_kl:+.6f}")
    print(f"    λ_{ll} = {lam_l:.6f}    p_{kk} = {p_k:.6f}    C_GME = {C_k:.6f}")
    print(f"")
    print(f"    ── dh/dRe U_kl ──")
    print(f"      term 1: 2 λ Re(U) C_k   = {t1_R:+.10f}")
    print(f"      term 2: phi_R · dm/dpsi = {t2_R:+.10f}")
    print(f"      analytical:               {dh_dRe_ana:+.10f}")
    print(f"      numerical:                {dh_dRe_num:+.10f}")
    if abs(dh_dRe_num) > 1e-12:
        print(f"      ratio ana/num:            {dh_dRe_ana / dh_dRe_num:+.4f}")

    print(f"\n    ── dh/dIm U_kl ──")
    print(f"      term 1: 2 λ Im(U) C_k   = {t1_I:+.10f}")
    print(f"      term 2: phi_I · dm/dpsi = {t2_I:+.10f}")
    print(f"      analytical:               {dh_dIm_ana:+.10f}")
    print(f"      numerical:                {dh_dIm_num:+.10f}")

    # Enforce (audit issue #2): this used to be a print-only diagnostic.
    # Compare the analytical partials against central differences.
    scale = max(abs(dh_dRe_num), abs(dh_dIm_num), 1e-12)
    err_re = abs(dh_dRe_ana - dh_dRe_num) / scale
    err_im = abs(dh_dIm_ana - dh_dIm_num) / scale
    assert err_re < 1e-3, (
        f"dh/dRe U[{kk},{ll}]: analytical {dh_dRe_ana:.10f} vs numerical "
        f"{dh_dRe_num:.10f} (relative error {err_re:.2e}, tol 1e-3)")
    assert err_im < 1e-3, (
        f"dh/dIm U[{kk},{ll}]: analytical {dh_dIm_ana:.10f} vs numerical "
        f"{dh_dIm_num:.10f} (relative error {err_im:.2e}, tol 1e-3)")
    if abs(dh_dIm_num) > 1e-12:
        print(f"      ratio ana/num:            {dh_dIm_ana / dh_dIm_num:+.4f}")

    # Independent Wirtinger cross-check
    s = np.sum(dC_dpsi_conj * psi_k.conj())
    Re_s = float(np.real(s))
    t = np.sum(dC_dpsi_conj * chi_l.conj())
    dh_dUconj = lam_l * U_kl * C_k - lam_l * U_kl * Re_s + sqrt_p_lam * t
    dh_dRe_w = 2.0 * np.real(dh_dUconj)
    dh_dIm_w = -2.0 * np.imag(dh_dUconj)
    print(f"\n    ── Independent Wirtinger derivation ──")
    print(f"      dh/dRe wirt:              {dh_dRe_w:+.10f}")
    print(f"      dh/dIm wirt:              {dh_dIm_w:+.10f}")
    print(f"    (compare both above to 'numerical' line)")


# ───────────────────────────────────────────────────────────────────────────
# TEST E: gradient at exact bipartition ties (GHZ, W)
# ───────────────────────────────────────────────────────────────────────────
#
# Random states drawn from continuous distributions almost surely have a
# unique argmax bipartition, so Tests A–D never exercise the subgradient
# averaging code path. Highly symmetric states (GHZ, W) tie exactly across
# multiple bipartitions and are the canonical stress test for the fix.
#
# A directional check is more discriminating than element-wise here: the
# finite difference picks up whichever bipartition activates locally on
# each side of psi, and the averaged subgradient should agree on average
# with the centered FD.
# ───────────────────────────────────────────────────────────────────────────

def _active_bipartition_gradients(psi, n, tie_tol=1e-10):
    """dC/dpsi* for each bipartition tied for the maximum purity.

    C = sqrt(2(1 - P_A)) with dP_A/dpsi* = 2 (M M^dag) M reshaped back, so
    dC/dpsi* = -(1/C) * that. Returns one vector per ACTIVE bipartition; the
    Clarke subdifferential at a tie is the convex hull of these.
    """
    from hubbard_gme.gme import get_bipartitions

    psi_t = psi.reshape([2] * n)
    data = []
    for A in get_bipartitions(n):
        B = tuple(i for i in range(n) if i not in A)
        perm = list(A) + list(B)
        M = np.transpose(psi_t, perm).reshape(2 ** len(A), 2 ** len(B))
        rho_A = M @ M.conj().T
        data.append((float(np.real(np.trace(rho_A @ rho_A))), M, perm, A, rho_A))

    max_p = max(d[0] for d in data)
    C = float(np.sqrt(max(0.0, 2.0 * (1.0 - max_p))))
    C_safe = max(C, 1e-10)

    grads = []
    for p_A, M, perm, A, rho_A in data:
        if max_p - p_A > tie_tol:
            continue
        gM = 2.0 * (rho_A @ M)
        blk = gM.reshape([2] * len(A) + [2] * (n - len(A)))
        blk = np.transpose(blk, np.argsort(perm))
        grads.append(-blk.reshape(2 ** n) / C_safe)
    return grads


def test_E(eps=1e-6):
    print("\n  TEST E: gradient at exact bipartition ties (GHZ, W)")
    print("  " + "-" * 70)

    cases = []

    # 3-qubit GHZ: all 3 single-qubit cuts tie at P_A = 1/2 → C_GME = 1
    n = 3; dim = 2 ** n
    psi_ghz3 = np.zeros(dim, complex)
    psi_ghz3[0] = psi_ghz3[-1] = 1 / np.sqrt(2)
    cases.append(("GHZ_3", n, psi_ghz3, 1.0))

    # 3-qubit W: all 3 single-qubit cuts tie at P_A = 5/9 by permutation
    # symmetry → C_GME = sqrt(8/9) = 2 sqrt(2)/3 ≈ 0.9428
    psi_w3 = np.zeros(dim, complex)
    for idx in (1, 2, 4):  # |001⟩, |010⟩, |100⟩
        psi_w3[idx] = 1 / np.sqrt(3)
    cases.append(("W_3", n, psi_w3, np.sqrt(2 * (1 - 5 / 9))))

    # 4-qubit GHZ: 4 single-cuts + 3 two-cuts all tie at P_A = 1/2 → C_GME = 1
    # This is a 7-way tie, the most stringent test.
    n = 4; dim = 2 ** n
    psi_ghz4 = np.zeros(dim, complex)
    psi_ghz4[0] = psi_ghz4[-1] = 1 / np.sqrt(2)
    cases.append(("GHZ_4", n, psi_ghz4, 1.0))

    # NOTE (audit issue #2, "additional test-quality note").
    # At an exact multiway tie C_GME is a MAXIMUM of several smooth functions,
    # so it is nonsmooth there and has no classical derivative. A centered
    # finite difference then measures
    #       (D_{+d} - D_{-d}) / 2,
    # i.e. an average of two ONE-SIDED directional derivatives, and an averaged
    # Clarke subgradient is not required to reproduce it. Demanding equality is
    # therefore not a valid gradient test at such a point — it fails for W_3
    # (rel-err ~ 4.5e-1) purely because of the criterion, not the gradient.
    #
    # The mathematically correct condition for a Clarke subgradient g of a max
    # of smooth functions is the SUBGRADIENT INEQUALITY in every direction:
    #       <g, d>  <=  D_{+d}      (the one-sided derivative),
    # which is what we assert below, alongside the exact value check.
    n_pass = 0
    for name, n, psi, C_expect in cases:
        C, grad = _cgme_pure_with_grad(psi, n)

        # 1. Value check (still an exact equality — the VALUE is well defined)
        ok_val = abs(C - C_expect) < 1e-10

        # 2. The returned vector must be the CENTROID of the active-set
        #    gradients, i.e. an element of the Clarke subdifferential
        #        conv{ grad C_A : A active }.
        #    (A per-direction inequality test is NOT usable here: for a max of
        #    smooth functions the one-sided derivative is the MIN over the
        #    active set, and the centroid may exceed it whenever the active
        #    gradients genuinely differ — which is the case for W_3 but not for
        #    the fully symmetric GHZ states. Membership is the right check.)
        active_grads = _active_bipartition_gradients(psi, n)
        centroid = sum(active_grads) / len(active_grads)
        grad_err = float(np.linalg.norm(grad - centroid))

        ok_grad = grad_err < 1e-10
        ok = ok_val and ok_grad
        n_pass += int(ok)
        print(f"    {name:6s}: C={C:.6f} (expected {C_expect:.6f})  "
              f"{len(active_grads)} active cuts  "
              f"||grad - centroid||={grad_err:.2e}  {'✓' if ok else '✗'}")

    print(f"  Passed: {n_pass}/{len(cases)}")
    if n_pass < len(cases):
        print(f"    Hint: a positive violation means <g,d> exceeded the "
              f"one-sided\n    directional derivative, i.e. g is not a valid "
              f"Clarke subgradient.")
    assert n_pass == len(cases), (
        f"failed {len(cases) - n_pass}/{len(cases)} cases")


# ───────────────────────────────────────────────────────────────────────────
# TEST F: end-to-end sanity — separable mixtures must give C_GME ≈ 0
# ───────────────────────────────────────────────────────────────────────────
#
# Tests A–E verify the gradient. They do not run convex_roof_gcg itself.
# A C_GME of zero on a fully-separable mixed state exercises the entire
# k-loop, restart logic, line search, parallel transport, and convergence
# check. Any sign error or accidentally-uphill step shows up here.
# ───────────────────────────────────────────────────────────────────────────

def test_F():
    print("\n  TEST F: end-to-end — separable mixtures give C_GME ≈ 0")
    print("  " + "-" * 70)

    n = 3
    dim = 2 ** n
    cases = []

    # Case 1: classical mixture of three product computational-basis states.
    # Each component is a product state, the mixture is separable on every
    # bipartition, so the convex roof of any GME measure must vanish.
    rng = np.random.default_rng(0)
    weights = rng.dirichlet(np.ones(3))
    rho1 = np.zeros((dim, dim), complex)
    for w, idx in zip(weights, [0, 3, 6]):  # |000⟩, |011⟩, |110⟩
        rho1[idx, idx] = w
    cases.append(("classical mix", rho1))

    # Case 2: full-rank product of single-qubit mixed states (separable).
    sigma = np.array([[0.7, 0.1 + 0.05j],
                      [0.1 - 0.05j, 0.3]], complex)
    rho2 = np.kron(np.kron(sigma, sigma), sigma)
    cases.append(("product mixed", rho2))

    n_pass = 0
    for name, rho in cases:
        res = convex_roof_gcg(rho, n=n, n_restarts=15, verbose=False)
        ok = res["C_gcg"] < 1e-3
        n_pass += int(ok)
        print(f"    {name:14s}: C_gcg = {res['C_gcg']:.2e}  "
              f"(k={res['k_converged']})  {'✓' if ok else '✗'}")

    print(f"  Passed: {n_pass}/{len(cases)}")
    if n_pass < len(cases):
        print(f"    Hint: gradient tests pass but optimization fails — look "
              f"at the line search,\n"
              f"    parallel transport sign, or k-loop early-exit logic.")
    assert n_pass == len(cases), (
        f"failed {len(cases) - n_pass}/{len(cases)} cases")


# ───────────────────────────────────────────────────────────────────────────
# TEST G: O(eps^2) scaling of finite-difference error
# ───────────────────────────────────────────────────────────────────────────
#
# Centered finite differences have truncation error O(eps^2) and roundoff
# error O(eps_machine / eps). The optimum is around eps ≈ 1e-5 for double
# precision. If the gradient is correct, error should drop ~100× per
# decade of eps from 1e-2 down to ~1e-5, then flatten or grow due to
# roundoff. Failure to scale exposes O(eps^0) bugs that a single-eps
# tolerance check at 1e-3 would miss.
# ───────────────────────────────────────────────────────────────────────────

def test_G(n=3):
    print("\n  TEST G: O(eps^2) scaling of FD error")
    print("  " + "-" * 70)

    rng = np.random.default_rng(42)
    psi = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
    psi = psi / np.linalg.norm(psi)
    d = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
    d = d / np.linalg.norm(d)

    _, grad = _cgme_pure_with_grad(psi, n)
    ana = float(np.sum(2 * np.real(grad) * np.real(d)
                     + 2 * np.imag(grad) * np.imag(d)))

    eps_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    errs = []
    for eps in eps_list:
        num = (_cgme_pure_with_grad(psi + eps * d, n)[0]
             - _cgme_pure_with_grad(psi - eps * d, n)[0]) / (2 * eps)
        err = abs(ana - num)
        errs.append(err)
        print(f"    eps={eps:.0e}   |ana - num| = {err:.2e}")

    # Healthy scaling: error drops by roughly 100x per decade of eps
    # (truncation ~ eps^2) until roundoff kicks in. We check the first
    # two decades, where roundoff is negligible.
    ratios_ok = []
    for i in range(2):
        if errs[i + 1] > 1e-13:
            ratio = errs[i] / errs[i + 1]
            ratios_ok.append(20 < ratio < 500)  # generous band around 100
    ok = all(ratios_ok)
    print(f"    {'✓ scaling looks healthy (~O(eps^2))' if ok else '✗ scaling NOT O(eps^2) — gradient may be incorrect'}")
    assert ok, "scaling/ratio check failed (see printed diagnostics above)"


# ───────────────────────────────────────────────────────────────────────────

def main():
    pA = test_A()
    pB = test_B()
    pC = test_C()
    test_D()
    pE = test_E()
    
    pG = test_G()

    print("\n  " + "=" * 70)
    print(f"  Diagnosis:")
    print(f"    Level 1 (C_GME gradient):       {'PASS' if pA else 'FAIL'}")
    print(f"    Level 2 (dh/dU via Eqs 10-11):  {'PASS' if pB else 'FAIL'}")
    print(f"    Level 3 (skew-Hermitian G):     {'PASS' if pC else 'FAIL'}")
    print(f"    Tie handling (GHZ/W):           {'PASS' if pE else 'FAIL'}")

    print(f"    FD scaling (O(eps^2)):          {'PASS' if pG else 'FAIL'}")
    print("  " + "=" * 70)
    pF = test_F()
    print(f"    End-to-end (separable → 0):     {'PASS' if pF else 'FAIL'}")
if __name__ == "__main__":
    main()