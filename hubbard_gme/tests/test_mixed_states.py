"""
test_mixed_states.py
====================
Benchmark tests for C_GME convex-roof on MIXED states.
Imports directly from the existing convex_roof.py module.

Five tests — four with EXACT proven analytical answers, one exploratory:

  Test A — Pure state sanity check
           rho = |GHZ><GHZ|
           Expected: convex roof = direct formula = 1.0

  Test M1 — Separable mixture (diagonal in computational basis)
            rho = 0.3|000><000| + 0.2|011><011| + 0.5|101><101|
            Expected: C_GME(rho) = 0  (exactly)

  Test M2 — GHZ+/GHZ- mixture  [PRIMARY BENCHMARK]
            rho(p) = p*|GHZ+><GHZ+| + (1-p)*|GHZ-><GHZ-|
            Expected: C_GME(rho) = |2p - 1|  (exactly)

            This is the GOLD-STANDARD mixed-state benchmark because:
            - The eigendecomposition gives C_GME = 1 (both eigenstates are GHZ)
            - The exact convex roof is |2p-1|.
            - In particular, at p=1/2 the state reduces to
              0.5|000><000| + 0.5|111><111|, so C_GME = 0,
              while the eigendecomposition still gives 1.
            - So if the optimizer reproduces the exact M2 curve, then it is not
              merely returning the eigendecomposition value.

  Test M3 — General GHZ-subspace cat state
            rho = alpha|000><000| + (1-alpha)|111><111|
                  + gamma|000><111| + gamma*|111><000|
            with |gamma|^2 <= alpha(1-alpha)  [positivity]
            Expected: C_GME(rho) = 2|gamma|  (exactly)

            Supports complex gamma. Positivity checked with abs(gamma)**2.

  Test D — p*|GHZ><GHZ| + (1-p)*|W><W|  (exploratory, no analytical answer)
           Note: This is only an exploratory consistency check against the
           eigendecomposition upper bound; it is not an exact correctness
           benchmark.

NOTE ON REPRODUCIBILITY:
  np.random.seed(42) is used for reproducibility. This only controls NumPy's
  global RNG. If convex_roof_numerical or convex_roof_stiefel use
  np.random.default_rng() internally without accepting a seed argument, then
  full reproducibility is not guaranteed. Check convex_roof.py to verify.

HOW TO RUN:
  python test_mixed_states.py

Or with pytest:
  pytest test_mixed_states.py -v
"""

import numpy as np
import sys
import os

# ── Import from your existing convex_roof module ──────────────────────────────
# Adds src/ to path so hubbard_gme package is found when running directly.
# When running via pytest from project root, this is not needed.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src'))

from hubbard_gme.convex_roof import (
    convex_roof_numerical,      # exp(iH) parametrisation
    convex_roof_stiefel,        # Euler-Hurwitz / Algorithm B
    convex_roof_gcg,            # U(k) GCG / Algorithm A — the PRODUCTION solver
    convex_roof_upper_bound,    # eigendecomposition upper bound
    _cgme_pure_vector,          # direct pure-state formula
)


# ══════════════════════════════════════════════════════════════════════════════
# DENSITY MATRIX VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def assert_valid_density_matrix(rho, name="rho", atol=1e-10):
    """
    Verify that rho is a valid density matrix (Hermitian, trace 1, positive semidefinite).
    Useful for catching silent mistakes in benchmark construction.

    Args:
        rho: density matrix to check
        name: name for error messages
        atol: absolute tolerance

    Raises:
        AssertionError if any condition fails
    """
    # Check Hermitian
    assert np.allclose(rho, rho.conj().T, atol=atol), \
        f"{name} is not Hermitian: ||rho - rho†|| = {np.max(np.abs(rho - rho.conj().T))}"

    # Check trace = 1
    trace = np.trace(rho)
    assert abs(trace - 1.0) < atol, \
        f"trace({name}) = {trace} != 1"

    # Check positive semidefinite
    evals = np.linalg.eigvalsh(rho)
    min_eval = np.min(evals)
    assert min_eval >= -atol, \
        f"{name} is not positive semidefinite, min eigenvalue = {min_eval}"


# ══════════════════════════════════════════════════════════════════════════════
# STATE BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def ket(*bits):
    """Computational basis state. ket(1,0,0) -> |100>."""
    n   = len(bits)
    idx = sum(b * (2**(n-1-i)) for i, b in enumerate(bits))
    v   = np.zeros(2**n, dtype=complex)
    v[idx] = 1.0
    return v


def normalise(v):
    """Normalize a vector. Raises AssertionError for zero vectors."""
    nrm = np.linalg.norm(v)
    assert nrm > 0, "Cannot normalize zero vector"
    return v / nrm


def dm(psi):
    """Pure state density matrix."""
    return np.outer(psi, psi.conj())


def mix(states_and_probs):
    """
    Mixed state density matrix from pure-state ensemble.
    states_and_probs: list of (probability, state_vector) tuples.

    Checks that:
      - probabilities are non-negative and sum to 1
      - all state vectors have the same dimension
    """
    # Probability sanity check (catches accidental mistakes in benchmark construction)
    probs = [p for p, _ in states_and_probs]
    assert all(p >= -1e-12 for p in probs), \
        f"Negative probability in mix: {probs}"
    assert abs(sum(probs) - 1.0) < 1e-10, \
        f"Probabilities in mix do not sum to 1: sum = {sum(probs)}"

    # Dimension consistency check
    d = len(states_and_probs[0][1])
    assert all(len(psi) == d for _, psi in states_and_probs), \
        "State vectors in mix() do not all have the same dimension"

    rho = np.zeros((d, d), dtype=complex)
    for p, psi in states_and_probs:
        rho += p * dm(psi)
    return rho


def make_ghz_plus(n=3):
    """|GHZ+> = (|00...0> + |11...1>) / sqrt(2)"""
    zeros = ket(*([0]*n))
    ones  = ket(*([1]*n))
    return normalise(zeros + ones)


def make_ghz_minus(n=3):
    """|GHZ-> = (|00...0> - |11...1>) / sqrt(2)"""
    zeros = ket(*([0]*n))
    ones  = ket(*([1]*n))
    return normalise(zeros - ones)


def make_w():
    """|W> = (|100> + |010> + |001>) / sqrt(3)"""
    return normalise(ket(1,0,0) + ket(0,1,0) + ket(0,0,1))


def make_cat_state(alpha, gamma):
    """
    General GHZ-subspace cat state:
      rho = alpha|000><000| + (1-alpha)|111><111|
            + gamma|000><111| + gamma*|111><000|

    Positivity requires |gamma|^2 <= alpha*(1-alpha).
    Expected C_GME = 2|gamma|.

    Supports complex gamma.
    """
    k000 = ket(0,0,0)
    k111 = ket(1,1,1)
    rho = (alpha * np.outer(k000, k000.conj())
           + (1 - alpha) * np.outer(k111, k111.conj())
           + gamma * np.outer(k000, k111.conj())
           + np.conj(gamma) * np.outer(k111, k000.conj()))
    return rho


def format_gamma(gamma):
    """
    Format gamma for display, handling both real and complex cases.
    Uses +.2f for imaginary part so negative values display correctly
    (e.g., "0.20-0.30j" instead of "0.20+-0.30j").
    """
    gamma_c = complex(gamma)
    if abs(gamma_c.imag) < 1e-12:
        return f"{gamma_c.real:.4f}"
    else:
        return f"{gamma_c.real:.2f}{gamma_c.imag:+.2f}j"


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: run both methods, check each separately
# ══════════════════════════════════════════════════════════════════════════════

def get_convex_roof_values(rho, n, n_restarts=40):
    """
    Run both methods and return their values separately.
    This allows validation of BOTH methods independently.

    Returns:
        (C_num, C_st, min_value)
        where min_value = min(C_num, C_st) is for reference/display only.
        Correctness checks should use C_num and C_st separately.
    """
    res_num = convex_roof_numerical(
        rho, n,
        n_restarts=n_restarts,
        method="L-BFGS-B",
        tol=1e-12,
    )
    C_num = res_num["C_numerical"]

    res_st = convex_roof_stiefel(
        rho, n,
        n_restarts=n_restarts,
        tol=1e-12,
        tol_k=1e-6,
        k_max_extra=4,
        verbose=False,
    )
    C_st = res_st["C_stiefel"]

    # GCG (Algorithm A) is the solver the production pipeline uses
    # (--thermal-solver gcg), so it must be exercised here too.
    res_gcg = convex_roof_gcg(
        rho, n,
        n_restarts=n_restarts,
        tol=1e-12,
        tol_k=1e-6,
        k_max_extra=4,
        verbose=False,
    )
    C_gcg = res_gcg["C_gcg"]

    # The convex roof is a MINIMUM, and each solver returns an upper bound on
    # it, so the smallest of the three is the best estimate. Returned for
    # reference/display only — correctness checks use the values separately.
    return C_num, C_st, C_gcg, min(C_num, C_st, C_gcg)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════════

def run_tests(n_restarts=40, tol_pass=1e-3):

    np.random.seed(42)
    all_passed = True

    print("=" * 80)
    print("  MIXED STATE BENCHMARK TESTS FOR C_GME CONVEX ROOF")
    print("  Methods: convex_roof_numerical  +  convex_roof_stiefel")
    print("=" * 80)

    # ── Test A — Pure state ───────────────────────────────────────────────────
    print()
    print("TEST A — Pure state sanity check: rho = |GHZ+><GHZ+|")
    print("  Expected: convex roof = direct formula = 1.0")
    print()

    ghz_p  = make_ghz_plus(3)
    rho    = dm(ghz_p)
    assert_valid_density_matrix(rho, "Test A: rho_GHZ")
    direct = _cgme_pure_vector(ghz_p, 3)

    C_num, C_st, C_gcg, _ = get_convex_roof_values(rho, 3, n_restarts)
    err_num = abs(C_num - direct)
    err_st  = abs(C_st - direct)
    err_gcg = abs(C_gcg - direct)
    passed_num = err_num < tol_pass
    passed_st  = err_st < tol_pass
    passed_gcg = err_gcg < tol_pass
    passed = passed_num and passed_st and passed_gcg
    all_passed = all_passed and passed

    print(f"  Direct formula            = {direct:.8f}")
    print(f"  Exp map (conv_roof_num)   = {C_num:.8f}  Error = {err_num:.2e}  {'✓' if passed_num else '✗'}")
    print(f"  Stiefel (conv_roof_st)    = {C_st:.8f}  Error = {err_st:.2e}  {'✓' if passed_st else '✗'}")
    print(f"  GCG     (conv_roof_gcg)    = {C_gcg:.8f}  Error = {err_gcg:.2e}  {'✓' if passed_gcg else '✗'}")
    print(f"  Result: {'PASSED ✓' if passed else 'FAILED ✗'}")

    # ── Test M1 — Separable diagonal mixture ──────────────────────────────────
    print()
    print("TEST M1 — Separable mixture: 0.3|000><000| + 0.2|011><011| + 0.5|101><101|")
    print("  Expected: C_GME(rho) = 0  (each component is a product state)")
    print()

    rho = mix([(0.3, ket(0,0,0)),
               (0.2, ket(0,1,1)),
               (0.5, ket(1,0,1))])
    assert_valid_density_matrix(rho, "Test M1: rho_separable")

    C_num, C_st, C_gcg, _ = get_convex_roof_values(rho, 3, n_restarts)
    err_num = abs(C_num - 0.0)
    err_st  = abs(C_st - 0.0)
    err_gcg = abs(C_gcg - 0.0)
    # Looser tolerance for zero targets (optimizer residual)
    passed_num = err_num < 1e-6
    passed_st  = err_st < 1e-6
    passed_gcg = err_gcg < 1e-6
    passed = passed_num and passed_st and passed_gcg
    all_passed = all_passed and passed

    print(f"  Expected                  = 0.00000000")
    print(f"  Exp map (conv_roof_num)   = {C_num:.8f}  Error = {err_num:.2e}  {'✓' if passed_num else '✗'}")
    print(f"  Stiefel (conv_roof_st)    = {C_st:.8f}  Error = {err_st:.2e}  {'✓' if passed_st else '✗'}")
    print(f"  GCG     (conv_roof_gcg)    = {C_gcg:.8f}  Error = {err_gcg:.2e}  {'✓' if passed_gcg else '✗'}")
    print(f"  Result: {'PASSED ✓' if passed else 'FAILED ✗'}")

    # ── Test M2 — GHZ+/GHZ- mixture (PRIMARY BENCHMARK) ───────────────────────
    print()
    print("TEST M2 — PRIMARY BENCHMARK: p*|GHZ+><GHZ+| + (1-p)*|GHZ-><GHZ-|")
    print("  C_GME(|GHZ+>) = C_GME(|GHZ->) = 1.0")
    print("  Expected: C_GME(rho) = |2p - 1|  (EXACT, proven)")
    print()
    print("  Note: eigendecomposition gives C_GME = 1 at all p.")
    print("  True answer requires optimizer to find optimal decomposition.")
    print()
    print(f"  {'p':>6}  {'Expected':>10}  {'Exp map':>10}  "
          f"{'Stiefel':>10}  {'Err_num':>10}  {'Err_st':>10}  Status")
    print("  " + "-"*80)

    m2_pass = True
    m2_p_values = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

    for p in m2_p_values:
        ghz_p_state = make_ghz_plus(3)
        ghz_m_state = make_ghz_minus(3)
        rho = mix([(p, ghz_p_state), (1-p, ghz_m_state)])
        assert_valid_density_matrix(rho, f"Test M2 at p={p}")

        C_num, C_st, C_gcg, _ = get_convex_roof_values(rho, 3, n_restarts)
        exp  = abs(2*p - 1)
        err_num = abs(C_num - exp)
        err_st  = abs(C_st - exp)
        err_gcg = abs(C_gcg - exp)
        ok_num = err_num < tol_pass
        ok_st  = err_st < tol_pass
        ok_gcg = err_gcg < tol_pass
        ok = ok_num and ok_st and ok_gcg
        if not ok:
            m2_pass = False
        status = "✓✓" if (ok_num and ok_st) else ("✓✗" if ok_num else ("✗✓" if ok_st else "✗✗"))
        print(f"  {p:>6.2f}  {exp:>10.6f}  {C_num:>10.6f}  "
              f"{C_st:>10.6f}  {err_num:>10.2e}  {err_st:>10.2e}  {status}")

    all_passed = all_passed and m2_pass
    print(f"\n  Result: {'PASSED ✓' if m2_pass else 'FAILED ✗'}")

    # ── Test M3 — General GHZ-subspace cat state ─────────────────────────────
    print()
    print("TEST M3 — General cat state: alpha|000><000| + (1-alpha)|111><111|")
    print("                            + gamma|000><111| + gamma*|111><000|")
    print("  Expected: C_GME(rho) = 2|gamma|  (EXACT)")
    print("  Positivity requires |gamma|^2 <= alpha(1-alpha).")
    print()
    print(f"  {'alpha':>7}  {'gamma':>15}  {'Expected':>10}  {'Exp map':>10}  "
          f"{'Stiefel':>10}  {'Err_num':>10}  {'Err_st':>10}  Status")
    print("  " + "-"*100)

    m3_pass = True
    m3_params = [
        (0.5, 0.5),                       # Reduces to M2 at p=1  (pure GHZ+)
        (0.5, 0.3),
        (0.5, 0.1),
        (0.3, 0.15),                      # from the guide
        (0.7, 0.2),
        (0.3, 0.3),                       # max: |gamma|^2 = alpha(1-alpha) = 0.21
        (0.5, 0.2j),                      # Complex gamma (pure imaginary)
        (0.6, 0.2 * np.exp(1j*np.pi/4)),  # Complex gamma (phase pi/4)
    ]

    for alpha, gamma in m3_params:
        # Check positivity using abs(gamma)**2 (handles complex gamma)
        if abs(gamma)**2 > alpha*(1-alpha) + 1e-12:
            print(f"  {alpha:>7.2f}  {format_gamma(gamma):>15}  [skipped: violates positivity]")
            continue

        rho = make_cat_state(alpha, gamma)
        assert_valid_density_matrix(rho, f"Test M3 (alpha={alpha}, gamma={gamma})")

        C_num, C_st, C_gcg, _ = get_convex_roof_values(rho, 3, n_restarts)
        exp  = 2 * abs(gamma)
        err_num = abs(C_num - exp)
        err_st  = abs(C_st - exp)
        err_gcg = abs(C_gcg - exp)
        ok_num = err_num < tol_pass
        ok_st  = err_st < tol_pass
        ok_gcg = err_gcg < tol_pass
        ok = ok_num and ok_st and ok_gcg
        if not ok:
            m3_pass = False
        status = "✓✓" if (ok_num and ok_st) else ("✓✗" if ok_num else ("✗✓" if ok_st else "✗✗"))
        print(f"  {alpha:>7.2f}  {format_gamma(gamma):>15}  {exp:>10.6f}  {C_num:>10.6f}  "
              f"{C_st:>10.6f}  {err_num:>10.2e}  {err_st:>10.2e}  {status}")

    all_passed = all_passed and m3_pass
    print(f"\n  Result: {'PASSED ✓' if m3_pass else 'FAILED ✗'}")

    # ── Test D — GHZ + W (exploratory) ───────────────────────────────────────
    print()
    print("TEST D — p*|GHZ+><GHZ+| + (1-p)*|W><W|  (exploratory, no analytical answer)")
    print("  Note: exploratory consistency check against the eigendecomposition")
    print("        upper bound; not an exact correctness benchmark.")
    print()
    print(f"  {'p':>6}  {'UB':>12}  {'Exp map':>10}  "
          f"{'Stiefel':>10}  {'UB-C_num':>10}  {'UB-C_st':>10}  Status")
    print("  " + "-"*85)

    ghz_p_state = make_ghz_plus(3)
    W           = make_w()

    testD_pass = True
    for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
        rho              = mix([(p, ghz_p_state), (1-p, W)])
        assert_valid_density_matrix(rho, f"Test D at p={p}")

        ub               = convex_roof_upper_bound(rho, 3)["C_UB"]
        C_num, C_st, C_gcg, _ = get_convex_roof_values(rho, 3, n_restarts)

        # Check both methods separately (not just min)
        ok_num = C_num <= ub + tol_pass
        ok_st  = C_st <= ub + tol_pass
        ok = ok_num and ok_st and ok_gcg
        if not ok:
            testD_pass = False
        status = "✓✓" if (ok_num and ok_st) else ("✓✗" if ok_num else ("✗✓" if ok_st else "✗✗"))
        print(f"  {p:>6.2f}  {ub:>12.6f}  {C_num:>10.6f}  "
              f"{C_st:>10.6f}  {ub-C_num:>10.6f}  {ub-C_st:>10.6f}  {status}")

    all_passed = all_passed and testD_pass
    print(f"\n  Result: {'PASSED ✓' if testD_pass else 'FAILED ✗'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"  OVERALL: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print("=" * 80)

    return all_passed


# ══════════════════════════════════════════════════════════════════════════════
# pytest-compatible wrappers
# ══════════════════════════════════════════════════════════════════════════════

def test_pure_state():
    """Test A — pure state sanity check."""
    np.random.seed(42)
    ghz_p  = make_ghz_plus(3)
    rho    = dm(ghz_p)
    assert_valid_density_matrix(rho, "test_pure_state")
    direct = _cgme_pure_vector(ghz_p, 3)
    C_num, C_st, C_gcg, _ = get_convex_roof_values(rho, 3, n_restarts=20)
    assert abs(C_num - direct) < 1e-3, f"Exp map failed: {C_num} vs {direct}"
    assert abs(C_st - direct) < 1e-3, f"Stiefel failed: {C_st} vs {direct}"


def test_separable_mixture():
    """Test M1 — separable mixture must give C_GME = 0."""
    np.random.seed(42)
    rho = mix([(0.3, ket(0,0,0)),
               (0.2, ket(0,1,1)),
               (0.5, ket(1,0,1))])
    assert_valid_density_matrix(rho, "test_separable_mixture")
    C_num, C_st, C_gcg, _ = get_convex_roof_values(rho, 3, n_restarts=20)
    assert abs(C_num) < 1e-6, f"Exp map failed: {C_num}"
    assert abs(C_st) < 1e-6, f"Stiefel failed: {C_st}"


def test_ghz_plus_minus_mixture():
    """Test M2 — PRIMARY BENCHMARK: p*|GHZ+><GHZ+| + (1-p)*|GHZ-><GHZ-|  =>  |2p-1|."""
    np.random.seed(42)
    ghz_p = make_ghz_plus(3)
    ghz_m = make_ghz_minus(3)
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        rho        = mix([(p, ghz_p), (1-p, ghz_m)])
        assert_valid_density_matrix(rho, f"test_ghz_plus_minus_mixture (p={p})")
        C_num, C_st, C_gcg, _ = get_convex_roof_values(rho, 3, n_restarts=20)
        expected   = abs(2*p - 1)
        assert abs(C_num - expected) < 1e-3, \
            f"M2 (exp map) failed at p={p}: got {C_num}, expected {expected}"
        assert abs(C_st - expected) < 1e-3, \
            f"M2 (Stiefel) failed at p={p}: got {C_st}, expected {expected}"


def test_cat_state():
    """Test M3 — general cat state: C_GME = 2|gamma| (including complex gamma)."""
    np.random.seed(42)
    # Real and complex gamma cases
    for alpha, gamma in [(0.5, 0.3), (0.3, 0.15), (0.7, 0.2),
                         (0.5, 0.2j), (0.6, 0.2 * np.exp(1j*np.pi/4))]:
        # Use abs(gamma)**2 for complex gamma support
        if abs(gamma)**2 > alpha*(1-alpha) + 1e-12:
            continue
        rho        = make_cat_state(alpha, gamma)
        assert_valid_density_matrix(rho, f"test_cat_state (alpha={alpha}, gamma={gamma})")
        C_num, C_st, C_gcg, _ = get_convex_roof_values(rho, 3, n_restarts=20)
        expected   = 2 * abs(gamma)
        assert abs(C_num - expected) < 1e-3, \
            f"M3 (exp map) failed at (alpha={alpha}, gamma={gamma}): got {C_num}, expected {expected}"
        assert abs(C_st - expected) < 1e-3, \
            f"M3 (Stiefel) failed at (alpha={alpha}, gamma={gamma}): got {C_st}, expected {expected}"


def test_ghz_plus_w_upper_bound():
    """Test D — convex roof must be below upper bound (for both methods separately)."""
    np.random.seed(42)
    ghz_p = make_ghz_plus(3)
    W     = make_w()
    for p in [0.25, 0.5, 0.75]:
        rho        = mix([(p, ghz_p), (1-p, W)])
        assert_valid_density_matrix(rho, f"test_ghz_plus_w_upper_bound (p={p})")
        ub         = convex_roof_upper_bound(rho, 3)["C_UB"]
        # Check both methods separately (not just min)
        C_num, C_st, C_gcg, _ = get_convex_roof_values(rho, 3, n_restarts=20)
        assert C_num <= ub + 1e-3, \
            f"Exp map exceeded UB at p={p}: {C_num} > {ub}"
        assert C_st <= ub + 1e-3, \
            f"Stiefel exceeded UB at p={p}: {C_st} > {ub}"


if __name__ == "__main__":
    run_tests(n_restarts=40)