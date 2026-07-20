"""
test_measure_param.py
---------------------
Verify the patched convex_roof_stiefel and convex_roof_gcg accept the
new `measure` parameter and produce sensible output for both choices.

Three checks:

  TEST 1: Old behaviour preserved
          Calling convex_roof_stiefel/gcg without measure arg should still
          run and produce C_GME results, identical to before.

  TEST 2: New MW path runs end-to-end
          measure="meyerwallach" should run without errors and produce a
          γ-based convex roof value.

  TEST 3: Sanity check on a pure state
          For ρ = |ψ⟩⟨ψ| (rank 1), convex roof must equal pure-state value.
          So convex_roof_stiefel(ρ_pure, n, measure="meyerwallach") should
          return γ(|ψ⟩) directly.

Run:
    python test_measure_param.py
"""

import numpy as np
import sys, os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hubbard_gme.convex_roof import (
    convex_roof_stiefel,
    convex_roof_gcg,
    _cgme_pure_vector_robust,
    _meyerwallach_pure_vector,
)


def make_random_mixed_rho(n, rank=3, seed=42):
    np.random.seed(seed)
    dim = 2 ** n
    A = np.random.randn(dim, rank) + 1j * np.random.randn(dim, rank)
    rho = A @ A.conj().T
    return rho / np.trace(rho).real


# ─────────────────────────────────────────────────────────────────────────
# TEST 1: default behaviour unchanged
# ─────────────────────────────────────────────────────────────────────────

def test_1_default_unchanged(n=3):
    print(f"\n  TEST 1: default behaviour preserved (no measure arg)")
    print(f"  " + "-" * 70)

    rho = make_random_mixed_rho(n, rank=3, seed=42)

    print(f"    Stiefel without measure arg …")
    t0 = time.time()
    res = convex_roof_stiefel(rho, n, n_restarts=3, k_max_extra=1, verbose=False)
    print(f"      C = {res['C_stiefel']:.6f}   measure = {res.get('measure', 'cgme')}   "
          f"time = {time.time() - t0:.1f}s")

    print(f"    GCG without measure arg …")
    t0 = time.time()
    res = convex_roof_gcg(rho, n, n_restarts=3, max_iter=100, k_max_extra=1, verbose=False)
    print(f"      C = {res['C_gcg']:.6f}   measure = {res.get('measure', 'cgme')}   "
          f"time = {time.time() - t0:.1f}s")
    print(f"    ✓ Both ran without error using default 'cgme'")


# ─────────────────────────────────────────────────────────────────────────
# TEST 2: MW path runs
# ─────────────────────────────────────────────────────────────────────────

def test_2_mw_runs(n=3):
    print(f"\n  TEST 2: meyerwallach measure runs end-to-end")
    print(f"  " + "-" * 70)

    rho = make_random_mixed_rho(n, rank=3, seed=42)

    print(f"    Stiefel with measure='meyerwallach' …")
    t0 = time.time()
    res = convex_roof_stiefel(rho, n, n_restarts=3, k_max_extra=1, verbose=False,
                              measure="meyerwallach")
    print(f"      γ = {res['C_stiefel']:.6f}   measure = {res['measure']}   "
          f"time = {time.time() - t0:.1f}s")

    print(f"    GCG with measure='meyerwallach' …")
    t0 = time.time()
    res = convex_roof_gcg(rho, n, n_restarts=3, max_iter=100, k_max_extra=1, verbose=False,
                          measure="meyerwallach")
    print(f"      γ = {res['C_gcg']:.6f}   measure = {res['measure']}   "
          f"time = {time.time() - t0:.1f}s")
    print(f"    ✓ Both ran with measure='meyerwallach'")


# ─────────────────────────────────────────────────────────────────────────
# TEST 3: convex roof on a pure state = pure-state value
# ─────────────────────────────────────────────────────────────────────────

def test_3_pure_state(n=3):
    """
    For a pure state ρ = |ψ⟩⟨ψ|, the only decomposition is {(1, |ψ⟩)},
    so the convex roof equals the pure-state value.

    This is a stringent correctness test: the optimiser shouldn't find
    anything below γ(|ψ⟩) because there's nothing below it.
    """
    print(f"\n  TEST 3: convex roof on pure state == pure-state value")
    print(f"  " + "-" * 70)

    np.random.seed(123)
    dim = 2 ** n
    psi = np.random.randn(dim) + 1j * np.random.randn(dim)
    psi = psi / np.linalg.norm(psi)
    rho_pure = np.outer(psi, psi.conj())

    gamma_pure = _meyerwallach_pure_vector(psi, n)
    cgme_pure = _cgme_pure_vector_robust(psi, n)
    print(f"    Pure state γ(|ψ⟩)         = {gamma_pure:.6f}")
    print(f"    Pure state C_GME(|ψ⟩)     = {cgme_pure:.6f}")

    # MW via Stiefel
    res = convex_roof_stiefel(rho_pure, n, n_restarts=3, k_max_extra=1, verbose=False,
                              measure="meyerwallach")
    diff = abs(res['C_stiefel'] - gamma_pure)
    print(f"    MW Stiefel convex roof    = {res['C_stiefel']:.6f}   "
          f"diff from γ = {diff:.2e}   {'✓' if diff < 1e-3 else '✗'}")

    # MW via GCG
    res = convex_roof_gcg(rho_pure, n, n_restarts=3, max_iter=100, k_max_extra=1, verbose=False,
                          measure="meyerwallach")
    diff = abs(res['C_gcg'] - gamma_pure)
    print(f"    MW GCG convex roof        = {res['C_gcg']:.6f}   "
          f"diff from γ = {diff:.2e}   {'✓' if diff < 1e-3 else '✗'}")

    # C_GME via Stiefel
    res = convex_roof_stiefel(rho_pure, n, n_restarts=3, k_max_extra=1, verbose=False,
                              measure="cgme")
    diff = abs(res['C_stiefel'] - cgme_pure)
    print(f"    CGME Stiefel convex roof  = {res['C_stiefel']:.6f}   "
          f"diff from C = {diff:.2e}   {'✓' if diff < 1e-3 else '✗'}")


# ─────────────────────────────────────────────────────────────────────────
# TEST 4: typo error handling
# ─────────────────────────────────────────────────────────────────────────

def test_4_typo_handling(n=3):
    print(f"\n  TEST 4: typo in measure arg raises helpful error")
    print(f"  " + "-" * 70)

    rho = make_random_mixed_rho(n, rank=2, seed=42)

    try:
        res = convex_roof_stiefel(rho, n, n_restarts=1, k_max_extra=0, verbose=False,
                                  measure="meyer_wallach")  # underscore typo
        print(f"    ✗ Expected ValueError, got result {res['C_stiefel']}")
    except ValueError as e:
        print(f"    ✓ Got ValueError as expected: {e}")
    except Exception as e:
        print(f"    ⚠ Got unexpected exception type: {type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────

def main():
    test_1_default_unchanged()
    test_2_mw_runs()
    test_3_pure_state()
    test_4_typo_handling()
    print()


if __name__ == "__main__":
    main()
