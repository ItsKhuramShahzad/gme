"""
tests/test_gme.py
-----------------
Unit tests for hubbard_gme.gme
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hubbard_gme.gme import (
    purity,
    is_pure_state,
    get_bipartitions,
    partial_trace_spins,
    compute_C_GME_pure,
    analyze_one_rdm_for_GME,
    Cgme_lower_bound,
    I_function,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _two_spin_singlet() -> np.ndarray:
    """
    |Ψ⁻⟩ = (|↑↓⟩ − |↓↑⟩)/√2  →  2-spin singlet (maximally entangled).
    Basis order: |↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩ → indices 0,1,2,3
    State vector: (0, 1/√2, −1/√2, 0)
    """
    psi = np.array([0.0, 1.0, -1.0, 0.0]) / np.sqrt(2)
    return np.outer(psi, psi.conj())


def _product_state_rho() -> np.ndarray:
    """
    |↑⟩⊗|↑⟩  →  ρ = |↑↑⟩⟨↑↑| (index 0, product state → no entanglement).
    """
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 1.0
    return rho


def _ghz_state_rho(n: int) -> np.ndarray:
    """
    GHZ state |GHZ⟩ = (|0…0⟩ + |1…1⟩)/√2.
    Encoding: |0⟩ = spin-up (bit 0), |1⟩ = spin-down (bit 1) at each site.
    For the big-endian convention used in gme.py, |0…0⟩ → index 0,
    |1…1⟩ → index 2ⁿ−1.
    """
    dim = 2 ** n
    psi = np.zeros(dim, dtype=complex)
    psi[0]      = 1.0 / np.sqrt(2)
    psi[dim - 1] = 1.0 / np.sqrt(2)
    return np.outer(psi, psi.conj())


# ── Purity ────────────────────────────────────────────────────────────────────

class TestPurity:
    def test_pure_state(self):
        rho = _product_state_rho()
        assert np.isclose(purity(rho), 1.0)

    def test_maximally_mixed_two_spins(self):
        rho = np.eye(4, dtype=complex) / 4
        assert np.isclose(purity(rho), 0.25)

    def test_is_pure_state_true(self):
        assert is_pure_state(_product_state_rho())

    def test_is_pure_state_false(self):
        assert not is_pure_state(np.eye(4, dtype=complex) / 4)


# ── Bipartitions ──────────────────────────────────────────────────────────────

class TestBipartitions:
    def test_n2_gives_one_partition(self):
        parts = get_bipartitions(2)
        assert len(parts) == 1
        assert parts[0] == (0,)

    def test_n3_gives_three_partitions(self):
        # (0,), (0,1), (0,2)
        parts = get_bipartitions(3)
        assert len(parts) == 3

    def test_n4_gives_seven_partitions(self):
        # 2^(n-1) - 1 = 7 for n=4
        parts = get_bipartitions(4)
        assert len(parts) == 7

    def test_zero_always_in_A(self):
        for n in [3, 4, 5]:
            for A in get_bipartitions(n):
                assert 0 in A


# ── Partial trace ─────────────────────────────────────────────────────────────

class TestPartialTrace:
    def test_trace_out_one_spin_gives_correct_shape(self):
        rho = _two_spin_singlet()
        rhoA = partial_trace_spins(rho, 2, keep=(0,))
        assert rhoA.shape == (2, 2)

    def test_singlet_single_qubit_rdm_is_maximally_mixed(self):
        """Tracing out one spin of a singlet → I/2."""
        rho = _two_spin_singlet()
        rhoA = partial_trace_spins(rho, 2, keep=(0,))
        expected = np.eye(2, dtype=complex) / 2
        assert np.allclose(rhoA, expected, atol=1e-10)

    def test_product_state_trace_gives_pure_rdm(self):
        """Tracing out spin-1 of |↑↑⟩ → |↑⟩⟨↑|."""
        rho = _product_state_rho()
        rhoA = partial_trace_spins(rho, 2, keep=(0,))
        expected = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        assert np.allclose(rhoA, expected, atol=1e-10)

    def test_trace_preserved(self):
        rho = _two_spin_singlet()
        rhoA = partial_trace_spins(rho, 2, keep=(0,))
        assert np.isclose(np.trace(rhoA).real, 1.0, atol=1e-10)

    def test_wrong_shape_raises(self):
        rho = np.eye(3, dtype=complex)
        with pytest.raises(ValueError):
            partial_trace_spins(rho, 2, keep=(0,))


# ── C_GME ─────────────────────────────────────────────────────────────────────

class TestCGME:
    def test_singlet_cgme_is_one(self):
        """Bell singlet is maximally entangled: C_GME should equal 1."""
        rho = _two_spin_singlet()
        # For 2 spins there is only one bipartition; GME formula applies.
        cgme = compute_C_GME_pure(rho, 2)
        assert np.isclose(cgme, 1.0, atol=1e-8)

    def test_product_state_cgme_is_zero(self):
        rho = _product_state_rho()
        cgme = compute_C_GME_pure(rho, 2)
        assert np.isclose(cgme, 0.0, atol=1e-8)

    def test_ghz3_cgme_positive(self):
        rho = _ghz_state_rho(3)
        cgme = compute_C_GME_pure(rho, 3)
        assert cgme > 0.0

    def test_cgme_nonnegative(self):
        for n in [2, 3]:
            rho = _ghz_state_rho(n)
            assert compute_C_GME_pure(rho, n) >= 0.0


# ── analyze_one_rdm_for_GME ────────────────────────────────────────────────────

class TestAnalyseOneRDM:
    def test_singlet_returns_gme(self):
        rho = _two_spin_singlet()
        info = analyze_one_rdm_for_GME(rho, (0, 1))
        assert info["status"] == "GME"

    def test_product_state_returns_no_gme(self):
        rho = _product_state_rho()
        info = analyze_one_rdm_for_GME(rho, (0, 1))
        assert info["status"] == "no_GME"

    def test_mixed_state_flagged(self):
        rho = np.eye(4, dtype=complex) / 4
        info = analyze_one_rdm_for_GME(rho, (0, 1))
        assert info["status"] == "mixed_state"

    def test_c_gme_none_when_no_gme(self):
        rho = _product_state_rho()
        info = analyze_one_rdm_for_GME(rho, (0, 1))
        assert info["C_GME"] is None

    def test_c_gme_positive_when_gme(self):
        rho = _two_spin_singlet()
        info = analyze_one_rdm_for_GME(rho, (0, 1))
        assert info["C_GME"] > 0.0


# ── Cgme_lower_bound ─────────────────────────────────────────────────────────

class TestCgmeLowerBound:
    def test_nonnegative(self):
        rho = _ghz_state_rho(3)
        C_LB, I = Cgme_lower_bound(rho, 3)
        assert C_LB >= 0.0

    def test_lower_bound_leq_upper_bound(self):
        """Lower bound ≤ C_GME (upper bound from full bipartition analysis)."""
        rho = _ghz_state_rho(3)
        C_LB, _ = Cgme_lower_bound(rho, 3)
        C_upper = compute_C_GME_pure(rho, 3)
        assert C_LB <= C_upper + 1e-10
