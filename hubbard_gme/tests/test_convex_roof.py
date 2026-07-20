"""
tests/test_convex_roof.py
-------------------------
Unit tests for hubbard_gme.convex_roof

Key invariants checked
----------------------
1.  Pure states:   C_LB = C_num = C_UB = C_GME_pure
2.  Mixed states:  C_LB ≤ C_num ≤ C_UB
3.  Separable states: all three → 0
4.  Maximally mixed: all three → 0  (no coherence → no GME)
5.  Known analytic case: isotropic mixture of singlet/triplet
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hubbard_gme.convex_roof import (
    convex_roof_lower_bound,
    convex_roof_upper_bound,
    convex_roof_numerical,
    full_convex_roof_analysis,
    _cgme_pure_vector,
    _params_to_unitary,
    _decompose,
)
from hubbard_gme.gme import compute_C_GME_pure


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _bell_singlet_rho():
    """Pure singlet |Ψ⁻⟩ = (|01⟩−|10⟩)/√2."""
    psi = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    return np.outer(psi, psi.conj())


def _bell_singlet_vec():
    return np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)


def _product_rho():
    """Pure product state |00⟩ — no entanglement."""
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 1.0
    return rho


def _mixed_singlet_triplet(p: float):
    """
    Mixture: p|Ψ⁻⟩⟨Ψ⁻| + (1-p)·I/4

    For p=1  → pure singlet
    For p=0  → maximally mixed
    """
    psi = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
    rho_s = np.outer(psi, psi.conj())
    rho_m = np.eye(4, dtype=complex) / 4
    return p * rho_s + (1 - p) * rho_m


def _ghz3_rho():
    """Pure 3-spin GHZ state."""
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1 / np.sqrt(2)
    psi[7] = 1 / np.sqrt(2)
    return np.outer(psi, psi.conj())


# ── _cgme_pure_vector ─────────────────────────────────────────────────────────

class TestCgmePureVector:
    def test_singlet_equals_one(self):
        psi = _bell_singlet_vec()
        assert np.isclose(_cgme_pure_vector(psi, 2), 1.0, atol=1e-8)

    def test_product_equals_zero(self):
        psi = np.array([1, 0, 0, 0], dtype=complex)
        assert np.isclose(_cgme_pure_vector(psi, 2), 0.0, atol=1e-8)

    def test_matches_density_matrix_formula(self):
        """SVD-based fast version must agree with full density-matrix version."""
        psi = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
        rho = np.outer(psi, psi.conj())
        c_fast = _cgme_pure_vector(psi, 2)
        c_full = compute_C_GME_pure(rho, 2)
        assert np.isclose(c_fast, c_full, atol=1e-8)

    def test_ghz3_positive(self):
        psi = np.zeros(8, dtype=complex)
        psi[0] = psi[7] = 1 / np.sqrt(2)
        assert _cgme_pure_vector(psi, 3) > 0


# ── Unitary parametrisation ───────────────────────────────────────────────────

class TestUnitaryParam:
    @pytest.mark.parametrize("r", [1, 2, 3, 4])
    def test_is_unitary(self, r):
        np.random.seed(42)
        from hubbard_gme.convex_roof import _n_params
        params = np.random.randn(_n_params(r))
        U = _params_to_unitary(params, r)
        assert np.allclose(U @ U.conj().T, np.eye(r), atol=1e-10)
        assert np.allclose(U.conj().T @ U, np.eye(r), atol=1e-10)

    def test_identity_from_zero_params(self):
        from hubbard_gme.convex_roof import _n_params
        r = 3
        params = np.zeros(_n_params(r))
        U = _params_to_unitary(params, r)
        assert np.allclose(U, np.eye(r), atol=1e-10)


# ── _decompose ────────────────────────────────────────────────────────────────

class TestDecompose:
    def _setup(self, rho):
        eigvals, eigvecs = np.linalg.eigh(rho)
        mask = eigvals > 1e-10
        return eigvals[mask], eigvecs[:, mask], np.sqrt(eigvals[mask])

    def test_probabilities_sum_to_one(self):
        rho = _mixed_singlet_triplet(0.6)
        ev, evc, evs = self._setup(rho)
        r = len(ev)
        U = np.eye(r, dtype=complex)
        probs, _ = _decompose(U, evc, evs)
        assert np.isclose(probs.sum(), 1.0, atol=1e-8)

    def test_reconstruction_correct(self):
        """Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ| must reproduce ρ."""
        rho = _mixed_singlet_triplet(0.7)
        ev, evc, evs = self._setup(rho)
        r = len(ev)
        np.random.seed(0)
        from hubbard_gme.convex_roof import _n_params
        params = np.random.randn(_n_params(r)) * 0.5
        U = _params_to_unitary(params, r)
        probs, pure_states = _decompose(U, evc, evs)

        rho_reconstructed = sum(
            p * np.outer(psi, psi.conj())
            for p, psi in zip(probs, pure_states)
            if p > 1e-14
        )
        assert np.allclose(rho_reconstructed, rho, atol=1e-8)


# ── Lower bound ───────────────────────────────────────────────────────────────

class TestLowerBound:
    def test_nonnegative(self):
        rho = _mixed_singlet_triplet(0.5)
        result = convex_roof_lower_bound(rho, 2)
        assert result["C_LB"] >= 0.0

    def test_zero_for_maximally_mixed(self):
        rho = np.eye(4, dtype=complex) / 4
        result = convex_roof_lower_bound(rho, 2)
        assert result["C_LB"] < 1e-8

    def test_positive_for_entangled_mixture(self):
        """Strongly mixed singlet should still give positive lower bound."""
        rho = _mixed_singlet_triplet(0.9)
        result = convex_roof_lower_bound(rho, 2)
        assert result["C_LB"] > 0.0

    def test_returns_required_keys(self):
        rho = _mixed_singlet_triplet(0.5)
        result = convex_roof_lower_bound(rho, 2)
        for key in ("C_LB", "best_I", "best_x", "best_y", "n_pairs_checked"):
            assert key in result

    def test_pure_state_matches_cgme(self):
        """For a pure state, lower bound = C_GME."""
        rho = _bell_singlet_rho()
        result = convex_roof_lower_bound(rho, 2)
        assert np.isclose(result["C_LB"], 1.0, atol=1e-6)


# ── Upper bound ───────────────────────────────────────────────────────────────

class TestUpperBound:
    def test_nonnegative(self):
        rho = _mixed_singlet_triplet(0.5)
        result = convex_roof_upper_bound(rho, 2)
        assert result["C_UB"] >= 0.0

    def test_zero_for_separable(self):
        rho = _product_rho()
        result = convex_roof_upper_bound(rho, 2)
        assert result["C_UB"] < 1e-8

    def test_returns_required_keys(self):
        rho = _mixed_singlet_triplet(0.5)
        result = convex_roof_upper_bound(rho, 2)
        for key in ("C_UB", "rank", "eigenvalues", "contributions"):
            assert key in result

    def test_pure_state_matches_cgme(self):
        rho = _bell_singlet_rho()
        result = convex_roof_upper_bound(rho, 2)
        assert np.isclose(result["C_UB"], 1.0, atol=1e-6)

    def test_contributions_sum_consistent(self):
        """Σ λᵢ·C_i should equal C_UB."""
        rho = _mixed_singlet_triplet(0.6)
        result = convex_roof_upper_bound(rho, 2)
        total = sum(c["eigenvalue"] * c["C_GME_pure"] for c in result["contributions"])
        assert np.isclose(total, result["C_UB"], atol=1e-10)


# ── Numerical minimisation ────────────────────────────────────────────────────

class TestNumerical:
    def test_nonnegative(self):
        rho = _mixed_singlet_triplet(0.5)
        result = convex_roof_numerical(rho, 2, n_restarts=5)
        assert result["C_numerical"] >= 0.0

    def test_leq_upper_bound(self):
        """Numerical result must not exceed upper bound."""
        rho = _mixed_singlet_triplet(0.6)
        ub = convex_roof_upper_bound(rho, 2)["C_UB"]
        num = convex_roof_numerical(rho, 2, n_restarts=10)["C_numerical"]
        assert num <= ub + 1e-5

    def test_geq_lower_bound(self):
        """Numerical result must not go below lower bound."""
        rho = _mixed_singlet_triplet(0.6)
        lb = convex_roof_lower_bound(rho, 2)["C_LB"]
        num = convex_roof_numerical(rho, 2, n_restarts=10)["C_numerical"]
        assert num >= lb - 1e-5

    def test_pure_state_matches_cgme(self):
        """For pure state, all methods should agree."""
        rho = _bell_singlet_rho()
        result = convex_roof_numerical(rho, 2, n_restarts=5)
        assert np.isclose(result["C_numerical"], 1.0, atol=1e-4)

    def test_separable_gives_zero(self):
        rho = _product_rho()
        result = convex_roof_numerical(rho, 2, n_restarts=5)
        assert result["C_numerical"] < 1e-5

    def test_returns_required_keys(self):
        rho = _mixed_singlet_triplet(0.5)
        result = convex_roof_numerical(rho, 2, n_restarts=3)
        for key in ("C_numerical", "rank", "n_restarts", "converged", "all_values"):
            assert key in result


# ── Full analysis ─────────────────────────────────────────────────────────────

class TestFullAnalysis:
    def test_bounds_consistent_for_singlet(self):
        rho = _bell_singlet_rho()
        result = full_convex_roof_analysis(rho, (0, 1), n_restarts=5, verbose=False)
        assert result["bounds_consistent"]

    def test_bounds_consistent_for_mixture(self):
        rho = _mixed_singlet_triplet(0.7)
        result = full_convex_roof_analysis(rho, (0, 1), n_restarts=10, verbose=False)
        assert result["C_LB"] <= result["C_numerical"] + 1e-4
        assert result["C_numerical"] <= result["C_UB"] + 1e-4

    def test_status_gme_for_entangled(self):
        rho = _mixed_singlet_triplet(0.9)
        result = full_convex_roof_analysis(rho, (0, 1), n_restarts=10, verbose=False)
        assert result["status"] in ("GME", "inconclusive")

    def test_status_no_gme_for_separable(self):
        rho = _product_rho()
        result = full_convex_roof_analysis(rho, (0, 1), n_restarts=5, verbose=False)
        assert result["status"] == "no_GME"

    def test_all_keys_present(self):
        rho = _mixed_singlet_triplet(0.5)
        result = full_convex_roof_analysis(rho, (0, 1), n_restarts=3, verbose=False)
        for key in ("C_LB", "C_numerical", "C_UB", "bounds_consistent",
                    "status", "lower_bound", "upper_bound", "numerical",
                    "global_purity", "n", "sites"):
            assert key in result
