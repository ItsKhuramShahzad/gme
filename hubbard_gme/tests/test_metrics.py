"""
tests/test_metrics.py
---------------------
Unit tests for hubbard_gme.metrics
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hubbard_gme.metrics import (
    compute_purity,
    compute_entropy,
    hilbert_schmidt_distance,
    trace_distance,
    rdm_metrics,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def pure_state():
    """Pure-state density matrix |0⟩⟨0|."""
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    return rho


@pytest.fixture
def maximally_mixed():
    """Maximally mixed state I/2."""
    return np.eye(2, dtype=complex) / 2.0


@pytest.fixture
def bell_state_rdm():
    """Single-qubit RDM from |Φ+⟩ = (|00⟩+|11⟩)/√2 → ρ_A = I/2."""
    return np.eye(2, dtype=complex) / 2.0


# ── Purity ────────────────────────────────────────────────────────────────────

class TestPurity:
    def test_pure_state_gives_one(self, pure_state):
        assert np.isclose(compute_purity(pure_state), 1.0)

    def test_maximally_mixed_gives_half(self, maximally_mixed):
        assert np.isclose(compute_purity(maximally_mixed), 0.5)

    def test_purity_between_zero_and_one(self, pure_state, maximally_mixed):
        for rho in [pure_state, maximally_mixed]:
            p = compute_purity(rho)
            assert 0.0 <= p <= 1.0 + 1e-10

    def test_purity_matches_eigenvalue_sum(self, maximally_mixed):
        rho = maximally_mixed
        p_matrix = compute_purity(rho)
        evals = np.linalg.eigvalsh(rho)
        p_evals = float(np.sum(evals ** 2))
        assert np.isclose(p_matrix, p_evals, atol=1e-10)


# ── Entropy ───────────────────────────────────────────────────────────────────

class TestEntropy:
    def test_pure_state_gives_zero(self, pure_state):
        assert np.isclose(compute_entropy(pure_state), 0.0, atol=1e-10)

    def test_maximally_mixed_gives_log2(self, maximally_mixed):
        expected = np.log(2)
        assert np.isclose(compute_entropy(maximally_mixed), expected, atol=1e-10)

    def test_entropy_nonnegative(self, pure_state, maximally_mixed):
        for rho in [pure_state, maximally_mixed]:
            assert compute_entropy(rho) >= -1e-12


# ── Hilbert-Schmidt distance ──────────────────────────────────────────────────

class TestHilbertSchmidtDistance:
    def test_identical_states_give_zero(self, pure_state):
        assert np.isclose(hilbert_schmidt_distance(pure_state, pure_state), 0.0)

    def test_symmetry(self, pure_state, maximally_mixed):
        d1 = hilbert_schmidt_distance(pure_state, maximally_mixed)
        d2 = hilbert_schmidt_distance(maximally_mixed, pure_state)
        assert np.isclose(d1, d2)

    def test_nonnegative(self, pure_state, maximally_mixed):
        assert hilbert_schmidt_distance(pure_state, maximally_mixed) >= 0.0

    def test_known_value(self, pure_state, maximally_mixed):
        # ||diag(1,0) − diag(0.5,0.5)||_F = ||diag(0.5,−0.5)||_F = √0.5
        expected = np.sqrt(0.5)
        assert np.isclose(
            hilbert_schmidt_distance(pure_state, maximally_mixed), expected, atol=1e-10
        )


# ── Trace distance ────────────────────────────────────────────────────────────

class TestTraceDistance:
    def test_identical_states_give_zero(self, pure_state):
        assert np.isclose(trace_distance(pure_state, pure_state), 0.0)

    def test_bounded_by_one(self, pure_state, maximally_mixed):
        d = trace_distance(pure_state, maximally_mixed)
        assert 0.0 <= d <= 1.0 + 1e-10

    def test_known_value(self, pure_state, maximally_mixed):
        # D_T(|0⟩⟨0|, I/2) = 0.5
        assert np.isclose(trace_distance(pure_state, maximally_mixed), 0.5, atol=1e-10)

    def test_symmetry(self, pure_state, maximally_mixed):
        d1 = trace_distance(pure_state, maximally_mixed)
        d2 = trace_distance(maximally_mixed, pure_state)
        assert np.isclose(d1, d2)


# ── Aggregate metrics dict ────────────────────────────────────────────────────

class TestRdmMetrics:
    def test_keys_present(self, pure_state):
        info = rdm_metrics(pure_state)
        for key in ("trace", "purity", "entropy", "eigvals"):
            assert key in info

    def test_trace_is_one(self, pure_state):
        info = rdm_metrics(pure_state)
        assert np.isclose(info["trace"], 1.0)

    def test_eigvals_length(self, maximally_mixed):
        info = rdm_metrics(maximally_mixed)
        assert len(info["eigvals"]) == 2
