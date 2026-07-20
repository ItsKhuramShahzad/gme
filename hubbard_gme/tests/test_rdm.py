"""
tests/test_rdm.py
-----------------
Unit tests for hubbard_gme.rdm

All tests use the smallest meaningful system: L=2, Ne=2 (half-filling).
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hubbard_gme.hamiltonian import compute_ground_and_excited_states
from hubbard_gme.rdm import (
    generate_site_clusters,
    compute_general_nRDM,
    compute_pure_and_excited_RDMs,
    compute_all_pure_rdms,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def l2_system():
    """Ground and excited states for L=2, Ne=2, uniform."""
    E0, psi0, E1, psi1, n_so = compute_ground_and_excited_states(
        L=2, t=1.0, U=4.0, mode="uniform", n_electrons=2
    )
    return dict(L=2, Ne=2, n_so=n_so, psi0=psi0, psi1=psi1, E0=E0, E1=E1)


# ── Cluster generation ────────────────────────────────────────────────────────

class TestGenerateSiteClusters:
    def test_correct_count(self):
        """L clusters of size n."""
        for L in [2, 4, 6]:
            for n in [1, 2]:
                clusters = generate_site_clusters(L, n)
                assert len(clusters) == L

    def test_periodic_boundary(self):
        """Last cluster wraps around."""
        clusters = generate_site_clusters(4, 2)
        assert (3, 0) in clusters

    def test_single_site_clusters_cover_all_sites(self):
        L = 6
        clusters = generate_site_clusters(L, 1)
        sites = {c[0] for c in clusters}
        assert sites == set(range(L))


# ── compute_general_nRDM ──────────────────────────────────────────────────────

class TestComputeGeneralNRDM:
    def test_1rdm_shape(self, l2_system):
        d = l2_system
        rho = compute_general_nRDM([0], d["psi0"], d["n_so"], d["Ne"])
        assert rho.shape == (2, 2)

    def test_2rdm_shape(self, l2_system):
        d = l2_system
        rho = compute_general_nRDM([0, 1], d["psi0"], d["n_so"], d["Ne"])
        assert rho.shape == (4, 4)

    def test_1rdm_trace_positive(self, l2_system):
        d = l2_system
        rho = compute_general_nRDM([0], d["psi0"], d["n_so"], d["Ne"])
        assert np.trace(rho).real > 0.0

    def test_1rdm_hermitian(self, l2_system):
        d = l2_system
        rho = compute_general_nRDM([0], d["psi0"], d["n_so"], d["Ne"])
        assert np.allclose(rho, rho.conj().T, atol=1e-10)


# ── compute_pure_and_excited_RDMs ─────────────────────────────────────────────

class TestPureExcitedRDMs:
    def test_trace_equals_one(self, l2_system):
        d = l2_system
        rho0, rho1 = compute_pure_and_excited_RDMs(
            [0], d["psi0"], d["psi1"], d["n_so"], d["Ne"]
        )
        assert np.isclose(np.trace(rho0).real, 1.0, atol=1e-8)
        assert np.isclose(np.trace(rho1).real, 1.0, atol=1e-8)

    def test_both_hermitian(self, l2_system):
        d = l2_system
        rho0, rho1 = compute_pure_and_excited_RDMs(
            [0, 1], d["psi0"], d["psi1"], d["n_so"], d["Ne"]
        )
        for rho in [rho0, rho1]:
            assert np.allclose(rho, rho.conj().T, atol=1e-10)

    def test_eigenvalues_nonnegative(self, l2_system):
        d = l2_system
        rho0, rho1 = compute_pure_and_excited_RDMs(
            [0], d["psi0"], d["psi1"], d["n_so"], d["Ne"]
        )
        for rho in [rho0, rho1]:
            assert np.all(np.linalg.eigvalsh(rho) >= -1e-10)


# ── compute_all_pure_rdms ─────────────────────────────────────────────────────

class TestComputeAllPureRdms:
    @pytest.fixture(scope="class")
    def all_rdms(self, l2_system):
        d = l2_system
        return compute_all_pure_rdms(
            L=d["L"], psi0=d["psi0"], psi1=d["psi1"],
            n_spin_orbitals=d["n_so"], n_electrons=d["Ne"],
            max_n=2
        )

    def test_returns_three_dicts(self, all_rdms):
        assert len(all_rdms) == 3

    def test_rdm_pure_has_correct_number_of_entries(self, all_rdms):
        rdm_pure, _, _ = all_rdms
        # L=2: 2 (1-body) + 2 (2-body) = 4 entries
        assert len(rdm_pure) == 4

    def test_metrics_keys_present(self, all_rdms):
        _, _, metrics = all_rdms
        for info in metrics.values():
            for key in ("trace", "purity", "entropy", "eigvals"):
                assert key in info

    def test_all_traces_close_to_one(self, all_rdms):
        rdm_pure, _, _ = all_rdms
        for rho in rdm_pure.values():
            assert np.isclose(np.trace(rho).real, 1.0, atol=1e-7)
