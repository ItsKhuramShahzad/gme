"""
tests/test_hamiltonian.py
-------------------------
Unit tests for hubbard_gme.hamiltonian
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openfermion.linalg import get_number_preserving_sparse_operator
from hubbard_gme.hamiltonian import (
    build_hubbard_hamiltonian,
    compute_ground_and_excited_states,
    double_occupancy_operator,
)


# ── Hamiltonian structure ─────────────────────────────────────────────────────

class TestBuildHubbardHamiltonian:
    def test_returns_fermion_operator(self):
        from openfermion import FermionOperator
        H = build_hubbard_hamiltonian(L=2, t=1.0, U=4.0, mode="uniform")
        assert isinstance(H, FermionOperator)

    def test_hermitian_sparse_matrix(self):
        """Sparse matrix in the 2-electron sector must be Hermitian."""
        L, Ne = 4, 4
        H = build_hubbard_hamiltonian(L=L, t=1.0, U=2.0, mode="uniform")
        H_sp = get_number_preserving_sparse_operator(H, 2*L, Ne, spin_preserving=True)
        H_dense = H_sp.toarray()
        assert np.allclose(H_dense, H_dense.conj().T, atol=1e-12)

    def test_nonzero_for_positive_t(self):
        H = build_hubbard_hamiltonian(L=2, t=1.0, U=0.0, mode="uniform")
        assert len(H.terms) > 0

    def test_u_only_when_t_zero(self):
        """U-only Hamiltonian: all off-diagonal elements should be zero."""
        L, Ne = 2, 2
        H = build_hubbard_hamiltonian(L=L, t=0.0, U=4.0, mode="uniform")
        H_sp = get_number_preserving_sparse_operator(H, 2*L, Ne, spin_preserving=True)
        H_dense = H_sp.toarray()
        off_diag = H_dense - np.diag(np.diag(H_dense))
        assert np.allclose(off_diag, 0.0, atol=1e-12)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            build_hubbard_hamiltonian(L=2, t=1.0, U=1.0, mode="bad_mode")

    @pytest.mark.parametrize("mode", ["uniform", "dimerized", "heisenberg"])
    def test_all_modes_build(self, mode):
        H = build_hubbard_hamiltonian(L=4, t=1.0, U=2.0, mode=mode)
        from openfermion import FermionOperator
        assert isinstance(H, FermionOperator)


# ── Diagonalisation ────────────────────────────────────────────────────────────

class TestDiagonalisation:
    @pytest.fixture(scope="class")
    def two_site_result(self):
        return compute_ground_and_excited_states(
            L=2, t=1.0, U=4.0, mode="uniform", n_electrons=2
        )

    def test_returns_five_values(self, two_site_result):
        assert len(two_site_result) == 5

    def test_E0_less_than_E1(self, two_site_result):
        E0, _, E1, _, _ = two_site_result
        assert E0 <= E1

    def test_psi0_normalized(self, two_site_result):
        _, psi0, _, _, _ = two_site_result
        assert np.isclose(np.linalg.norm(psi0), 1.0, atol=1e-10)

    def test_psi1_normalized(self, two_site_result):
        _, _, _, psi1, _ = two_site_result
        assert np.isclose(np.linalg.norm(psi1), 1.0, atol=1e-10)

    def test_n_spin_orbitals_correct(self, two_site_result):
        *_, n_so = two_site_result
        assert n_so == 4   # 2 * L

    def test_two_site_ground_energy_known(self):
        """
        2-site half-filled Hubbard with PBC.

        With periodic boundary conditions on L=2 sites the loop over j=0,1
        adds BOTH hops (0→1) and (1→0), doubling the effective hopping in
        the Hamiltonian matrix.  The exact singlet ground-state energy is:

            E₀ = U/2 − √((U/2)² + (4t)²)

        Derivation: in the singlet sub-space {|↑↓,0⟩, |0,↑↓⟩} the
        off-diagonal element is −2t (from the doubled hop) × √2 from the
        singlet combination, giving an effective coupling of 4t.
        """
        L, t, U = 2, 1.0, 4.0
        E_analytic = U / 2 - np.sqrt((U / 2) ** 2 + (4 * t) ** 2)
        E0, *_ = compute_ground_and_excited_states(
            L=L, t=t, U=U, mode="uniform", n_electrons=2
        )
        assert np.isclose(E0, E_analytic, atol=1e-8)


# ── Double occupancy operator ─────────────────────────────────────────────────

class TestDoubleOccupancyOperator:
    def test_non_interacting_limit(self):
        """
        For U=0, t≠0 at half-filling: ⟨D⟩ should be approximately 0.25 per site
        (random occupation → independent probabilities of ↑ and ↓ = 0.5 each).
        """
        L, Ne = 4, 4
        n_so = 2 * L
        E0, psi0, _, _, _ = compute_ground_and_excited_states(
            L=L, t=1.0, U=0.0, mode="uniform", n_electrons=Ne
        )
        D_op = double_occupancy_operator(n_so, L)
        D_sp = get_number_preserving_sparse_operator(D_op, n_so, Ne, spin_preserving=True)
        d = float(np.vdot(psi0, D_sp @ psi0).real)
        # Should be close to 1.0 total (0.25/site) for non-interacting half-filling
        assert 0.5 <= d <= 1.5, f"Unexpected double occupancy {d:.4f}"
