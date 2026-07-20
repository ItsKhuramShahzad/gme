"""
molecular.py
------------
Core molecular routines for the hubbard_gme package:

    build_molecular_hamiltonian(mol_data, cas, basis) -> H_sparse + metadata
    compute_molecular_ground_state(H_sparse, k=2)     -> E0, E1, psi0
    compute_molecular_spin_docc(psi0, norb_cas, ...)  -> S^2, S, Sz, label, <D>
    compute_molecular_nrdm(sites, psi, n_so, ne)      -> (2^n, 2^n) matrix
    compute_all_molecular_rdms(psi, ...)              -> {sites: rho}
    compute_rdm_properties(rho)                       -> trace, purity, ...

Notes
-----
* RDMs are normalised so Tr(rho) = 1.
* The RDM return dict uses tuple-of-site-indices as key, matching the
  convention used throughout hubbard_gme (rdm.py, visualization.py, etc.).
* The HDF5 cache file created by openfermionpyscf is deleted after each
  Hamiltonian build so that repeated runs do not fill the disk.
"""

from __future__ import annotations

import os
import warnings
from itertools import combinations
from typing import Dict, Tuple

import numpy as np
from scipy.sparse.linalg import eigsh

import openfermion
import openfermionpyscf
from openfermion.transforms import get_fermion_operator
from openfermion.linalg import get_number_preserving_sparse_operator
from openfermion.ops import FermionOperator
from openfermion.hamiltonians import sz_operator, s_squared_operator

from hubbard_gme.hamiltonian import double_occupancy_operator


# ──────────────────────────────────────────────────────────────────────────────
# Hamiltonian
# ──────────────────────────────────────────────────────────────────────────────

def build_molecular_hamiltonian(
    mol_data: dict,
    cas: dict,
    basis: str = "cc-pVDZ",
    delete_hdf5: bool = True,
) -> Tuple:
    """
    Build the active-space Hamiltonian in spin-orbital basis.

    Parameters
    ----------
    mol_data    : entry from MOLECULES dict
    cas         : {ncore, nele_cas, norb_cas}
    basis       : basis set name
    delete_hdf5 : remove openfermionpyscf HDF5 cache after build

    Returns
    -------
    H_sparse     : scipy sparse matrix (number-preserving, spin-preserving)
    hf_energy    : float (Hartree)
    n_so         : int, number of spin-orbitals in active space (2 * norb_cas)
    ne           : int, active electrons
    norb_cas     : int
    hilbert_dim  : int
    """
    charge = mol_data.get("charge", 0)
    mult   = mol_data.get("multiplicity", 1)

    ncore    = cas["ncore"]
    norb_cas = cas["norb_cas"]
    nele_cas = cas["nele_cas"]

    n_so = 2 * norb_cas
    ne   = nele_cas

    mol      = openfermion.MolecularData(mol_data["geometry"], basis, mult, charge)
    # print("Geometry", mol_data["geometry"])
    # print("mol", mol)
    molecule = openfermionpyscf.run_pyscf(mol, run_scf=True, run_fci=False)

    # Delete the HDF5 cache file immediately — prevents disk from filling up
    if delete_hdf5:
        hdf5_path = mol.filename + ".hdf5"
        if os.path.exists(hdf5_path):
            try:
                os.remove(hdf5_path)
            except OSError:
                pass

    occupied_indices = list(range(ncore))
    active_indices   = list(range(ncore, ncore + norb_cas))

    molecular_ham = molecule.get_molecular_hamiltonian(
        occupied_indices=occupied_indices,
        active_indices=active_indices,
    )
    fermion_ham = get_fermion_operator(molecular_ham)

    H_sparse = get_number_preserving_sparse_operator(
        fermion_ham,
        num_qubits=n_so,
        num_electrons=ne,
        spin_preserving=True,
    )

    hilbert_dim = H_sparse.shape[0]
    return H_sparse, float(molecule.hf_energy), n_so, ne, norb_cas, hilbert_dim


# ──────────────────────────────────────────────────────────────────────────────
# Ground state
# ──────────────────────────────────────────────────────────────────────────────

def compute_molecular_ground_state(H_sparse, k: int = 2):
    """
    Diagonalise the sparse Hamiltonian for the k lowest eigenpairs.

    Returns
    -------
    E0   : float   — ground-state energy (Hartree)
    E1   : float   — first-excited-state energy (Hartree, = E0 if k < 2)
    psi0 : ndarray — ground-state vector
    """
    dim = H_sparse.shape[0]
    k   = max(1, min(k, dim - 1))
    energies, vecs = eigsh(H_sparse, k=k, which="SA")

    order    = np.argsort(energies)
    energies = energies[order]
    vecs     = vecs[:, order]

    E0 = float(energies[0])
    E1 = float(energies[1]) if k >= 2 else E0
    psi0 = vecs[:, 0]
    return E0, E1, psi0


def compute_molecular_k_eigenstates(H_sparse, k: int = 10):
    """
    Diagonalise for the k lowest eigenpairs (for thermal Gibbs ensembles).

    Returns
    -------
    energies : (k,) ndarray, ascending (Hartree)
    vecs     : (dim, k) ndarray, column m = eigenvector for energies[m]
    """
    dim = H_sparse.shape[0]
    k   = max(1, min(k, dim - 1))
    energies, vecs = eigsh(H_sparse, k=k, which="SA")
    order = np.argsort(energies)
    return energies[order], vecs[:, order]


# ──────────────────────────────────────────────────────────────────────────────
# Spin + double occupancy
# ──────────────────────────────────────────────────────────────────────────────

_SPIN_LABELS = {
    0.0: "Singlet (S=0)",
    0.5: "Doublet (S=1/2)",
    1.0: "Triplet (S=1)",
    1.5: "Quartet (S=3/2)",
    2.0: "Quintet (S=2)",
    2.5: "Sextet (S=5/2)",
    3.0: "Septet (S=3)",
}


def compute_molecular_spin_docc(psi0, norb_cas: int, n_so: int, ne: int):
    """
    Compute <S^2>, S, <Sz>, spin label, and double-occupancy <D>.

    Same approach as hubbard_gme.spin but without the pandas wrapper.
    """
    Sz_sparse = 0.5 * get_number_preserving_sparse_operator(
        sz_operator(norb_cas), n_so, ne, spin_preserving=True,
    )
    Sz_val = float(np.real(np.vdot(psi0, Sz_sparse @ psi0)))

    S2_sparse = get_number_preserving_sparse_operator(
        s_squared_operator(norb_cas), n_so, ne, spin_preserving=True,
    )
    S2_val = float(np.real(np.vdot(psi0, S2_sparse @ psi0)))
    S_val  = float(0.5 * (-1.0 + np.sqrt(max(0.0, 1.0 + 4.0 * S2_val))))

    spin_label = next(
        (v for k, v in _SPIN_LABELS.items() if abs(S_val - k) < 1e-3),
        f"S={S_val:.3f}",
    )

    D_sparse = get_number_preserving_sparse_operator(
        double_occupancy_operator(n_so, norb_cas), n_so, ne, spin_preserving=True,
    )
    D_total = float(np.real(np.vdot(psi0, D_sparse @ psi0)))

    return S2_val, S_val, Sz_val, spin_label, D_total


# ──────────────────────────────────────────────────────────────────────────────
# n-body RDMs
# ──────────────────────────────────────────────────────────────────────────────

def compute_molecular_nrdm(
    orbital_indices,
    psi,
    n_spin_orbitals: int,
    n_electrons: int,
    spin_preserving: bool = True,
    min_raw_trace: float = 1e-10,
):
    """
    ...
    Returns
    -------
    rho : (2^n, 2^n) ndarray, normalised so Tr(rho) = 1,
          OR None if the raw (pre-normalisation) trace is below
          ``min_raw_trace``. A near-zero raw trace means the interleaved
          n-body RDM is ill-defined for this state (typically happens
          when n == norb_cas and the state has significant double
          occupancy — diagonal elements measure the probability of
          "one electron per orbital everywhere").
    """
    n   = len(orbital_indices)
    dim = 2 ** n
    rho = np.zeros((dim, dim), dtype=complex)

    for bra_idx in range(dim): # bra_spins
        for ket_idx in range(dim): # ket_spins
            bra_spins = [(bra_idx >> k) & 1 for k in range(n)]
            ket_spins = [(ket_idx >> k) & 1 for k in range(n)]
            op_terms = []
            for body in range(n):
                site = orbital_indices[body]
                p = 2 * site + bra_spins[body] # d†_{i,σ'_i}
                q = 2 * site + ket_spins[body] # d_{i,σ_i}
                op_terms.append((p, 1))
                op_terms.append((q, 0))
            op = FermionOperator(tuple(op_terms))
            op_sparse = get_number_preserving_sparse_operator(
                op, n_spin_orbitals, n_electrons, spin_preserving=spin_preserving,
            )
            rho[bra_idx, ket_idx] = np.vdot(psi, op_sparse @ psi) # Γ^(n)_{σ,σ'}

    raw_trace = float(np.real(np.trace(rho)))
    if abs(raw_trace) < min_raw_trace:
        warnings.warn(
            f"RDM over orbitals {list(orbital_indices)}: raw trace "
            f"{raw_trace:.2e} < {min_raw_trace:.0e}. "
            f"This happens when the body order equals the number of "
            f"orbitals and the state has significant double/empty "
            f"occupancy. The interleaved n-RDM is ill-defined here — "
            f"skipping this cluster.",
            stacklevel=2,
        )
        return None

    return rho / raw_trace

def compute_all_molecular_rdms(
    psi,
    norb_cas: int,
    n_spin_orbitals: int,
    n_electrons: int,
    max_n_body: int = 3,
    min_n_body: int = 1,
) -> Dict[Tuple[int, ...], np.ndarray]:
    """
    Compute every k-body RDM for k = min_n_body ... max_n_body over all
    orbital combinations in the active space.

    Returns
    -------
    {tuple_of_sites: rho}
    """
    rdm_dict: Dict[Tuple[int, ...], np.ndarray] = {}

    for n_body in range(min_n_body, max_n_body + 1):
        if n_body > n_electrons:
            # Cannot trace a k-body object on a state with fewer electrons
            continue
        if n_body > norb_cas:
            continue
        for sites in combinations(range(norb_cas), n_body):
            rho = compute_molecular_nrdm(
                list(sites), psi, n_spin_orbitals, n_electrons,
            )
            if rho is None:       # degenerate, warning already emitted
                continue
            rdm_dict[tuple(sites)] = rho

    return rdm_dict


# ──────────────────────────────────────────────────────────────────────────────
# RDM properties
# ──────────────────────────────────────────────────────────────────────────────

def compute_rdm_properties(rho: np.ndarray) -> dict:
    """
    Compute trace, purity, von Neumann entropy, eigenvalues, rank,
    and a qualitative 'state_type' label.
    """
    trace   = float(np.real(np.trace(rho)))
    purity  = float(np.real(np.trace(rho @ rho)))
    eigvals = np.linalg.eigvalsh(rho)
    entropy = float(-sum(v * np.log(v) for v in eigvals if v > 1e-12))
    rank    = int(np.sum(eigvals > 1e-10))

    if   abs(purity - 1.0) < 1e-6: state_type = "PURE"
    elif purity > 0.5:             state_type = "WEAKLY MIXED"
    elif purity > 0.2:             state_type = "MIXED"
    else:                          state_type = "STRONGLY MIXED"

    return dict(
        trace=trace,
        purity=purity,
        entropy=entropy,
        eigvals=eigvals,
        rank=rank,
        state_type=state_type,
    )


def build_metrics_dict(rdm_dict: Dict[Tuple[int, ...], np.ndarray]) -> dict:
    """Apply compute_rdm_properties to every RDM in rdm_dict."""
    return {sites: compute_rdm_properties(rho) for sites, rho in rdm_dict.items()}