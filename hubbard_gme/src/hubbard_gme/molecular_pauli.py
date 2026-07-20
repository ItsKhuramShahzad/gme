"""
molecular_pauli.py
==================

An ALTERNATIVE construction of the molecular n-orbital reduced density matrices
that works **directly on Pauli (sigma) strings**, following the Jordan-Wigner
notes literally, instead of number-preserving fermionic creation/annihilation
operators (as in :func:`molecular.compute_molecular_nrdm`).

Idea (exactly the notes)
------------------------
Every RDM element is

    rho[bra, ket] = <psi|  a†_p1 a_q1  a†_p2 a_q2  ...  |psi>
                  = <psi|  ( sum_k  c_k * P_k )  |psi>            (Jordan-Wigner)
                  = sum_k  c_k * <psi| P_k |psi> ,

where each ``P_k`` is a Pauli string (tensor product of I/X/Y/Z) and ``c_k`` its
Jordan-Wigner coefficient (carrying the +/- sign of the notes).  The
expectation values ``<psi| P_k |psi>`` are evaluated by **direct single-qubit
Pauli action** on the full statevector -- i.e. exactly the quantities you would
*measure* on quantum hardware.

This requires ``psi`` as a FULL 2**N qubit statevector in the Jordan-Wigner
computational basis (N = number of spin-orbitals), not the number-preserving
reduced vector used by the fermionic method.  :func:`full_ground_state` builds
it for a molecule; :func:`nrdm_from_pauli` turns it into an RDM.

The result is identical (to machine precision) to the fermionic method -- see
``python -m hubbard_gme.molecular_pauli`` for the self-check.  Nothing here is
imported by the existing pipeline; it is a standalone, notes-faithful route.
"""
from __future__ import annotations

import os
from itertools import combinations
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import eigsh

from openfermion import (
    FermionOperator, get_fermion_operator, get_sparse_operator, jordan_wigner,
    number_operator,
)
from openfermion.hamiltonians import sz_operator


# ──────────────────────────────────────────────────────────────────────────────
# 1. Pauli-string expectation values (the core "works on Pauli strings" piece)
# ──────────────────────────────────────────────────────────────────────────────

def _apply_single_qubit_pauli(psi_tensor: np.ndarray, qubit: int,
                              which: str) -> np.ndarray:
    """
    Apply one single-qubit Pauli (X/Y/Z) to ``psi_tensor`` (shape [2]*N) on the
    given qubit axis.  Qubit 0 is the leftmost/most-significant factor, matching
    OpenFermion's ``get_sparse_operator`` tensor ordering.
    """
    t = np.moveaxis(psi_tensor, qubit, 0)     # bring qubit axis to front: (2, ...)
    r = np.empty_like(t)
    if which == "X":                          # X = [[0,1],[1,0]]
        r[0], r[1] = t[1], t[0]
    elif which == "Y":                        # Y = [[0,-i],[i,0]]
        r[0], r[1] = -1j * t[1], 1j * t[0]
    elif which == "Z":                        # Z = [[1,0],[0,-1]]
        r[0], r[1] = t[0], -t[1]
    else:
        raise ValueError(f"unknown Pauli {which!r}")
    return np.moveaxis(r, 0, qubit)


def pauli_expectation(psi_full: np.ndarray, term: tuple, n_qubits: int) -> complex:
    """
    <psi| P |psi> for a single Pauli string ``term`` = ((q, 'X'/'Y'/'Z'), ...).
    The empty tuple () is the identity, with expectation <psi|psi> = 1 (assuming
    a normalised state).  Evaluated by direct single-qubit action -- exactly the
    Pauli operators one measures on hardware.
    """
    t = psi_full.reshape([2] * n_qubits)
    for qubit, which in term:
        t = _apply_single_qubit_pauli(t, qubit, which)
    return complex(np.vdot(psi_full, t.reshape(-1)))


def fermion_string_expectation(fermion_op: FermionOperator, psi_full: np.ndarray,
                               n_qubits: int) -> complex:
    """
    <psi| O |psi> for a fermionic operator O, evaluated through Jordan-Wigner:
    map O -> sum_k c_k P_k, then sum c_k <psi|P_k|psi>.  This is the notes'
    recipe: a fermionic string becomes a signed sum of Pauli-string measurements.
    """
    qop = jordan_wigner(fermion_op)
    val = 0.0 + 0.0j
    for term, coeff in qop.terms.items():
        if term == ():                        # identity Pauli
            val += coeff
        else:
            val += coeff * pauli_expectation(psi_full, term, n_qubits)
    return val


# ──────────────────────────────────────────────────────────────────────────────
# 2. n-orbital RDM assembled purely from Pauli-string expectations
# ──────────────────────────────────────────────────────────────────────────────

def nrdm_from_pauli(orbital_indices, psi_full: np.ndarray, n_qubits: int,
                    min_raw_trace: float = 1e-10) -> Optional[np.ndarray]:
    """
    Build the (2**n x 2**n) n-orbital RDM over ``orbital_indices`` from
    Pauli-string expectation values on the full statevector ``psi_full``.

    Mirrors :func:`molecular.compute_molecular_nrdm` element for element, but
    every entry is assembled as  sum_k c_k <psi|P_k|psi>  (Jordan-Wigner /
    notes), never touching a number-preserving fermionic sparse operator.
    Returns the trace-normalised RDM, or None if the raw trace is ~0.
    """
    n = len(orbital_indices)
    dim = 2 ** n
    rho = np.zeros((dim, dim), dtype=complex)

    for bra_idx in range(dim):
        bra_spins = [(bra_idx >> k) & 1 for k in range(n)]
        for ket_idx in range(dim):
            ket_spins = [(ket_idx >> k) & 1 for k in range(n)]
            op_terms = []
            for body in range(n):
                site = orbital_indices[body]
                p = 2 * site + bra_spins[body]        # creation spin-orbital
                q = 2 * site + ket_spins[body]        # annihilation spin-orbital
                op_terms.append((p, 1))
                op_terms.append((q, 0))
            rho[bra_idx, ket_idx] = fermion_string_expectation(
                FermionOperator(tuple(op_terms)), psi_full, n_qubits)

    raw_trace = float(np.real(np.trace(rho)))
    if abs(raw_trace) < min_raw_trace:
        return None
    return rho / raw_trace


def all_nrdms_from_pauli(psi_full: np.ndarray, norb_cas: int, n_qubits: int,
                         n_electrons: int, max_n_body: int = 3,
                         min_n_body: int = 1
                         ) -> Dict[Tuple[int, ...], np.ndarray]:
    """Pauli-route analogue of ``molecular.compute_all_molecular_rdms``."""
    rdm_dict: Dict[Tuple[int, ...], np.ndarray] = {}
    for n_body in range(min_n_body, max_n_body + 1):
        if n_body > n_electrons or n_body > norb_cas:
            continue
        for sites in combinations(range(norb_cas), n_body):
            rho = nrdm_from_pauli(list(sites), psi_full, n_qubits)
            if rho is not None:
                rdm_dict[tuple(sites)] = rho
    return rdm_dict


# ──────────────────────────────────────────────────────────────────────────────
# 3. Full-space (2**N) ground state in the Jordan-Wigner computational basis
# ──────────────────────────────────────────────────────────────────────────────

def _active_fermion_hamiltonian(mol_data: dict, cas: dict, basis: str,
                                delete_hdf5: bool = True):
    """Active-space FermionOperator Hamiltonian (same build as molecular.py)."""
    import openfermion
    import openfermionpyscf

    charge = mol_data.get("charge", 0)
    mult = mol_data.get("multiplicity", 1)
    ncore, norb, nele = cas["ncore"], cas["norb_cas"], cas["nele_cas"]

    mol = openfermion.MolecularData(mol_data["geometry"], basis, mult, charge)
    molecule = openfermionpyscf.run_pyscf(mol, run_scf=True, run_fci=False)
    if delete_hdf5:
        h5 = mol.filename + ".hdf5"
        if os.path.exists(h5):
            try:
                os.remove(h5)
            except OSError:
                pass
    occ = list(range(ncore))
    act = list(range(ncore, ncore + norb))
    mham = molecule.get_molecular_hamiltonian(occupied_indices=occ,
                                              active_indices=act)
    return get_fermion_operator(mham), 2 * norb, nele, norb


def full_ground_state(mol_data: dict, cas: dict, basis: str = "cc-pVDZ",
                      penalty: float = 25.0) -> Tuple[np.ndarray, int, int, int]:
    """
    Ground state as a FULL 2**N statevector in the Jordan-Wigner basis, in the
    (N = n_electrons, Sz = 0) sector -- the same physical state the fermionic
    number/spin-preserving method uses.

    The sector is enforced with quadratic penalties  penalty*(N - ne)^2  and
    penalty*(Sz)^2  added to the Hamiltonian, so the constrained ground state
    lands in the right sector while its energy inside that sector is unchanged.

    Returns (psi_full, n_qubits, n_electrons, norb_cas).
    """
    fh, n_so, ne, norb = _active_fermion_hamiltonian(mol_data, cas, basis)

    H = get_sparse_operator(fh, n_qubits=n_so)
    Nop = get_sparse_operator(number_operator(n_so), n_qubits=n_so)
    Szop = get_sparse_operator(sz_operator(norb), n_qubits=n_so)
    I = identity(2 ** n_so, format="csc", dtype=complex)

    dN = Nop - ne * I
    H_pen = H + penalty * (dN @ dN) + penalty * (Szop @ Szop)

    energies, vecs = eigsh(H_pen, k=2, which="SA")
    psi = vecs[:, int(np.argmin(energies))]
    psi = psi / np.linalg.norm(psi)
    return psi, n_so, ne, norb


# ──────────────────────────────────────────────────────────────────────────────
# 4. Self-check:  Pauli route  ==  fermionic route  (machine precision)
# ──────────────────────────────────────────────────────────────────────────────

def _selfcheck_random(n_orb: int = 3, cluster=(0, 1, 2), seed: int = 0) -> float:
    """
    Validate the Pauli machinery WITHOUT pyscf: on a random full-space state,
    the Pauli-assembled RDM must equal the full-space fermionic RDM element by
    element.  Returns the max absolute difference.
    """
    n_qubits = 2 * n_orb
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(2 ** n_qubits) + 1j * rng.standard_normal(2 ** n_qubits)
    psi /= np.linalg.norm(psi)

    n = len(cluster)
    dim = 2 ** n
    rho_ref = np.zeros((dim, dim), dtype=complex)
    for b in range(dim):
        bs = [(b >> k) & 1 for k in range(n)]
        for k_ in range(dim):
            ks = [(k_ >> t) & 1 for t in range(n)]
            terms = []
            for body in range(n):
                s = cluster[body]
                terms.append((2 * s + bs[body], 1))
                terms.append((2 * s + ks[body], 0))
            op = get_sparse_operator(FermionOperator(tuple(terms)),
                                     n_qubits=n_qubits)
            rho_ref[b, k_] = np.vdot(psi, op @ psi)
    rho_ref /= np.real(np.trace(rho_ref))

    rho_pauli = nrdm_from_pauli(list(cluster), psi, n_qubits)
    return float(np.max(np.abs(rho_pauli - rho_ref)))


if __name__ == "__main__":
    print("Self-check: Pauli-string RDM vs full-space fermionic RDM "
          "(random state, no pyscf)")
    for nb, cl in [(1, (0,)), (2, (0, 1)), (3, (0, 1, 2))]:
        d = _selfcheck_random(n_orb=3, cluster=cl)
        print(f"  {nb}-body cluster {cl}:  max|Δ| = {d:.2e}  "
              f"{'OK' if d < 1e-10 else 'FAIL'}")
