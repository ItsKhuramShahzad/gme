"""
molecular_localize.py
=====================

Build the active-space Hamiltonian in a LOCALIZED orbital basis (PySCF's
built-in pyscf.lo — Boys / Pipek-Mezey / IBO), so the n-RDMs and GME describe
entanglement between *localized* (atom/bond) orbitals instead of the canonical,
delocalized ones.

Localization is a unitary rotation WITHIN the active space, so the CASCI
ground-state energy is unchanged — only the orbital basis (hence the RDMs and
GME) differs. :func:`build_localized_molecular_hamiltonian` therefore returns a
Hamiltonian whose ground-state energy must match the canonical one; the module
self-check verifies exactly that.

The returned H_sparse plugs straight into the existing machinery
(compute_molecular_ground_state, compute_molecular_nrdm, run_convex_roof_on_rdm).
"""
from __future__ import annotations

import numpy as np
from pyscf import ao2mo, gto, lo, mcscf, scf

from openfermion import InteractionOperator, get_fermion_operator
from openfermion.linalg import get_number_preserving_sparse_operator

try:                                    # location moved across openfermion versions
    from openfermion.chem.molecular_data import spinorb_from_spatial
except Exception:                       # pragma: no cover
    from openfermion.ops.representations import spinorb_from_spatial


_METHODS = {"boys": "Boys", "pm": "Pipek-Mezey", "ibo": "IBO"}


def localize_active_orbitals(mol, mf, active_indices, method="pm"):
    """Return localized coefficients (nao x n_active) for the active MOs."""
    C = mf.mo_coeff[:, active_indices]
    if method == "boys":
        return lo.Boys(mol, C).kernel()
    if method == "pm":
        return lo.PipekMezey(mol, C).kernel()
    if method == "ibo":
        a = lo.iao.iao(mol, C)
        a = lo.vec_lowdin(a, mf.get_ovlp())
        return lo.ibo.ibo(mol, C, iaos=a)
    raise ValueError(f"unknown localization method {method!r}")


def build_localized_molecular_hamiltonian(mol_data, cas, basis="cc-pVDZ",
                                          method="pm"):
    """
    Active-space Hamiltonian in a localized orbital basis.

    Returns a dict with:
      H_sparse   number-preserving sparse Hamiltonian (localized basis)
      hf_energy, n_so, ne, norb, ecore
      mol, mf, loc_coeff   (for cube/visualization or Mulliken analysis)
      method
    """
    ncore = int(cas["ncore"])
    norb = int(cas["norb_cas"])
    ne = int(cas["nele_cas"])
    n_so = 2 * norb

    atoms = [[el, tuple(xyz)] for (el, xyz) in mol_data["geometry"]]
    mol = gto.M(atom=atoms, basis=basis,
                charge=int(mol_data.get("charge", 0)),
                spin=int(mol_data.get("multiplicity", 1)) - 1, verbose=0)
    mf = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
    mf.kernel()

    active = list(range(ncore, ncore + norb))
    loc = localize_active_orbitals(mol, mf, active, method)

    mo_new = mf.mo_coeff.copy()
    mo_new[:, active] = loc

    # active-space integrals in the localized basis (PySCF folds the frozen core)
    mc = mcscf.CASCI(mf, norb, ne)
    mc.mo_coeff = mo_new
    h1, ecore = mc.get_h1eff()                    # (norb,norb) + core constant
    eri = ao2mo.restore(1, mc.get_h2eff(), norb)  # chemist (pq|rs), full 4-index

    # OpenFermion convention: h[p,q,r,s] = (ps|qr) = chemist.transpose(0,2,3,1),
    # and get_molecular_hamiltonian multiplies the two-body tensor by 1/2.
    two_body_of = np.asarray(eri.transpose(0, 2, 3, 1), order="C")
    one_body, two_body = spinorb_from_spatial(np.asarray(h1), two_body_of)
    ham = InteractionOperator(float(ecore), one_body, 0.5 * two_body)
    fermion_ham = get_fermion_operator(ham)

    H_sparse = get_number_preserving_sparse_operator(
        fermion_ham, num_qubits=n_so, num_electrons=ne, spin_preserving=True)

    return {
        "H_sparse": H_sparse, "hf_energy": float(mf.e_tot),
        "n_so": n_so, "ne": ne, "norb": norb, "ecore": float(ecore),
        "mol": mol, "mf": mf, "loc_coeff": loc, "method": method,
    }


# ── self-check: localized E0 must equal canonical E0 ──────────────────────────

def _selfcheck(mol_name="Methylene", method="pm"):
    from scipy.sparse.linalg import eigsh
    from run_molecules_batch import load_molecules, pick_cas, DEFAULT_MOLECULES_PY
    from hubbard_gme.molecular import (
        build_molecular_hamiltonian, compute_molecular_ground_state)

    mol_data = load_molecules(DEFAULT_MOLECULES_PY)[mol_name]
    cas, _ = pick_cas(mol_data["valid_active_spaces"])

    # canonical
    Hc, *_ = build_molecular_hamiltonian(mol_data, cas, basis="cc-pVDZ")
    E0c, _, _ = compute_molecular_ground_state(Hc, k=2)

    # localized
    info = build_localized_molecular_hamiltonian(mol_data, cas,
                                                 basis="cc-pVDZ", method=method)
    E0l = float(np.sort(eigsh(info["H_sparse"], k=2, which="SA")[0])[0])

    d = abs(E0c - E0l)
    print(f"{mol_name}  CAS({cas['nele_cas']},{cas['norb_cas']})  {_METHODS[method]}")
    print(f"  canonical  E0 = {E0c:.8f} Ha")
    print(f"  localized  E0 = {E0l:.8f} Ha")
    print(f"  |Δ|          = {d:.2e}   {'OK (energy invariant)' if d < 1e-6 else 'MISMATCH!'}")
    return d


if __name__ == "__main__":
    import sys
    m = sys.argv[1] if len(sys.argv) > 1 else "Methylene"
    _selfcheck(m)
