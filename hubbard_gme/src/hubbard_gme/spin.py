"""
spin.py
-------
Spin observables for a Hubbard ground state:

    - ⟨S_z⟩  (z-projection of total spin)
    - ⟨S²⟩   (total-spin squared → determines S and multiplicity)
    - Double occupancy D = Σ_j ⟨n_{j↑} n_{j↓}⟩

Returns a Pandas DataFrame so results slot naturally into further analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from openfermion.hamiltonians import sz_operator, s_squared_operator
from openfermion.linalg import get_number_preserving_sparse_operator


# ── Spin state classification ────────────────────────────────────────────────

_SPIN_LABEL: dict[float, str] = {
    0.0: "Singlet (S=0)",
    0.5: "Doublet (S=1/2)",
    1.0: "Triplet (S=1)",
    1.5: "Quartet (S=3/2)",
    2.0: "Quintet (S=2)",
    2.5: "Sextet (S=5/2)",
    3.0: "Septet (S=3)",
}


def _classify_spin(s_value: float, atol: float = 1e-3) -> str:
    for s_ref, label in _SPIN_LABEL.items():
        if np.isclose(s_value, s_ref, atol=atol):
            return label
    return f"Unknown (S≈{s_value:.3f})"


# ── Main entry point ─────────────────────────────────────────────────────────

def compute_spin_properties(
    U: float,
    t: float,
    L: int,
    n_spin_orbitals: int,
    eigvec: np.ndarray,
    actual_electrons: int,
    ground_state_eng: float,
    excited_state1_eng: float,
    double_occ: float,
) -> pd.DataFrame:
    """
    Compute ⟨S_z⟩ and ⟨S²⟩ for *eigvec* and return a summary DataFrame.

    Parameters
    ----------
    U, t              : Hubbard parameters (stored in output for labelling).
    L                 : Number of sites.
    n_spin_orbitals   : 2 * L.
    eigvec            : Wavefunction in the number-preserving basis.
    actual_electrons  : Total electron count.
    ground_state_eng  : E₀ (stored in output).
    excited_state1_eng: E₁ (stored in output).
    double_occ        : ⟨D̂⟩ total (stored in output).

    Returns
    -------
    pd.DataFrame with one row containing all properties.
    """
    # ── ⟨S_z⟩ ──
    sz_op = sz_operator(L)
    Sz_sparse =  get_number_preserving_sparse_operator(
        sz_op, n_spin_orbitals, actual_electrons, spin_preserving=True
    )
    exp_sz = float(np.vdot(eigvec, Sz_sparse.dot(eigvec)).real)

    # ── ⟨S²⟩ ──
    s2_op = s_squared_operator(L)
    S2_sparse = get_number_preserving_sparse_operator(
        s2_op, n_spin_orbitals, actual_electrons, spin_preserving=True
    )
    exp_s2 = float(np.vdot(eigvec, S2_sparse.dot(eigvec)).real)

    # ── Derive S from ⟨S²⟩ = S(S+1) ──
    s_value = 0.5 * (-1.0 + np.sqrt(1.0 + 4.0 * exp_s2))
    spin_label = _classify_spin(s_value)
    multiplicity = 2.0 * s_value + 1.0

    return pd.DataFrame(
        {
            "U":                    [U],
            "t":                    [t],
            'Ne ("ψ₀")':           [actual_electrons],
            "ψ₀ Energy":            [ground_state_eng],
            "ψ₁ Energy":            [excited_state1_eng],
            "⟨S²⟩":                 [exp_s2],
            "S":                    [s_value],
            "⟨Sz⟩":                 [exp_sz],
            "Spin State":           [spin_label],
            "⟨D⟩ total":            [double_occ],
            "⟨D⟩ / site":           [double_occ / L],
            "Multiplicity (2S+1)":  [multiplicity],
        }
    )
