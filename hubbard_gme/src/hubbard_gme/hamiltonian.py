"""
hamiltonian.py
--------------
Construction of the Hubbard Hamiltonian in second quantization and
exact diagonalization within the number-preserving subspace.

Spin-orbital mapping used throughout:
    orbital index p = 2*j + s
    s = 0  -> spin-up  (up)
    s = 1  -> spin-down (dn)

Supported topologies
--------------------
uniform   : 1-D ring, all bonds = t, PBC
             H = -t Σ_j (c†_j c_{j+1} + h.c.) + U Σ_j n_{jup} n_{jdn}

open      : 1-D open chain, all bonds = t, NO PBC
             Same as uniform but bond (L-1)→0 is removed.

nnn       : 1-D ring + next-nearest-neighbour hopping t'
             H = -t Σ_j c†_j c_{j+1} - t' Σ_j c†_j c_{j+2} + h.c. + U terms
             Extra param: t_prime (default 0.3*t)

ssh       : Su-Schrieffer-Heeger model, alternating t1/t2, OPEN BC
             Bonds: t1 on (0-1),(2-3),...  t2 on (1-2),(3-4),...
             Models polyacetylene / topological insulators.
             Extra param: t2 (default 0.5*t), t1 = t

ladder    : Two-leg ladder (requires even L)
             Leg-0 sites: 0 ... L/2-1
             Leg-1 sites: L/2 ... L-1
             Leg bonds: j <-> j+1 in each leg (open ends per leg)
             Rung bonds: j <-> j+L/2
             Extra param: t_perp (default t)
"""

from __future__ import annotations

import numpy as np
from openfermion import FermionOperator
from openfermion.linalg import get_number_preserving_sparse_operator
from scipy.sparse.linalg import eigsh

# All valid topology names
VALID_MODES = ("uniform", "open", "nnn", "ssh", "ladder",
               "dimerized", "heisenberg")


# ── Hopping pattern builders ─────────────────────────────────────────────────

def _bonds_uniform(L: int, t: float) -> list[tuple[int, int, float]]:
    """Ring: all NN bonds equal t with PBC."""
    bonds = []
    for j in range(L):
        k = (j + 1) % L
        bonds.append((j, k, t))
    return bonds


def _bonds_open(L: int, t: float) -> list[tuple[int, int, float]]:
    """Open chain: all NN bonds equal t, no PBC."""
    bonds = []
    for j in range(L - 1):
        bonds.append((j, j + 1, t))
    return bonds


def _bonds_nnn(L: int, t: float, t_prime: float) -> list[tuple[int, int, float]]:
    """Ring: NN hopping t + NNN hopping t'."""
    bonds = []
    for j in range(L):
        bonds.append((j, (j + 1) % L, t))        # NN
        bonds.append((j, (j + 2) % L, t_prime))  # NNN
    return bonds


def _bonds_ssh(L: int, t1: float, t2: float) -> list[tuple[int, int, float]]:
    """
    SSH model (open chain, alternating bonds).
    Bond pattern: t1-t2-t1-t2-...
    t1 on (0-1), (2-3), (4-5), ...
    t2 on (1-2), (3-4), (5-6), ...
    """
    bonds = []
    for j in range(L - 1):
        amp = t1 if j % 2 == 0 else t2
        bonds.append((j, j + 1, amp))
    return bonds


def _bonds_ladder(L: int, t: float, t_perp: float) -> list[tuple[int, int, float]]:
    """
    Two-leg ladder topology (L must be even).
    Leg-0: sites 0 .. L/2-1  (bottom)
    Leg-1: sites L/2 .. L-1  (top)
    Leg hopping: t along each leg (open ends)
    Rung hopping: t_perp connecting site j (leg-0) to site j+L/2 (leg-1)
    """
    if L % 2 != 0:
        raise ValueError(f"ladder mode requires even L, got L={L}")
    half = L // 2
    bonds = []
    # Bottom leg
    for j in range(half - 1):
        bonds.append((j, j + 1, t))
    # Top leg
    for j in range(half, L - 1):
        bonds.append((j, j + 1, t))
    # Rungs
    for j in range(half):
        bonds.append((j, j + half, t_perp))
    return bonds


def _bonds_dimerized(L: int, t: float) -> list[tuple[int, int, float]]:
    """Legacy dimerized: active bonds (0-1),(2-3),... PBC implied but interdimer=0."""
    bonds = []
    for d in range(L // 2):
        a, b = 2 * d, 2 * d + 1
        bonds.append((a, b, t))
    return bonds


# ── Main Hamiltonian builder ──────────────────────────────────────────────────

def build_hubbard_hamiltonian(
    L: int,
    t: float,
    U: float,
    mode: str = "uniform",
    t_prime: float = None,
    t2:      float = None,
    t_perp:  float = None,
    V: float =0.0,
) -> FermionOperator:
    """
    Build the Hubbard Hamiltonian for the chosen topology.

        H = -Σ_{bonds(j,k)} t_{jk} Σ_σ (c†_{jσ} c_{kσ} + h.c.)
            + U Σ_j n_{jup} n_{jdn}

    Parameters
    ----------
    L       : Number of lattice sites.
    t       : Primary hopping amplitude.
    U       : On-site Coulomb repulsion.
    mode    : Topology — see module docstring for full list.
    t_prime : NNN hopping (nnn mode, default 0.3*t).
    t2      : Weak bond hopping (ssh mode, default 0.5*t).
    t_perp  : Rung hopping (ladder mode, default t).
    V       : On-site potential (default 0.0).

    Returns
    -------
    FermionOperator
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"Unknown mode '{mode}'. Valid options: {VALID_MODES}"
        )

    # ── Build bond list ──────────────────────────────────────────────────
    if mode == "uniform":
        bonds = _bonds_uniform(L, t)

    elif mode == "open":
        bonds = _bonds_open(L, t)

    elif mode == "nnn":
        tp = t_prime if t_prime is not None else 0.3 * t
        bonds = _bonds_nnn(L, t, tp)

    elif mode == "ssh":
        t2_ = t2 if t2 is not None else 0.5 * t
        bonds = _bonds_ssh(L, t1=t, t2=t2_)

    elif mode == "ladder":
        tp_ = t_perp if t_perp is not None else t
        bonds = _bonds_ladder(L, t, tp_)

    elif mode in ("dimerized", "heisenberg"):
        # legacy modes kept for backward compatibility
        bonds = _bonds_dimerized(L, t)

    # ── Assemble hopping terms ───────────────────────────────────────────
    H = FermionOperator()

    for j, k, tjk in bonds:
        if abs(tjk) < 1e-14:
            continue
        # Spin-up   (orbitals 2j, 2k)
        H += -tjk * FermionOperator(((2*j,   1), (2*k,   0)))
        H += -tjk * FermionOperator(((2*k,   1), (2*j,   0)))
        # Spin-down (orbitals 2j+1, 2k+1)
        H += -tjk * FermionOperator(((2*j+1, 1), (2*k+1, 0)))
        H += -tjk * FermionOperator(((2*k+1, 1), (2*j+1, 0)))

    # ── On-site interaction terms ────────────────────────────────────────
    for j in range(L):
        n_up = FermionOperator(((2*j,   1), (2*j,   0)))
        n_dn = FermionOperator(((2*j+1, 1), (2*j+1, 0)))
        H += U * n_up * n_dn

    # ── Nearest-neighbor potential terms ──────────────────────────────────
    if abs(V) > 1e-14:
        for j in range(L):
            k = (j+1) %L
            n_j= (FermionOperator(((2*j,1), (2*j,0))) + FermionOperator(((2*j+1,1), (2*j+1,0))))
            n_k= (FermionOperator(((2*k,1),(2*k,0))) + FermionOperator(((2*k+1,1), (2*k+1,0))))
            H += V * n_j * n_k
 
    return H


# ── Diagonalization ──────────────────────────────────────────────────────────

def _diagonalize(H, n_spin_orbitals, n_electrons, k, spin_preserving=True):
    H_sparse = get_number_preserving_sparse_operator(
        H, n_spin_orbitals, n_electrons, spin_preserving=spin_preserving
    )
    dim = H_sparse.shape[0]
    k   = min(k, dim - 1)
    E, vecs = eigsh(H_sparse, k=k, which="SA")
    order   = np.argsort(E)
    return E[order], vecs[:, order], n_spin_orbitals


def compute_ground_and_excited_states(
    L: int,
    t: float,
    U: float,
    mode: str,
    n_electrons: int,
    t_prime: float = None,
    t2:      float = None,
    t_perp:  float = None,
    spin_preserving: bool = True,
    V:       float = 0.0,
) -> tuple[float, np.ndarray, float, np.ndarray, int]:
    """
    Return ground state and first excited state.
    Returns: E0, psi0, E1, psi1, n_spin_orbitals
    """
    n_so = 2 * L
    H    = build_hubbard_hamiltonian(L, t, U, mode, t_prime, t2, t_perp, V=V)
    E, vecs, n_so = _diagonalize(H, n_so, n_electrons, k=2, spin_preserving=spin_preserving)
    return E[0], vecs[:, 0], E[1], vecs[:, 1], n_so


def compute_k_eigenstates(
    L: int,
    t: float,
    U: float,
    mode: str,
    n_electrons: int,
    k: int = 4,
    t_prime: float = None,
    t2:      float = None,
    t_perp:  float = None,
    spin_preserving: bool = True,
    V:       float = 0.0,
    
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Return the k lowest eigenstates.
    Returns: energies (k,), eigvecs (dim, k), n_spin_orbitals
    """
    n_so = 2 * L
    H    = build_hubbard_hamiltonian(L, t, U, mode, t_prime, t2, t_perp, V=V    )
    return _diagonalize(H, n_so, n_electrons, k, spin_preserving=spin_preserving)


def compute_state_at_index(
    L: int,
    t: float,
    U: float,
    mode: str,
    n_electrons: int,
    state_idx: int,
    k_min: int = 8,
    t_prime: float = None,
    t2:      float = None,
    t_perp:  float = None,
    spin_preserving: bool = True,
    V:       float = 0.0,
) -> tuple[float, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compute a specific eigenstate by 0-indexed position.

    Used for "first excited singlet" type queries where the user asks
    for, e.g. eigenstate number 5 (1-indexed) = index 4 (0-indexed).

    Parameters
    ----------
    L, t, U, mode, n_electrons : Hubbard parameters
    state_idx                  : 0-indexed eigenstate position (0 = ground)
    k_min                      : minimum number of states to diagonalise
                                 (k = max(state_idx + 2, k_min) so that
                                 enough states are computed for spin
                                 classification)
    t_prime, t2, t_perp        : topology extras

    Returns
    -------
    E_target        : energy of the chosen state
    psi_target      : eigenvector at position state_idx
    E_next          : energy of state at state_idx + 1 (used as 'psi1' for
                      backward compatibility with the existing pipeline)
    psi_next        : eigenvector at state_idx + 1
    all_energies    : array of all k computed energies (for spin table)
    all_eigvecs     : matrix of all k eigenvectors as columns
    n_spin_orbitals : 2 * L
    """
    if state_idx < 0:
        raise ValueError(f"state_idx must be >= 0, got {state_idx}")

    k = max(state_idx + 2, k_min)
    energies, eigvecs, n_so = compute_k_eigenstates(
        L=L, t=t, U=U, mode=mode, n_electrons=n_electrons,
        k=k, t_prime=t_prime, t2=t2, t_perp=t_perp, V=V,
        spin_preserving=spin_preserving,
    )

    if state_idx >= energies.shape[0]:
        raise ValueError(
            f"state_idx={state_idx} out of range — only {energies.shape[0]} "
            f"states available. Try increasing k_min."
        )

    next_idx = min(state_idx + 1, energies.shape[0] - 1)

    return (
        float(energies[state_idx]),
        eigvecs[:, state_idx],
        float(energies[next_idx]),
        eigvecs[:, next_idx],
        energies,
        eigvecs,
        n_so,
    )


def classify_eigenstates_spin(
    eigvecs: np.ndarray,
    energies: np.ndarray,
    L: int,
    n_electrons: int,
    n_spin_orbitals: int,
    spin_preserving: bool = True,
) -> list[dict]:
    """
    For each eigenstate, compute <S^2>, S, and a spin-state label.

    Useful for confirming which state is the first excited singlet.

    Returns
    -------
    list of dicts with keys:
        idx, energy, S2, S, multiplicity, spin_label

    Note: <S_z> per state is NOT reported here — within a degenerate energy
    block the solver returns arbitrary superpositions of the (2S+1) components,
    so a per-state <S_z> is meaningless. To resolve S_z, diagonalise the S_z
    operator inside each degenerate block (see main.py --spectrum).
    """
    from openfermion.hamiltonians import s_squared_operator
    from openfermion.linalg import get_number_preserving_sparse_operator

    _LABELS = {
        0.0: "Singlet (S=0)",
        0.5: "Doublet (S=1/2)",
        1.0: "Triplet (S=1)",
        1.5: "Quartet (S=3/2)",
        2.0: "Quintet (S=2)",
        2.5: "Sextet (S=5/2)",
        3.0: "Septet (S=3)",
    }

    S2_sparse = get_number_preserving_sparse_operator(
        s_squared_operator(L), n_spin_orbitals, n_electrons,
        spin_preserving=spin_preserving,
    )

    info = []
    for i in range(eigvecs.shape[1]):
        psi   = eigvecs[:, i]
        S2    = float(np.real(np.vdot(psi, S2_sparse @ psi)))
        S     = float(0.5 * (-1.0 + np.sqrt(max(0.0, 1.0 + 4.0 * S2))))
        mult  = 2.0 * S + 1.0
        label = next(
            (v for k_, v in _LABELS.items() if abs(S - k_) < 1e-3),
            f"S={S:.3f}",
        )
        info.append({
            "idx":          i,
            "energy":       float(energies[i]),
            "S2":           S2,
            "S":            S,
            "multiplicity": mult,
            "spin_label":   label,
        })

    return info


def print_spin_classification(
    energies: np.ndarray,
    eigvecs: np.ndarray,
    *,
    L: int,
    n_electrons: int,
    n_spin_orbitals: int,
    spin_preserving: bool = True,
    k: int = 16,
    header: str = "Eigenstate spin classification",
) -> list[dict]:
    """
    Classify the first ``k`` eigenstates by <S^2> and print the table.

    Single shared implementation used by EVERY analysis path (ground-state
    scans, thermal, tT-surface, and the excited-singlet search), so the table
    looks identical everywhere. Returns the spin table so callers that need it
    (e.g. locating the first singlet) can reuse it without re-classifying.
    """
    k = int(min(k, energies.shape[0]))
    spin_table = classify_eigenstates_spin(
        eigvecs=eigvecs[:, :k], energies=energies[:k],
        L=L, n_electrons=n_electrons, n_spin_orbitals=n_spin_orbitals,
        spin_preserving=spin_preserving,
    )
    print(f"\n  {header}:")
    print(f"  {'idx':>3}  {'E':>14}  {'<S²>':>8}  {'S':>5}  label")
    print("  " + "-" * 50)
    for row in spin_table:
        print(f"  {row['idx']:>3}  {row['energy']:>14.8f}  "
              f"{row['S2']:>8.4f}  {row['S']:>5.3f}  {row['spin_label']}")
    return spin_table


# ── Convenience operators ────────────────────────────────────────────────────

def compute_all_eigenstates(
    L: int,
    t: float,
    U: float,
    mode: str,
    n_electrons: int,
    t_prime: float = None,
    t2:      float = None,
    t_perp:  float = None,
    spin_preserving: bool = True,
    V:       float = 0.0,
    classify_spin: bool = False,
    classify_k:    int = 16,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Full dense diagonalization — returns ALL eigenstates.

    classify_spin : if True, print the <S^2> spin-classification table for the
                    first ``classify_k`` eigenstates (the same table the
                    excited-singlet search prints). Opt-in so quiet/internal
                    calls stay silent.

    Uses numpy.linalg.eigh on the dense Hamiltonian matrix.
    Suitable for small systems (L ≤ 8, dim ≤ ~1000).

    spin_preserving=True  : restrict to Sz=0 sector (n_up = n_down = Ne/2)
    spin_preserving=False : full Ne-electron space (all Sz sectors combined)

    Returns
    -------
    energies : (dim,) array of all eigenvalues, ascending
    eigvecs  : (dim, dim) array, column k = eigenvector for energies[k]
    n_spin_orbitals : 2 * L
    """
    n_so = 2 * L
    H    = build_hubbard_hamiltonian(L, t, U, mode, t_prime, t2, t_perp, V=V)
    H_sp = get_number_preserving_sparse_operator(
        H, n_so, n_electrons, spin_preserving=spin_preserving
    )
    energies, eigvecs = np.linalg.eigh(H_sp.toarray())
    if classify_spin:
        print_spin_classification(
            energies, eigvecs, L=L, n_electrons=n_electrons,
            n_spin_orbitals=n_so, spin_preserving=spin_preserving,
            k=classify_k,
        )
    return energies, eigvecs, n_so


def double_occupancy_operator(n_spin_orbitals: int, L: int) -> FermionOperator:
    """Return D = Σ_j n_{jup} n_{jdn}."""
    D = FermionOperator()
    for j in range(L):
        up = 2 * j
        dn = 2 * j + 1
        D += FermionOperator(((up, 1), (up, 0), (dn, 1), (dn, 0)), 1.0)
    return D