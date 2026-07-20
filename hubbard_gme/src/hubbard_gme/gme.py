"""
gme.py
------
Genuine Multipartite Entanglement (GME) analysis for n-spin RDMs.

Key concepts
------------
Given an n-spin RDM ρ extracted from the many-body ground state:

1.  **Global purity**   γ = Tr(ρ²)
    - γ = 1 → pure n-spin state (GME analysis applies directly)
    - γ < 1 → mixed state (convex-roof extension needed; flagged here)

2.  **Bipartition purity**   γ_A = Tr(ρ_A²), where ρ_A = Tr_B(ρ)
    - γ_A = 1 for some A|B → ρ = ρ_A ⊗ ρ_B → no GME on that cut

3.  **C_GME** (pure state formula from Troiani et al.)
    C_GME = √(2 · min_{all cuts} (1 − Tr(ρ_A²)))
           = √(2 · (1 − max_{all cuts} Tr(ρ_A²)))

4.  **C_GME lower bound** — two implementations available:

    (a) Paper method  [Ma et al., PRA 2011]  — sum over all bipartitions
        I_{x,y}(ρ) = |ρ_{x,y}| − Σ_A √(ρ_{x^A,x^A} · ρ_{y^A,y^A})
        C_GME ≥ max_{x≠y} 2·I_{x,y}(ρ)

    (b) Supervisor method  [eqs. 6-9 of notes]  — sum over sites k only
        A    = |ρ_{z2,z1}|²                           eq.(6)
        B_k  = ρ_{α_k,α_k} · ρ_{β_k,β_k}  per site k eq.(7)
        I_Φ  = 2·(√A − Σ_k √B_k)                     eq.(8)
        I    = max_{Φ} I_Φ  ≤  C_GME(ρ)              eq.(9)

    Both are valid lower bounds. Use supervisor_gme_lower_bound() for
    the supervisor's method and I_function / Cgme_lower_bound() for
    the paper method.

References
----------
F. Troiani et al., Phys. Rev. A 99, 022323 (2019).
Z.-H. Ma et al., Phys. Rev. A 83, 062325 (2011).
"""

from __future__ import annotations

import itertools
import numpy as np
from typing import Dict, List, Optional, Tuple


# ── Purity helpers ────────────────────────────────────────────────────────────

def purity(rho: np.ndarray) -> float:
    """Compute Tr(ρ²)."""
    return float(np.trace(rho @ rho).real)


def is_pure_state(rho: np.ndarray, tol: float = 1e-8) -> bool:
    """Return True if γ = Tr(ρ²) ≈ 1."""
    return abs(purity(rho) - 1.0) < tol


# ── Bipartitions ──────────────────────────────────────────────────────────────

def get_bipartitions(n: int) -> list[tuple[int, ...]]:
    """
    Return all unique non-trivial bipartitions A | B of n spins,
    with the convention that 0 ∈ A (avoids double-counting A↔B).

    Example n=3:
        A=(0,)  B=(1,2)
        A=(0,1) B=(2,)
        A=(0,2) B=(1,)
    """
    parts: list[tuple[int, ...]] = []
    sites = list(range(n))
    for r in range(1, n):
        for A in itertools.combinations(sites, r):
            if 0 in A:
                parts.append(A)
    return parts


def print_bipartitions(n: int) -> None:
    """Pretty-print all unique bipartitions for n spins."""
    parts = get_bipartitions(n)
    print(f"\nAll unique bipartitions for n = {n} spins:")
    for A in parts:
        B = tuple(i for i in range(n) if i not in A)
        print(f"  A = {A}  |  B = {B}")


# ── Partial trace (spin space) ────────────────────────────────────────────────

def partial_trace_spins(
    rho: np.ndarray,
    n: int,
    keep: tuple[int, ...],
) -> np.ndarray:
    """
    Trace out the spin sites NOT listed in *keep* from a 2ⁿ × 2ⁿ density matrix.

    Parameters
    ----------
    rho  : (2ⁿ, 2ⁿ) density matrix.
    n    : Total number of spins.
    keep : Tuple of spin indices to retain.

    Returns
    -------
    np.ndarray of shape (2^|keep|, 2^|keep|).
    """
    if rho.shape[0] != 2 ** n:
        raise ValueError(
            f"rho has shape {rho.shape}, expected ({2**n}, {2**n}) for n={n}."
        )
    keep = tuple(keep)
    trace_out = sorted([i for i in range(n) if i not in keep], reverse=True)

    # Reshape to tensor with explicit bra/ket indices for each spin
    rho_t = rho.reshape([2] * n + [2] * n)

    for s in trace_out:
        n_cur = rho_t.ndim // 2
        rho_t = np.trace(rho_t, axis1=s, axis2=s + n_cur)

    k = len(keep)
    return rho_t.reshape((2 ** k, 2 ** k))


# ── GME for a single RDM ──────────────────────────────────────────────────────

def analyze_one_rdm_for_GME(
    rho: np.ndarray,
    sites: tuple[int, ...],
    tol: float = 1e-8,
) -> Dict:
    """
    Full GME analysis for one n-spin RDM.

    Returns a dictionary with:
        sites          : tuple of site indices
        n              : number of spins
        global_purity  : Tr(ρ²)
        bipartitions   : list of {A, B, purity_A} for every cut
        separable_cuts : cuts where Tr(ρ_A²) ≈ 1
        C_GME          : float or None
        status         : 'GME' | 'no_GME' | 'mixed_state'
    """
    n = len(sites)
    info: Dict = {
        "sites":         sites,
        "n":             n,
        "global_purity": purity(rho),
        "bipartitions":  [],
        "separable_cuts":[],
        "C_GME":         None,
        "status":        None,
    }

    if info["global_purity"] < 1.0 - tol:
        info["status"] = "mixed_state"
        return info

    purities: list[float] = []

    for A in get_bipartitions(n):
        B = tuple(i for i in range(n) if i not in A)
        rhoA = partial_trace_spins(rho, n, A)
        purA = purity(rhoA)

        A_sites = tuple(sites[i] for i in A)
        B_sites = tuple(sites[i] for i in B)

        info["bipartitions"].append({"A": A_sites, "B": B_sites, "purity_A": purA})
        purities.append(purA)

        if abs(purA - 1.0) < tol:
            info["separable_cuts"].append((A_sites, B_sites))

    if info["separable_cuts"]:
        info["status"] = "no_GME"
    else:
        pur_max = max(purities)
        info["C_GME"] = float(np.sqrt(2.0 * (1.0 - pur_max)))
        info["status"] = "GME"

    return info


def check_GME_for_one_rdm(
    rho: np.ndarray,
    sites: tuple[int, ...],
    tol: float = 1e-8,
    verbose: bool = True,
) -> Optional[float]:
    """
    Convenience wrapper: analyse one RDM and print a human-readable report.

    Returns C_GME (float) if GME exists, 0.0 if no GME, None if mixed.
    """
    info = analyze_one_rdm_for_GME(rho, sites, tol)
    n = info["n"]

    if verbose:
        print("\n" + "=" * 65)
        print(f"  Subsystem sites : {sites}")
        print(f"  Global purity   : {info['global_purity']:.8f}")

    if info["status"] == "mixed_state":
        if verbose:
            print("  → Mixed state: pure-state GME formula not applicable.")
        return None

    if verbose:
        print("  State is PURE → checking bipartitions:")
        for entry in info["bipartitions"]:
            marker = "  ← separable!" if abs(entry["purity_A"] - 1.0) < tol else ""
            print(
                f"    A={entry['A']} | B={entry['B']}  "
                f"Tr(ρ_A²) = {entry['purity_A']:.8f}{marker}"
            )

    if info["status"] == "no_GME":
        if verbose:
            print("\n  ✗ NO GME  (separable bipartition found)")
        return 0.0
    else:
        if verbose:
            print(f"\n  ✓ GME EXISTS   C_GME = {info['C_GME']:.8f}")
        return info["C_GME"]


def check_GME_for_all_rdms(
    rdm_dict: dict,
    tol: float = 1e-8,
    verbose: bool = True,
) -> dict[tuple, Optional[float]]:
    """
    Run GME analysis for every n ≥ 3 entry in *rdm_dict*.

    Returns
    -------
    dict mapping sites_tuple → C_GME value (None if mixed, 0 if no GME).
    """
    results: dict = {}
    for sites, rho in rdm_dict.items():
        n = len(sites)
        if n < 3:
            continue
        if rho.ndim != 2 or rho.shape[0] != 2 ** n:
            print(f"  [WARNING] Skipping invalid RDM shape {rho.shape} for sites {sites}")
            continue
        results[sites] = check_GME_for_one_rdm(rho, sites, tol, verbose=verbose)
    return results


def compute_C_GME_pure(rho: np.ndarray, n: int) -> float:
    """
    Compute C_GME for a PURE n-spin state using the Troiani et al. formula.

        C_GME = √(2 · (1 − max_{A} Tr(ρ_A²)))
    """
    purities = [purity(partial_trace_spins(rho, n, A)) for A in get_bipartitions(n)]
    return float(np.sqrt(2.0 * (1.0 - max(purities))))


# ══════════════════════════════════════════════════════════════════════════════
# PAPER METHOD  —  Lower bound via coherence witness  (Ma et al. 2011)
# Sums over ALL non-trivial bipartitions A.
# ══════════════════════════════════════════════════════════════════════════════

def _bits_to_index(bits: list[int]) -> int:
    """Big-endian binary → integer index."""
    idx = 0
    for b in bits:
        idx = (idx << 1) | int(b)
    return idx


def _swap_on_subset(
    x_bits: list[int], y_bits: list[int], A: tuple[int, ...]
) -> tuple[list[int], list[int]]:
    """Swap the bits at positions in A between x and y."""
    A_set = set(A)
    xA = x_bits.copy()
    yA = y_bits.copy()
    for k in A_set:
        xA[k], yA[k] = yA[k], xA[k]
    return xA, yA


def I_function(
    rho: np.ndarray,
    n: int,
    x_bits: list[int],
    y_bits: list[int],
    tol: float = 1e-12,
) -> float:
    """
    Compute the GME coherence witness  [Paper method — Ma et al. eq.25]:

        I_{x,y}(ρ) = |ρ_{x,y}| − Σ_A √(ρ_{x^A, x^A} · ρ_{y^A, y^A})

    where the sum is over ALL non-trivial bipartitions A of the n spins.
    For n spins there are 2^(n-1) - 1 bipartitions.

    Parameters
    ----------
    rho    : (2ⁿ, 2ⁿ) density matrix
    n      : number of spins
    x_bits : bra spin string, e.g. [0,0,0] for all-up
    y_bits : ket spin string, e.g. [1,1,1] for all-down

    Returns
    -------
    float : witness value (positive = entanglement detected)
    """
    if rho.shape != (2 ** n, 2 ** n):
        raise ValueError(f"Expected ({2**n}, {2**n}), got {rho.shape}.")

    x_idx = _bits_to_index(x_bits)
    y_idx = _bits_to_index(y_bits)

    global_coh = abs(rho[x_idx, y_idx])

    sub = 0.0
    for A in get_bipartitions(n):
        xA_bits, yA_bits = _swap_on_subset(x_bits, y_bits, A)
        xA = _bits_to_index(xA_bits)
        yA = _bits_to_index(yA_bits)
        px = max(float(np.real(rho[xA, xA])), 0.0)
        py = max(float(np.real(rho[yA, yA])), 0.0)
        sub += np.sqrt(px * py)

    I = global_coh - sub
    return 0.0 if abs(I) < tol else float(I)


def Cgme_lower_bound(
    rho: np.ndarray,
    n: int,
    x_bits: Optional[list[int]] = None,
    y_bits: Optional[list[int]] = None,
) -> tuple[float, float]:
    """
    Paper method — lower bound with a FIXED choice of (x, y).

    Default choice: x = 00…0, y = 11…1.
    For the optimised bound over all pairs, use convex_roof_lower_bound().

    Returns
    -------
    C_LB : lower bound on C_GME
    I    : raw coherence-witness value
    """
    if x_bits is None:
        x_bits = [0] * n
    if y_bits is None:
        y_bits = [1] * n
    I = I_function(rho, n, x_bits, y_bits)
    return max(0.0, 2.0 * I), I


# ══════════════════════════════════════════════════════════════════════════════
# SUPERVISOR METHOD  —  Lower bound via eqs. (6-9) of supervisor notes
# Sums over individual sites k only (n terms, not 2^(n-1)-1).
# ══════════════════════════════════════════════════════════════════════════════

def _supervisor_swap_site_k(
    z1: list[int], z2: list[int], k: int
) -> tuple[list[int], list[int]]:
    """
    Swap only position k between z1 and z2.

    Returns alpha_k and beta_k from supervisor eq.(7):
        alpha_k = z1 with position k replaced by z2[k]
        beta_k  = z2 with position k replaced by z1[k]

    Example (n=3, k=1, z1=[0,0,0], z2=[1,1,1]):
        alpha_1 = [1,0,0]
        beta_1  = [0,1,1]
    """
    alpha_k = z1.copy()
    beta_k  = z2.copy()
    alpha_k[k] = z2[k]
    beta_k[k]  = z1[k]
    return alpha_k, beta_k


def supervisor_compute_A(
    rho: np.ndarray,
    z1: list[int],
    z2: list[int],
) -> float:
    """
    Supervisor eq.(6):
        A = <Phi| rho^(⊗2) Pi_{1..n} |Phi>  =  |rho_{z2, z1}|²

    Just the absolute value squared of one off-diagonal element.
    Row = z2, Column = z1.

    Physical meaning: quantum coherence between state z1 and state z2.
    """
    row = _bits_to_index(z2)
    col = _bits_to_index(z1)
    return abs(rho[row, col]) ** 2


def supervisor_compute_Bk(
    rho: np.ndarray,
    z1: list[int],
    z2: list[int],
    k: int,
) -> float:
    """
    Supervisor eq.(7) for site k:
        B_k = <Phi| Pi_k rho^(⊗2) Pi_k |Phi>
            = rho_{alpha_k, alpha_k} * rho_{beta_k, beta_k}

    Two diagonal elements multiplied together.

    Physical meaning: classical probability upper bound for site k bipartition.
    """
    alpha_k, beta_k = _supervisor_swap_site_k(z1, z2, k)
    ia = _bits_to_index(alpha_k)
    ib = _bits_to_index(beta_k)
    pa = max(float(np.real(rho[ia, ia])), 0.0)
    pb = max(float(np.real(rho[ib, ib])), 0.0)
    return pa * pb


def supervisor_compute_I_phi(
    rho: np.ndarray,
    n: int,
    z1: list[int],
    z2: list[int],
) -> float:
    """
    Supervisor eq.(8):
        I_Phi = 2 * ( sqrt(A)  -  sqrt(B_1)  - ... - sqrt(B_n) )

    For a given choice of |Phi> = |z1> ⊗ |z2>.

    Returns
    -------
    float : positive = entanglement detected for this |Phi>
            negative = no detection for this |Phi>, try another
    """
    A      = supervisor_compute_A(rho, z1, z2)
    sqrt_A = np.sqrt(A)                       # = |rho_{z2, z1}|

    sum_sqrt_Bk = 0.0
    for k in range(n):
        Bk = supervisor_compute_Bk(rho, z1, z2, k)
        sum_sqrt_Bk += np.sqrt(Bk)

    return 2.0 * (sqrt_A - sum_sqrt_Bk)


def supervisor_gme_lower_bound(
    rho: np.ndarray,
    n: int,
) -> Dict:
    """
    Supervisor method — lower bound on C_GME via eqs.(6-9).

        I = max_{|Phi>} I_Phi  <=  C_GME(rho)

    Loops over ALL ordered pairs (z1, z2) where z1 ≠ z2.
    Each pair is a different choice of |Phi> = |z1> ⊗ |z2>.

    For n spins:
        - z1 and z2 each have 2^n possible values
        - total pairs checked: 2^n * (2^n - 1)
        - e.g. n=3: 8 * 7 = 56 pairs

    Parameters
    ----------
    rho : (2ⁿ, 2ⁿ) density matrix (pure or mixed)
    n   : number of spins

    Returns
    -------
    dict:
        C_LB     : lower bound on C_GME  =  max(0, best I_Phi)
        best_I   : raw best I_Phi before factor of 2
        best_z1  : z1 string giving the best result
        best_z2  : z2 string giving the best result
        n_pairs  : number of pairs checked
        method   : 'supervisor_eqs6-9'
    """
    dim = 2 ** n

    # all 2^n spin strings of length n, big-endian
    # e.g. n=2: [0,0], [0,1], [1,0], [1,1]
    all_strings = [
        [(k >> (n - 1 - b)) & 1 for b in range(n)]
        for k in range(dim)
    ]

    best_val = 0.0
    best_z1  = all_strings[0]
    best_z2  = all_strings[-1]
    n_pairs  = 0

    for zi in range(dim):
        for zj in range(dim):
            if zi == zj:
                continue

            z1 = all_strings[zi]
            z2 = all_strings[zj]

            I_phi = supervisor_compute_I_phi(rho, n, z1, z2)
            n_pairs += 1

            if I_phi > best_val:
                best_val = I_phi
                best_z1  = z1
                best_z2  = z2

    return {
        "C_LB":    max(0.0, best_val),
        "best_I":  float(best_val),
        "best_z1": best_z1,
        "best_z2": best_z2,
        "n_pairs": n_pairs,
        "method":  "supervisor_eqs6-9",
    }


def supervisor_test_pure_state(
    rho_or_psi: np.ndarray,
    n: int,
    verbose: bool = True,
) -> Dict:
    """
    Supervisor's verification test: run on a pure state (n = N case).

    Accepts EITHER:
      (a) a 2^n spin-space pure state vector  (shape: (2^n,))
      (b) a 2^n x 2^n pure state density matrix  (shape: (2^n, 2^n))

    In the n = N case from the pipeline, always pass the n-body RDM
    (option b) since psi0 lives in Fock space, not spin space.

    For a pure state the lower bound should be close to the exact C_GME
    from eq.(1). Use this to verify your implementation is correct.

    Parameters
    ----------
    rho_or_psi : either a 1D spin-space vector of length 2^n
                 OR a 2D density matrix of shape (2^n, 2^n)
    n          : number of spins
    verbose    : print comparison report

    Returns
    -------
    dict:
        C_exact          : exact C_GME from eq.(1)
        C_LB             : lower bound from supervisor method
        difference       : |C_exact - C_LB|
        is_valid         : True if C_LB <= C_exact + tol
        supervisor_result: full dict from supervisor_gme_lower_bound
    """
    dim = 2 ** n

    # ── Accept either a vector or a matrix ──────────────────────────────────
    if rho_or_psi.ndim == 1:
        # Pure state vector — must be length 2^n (spin space)
        if len(rho_or_psi) != dim:
            raise ValueError(
                f"Expected spin-space vector of length {dim} for n={n} spins, "
                f"got length {len(rho_or_psi)}.\n"
                f"If you have a Fock-space wavefunction, pass the n-body RDM "
                f"(the 2D matrix) instead."
            )
        psi = rho_or_psi
        rho = np.outer(psi, psi.conj())
    elif rho_or_psi.ndim == 2:
        # Density matrix — must be (2^n, 2^n)
        if rho_or_psi.shape != (dim, dim):
            raise ValueError(
                f"Expected density matrix of shape ({dim}, {dim}) for n={n} spins, "
                f"got shape {rho_or_psi.shape}."
            )
        rho = rho_or_psi
        # Check it is pure
        pur = float(np.real(np.trace(rho @ rho)))
        if abs(pur - 1.0) > 1e-4:
            raise ValueError(
                f"Density matrix is not pure: Tr(rho^2) = {pur:.6f}.\n"
                f"supervisor_test_pure_state only works for pure state RDMs (n = N case)."
            )
    else:
        raise ValueError(f"Expected 1D vector or 2D matrix, got shape {rho_or_psi.shape}.")

    # ── Exact C_GME from eq.(1) — same formula as compute_C_GME_pure ────────
    # C_GME = sqrt( 2 * min_A (1 - Tr(rho_A^2)) )
    # This is bounded by 1 for qubit systems — consistent with regular PDF.
    min_val = np.inf
    for A in get_bipartitions(n):
        rho_A = partial_trace_spins(rho, n, A)
        pur_A = float(np.real(np.trace(rho_A @ rho_A)))
        val   = 1.0 - pur_A          # (1 - Tr(rho_A^2))
        if val < min_val:
            min_val = val
    C_exact = float(np.sqrt(max(2.0 * min_val, 0.0)))

    # ── Lower bound from supervisor method eqs.(6-9) ─────────────────────────
    result = supervisor_gme_lower_bound(rho, n)
    C_LB   = result["C_LB"]

    tol   = 1e-6
    valid = C_LB <= C_exact + tol

    if verbose:
        pur_rho = float(np.real(np.trace(rho @ rho)))
        print("\n" + "=" * 55)
        print("  Supervisor method — pure state verification test")
        print("=" * 55)
        print(f"  n spins          = {n}")
        print(f"  Tr(rho²)         = {pur_rho:.10f}  (should be 1.0)")
        print(f"  Exact C_GME      = {C_exact:.8f}  (from eq.1)")
        print(f"  Lower bound      = {C_LB:.8f}  (from eqs.6-9)")
        print(f"  Difference       = {abs(C_exact - C_LB):.2e}")
        print(f"  C_LB ≤ C_exact ? = {'YES ✓' if valid else 'NO ✗  — check implementation'}")
        print(f"  Best |Phi>       = z1={result['best_z1']}  z2={result['best_z2']}")
        print(f"  Pairs checked    = {result['n_pairs']}")

    return {
        "C_exact":           float(C_exact),
        "C_LB":              float(C_LB),
        "difference":        float(abs(C_exact - C_LB)),
        "is_valid":          bool(valid),
        "supervisor_result": result,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONTINUOUS OPTIMIZATION — supervisor's suggestion
# Maximize I_Phi over general separable states |Phi> = |psi1> x |psi2>
# where each qubit k is parameterized by (theta_k, phi_k) on the Bloch sphere:
#   |phi_k> = cos(theta_k/2)|0> + exp(i*phi_k)*sin(theta_k/2)|1>
# ══════════════════════════════════════════════════════════════════════════════


def _make_product_state(
    thetas: list,
    phis: list,
) -> np.ndarray:
    """
    Build an n-qubit product state as a 2^n vector.
    Each qubit k: |phi_k> = cos(t/2)|0> + exp(i*p)*sin(t/2)|1>

    theta=0   → |0>  (spin up)
    theta=pi  → |1>  (spin down)
    """
    n = len(thetas)
    q = np.array([np.cos(thetas[0] / 2.0),
                  np.exp(1j * phis[0]) * np.sin(thetas[0] / 2.0)],
                 dtype=complex)
    for k in range(1, n):
        qk = np.array([np.cos(thetas[k] / 2.0),
                       np.exp(1j * phis[k]) * np.sin(thetas[k] / 2.0)],
                      dtype=complex)
        q = np.kron(q, qk)
    return q


def I_phi_continuous(
    rho: np.ndarray,
    n: int,
    thetas1: list,
    phis1: list,
    thetas2: list,
    phis2: list,
) -> float:
    """
    Compute I_Phi for a general product state |Phi> = |psi1> x |psi2>.

    Each qubit k of psi1 is parameterized by (thetas1[k], phis1[k]).
    Each qubit k of psi2 is parameterized by (thetas2[k], phis2[k]).

    Generalises supervisor eqs.(6-8) to continuous |Phi>:

        A   = |<psi1| rho |psi2>|^2
        B_k = <alpha_k| rho |alpha_k> * <beta_k| rho |beta_k>
              where alpha_k = psi1 with qubit k replaced by psi2's qubit k
                    beta_k  = psi2 with qubit k replaced by psi1's qubit k
        I   = 2 * (sqrt(A) - sum_k sqrt(B_k))

    When thetas1=[0,...,0] and thetas2=[pi,...,pi] and all phis=0,
    this recovers exactly the basis-state formula supervisor_compute_I_phi.
    """
    psi1 = _make_product_state(thetas1, phis1)
    psi2 = _make_product_state(thetas2, phis2)

    # A = |<psi1| rho |psi2>|^2
    A      = abs(psi1.conj() @ rho @ psi2) ** 2
    sqrt_A = np.sqrt(max(A, 0.0))

    # B_k for each site k
    sum_sqrt_Bk = 0.0
    for k in range(n):
        # alpha_k: psi1 with qubit k swapped to psi2's qubit k
        t1k = list(thetas1); t1k[k] = thetas2[k]
        p1k = list(phis1);   p1k[k] = phis2[k]
        alpha_k = _make_product_state(t1k, p1k)

        # beta_k: psi2 with qubit k swapped to psi1's qubit k
        t2k = list(thetas2); t2k[k] = thetas1[k]
        p2k = list(phis2);   p2k[k] = phis1[k]
        beta_k = _make_product_state(t2k, p2k)

        Bk = ((alpha_k.conj() @ rho @ alpha_k).real *
              (beta_k.conj()  @ rho @ beta_k).real)
        sum_sqrt_Bk += np.sqrt(max(Bk, 0.0))

    return float(2.0 * (sqrt_A - sum_sqrt_Bk))


def supervisor_gme_lower_bound_continuous(
    rho: np.ndarray,
    n: int,
    real_only: bool = False,
    n_restarts: int = 5,
    seed: int = 42,
    verbose: bool = False,
) -> Dict:
    """
    Improved lower bound on C_GME by maximizing I_Phi over continuous
    product states |Phi> = |psi1> x |psi2>, as suggested by supervisor.

    Each qubit k parameterized by (theta_k, phi_k) on the Bloch sphere:
        |phi_k> = cos(theta_k/2)|0> + exp(i*phi_k)*sin(theta_k/2)|1>

    Parameters
    ----------
    rho        : (2^n, 2^n) density matrix
    n          : number of spins
    real_only  : If True, set all phi_k = 0 (real states only).
                 Supervisor's suggested intermediate step.
                 Uses n parameters instead of 2n.
    n_restarts : number of random restarts for optimization
    seed       : random seed for reproducibility
    verbose    : print progress

    Returns
    -------
    dict:
        C_LB          : improved lower bound on C_GME
        best_I        : raw best I_Phi value
        best_thetas1  : optimal theta parameters for psi1
        best_phis1    : optimal phi parameters for psi1
        best_thetas2  : optimal theta parameters for psi2
        best_phis2    : optimal phi parameters for psi2
        method        : 'continuous_real' or 'continuous_complex'
        C_LB_basis    : comparison — basis state bound (current method)
    """
    from scipy.optimize import differential_evolution

    # First get basis-state bound for comparison
    basis_result = supervisor_gme_lower_bound(rho, n)
    C_LB_basis   = basis_result["C_LB"]

    # Define cost function
    if real_only:
        # n parameters: thetas1, thetas2 (all phis=0)
        def neg_I(params):
            t1 = list(params[:n])
            t2 = list(params[n:])
            p  = [0.0] * n
            return -I_phi_continuous(rho, n, t1, p, t2, p)

        bounds = [(0.0, np.pi)] * n * 2
        method = "continuous_real"

    else:
        # 4n parameters: thetas1, phis1, thetas2, phis2
        def neg_I(params):
            t1 = list(params[0:n]);     p1 = list(params[n:2*n])
            t2 = list(params[2*n:3*n]); p2 = list(params[3*n:4*n])
            return -I_phi_continuous(rho, n, t1, p1, t2, p2)

        bounds = ([(0.0, np.pi)] * n + [(0.0, 2*np.pi)] * n) * 2
        method = "continuous_complex"

    # Global optimization with differential evolution
    # Use popsize proportional to problem dimension for better coverage
    popsize   = max(15, 3 * len(bounds))
    best_val  = -np.inf
    best_res  = None

    np.random.seed(seed)
    for restart in range(n_restarts):
        s = seed + restart * 17
        res = differential_evolution(
            neg_I, bounds,
            seed=s,
            maxiter=1000,
            tol=1e-9,
            popsize=popsize,
            workers=1,
            polish=True,
        )
        if -res.fun > best_val:
            best_val = -res.fun
            best_res = res

    C_LB = max(0.0, best_val)

    # Extract best parameters — always from the best restart found
    p = best_res.x if best_res is not None else np.zeros(len(bounds))

    if real_only:
        t1_best = list(p[:n]);  p1_best = [0.0]*n
        t2_best = list(p[n:]);  p2_best = [0.0]*n
    else:
        t1_best = list(p[0:n]);     p1_best = list(p[n:2*n])
        t2_best = list(p[2*n:3*n]); p2_best = list(p[3*n:4*n])

    if verbose:
        print(f"\n  Continuous optimization ({method}):")
        print(f"    C_LB (basis states) = {C_LB_basis:.8f}")
        print(f"    C_LB ({method:16s}) = {C_LB:.8f}")
        print(f"    Improvement = {C_LB - C_LB_basis:.8f}")

    return {
        "C_LB":         C_LB,
        "best_I":       best_val,
        "best_thetas1": t1_best,
        "best_phis1":   p1_best,
        "best_thetas2": t2_best,
        "best_phis2":   p2_best,
        "method":       method,
        "C_LB_basis":   C_LB_basis,
    }