"""
scan.py
-------
Parameter scan engine for the Hubbard model GME analysis.

For each (U, t) point:
    1. Compute k lowest eigenstates via exact diagonalisation
    2. Compute n-body RDMs for every eigenstate and every contiguous cluster
    3. Compute C_GME, purity, entropy for each (eigenstate, cluster) pair
    4. Compute energy gaps, double occupancy, spin properties

Results are returned as a pandas DataFrame and optionally saved as CSV.

Public functions
----------------
run_U_scan(...)   — vary U at fixed t
run_Ut_scan(...)  — vary U/t ratio (scales both)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from .hamiltonian import (
    compute_k_eigenstates,
    double_occupancy_operator,
)
from .rdm import compute_general_nRDM, generate_site_clusters
from .gme import compute_C_GME_pure, purity as gme_purity
from .metrics import compute_entropy
from openfermion.linalg import get_number_preserving_sparse_operator


# ── Internal helpers ──────────────────────────────────────────────────────────

def _double_occ(psi: np.ndarray, n_spin_orbitals: int,
                L: int, n_electrons: int) -> float:
    D_op = double_occupancy_operator(n_spin_orbitals, L)
    D_sp = get_number_preserving_sparse_operator(
        D_op, n_spin_orbitals, n_electrons, spin_preserving=True
    )
    return float(np.vdot(psi, D_sp @ psi).real)


def _rdm_for_psi(psi: np.ndarray, sites: tuple,
                 n_spin_orbitals: int, n_electrons: int) -> np.ndarray:
    """Return normalised n-body RDM for given psi and sites."""
    rho = compute_general_nRDM(
        list(sites), psi, n_spin_orbitals, n_electrons
    )
    tr = np.trace(rho).real
    return rho / tr if abs(tr) > 1e-14 else rho


def _scan_one_point(
    L: int,
    t: float,
    U: float,
    mode: str,
    n_electrons: int,
    n_body: int,
    k_states: int,
    t_prime: float = None,
    t2:      float = None,
    t_perp:  float = None,
) -> list[dict]:
    """
    Run one (L, t, U) point and return a list of result rows —
    one row per (eigenstate index, cluster).
    """
    energies, eigvecs, n_so = compute_k_eigenstates(
        L=L, t=t, U=U, mode=mode, n_electrons=n_electrons, k=k_states,
        t_prime=t_prime, t2=t2, t_perp=t_perp,
    )

    k_actual = energies.shape[0]
    clusters = generate_site_clusters(L, n_body)
    rows: list[dict] = []

    for state_idx in range(k_actual):
        psi   = eigvecs[:, state_idx]
        E     = float(energies[state_idx])
        gap   = float(energies[state_idx] - energies[0])   # E_i - E_0
        d_occ = _double_occ(psi, n_so, L, n_electrons)

        for sites in clusters:
            rho  = _rdm_for_psi(psi, sites, n_so, n_electrons)
            pur  = float(gme_purity(rho))
            ent  = float(compute_entropy(rho))
            cgme = float(compute_C_GME_pure(rho, n_body)) if pur > 1 - 1e-6 else float("nan")

            rows.append({
                # System parameters
                "L":           L,
                "t":           t,
                "U":           U,
                "U_over_t":    U / t if abs(t) > 1e-14 else float("inf"),
                "mode":        mode,
                "Ne":          n_electrons,
                "n_body":      n_body,
                # Eigenstate info
                "state_idx":   state_idx,
                "label":       f"ψ{state_idx}",
                "energy":      E,
                "gap":         gap,          # E_i - E_0
                "d_occ_total": d_occ,
                "d_occ_site":  d_occ / L,
                # RDM info
                "sites":       str(sites),
                "purity":      pur,
                "entropy":     ent,
                "C_GME":       cgme,
                "C_GME_valid": not np.isnan(cgme),
            })

    return rows


# ── Scan averaged over clusters ───────────────────────────────────────────────

def _aggregate_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-(U, state_idx) summary statistics by averaging over clusters.

    Returns a DataFrame with one row per (U, t, state_idx).
    """
    group_cols = ["L", "t", "U", "U_over_t", "mode", "Ne", "n_body",
                  "state_idx", "label", "energy", "gap",
                  "d_occ_total", "d_occ_site"]

    agg = (
        df
        .groupby(group_cols, as_index=False)
        .agg(
            C_GME_mean=("C_GME",   "mean"),
            C_GME_max= ("C_GME",   "max"),
            C_GME_min= ("C_GME",   "min"),
            purity_mean=("purity", "mean"),
            entropy_mean=("entropy","mean"),
        )
    )
    return agg


# ── Public scan functions ─────────────────────────────────────────────────────

def run_U_scan(
    L:          int,
    t:          float,
    U_values:   list,
    mode:       str   = "uniform",
    n_electrons: int  = None,
    n_body:     int   = 3,
    k_states:   int   = 4,
    save_csv:   Optional[str] = None,
    verbose:    bool  = True,
    t_prime:    float = None,
    t2:         float = None,
    t_perp:     float = None,
) -> tuple:
    if n_electrons is None:
        n_electrons = L

    all_rows: list[dict] = []

    for i, U in enumerate(U_values):
        if verbose:
            print(f"  U scan [{i+1}/{len(U_values)}]  U={U:.3f}  U/t={U/t:.2f} …",
                  flush=True)
        rows = _scan_one_point(L, t, U, mode, n_electrons, n_body, k_states,
                               t_prime=t_prime, t2=t2, t_perp=t_perp)
        all_rows.extend(rows)

    df_full = pd.DataFrame(all_rows)
    df_agg  = _aggregate_clusters(df_full)

    if save_csv:
        df_full.to_csv(save_csv, index=False)
        if verbose:
            print(f"  Full scan saved → {save_csv}")

    return df_full, df_agg


def run_Ut_scan(
    L:           int,
    Ut_values:   list,
    t_fixed:     float = 1.0,
    mode:        str   = "uniform",
    n_electrons: int   = None,
    n_body:      int   = 3,
    k_states:    int   = 4,
    save_csv:    Optional[str] = None,
    verbose:     bool  = True,
    t_prime:     float = None,
    t2:          float = None,
    t_perp:      float = None,
) -> tuple:
    """Scan over U/t ratio by varying U at fixed t."""
    U_values = [ut * t_fixed for ut in Ut_values]
    return run_U_scan(
        L=L, t=t_fixed, U_values=U_values,
        mode=mode, n_electrons=n_electrons,
        n_body=n_body, k_states=k_states,
        save_csv=save_csv, verbose=verbose,
        t_prime=t_prime, t2=t2, t_perp=t_perp,
    )
