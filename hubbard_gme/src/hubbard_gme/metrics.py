"""
metrics.py
----------
Quantum-information metrics for density matrices:

    - Purity          γ = Tr(ρ²)
    - Von Neumann entropy  S = -Tr(ρ log ρ)
    - Hilbert-Schmidt distance  ||ρ_A - ρ_B||_F
    - Trace distance  ½ ||ρ_A - ρ_B||_1
    - Aggregate rdm_metrics() dictionary
"""

from __future__ import annotations

import numpy as np
from typing import Dict


# ── Single-matrix measures ───────────────────────────────────────────────────

def compute_purity(rho: np.ndarray) -> float:
    """
    Purity γ = Tr(ρ²).

    Returns
    -------
    float
        1.0 for a pure state; < 1.0 for mixed.
    """
    return float(np.real(np.trace(rho @ rho)))


def compute_entropy(rho: np.ndarray) -> float:
    """
    Von Neumann entropy S(ρ) = -Tr(ρ log ρ).

    Uses eigenvalue decomposition; tiny negative eigenvalues are clipped
    to avoid log(0) issues.

    Returns
    -------
    float
        0.0 for a pure state; > 0.0 for mixed.
    """
    evals = np.linalg.eigvalsh(rho)
    evals = np.clip(evals, 1e-14, 1.0)
    return float(-np.sum(evals * np.log(evals)))


# ── Two-matrix distances ─────────────────────────────────────────────────────

def hilbert_schmidt_distance(rhoA: np.ndarray, rhoB: np.ndarray) -> float:
    """
    Hilbert-Schmidt (Frobenius) distance:
        D_HS = ||ρ_A - ρ_B||_F = sqrt(Tr[(ρ_A - ρ_B)²])

    Returns
    -------
    float
        0 if identical; > 0 otherwise.
    """
    return float(np.linalg.norm(rhoA - rhoB, ord="fro"))


def trace_distance(rhoA: np.ndarray, rhoB: np.ndarray) -> float:
    """
    Trace distance:
        D_T = ½ ||ρ_A - ρ_B||_1  = ½ Σ|λ_i(ρ_A - ρ_B)|

    Operationally: maximum probability difference under any measurement.

    Returns
    -------
    float
        In [0, 1]; 0 if identical.
    """
    evals = np.linalg.eigvalsh(rhoA - rhoB)
    return float(0.5 * np.sum(np.abs(evals)))


# ── Aggregate helper ─────────────────────────────────────────────────────────

def rdm_metrics(rho: np.ndarray) -> Dict:
    """
    Compute all single-matrix metrics for *rho*.

    Returns
    -------
    dict with keys:
        trace   : float
        purity  : float
        entropy : float
        eigvals : np.ndarray (sorted ascending)
    """
    return {
        "trace":   float(np.trace(rho).real),
        "purity":  compute_purity(rho),
        "entropy": compute_entropy(rho),
        "eigvals": np.linalg.eigvalsh(rho),
    }


# ── Human-readable report ────────────────────────────────────────────────────

def short_metrics_report(info: Dict, label: str = "RDM") -> None:
    """Print a concise metric summary to stdout."""
    print(f"\n{'─'*40}")
    print(f"  {label}")
    print(f"{'─'*40}")
    print(f"  Trace   = {info['trace']:.8f}")
    print(f"  Purity  = {info['purity']:.8f}")
    print(f"  Entropy = {info['entropy']:.8f}")
    print(f"  λ_min   = {info['eigvals'].min():.4e}")
    print(f"  λ_max   = {info['eigvals'].max():.4e}")
    print(f"{'─'*40}\n")
