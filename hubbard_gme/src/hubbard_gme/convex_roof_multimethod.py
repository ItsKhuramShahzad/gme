"""
convex_roof_multimethod.py
--------------------------
Multi-method convex-roof C_GME for mixed n-spin RDMs.

Methods
-------
  1. L-BFGS-B        — Gradient-based local search
  2. Nelder-Mead      — Simplex, derivative-free
  3. Powell           — Direction-set, derivative-free
  4. COBYLA           — Linear-approx constrained
  5. Differential-Evo — Global stochastic (slowest, most thorough)
  6. Simulated-Ann    — Basin hopping (global, moderate)

Analytical bounds (no optimisation)
  C_LB  — Paper coherence witness lower bound
  C_UB  — Eigendecomposition upper bound

Public API
----------
    run_all_methods(rho, n, n_restarts, methods, verbose)
        -> dict for one RDM

    multi_method_convex_roof_analysis(rho_dict, n_restarts, methods, verbose)
        -> {sites: result_dict} for all RDMs
"""

from __future__ import annotations

import time
import warnings
import numpy as np
from scipy.linalg import expm as _expm
from scipy.optimize import minimize, differential_evolution, basinhopping
from typing import Dict, List, Tuple
from .gme import compute_C_GME_pure, supervisor_gme_lower_bound

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_bipartitions(n: int) -> List[Tuple[int, ...]]:
    parts = []
    for mask in range(1, (1 << n) - 1):
        A = tuple(i for i in range(n) if (mask >> i) & 1)
        B = tuple(i for i in range(n) if not ((mask >> i) & 1))
        if len(A) <= len(B) and (B, A) not in [(p, q) for p, q in parts]:
            parts.append((A, B))
    return [a for a, _ in parts]


def _partial_trace(rho: np.ndarray, n: int, keep: Tuple[int, ...]) -> np.ndarray:
    trace_over = tuple(i for i in range(n) if i not in keep)
    dim_keep   = 2 ** len(keep)
    dim_trace  = 2 ** len(trace_over)
    perm       = list(keep) + list(trace_over)
    inv_perm   = [perm.index(i) for i in range(n)]
    rho_t = rho.reshape([2] * (2 * n))
    rho_t = np.transpose(rho_t, inv_perm + [p + n for p in inv_perm])
    rho_t = rho_t.reshape(dim_keep, dim_trace, dim_keep, dim_trace)
    return np.trace(rho_t, axis1=1, axis2=3)


def _purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


def _cgme_pure_vector(psi: np.ndarray, n: int) -> float:
    psi_t   = psi.reshape([2] * n)
    max_pur = 0.0
    for A in _get_bipartitions(n):
        B    = tuple(i for i in range(n) if i not in A)
        perm = list(A) + list(B)
        M    = np.transpose(psi_t, perm).reshape(2 ** len(A), 2 ** len(B))
        sv   = np.linalg.svd(M, compute_uv=False)
        pur  = float(np.sum(sv ** 4))
        if pur > max_pur:
            max_pur = pur
    return float(np.sqrt(max(0.0, 2.0 * (1.0 - max_pur))))


def _n_params(r: int) -> int:
    # Full complex unitary U(r): r diagonal reals + r*(r-1)/2 complex off-diagonals
    # Total real parameters = r + 2 * r*(r-1)/2 = r^2
    return r * r


def _params_to_unitary(params: np.ndarray, r: int) -> np.ndarray:
    """
    Build r x r unitary U = exp(i*H) where H is Hermitian.
    params layout:
      first r values      -> diagonal of H (real)
      next r*(r-1) values -> off-diagonal pairs (re, im) for i<j
    This is the full U(r) parametrisation — same as the original convex_roof.py.
    """
    H = np.zeros((r, r), dtype=complex)
    idx = 0
    # Diagonal: real
    for i in range(r):
        H[i, i] = params[idx]
        idx += 1
    # Off-diagonal: complex, Hermitian H[j,i] = conj(H[i,j])
    for i in range(r):
        for j in range(i + 1, r):
            H[i, j] =  params[idx] + 1j * params[idx + 1]
            H[j, i] =  params[idx] - 1j * params[idx + 1]
            idx += 2
    return _expm(1j * H)


def _decompose(U, eigvecs, eigvals_sqrt):
    V     = eigvecs @ (U * eigvals_sqrt).T
    probs = np.real(np.einsum("ij,ij->j", V.conj(), V))
    probs = np.clip(probs, 0.0, None)
    safe  = probs > 1e-14
    pst   = np.zeros_like(V.T)
    pst[safe] = V.T[safe] / np.sqrt(probs[safe, np.newaxis])
    return probs, pst


def _cost_fn(params, r, eigvecs, eigvals_sqrt, n):
    U = _params_to_unitary(params, r)
    probs, pst = _decompose(U, eigvecs, eigvals_sqrt)
    return sum(
        probs[i] * _cgme_pure_vector(pst[i], n)
        for i in range(r) if probs[i] > 1e-14
    )


# ─────────────────────────────────────────────────────────────────────────────
# Analytical bounds
# ─────────────────────────────────────────────────────────────────────────────

def _lower_bound(rho: np.ndarray, n: int) -> float:
    dim     = 2 ** n
    strings = [[(k >> (n - 1 - b)) & 1 for b in range(n)] for k in range(dim)]
    best_I  = 0.0
    for xi in range(dim):
        for yi in range(dim):
            if xi == yi:
                continue
            x, y    = strings[xi], strings[yi]
            xidx    = sum(x[b] * (2 ** (n - 1 - b)) for b in range(n))
            yidx    = sum(y[b] * (2 ** (n - 1 - b)) for b in range(n))
            offdiag = abs(rho[xidx, yidx])
            penalty = 0.0
            for A in _get_bipartitions(n):
                xA  = tuple(x[a] for a in A)
                yA  = tuple(y[a] for a in A)
                xAi = sum(xA[k] * (2 ** (len(A) - 1 - k)) for k in range(len(A)))
                yAi = sum(yA[k] * (2 ** (len(A) - 1 - k)) for k in range(len(A)))
                rA  = _partial_trace(rho, n, A)
                penalty += np.sqrt(abs(rA[xAi, xAi]) * abs(rA[yAi, yAi]))
            I = offdiag - penalty
            if I > best_I:
                best_I = I
    return float(max(0.0, 2.0 * best_I))


def _upper_bound(rho: np.ndarray, n: int) -> Tuple[float, int]:
    eigvals, eigvecs = np.linalg.eigh(rho)
    mask = eigvals > 1e-10
    c_ub = sum(
        float(lam) * _cgme_pure_vector(phi, n)
        for lam, phi in zip(eigvals[mask], eigvecs[:, mask].T)
    )
    return float(c_ub), int(np.sum(mask))


# ─────────────────────────────────────────────────────────────────────────────
# Single-method runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_single_method(
    rho:         np.ndarray,
    n:           int,
    method_name: str,
    n_restarts:  int   = 50,
    tol:         float = 1e-9,
) -> Dict:
    eigvals, eigvecs = np.linalg.eigh(rho)
    mask         = eigvals > 1e-10
    eigvals_r    = eigvals[mask]
    eigvecs_r    = eigvecs[:, mask]
    eigvals_sqrt = np.sqrt(eigvals_r)
    r            = int(np.sum(mask))
    np_params    = _n_params(r)

    # rank-1 = pure state: no optimisation needed, C_GME is exact
    if r == 1:
        psi   = eigvecs_r[:, 0]
        c_val = _cgme_pure_vector(psi, n)
        return {
            "method":     method_name,
            "C_GME":      c_val,
            "time_s":     0.0,
            "n_restarts": 0,
            "n_evals":    0,
            "spread":     0.0,
            "all_values": [c_val],
            "decomp":     [{"component": 1, "weight_p": 1.0,
                            "C_GME_i": c_val, "contribution": c_val}],
            "rank":       1,
        }

    t0        = time.perf_counter()
    best_cost = np.inf
    best_x    = None
    all_vals: List[float] = []
    last_res  = None

    _scale = {"L-BFGS-B": 1.0, "Nelder-Mead": 0.5, "Powell": 0.4, "COBYLA": 0.3}
    eff_restarts = max(1, int(n_restarts * _scale.get(method_name, 1.0)))

    if method_name in ("L-BFGS-B", "Nelder-Mead", "Powell", "COBYLA"):
        maxiter = {"L-BFGS-B": 10_000, "Nelder-Mead": 5_000,
                   "Powell": 5_000, "COBYLA": 2_000}[method_name]
        for _ in range(eff_restarts):
            x0 = np.random.randn(np_params) * 0.5
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    _cost_fn, x0,
                    args=(r, eigvecs_r, eigvals_sqrt, n),
                    method=method_name,
                    tol=tol,
                    options={"maxiter": maxiter},
                )
            val = float(res.fun)
            all_vals.append(val)
            if val < best_cost:
                best_cost = val
                best_x    = res.x.copy()
                last_res  = res

    elif method_name == "Differential-Evo":
        bounds = [(-np.pi, np.pi)] * np_params
        for seed in range(max(1, n_restarts // 5)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = differential_evolution(
                    _cost_fn, bounds,
                    args=(r, eigvecs_r, eigvals_sqrt, n),
                    maxiter=200, tol=tol, seed=seed,
                    popsize=8, workers=1,
                )
            val = float(res.fun)
            all_vals.append(val)
            if val < best_cost:
                best_cost = val
                best_x    = res.x.copy()
                last_res  = res

    elif method_name == "Simulated-Ann":
        x0 = np.random.randn(np_params) * 0.5
        min_kwargs = {
            "method": "L-BFGS-B",
            "args":   (r, eigvecs_r, eigvals_sqrt, n),
            "tol":    tol,
            "options":{"maxiter": 500},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = basinhopping(
                _cost_fn, x0,
                niter=max(n_restarts * 3, 60),
                minimizer_kwargs=min_kwargs,
                seed=42,
            )
        best_cost = float(res.fun)
        best_x    = res.x.copy()
        all_vals  = [best_cost]
        last_res  = res

    else:
        raise ValueError(f"Unknown method: {method_name!r}")

    elapsed = time.perf_counter() - t0

    decomp = []
    if best_x is not None:
        U = _params_to_unitary(best_x, r)
        probs, pst = _decompose(U, eigvecs_r, eigvals_sqrt)
        for i in range(r):
            if probs[i] < 1e-12:
                continue
            C_i = _cgme_pure_vector(pst[i], n)
            decomp.append({
                "component":    i + 1,
                "weight_p":     float(probs[i]),
                "C_GME_i":      float(C_i),
                "contribution": float(probs[i] * C_i),
            })

    spread = float(max(all_vals) - min(all_vals)) if len(all_vals) > 1 else 0.0

    return {
        "method":     method_name,
        "C_GME":      float(max(0.0, best_cost)),
        "time_s":     float(elapsed),
        "n_restarts": n_restarts,
        "n_evals":    len(all_vals),
        "spread":     spread,
        "all_values": all_vals,
        "decomp":     decomp,
        "rank":       r,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

ALL_METHODS = [
    "L-BFGS-B",
    # "Nelder-Mead",
    # "Powell",
    # "COBYLA",
    # "Differential-Evo",
    # "Simulated-Ann",
]


def run_all_methods(
    rho:        np.ndarray,
    n:          int,
    n_restarts: int       = 50,
    methods:    List[str] = None,
    verbose:    bool      = True,
    L: int = 4, 
) -> Dict:
    """
    Run all methods on a single RDM.

    Returns
    -------
    dict with keys:
        n, purity, rank, C_LB, C_UB, bipartitions,
        methods      : {name -> {C_GME, time_s, n_restarts, n_evals,
                                  spread, all_values, decomp, rank}}
        best_method  : str
        best_C_GME   : float
    """
    if methods is None:
        methods = ALL_METHODS

    pur        = _purity(rho)
    C_LB       = supervisor_gme_lower_bound (rho, n) #_lower_bound(rho, n)
    C_UB, rank = _upper_bound(rho, n)
    #    return {
    #     "C_LB":    max(0.0, best_val),
    #     "best_I":  float(best_val),
    #     "best_z1": best_z1,
    #     "best_z2": best_z2,
    #     "n_pairs": n_pairs,
    #     "method":  "supervisor_eqs6-9",
    # }
    C_LB= C_LB["C_LB"]
    C_GME=0.0
    if np.allclose(L, n):
        C_GME= compute_C_GME_pure(rho,n)
    bipartitions = []
    for A in _get_bipartitions(n):
        B  = tuple(i for i in range(n) if i not in A)
        rA = _partial_trace(rho, n, A)
        bipartitions.append({"A": A, "B": B, "purity_A": float(_purity(rA))})

    if verbose:
        print(f"    purity={pur:.6f}  rank={rank}  "
              f"C_LB={C_LB:.6f}  C_UB={C_UB:.6f}")

    method_results = {}
    for mname in methods:
        if verbose:
            print(f"      [{mname:<16}] ", end="", flush=True)
        method_results[mname] = _run_single_method(
            rho, n, mname, n_restarts=n_restarts
        )
        if verbose:
            r = method_results[mname]
            print(f"C_GME={r['C_GME']:.8f}  t={r['time_s']:.2f}s")

    best_method = min(method_results, key=lambda m: method_results[m]["C_GME"])
    best_C      = method_results[best_method]["C_GME"]

    return {
        "n":            n,
        "purity":       pur,
        "rank":         rank,
        "C_LB":         C_LB,
        "C_GME":        C_GME,
        "C_UB":         C_UB,
        "bipartitions": bipartitions,
        "methods":      method_results,
        "best_method":  best_method,
        "best_C_GME":   best_C,
    }


def multi_method_convex_roof_analysis(
    rho_dict:   Dict[tuple, np.ndarray],
    n_restarts: int       = 50,
    methods:    List[str] = None,
    verbose:    bool      = True,
    L: int =4,
) -> Dict[tuple, Dict]:
    """Run all methods on every RDM. Returns {sites: run_all_methods_result}."""
    results = {}
    total   = len(rho_dict)
    for idx, (sites, rho) in enumerate(rho_dict.items()):
        if verbose:
            print(f"\n  [{idx+1}/{total}] sites={sites}")
        results[sites] = run_all_methods(
            rho, len(sites),
            n_restarts=n_restarts,
            methods=methods,
            verbose=verbose,
            L=L,
        )
    return results