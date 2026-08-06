"""
convex_roof.py
--------------
Unified convex-roof extension of C_GME for MIXED and PURE n-spin RDMs.

When the extracted n-body RDM ρ is mixed (Tr(ρ²) < 1), the pure-state
formula C_GME = √(2·(1 − max_A Tr(ρ_A²))) is NOT applicable directly.

The true definition is the convex roof:

    C_GME(ρ) = min_{decompositions} Σᵢ pᵢ · C_GME(|ψᵢ⟩)

minimised over ALL pure-state decompositions  ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|.

This module implements FOUR complementary bounds and methods:
  1. Lower bound (paper)      — Coherence witness I_{x,y}(ρ)
  2. Lower bound (supervisor) — Optimize over (z1,z2) pairs via eqs.(6-9)
  3. Upper bound              — Eigendecomposition ρ = Σλᵢ|φᵢ⟩⟨φᵢ|
  4. Numerical (exact)        — Unitary optimization via scipy
  5. CVXPY SDP relaxation     — SDP lower bound (when cvxpy available)

USAGE EXAMPLES
--------------
# For mixed states (n < N)
result = full_convex_roof_analysis(rho, sites=(0,1,2), n_restarts=20)
C_LB = result['C_LB_best']
C_num = result['C_numerical']

# For pure states (n = N) — use direct C_GME formula
from .gme import compute_C_GME_pure
C_exact = compute_C_GME_pure(rho, n)

# All-in-one analysis
result = convex_roof_analysis_complete(rho, sites, use_cvxpy=True)
"""

from __future__ import annotations

import warnings
import inspect
import time
import hashlib
import numpy as np
from scipy.linalg import expm
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize, minimize_scalar, differential_evolution
from .gme import (
    get_bipartitions,
    partial_trace_spins,
    purity,
    supervisor_gme_lower_bound,
    I_function,
)
from scipy.optimize import minimize as sp_minimize
N_STARTS = 200

# ── Near-tie policy (audit issue #7) ─────────────────────────────────────────
# C_GME = sqrt(2(1 - P_max)) is a max over bipartitions, so it is nonsmooth
# wherever several bipartitions tie for the maximum purity. Two decisions must
# be made, and they are deliberately SEPARATE:
#
#   VALUE policy    : the measure value is the EXACT maximum purity. We do not
#                     average purities into the value — averaging would replace
#                     the measure by a nearby different one, and the effect is
#                     amplified near C = 0 because C = sqrt(2(1-P)).
#   GRADIENT policy : at a tie the classical derivative does not exist, so we
#                     use the averaged Clarke subgradient over the active set
#                     (all bipartitions within TIE_TOL of the maximum).
#
# One shared tolerance defines "active" for both, so the solvers cannot drift
# apart. See tests/test_tie_policy.py.
TIE_TOL = 1e-10

# ── Helper functions ───────────────────────────────────────────────────

def compute_lower_bounds(rho, n):
    """Paper and supervisor lower bounds."""
    lb1 = convex_roof_lower_bound(rho, n)
    lb2 = supervisor_gme_lower_bound(rho, n)
    return {
        "paper":      lb1,
        "supervisor": lb2,
        "best":       max(lb1["C_LB"], lb2["C_LB"]),
    }
def C_GME_pure_dm(rho_pure, n):
    """C_GME for a pure state given as density matrix."""
    purities = [purity(partial_trace_spins(rho_pure, n, A))
                for A in get_bipartitions(n)]
    return float(np.sqrt(max(2.0*(1.0 - max(purities)), 0.0)))

def unitary_convex_roof(rho, n, n_starts=N_STARTS):
    """
    Convex roof via unitary optimization.
    rho = W W†  where W = V sqrt(D) U†
    Minimizes sum_i p_i C_GME(|psi_i>) over unitary U.
    """
    eigvals, eigvecs = np.linalg.eigh(rho)
    mask    = eigvals > 1e-10
    eigvals = eigvals[mask]
    eigvecs = eigvecs[:, mask]
    rank    = len(eigvals)
    W0      = eigvecs @ np.diag(np.sqrt(eigvals))

    def cost(params):
        A = params.reshape(rank, rank)
        A = A - A.T
        try:
            U = np.linalg.solve(np.eye(rank) + A, np.eye(rank) - A)
        except np.linalg.LinAlgError:
            return 1e10
        W     = W0 @ U
        total = 0.0
        for i in range(rank):
            col = W[:, i]
            p_i = float(np.real(col.conj() @ col))
            if p_i < 1e-12:
                continue
            phi   = col / np.sqrt(p_i)
            rho_i = np.outer(phi, phi.conj())
            total += p_i * C_GME_pure_dm(rho_i, n)
        return total

    best_C = np.inf
    best_params = None
    for start in range(n_starts):
        # audit #9: local Generator, no global RNG mutation (still deterministic)
        _rng = np.random.default_rng(start)
        A0 = _rng.standard_normal((rank, rank)) * 0.3
        params0 = (A0 - A0.T).flatten()
        res = sp_minimize(cost, params0, method="L-BFGS-B",
                          options={"maxiter": 300, "ftol": 1e-13})
        if res.fun < best_C:
            best_C = res.fun
            best_params = res.x.copy()
        if (start + 1) % 50 == 0:
            print(f"    restart {start+1}/{n_starts}  best={best_C:.8f}")

    # Extract decomposition
    A = best_params.reshape(rank, rank)
    A = A - A.T
    U = np.linalg.solve(np.eye(rank) + A, np.eye(rank) - A)
    W = W0 @ U
    decomp = []
    for i in range(rank):
        col  = W[:, i]
        p_i  = float(np.real(col.conj() @ col))
        if p_i < 1e-12:
            continue
        phi   = col / np.sqrt(p_i)
        rho_i = np.outer(phi, phi.conj())
        C_i   = C_GME_pure_dm(rho_i, n)
        decomp.append(_component_decomp_entry(i, p_i, C_i, phi))

    return {"C_GME": best_C, "rank": rank, "decomp": decomp}
# ── Fast pure-state C_GME via SVD ────────────────────────────────────────────

def _cgme_pure_vector(psi, n):
    psi_tensor = psi.reshape([2] * n)
    max_purity = 0.0
    for A in get_bipartitions(n):
        B = tuple(i for i in range(n) if i not in A)
        perm = list(A) + list(B)
        M = np.transpose(psi_tensor, perm).reshape(2 ** len(A), 2 ** len(B))
        rho_A = M @ M.conj().T            # 2^|A| × 2^|A|
        pur_A = float(np.real(np.trace(rho_A @ rho_A)))
        if pur_A > max_purity:
            max_purity = pur_A
    return float(np.sqrt(max(0.0, 2.0 * (1.0 - max_purity))))

def _cgme_pure_vector_robust(psi, n, tie_tol=TIE_TOL):
    """
    C_GME of a pure state, evaluated under the shared near-tie VALUE policy.

    The value is the EXACT maximum purity over bipartitions (audit issue #7).
    Earlier revisions averaged the purities of all bipartitions within
    ``tie_tol`` of the maximum; that made this function disagree with the
    gradient-side routine, which keeps the strict maximum. Because
    C = sqrt(2(1-P)), a small purity perturbation is amplified near C = 0, so
    the two solvers could evaluate slightly different objectives exactly where
    it matters most.

    Tie handling now lives entirely in the GRADIENT (an averaged Clarke
    subgradient over the active set) — see ``_cgme_pure_with_grad`` and the
    TIE_TOL note at the top of this module.

    ``tie_tol`` is retained for API compatibility and is used only to report
    how many bipartitions are active; it no longer changes the returned value.
    """
    psi_tensor = psi.reshape([2] * n)
    purities = []
    for A in get_bipartitions(n):
        B = tuple(i for i in range(n) if i not in A)
        perm = list(A) + list(B)
        M = np.transpose(psi_tensor, perm).reshape(2 ** len(A), 2 ** len(B))
        rho_A = M @ M.conj().T
        pur_A = float(np.real(np.trace(rho_A @ rho_A)))
        purities.append(pur_A)
    
    # VALUE policy: exact maximum (no averaging) — see TIE_TOL note above.
    max_p = max(purities)
    return float(np.sqrt(max(0.0, 2.0 * (1.0 - max_p))))
def _cgme_pure_dm(rho_pure: np.ndarray, n: int) -> float:
    """C_GME for a pure state given as density matrix."""
    purities = [purity(partial_trace_spins(rho_pure, n, A))
                for A in get_bipartitions(n)]
    return float(np.sqrt(max(2.0*(1.0 - max(purities)), 0.0)))


# ── Unitary parametrisation ───────────────────────────────────────────────────

def _n_params(r: int) -> int:
    """Number of real parameters needed to specify an r×r unitary via exp(iH)."""
    return r * r


def _params_to_unitary(params: np.ndarray, r: int) -> np.ndarray:
    """Build an r×r unitary U = exp(i·H) from r² real parameters."""
    H = np.zeros((r, r), dtype=complex)
    idx = 0
    for i in range(r): # First loop diagonal elements
        H[i, i] = params[idx]
        idx += 1

    for i in range(r):  # Second loop Off Diagonal (2 values per pair)
        for j in range(i + 1, r):
            H[i, j] = params[idx] + 1j * params[idx + 1]
            H[j, i] = params[idx] - 1j * params[idx + 1]
            idx += 2
    return expm(1j * H)


# def _decompose(
#     U: np.ndarray,
#     eigvecs: np.ndarray,
#     eigvals_sqrt: np.ndarray,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Given unitary U (r×r) and eigendecomposition of ρ, return the
#     corresponding pure-state decomposition.
#     """
#     V = eigvecs @ (U * eigvals_sqrt).T

#     probs = np.real(np.einsum('ij,ij->j', V.conj(), V))
#     probs = np.clip(probs, 0.0, None)

#     safe = probs > 1e-14
#     pure_states = np.zeros_like(V.T)
#     pure_states[safe] = (V.T[safe]) / np.sqrt(probs[safe, np.newaxis])

#     return probs, pure_states
def _decompose(U, eigvecs, eigvals_sqrt):
    
    # Step 1: Build W0 = eigvecs * sqrt(D)
    #         then W  = W0 * U
    W0 = eigvecs * eigvals_sqrt        # multiply each column j by sqrt(lambda_j)
    W  = W0 @ U                        # W is (2^n x r) matrix

    # Step 2: Extract pure states from columns of W
    r = W.shape[1]                     # number of columns = rank of rho
    probs       = np.zeros(r)
    pure_states = np.zeros_like(W)

    for i in range(r):
        col    = W[:, i]               # i-th column of W = unnormalised pure state
        p_i    = np.real(col.conj() @ col)   # p_i = <col|col> = squared norm
        p_i    = max(p_i, 0.0)         # clip negative numerical noise to zero

        probs[i] = p_i

        if p_i > 1e-14:                # only normalise if probability is non-zero
            pure_states[:, i] = col / np.sqrt(p_i)   # |psi_i> = col / sqrt(p_i)

    return probs, pure_states.T        # return probabilities and normalised pure states

def _cost(
    params: np.ndarray,
    r: int,
    eigvecs: np.ndarray,
    eigvals_sqrt: np.ndarray,
    n: int,
) -> float:
    """f(U) = Σᵢ pᵢ · C_GME(|ψᵢ⟩)"""
    U = _params_to_unitary(params, r)
    probs, pure_states = _decompose(U, eigvecs, eigvals_sqrt)

    total = 0.0
    for i in range(r):
        if probs[i] < 1e-14:
            continue
        total += probs[i] * _cgme_pure_vector_robust(pure_states[i], n)
    return total


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 4: NUMERICAL (Unitary Optimization)
# ══════════════════════════════════════════════════════════════════════════════

def convex_roof_numerical(
    rho:        np.ndarray,
    n:          int,
    n_restarts: int   = 20,
    method:     str   = "L-BFGS-B",
    tol:        float = 1e-9,
    use_de:     bool  = False,
) -> Dict:
    """
    Numerical convex-roof minimisation over U ∈ U(r).

    Parametrises all decompositions ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ| via unitary
    transformation of the eigendecomposition.

    Returns
    -------
    dict:
        C_numerical : float  — best (minimum) value found
        rank        : int
        n_restarts  : int
        converged   : bool
        decomp      : list of {p, C_GME} for each pure state
        all_values  : list of all restart results
    """
    eigvals, eigvecs = np.linalg.eigh(rho)
    mask         = eigvals > 1e-10 # return an array of true and false.
    eigvals_r    = eigvals[mask]
    eigvecs_r    = eigvecs[:, mask]
    eigvals_sqrt = np.sqrt(eigvals_r)
    r            = int(np.sum(mask)) # count of all values with true.
    np_params    = _n_params(r)

    best_cost   = np.inf
    best_res    = None
    all_values: list[float] = []

    for _ in range(n_restarts):
        x0 = np.random.randn(np_params) * 0.3

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                _cost, x0,
                args=(r, eigvecs_r, eigvals_sqrt, n),
                method=method,
                tol=tol,
                options={"maxiter": 20_000, "maxfun": 50_000},
            )

        val = float(res.fun)
        all_values.append(val)
        if val < best_cost:
            best_cost = val
            best_res  = res

    if use_de:
        bounds = [(-np.pi, np.pi)] * np_params
        de_res = differential_evolution(
            _cost,
            bounds,
            args=(r, eigvecs_r, eigvals_sqrt, n),
            maxiter=500,
            tol=tol,
            seed=42,
        )
        if de_res.fun < best_cost:
            best_cost = float(de_res.fun)
            best_res  = de_res
        all_values.append(float(de_res.fun))

    # Extract decomposition from best result
    if best_res is not None:
        U = _params_to_unitary(best_res.x, r)
        probs, pure_states = _decompose(U, eigvecs_r, eigvals_sqrt)
        decomp = []
        for i in range(r):
            if probs[i] < 1e-12:
                continue
            C_i = _cgme_pure_vector_robust(pure_states[i], n)
            decomp.append(_component_decomp_entry(i, probs[i], C_i, pure_states[i]))
    else:
        decomp = []

    return {
        "C_numerical": max(0.0, best_cost),
        "rank":        r,
        "n_restarts":  n_restarts,
        "converged":   best_res.success if best_res else False,
        "message":     best_res.message if best_res else "No result",
        "decomp":      decomp,
        "all_values":  all_values,
    }


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1 & 2: LOWER BOUNDS (Paper & Supervisor)
# ══════════════════════════════════════════════════════════════════════════════

def convex_roof_lower_bound(rho: np.ndarray, n: int) -> Dict:
    """
    Paper method lower bound on C_GME via coherence witness I_{x,y}(ρ).

        C_GME(ρ) ≥  max_{x≠y}  2 · I_{x,y}(ρ)

    where I_{x,y}(ρ) = |ρ_{x,y}| − Σ_A √(ρ_{x^A,x^A} · ρ_{y^A,y^A})
    and the sum is over ALL non-trivial bipartitions A.

    Returns
    -------
    dict:
        C_LB           : float — best lower bound found
        best_I         : float — raw witness value
        best_x         : list  — best bra bit-string
        best_y         : list  — best ket bit-string
        method         : 'paper_witness'
    """
    dim = 2 ** n
    strings = [[(k >> (n - 1 - b)) & 1 for b in range(n)] for k in range(dim)]

    best_I = 0.0
    best_x = strings[0]
    best_y = strings[-1]

    n_pairs_checked = 0
    for xi in range(dim):
        for yi in range(dim):
            if xi == yi:
                continue
            n_pairs_checked += 1
            I = I_function(rho, n, strings[xi], strings[yi])
            if I > best_I:
                best_I = I
                best_x = strings[xi]
                best_y = strings[yi]

    return {
        "C_LB":    max(0.0, 2.0 * best_I),
        "best_I":  float(best_I),
        "best_x":  best_x,
        "best_y":  best_y,
        # documented in the docstring but previously not returned
        "n_pairs_checked": int(n_pairs_checked),
        "method":  "paper_witness",
    }


def convex_roof_lower_bound_supervisor(rho: np.ndarray, n: int) -> Dict:
    """
    Supervisor method lower bound via eqs.(6-9).
    Delegates to gme.supervisor_gme_lower_bound().
    """
    return supervisor_gme_lower_bound(rho, n)


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 3: UPPER BOUND (Eigendecomposition)
# ══════════════════════════════════════════════════════════════════════════════

def convex_roof_upper_bound(rho: np.ndarray, n: int) -> Dict:
    """
    Upper bound on C_GME via the eigendecomposition.

        C_UB = Σᵢ λᵢ · C_GME(|φᵢ⟩)

    Returns
    -------
    dict:
        C_UB          : float
        rank          : int
        eigenvalues   : list
        contributions : list of {eigenvalue, C_GME_pure} dicts
    """
    eigvals, eigvecs = np.linalg.eigh(rho)
    mask          = eigvals > 1e-10
    eigvals_r     = eigvals[mask]
    eigvecs_r     = eigvecs[:, mask]

    c_ub          = 0.0
    contributions = []

    for lam, phi in zip(eigvals_r, eigvecs_r.T):
        c_i = _cgme_pure_vector_robust(phi, n)
        c_ub += float(lam) * c_i
        contributions.append({
            "eigenvalue": float(lam),
            "C_GME_pure": float(c_i),
        })

    return {
        "C_UB":          float(c_ub),
        "rank":          int(np.sum(mask)),
        "eigenvalues":   eigvals_r.tolist(),
        "contributions": contributions,
    }


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 5: CVXPY SDP RELAXATION (Optional — requires cvxpy)
# ══════════════════════════════════════════════════════════════════════════════

def convex_roof_sdp_relaxation(rho: np.ndarray, n: int) -> Dict:
    """
    SDP relaxation lower bound on convex roof using CVXPY.

    **Requires:** pip install cvxpy

    This is a LOWER BOUND on the true convex roof, obtained by relaxing
    the rank-1 constraint on the pure-state components.

    Returns
    -------
    dict:
        C_sdp      : float or None — SDP lower bound
        status     : str  — solver status
        rank       : int  — rank of input rho
        decomp     : list of {p, t, C_GME_bound}
    """
    try:
        import cvxpy as cp
    except ImportError:
        return {
            "C_sdp": None,
            "status": "cvxpy_not_installed",
            "rank": None,
            "decomp": [],
        }

    dim   = 2**n
    rank  = int(np.sum(np.linalg.eigvalsh(rho) > 1e-10))
    biparts = get_bipartitions(n)

    # Variables: Y_i (PSD matrices), t_i (purity bounds)
    Y = [cp.Variable((dim, dim), hermitian=True) for _ in range(rank)]
    t = [cp.Variable(nonneg=True) for _ in range(rank)]

    constraints = []

    # sum_i Y_i = rho
    constraints.append(cp.sum(Y) == rho)

    # Y_i >= 0
    for i in range(rank):
        constraints.append(Y[i] >> 0)

    # Schur complement for each bipartition
    for i in range(rank):
        p_i = cp.real(cp.trace(Y[i]))
        for A in biparts:
            B = tuple(j for j in range(n) if j not in A)
            dimA = 2**len(A)
            dimB = 2**len(B)

            # Partial trace: rho_A_i
            rho_A_i = cp.Variable((dimA, dimA), hermitian=True)
            for a1 in range(dimA):
                for a2 in range(dimA):
                    constraints.append(
                        rho_A_i[a1, a2] ==
                        cp.sum([Y[i][a1*dimB+b, a2*dimB+b]
                                for b in range(dimB)])
                    )

            # Schur: [t_i * I, rho_A_i; rho_A_i†, I] >= 0
            block = cp.bmat([
                [t[i] * np.eye(dimA), rho_A_i],
                [rho_A_i.H, np.eye(dimA)],
            ])
            constraints.append(block >> 0)

    # Objective: minimize sum_i (1 - t_i)
    objective = cp.Minimize(cp.sum([1 - t[i] for i in range(rank)]))

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False, eps=1e-6, max_iters=10000)
        status = prob.status

        if prob.value is not None and status in ["optimal", "optimal_inaccurate"]:
            obj_val = float(prob.value)
            avg_t = 1.0 - obj_val / rank
            C_sdp = float(np.sqrt(max(2.0 * (1.0 - avg_t), 0.0)))

            decomp_sdp = []
            for i in range(rank):
                p_i_val = float(np.real(np.trace(Y[i].value))) if Y[i].value is not None else 0.0
                t_i_val = float(t[i].value) if t[i].value is not None else 0.0
                C_i_val = float(np.sqrt(max(2.0*(1.0 - t_i_val), 0.0)))
                decomp_sdp.append({"p": p_i_val, "t": t_i_val, "C_GME_bound": C_i_val})

            return {
                "C_sdp": C_sdp,
                "status": status,
                "rank": rank,
                "decomp": decomp_sdp,
            }
        else:
            return {
                "C_sdp": None,
                "status": status,
                "rank": rank,
                "decomp": [],
            }
    except Exception as e:
        return {
            "C_sdp": None,
            "status": f"error: {str(e)}",
            "rank": rank,
            "decomp": [],
        }

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 5: ALGORITHM B — EULER-HURWITZ / STIEFEL MANIFOLD
# (Rothlisberger, Lehmann, Loss — Phys. Rev. A 80, 042301 (2009))
# ══════════════════════════════════════════════════════════════════════════════




def _givens_matrix(k: int, s: int, theta: float, phi: float) -> np.ndarray:
    """
    Build the k×k complex Givens rotation matrix G_s(theta, phi).
    Acts on rows s and s+1 only (0-indexed).
    Equation (27) of Rothlisberger et al.
    """
    G = np.eye(k, dtype=complex)
    G[s,   s  ] =  np.exp( 1j * phi) * np.cos(theta)
    G[s,   s+1] =  np.exp(-1j * phi) * np.sin(theta)
    G[s+1, s  ] = -np.exp( 1j * phi) * np.sin(theta)
    G[s+1, s+1] =  np.exp(-1j * phi) * np.cos(theta)
    return G
 
 
def _n_angles_stiefel(k: int, r: int) -> int:
    """
    Total number of real Euler-Hurwitz angles for St(k,r).
    dim St(k,r) = 2kr - r^2
    Split: n_givens theta + n_givens phi + r chi = 2*n_givens + r
    where n_givens = sum_{j=0}^{r-1} (k-1-j)
    """
    n_givens = sum(k - 1 - j for j in range(r))
    return 2 * n_givens + r



def _params_to_stiefel(params: np.ndarray, k: int, r: int) -> np.ndarray:
    """
    Convert flat real angle vector to U in St(k,r).
 
    Implements inverse of the zeroing procedure (Eq. 32):
      1. Fill k×r matrix B with diagonal phases e^{i*chi_i}
      2. Apply inverse Givens rotations from the left in reverse order
 
    params layout: [theta_0..theta_{n_g-1} | phi_0..phi_{n_g-1} | chi_0..chi_{r-1}]
 
    Returns U : k×r complex matrix with U†U = I_{r×r}
    """
    n_givens = sum(k - 1 - j for j in range(r))
    thetas   = params[:n_givens]
    phis     = params[n_givens: 2 * n_givens]
    chis     = params[2 * n_givens: 2 * n_givens + r]
 
    # Step 1 — diagonal phase matrix
    B = np.zeros((k, r), dtype=complex)
    for i in range(r):
        B[i, i] = np.exp(1j * chis[i])
 
    # Step 2 — apply inverse Givens rotations in reverse order
    idx = n_givens - 1
    U   = B.copy()
 
    for i in range(r - 1, -1, -1):
        col = r - 1 - i
        for j in range(r - i, k):
            row = k - 1 - (j - (r - i))
            if row <= col:
                break
            s     = row - 1
            theta = thetas[idx]
            phi   = phis[idx]
            idx  -= 1
            G     = _givens_matrix(k, s, theta, phi)
            U     = G @ U
 
    return U


def _decompose_stiefel(
    U:            np.ndarray,
    eigvecs:      np.ndarray,
    eigvals_sqrt: np.ndarray,
) -> tuple:
    """
    Given U ∈ St(k,r), return the k-component pure-state decomposition of ρ.
 
    Follows Eq.(5)-(7) of Rothlisberger et al.:
        W  = W0 @ U.T    shape (d, k)
        p_i = ||W[:,i]||^2
        |psi_i> = W[:,i] / sqrt(p_i)
    """
    k  = U.shape[0]
    W0 = eigvecs * eigvals_sqrt    # d×r
    W  = W0 @ U.T                  # d×k
 
    probs       = np.zeros(k)
    pure_states = np.zeros((k, W.shape[0]), dtype=complex)
 
    for i in range(k):
        col = W[:, i]
        p_i = float(np.real(col.conj() @ col))
        p_i = max(p_i, 0.0)
        probs[i] = p_i
        if p_i > 1e-14:
            pure_states[i] = col / np.sqrt(p_i)
 
    return probs, pure_states
 
 
def _cost_stiefel(
    params:       np.ndarray,
    k:            int,
    r:            int,
    eigvecs:      np.ndarray,
    eigvals_sqrt: np.ndarray,
    n:            int,
    measure:      str = "cgme",
) -> float:
    """h(U) = Σ_i p_i · m(|ψ_i⟩)   Eq.(9) of Rothlisberger et al."""
    m_value = _resolve_measure(measure)["value"]
    U = _params_to_stiefel(params, k, r)
    probs, pure_states = _decompose_stiefel(U, eigvecs, eigvals_sqrt)

    total = 0.0
    for i in range(k):
        if probs[i] < 1e-14:
            continue
        total += probs[i] * m_value(pure_states[i], n)
    return total

 
def _optimise_single_k(
    k:            int,
    r:            int,
    eigvecs:      np.ndarray,
    eigvals_sqrt: np.ndarray,
    n:            int,
    n_restarts:   int   = 20,
    max_iter:    int | None = None,
    optimizer:    str        = "L-BFGS-B",
    tol:          float = 1e-9,
    use_de:       bool  = False,
    measure: str = "cgme"
) -> Dict:
    """
    Run L-BFGS-B with n_restarts random initial angles for a fixed k.
    Records every restart result for later visualisation.
    """
    n_params    = _n_angles_stiefel(k, r)
    m_value = _resolve_measure(measure)["value"]
    best_cost   = np.inf
    best_params = None
    best_status: Dict[str, object] = {}
    best_restart_index = None
    restart_log: List[float] = []

    # Adaptive iteration cap.  L-BFGS-B with a few thousand iters is plenty
    # for these problems; 20_000 was overkill and almost never reached.
    # Scale with parameter count so easy problems stay fast.
    if n_params <= 50:
        maxiter, maxfun = 2_000, 5_000
    elif n_params <= 200:
        maxiter, maxfun = 3_000, 8_000
    else:
        maxiter, maxfun = 5_000, 15_000
    if max_iter is not None:
        maxiter = int(max_iter)
        maxfun = max(5 * maxiter, maxiter + 100)
    n_givens = sum(k - 1 - j for j in range(r))
    best_history: List[float] = []   # ← new

    for _ in range(n_restarts):
        # x0 = np.random.uniform(-np.pi, np.pi, n_params)
        x0_theta = np.random.uniform(0, np.pi/2, n_givens)
        x0_phi   = np.random.uniform(-np.pi, np.pi, n_givens)
        x0_chi   = np.random.uniform(-np.pi, np.pi, r)

        x0 = np.concatenate([x0_theta, x0_phi, x0_chi])
        
        bounds = (
            [(0, np.pi/2)] * n_givens +
            [(-np.pi, np.pi)] * n_givens +
            [(-np.pi, np.pi)] * r
        )
        this_history = [_cost_stiefel(x0, k, r, eigvecs, eigvals_sqrt, n, measure)]
        def callback(xk, this_history=this_history):
            this_history.append(_cost_stiefel(xk, k, r, eigvecs, eigvals_sqrt, n, measure))

        options = {}

        if optimizer == "L-BFGS-B":
            options = {
                "maxiter": maxiter,
                "maxfun": maxfun,
            }

        elif optimizer == "Powell":
            options = {
                "maxiter": maxiter,
                "maxfev": maxfun,
                "xtol": tol,
                "ftol": tol,
            }

        elif optimizer == "SLSQP":
            options = {
                "maxiter": maxiter,
                "ftol": tol,
            }

        elif optimizer == "TNC":
            options = {
                "maxfun": maxfun,
            }

        elif optimizer == "trust-constr":
            options = {
                "maxiter": maxiter,
                "gtol": tol,
                "xtol": tol,
            }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                            _cost_stiefel,
                            x0,
                            args=(k, r, eigvecs, eigvals_sqrt, n    , measure   ),
                            method=optimizer,
                            bounds=bounds,
                            options=options,
                            callback=callback,
                        )

        val = float(res.fun)
        restart_log.append(val)
        if val < best_cost:
            best_cost   = val
            best_params = res.x.copy()
            best_restart_index = len(restart_log) - 1
            best_status = {
                "success": bool(getattr(res, "success", False)),
                "message": str(getattr(res, "message", "")),
                "status": int(getattr(res, "status", -999)) if getattr(res, "status", None) is not None else None,
                "nit": int(getattr(res, "nit", 0)) if getattr(res, "nit", None) is not None else None,
                "nfev": int(getattr(res, "nfev", 0)) if getattr(res, "nfev", None) is not None else None,
                "fun": float(val),
            }
            best_history = this_history.copy()       # ← new: save history of best restart
            if len(best_history) == 0 or abs(best_history[-1] - val) > 1e-12:
                best_history.append(val)   
 
    if use_de:
        bounds = [(-np.pi, np.pi)] * n_params
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            de_res = differential_evolution(
                _cost_stiefel,
                bounds,
                args=(k, r, eigvecs, eigvals_sqrt, n, measure),
                maxiter=500, tol=tol, seed=42,
            )
        val = float(de_res.fun)
        restart_log.append(val)
        if val < best_cost:
            best_cost   = val
            best_params = de_res.x.copy()
            best_restart_index = len(restart_log) - 1
            best_status = {
                "success": bool(getattr(de_res, "success", False)),
                "message": str(getattr(de_res, "message", "")),
                "status": None,
                "nit": int(getattr(de_res, "nit", 0)) if getattr(de_res, "nit", None) is not None else None,
                "nfev": int(getattr(de_res, "nfev", 0)) if getattr(de_res, "nfev", None) is not None else None,
                "fun": float(val),
            }


 
    # extract final decomposition
    U = _params_to_stiefel(best_params, k, r)
    probs, pure_states = _decompose_stiefel(U, eigvecs, eigvals_sqrt)
 
    decomp = []
    for i in range(k):
        if probs[i] < 1e-12:
            continue
        C_i = m_value(pure_states[i], n)
        decomp.append(_component_decomp_entry(i, probs[i], C_i, pure_states[i]))
 
    return {
        "k":           k,
        "C_GME":       max(0.0, best_cost),
        "U":           U,
        "params":      best_params,
        "decomp":      decomp,
        "restart_log": restart_log,
        "history": best_history,  # ← new
        "n_params":    n_params,
        "n_restarts":  n_restarts,
        # Exact optimiser limits actually passed to scipy for this k.
        # If CLI max_iter was None, Stiefel uses the adaptive maxiter/maxfun
        # chosen above from n_params; save the resolved numbers, not None/auto.
        "max_iter_input": max_iter,
        "max_iter": maxiter,
        "max_iter_effective": maxiter,
        "maxfun_effective": maxfun,
        "tol": float(tol),
        "optimizer": optimizer,
        "measure": measure,
        "converged": bool(best_status.get("success", False)),
        "completed_successfully": bool(np.isfinite(best_cost)),
        "optimizer_success": bool(best_status.get("success", False)),
        "optimizer_message": best_status.get("message"),
        "optimizer_status": best_status.get("status"),
        "best_restart_index": best_restart_index,
        "best_optimizer_nit": best_status.get("nit"),
        "best_optimizer_nfev": best_status.get("nfev"),
    }
 
 
def convex_roof_stiefel(
    rho:         np.ndarray,
    n:           int,
    n_restarts:  int   = 15,
    tol:         float = 1e-9,
    tol_k:       float = 1e-6,
    k_max_extra: int   = 4,
    max_iter: int | None = None,
    optimizer:    str       = "L-BFGS-B",
    use_de:      bool  = False,
    verbose:     bool  = True,
    auto_scale:  bool  = False,
    measure: str = "cgme"
) -> Dict:
    """
    Algorithm B from Rothlisberger, Lehmann, Loss (2009).

    Parametrises St(k,r) via Euler-Hurwitz angles and minimises
    h(U) using L-BFGS-B.  The cardinality k increases from r until
    consecutive results agree within tol_k.

    Parameters
    ----------
    rho          : density matrix (2^n × 2^n)
    n            : number of qubits/spins
    n_restarts   : random restarts per k (auto-scaled by rank if auto_scale=True)
    tol          : L-BFGS-B convergence tolerance
    tol_k        : outer k-loop convergence tolerance
    k_max_extra  : max extra states beyond rank r to try (default 3, was 8;
                   the paper uses 2-3 in practice)
    use_de       : also run Differential Evolution per k
    verbose      : print progress
    auto_scale   : if True, reduce restarts and L-BFGS-B iterations for
                   high-rank states to keep runtime bounded.  The cost
                   function is O(k * 2^n) per evaluation, so high-rank
                   states are exponentially more expensive.

    Returns
    -------
    dict:
        C_stiefel    : float — final converged C_GME
        rank         : int   — rank r of rho
        k_converged  : int   — k at convergence
        k_records    : list  — full record per k (for visualisation)
        best_record  : dict  — record at converged k
    """
    # n_restarts=30
    # k_max_extra=10
    # eigendecomposition — done once
    eigvals, eigvecs = np.linalg.eigh(rho)
    mask         = eigvals > 1e-10
    eigvals_r    = eigvals[mask]
    eigvecs_r    = eigvecs[:, mask]
    eigvals_sqrt = np.sqrt(eigvals_r)
    r            = int(np.sum(mask))

    # ── Auto-scale n_restarts based on rank ───────────────────────────────────
    # Cost function evaluates exp(iH) over a r×r matrix and an SVD over
    # 2^len(A) × 2^len(B) for each bipartition.  Total parameters scale as
    # 2*k*r - r^2.  For r >= 16 (which arises for n=4 fully-mixed RDMs and
    # almost any n=5), 20 restarts × 9 k-values × 20k iterations is 1+ hour
    # per cluster.  Scale down for high rank.
    n_restarts_actual = n_restarts
    if auto_scale:
        if   r <= 4:   n_restarts_actual = n_restarts             # keep full
        if   r <= 8:   n_restarts_actual = n_restarts             # keep full
        elif r <= 16:   n_restarts_actual = max(10, n_restarts // 2)
        elif r <= 24:  n_restarts_actual = max(6,  n_restarts // 3)
        else:          n_restarts_actual = max(3,  n_restarts // 6)

    if verbose:
        scale_note = (f"  (auto-scaled from {n_restarts})"
                      if auto_scale and n_restarts_actual != n_restarts else "")
        print(f"    [Stiefel] rank r = {r}   "
              f"n_restarts = {n_restarts_actual}{scale_note}   "
              f"k_max_extra = {k_max_extra}")

    k_records:   List[Dict] = []
    prev_C:      float      = np.inf
    best_record: Dict       = {}
    k_history = []
    for extra in range(k_max_extra + 1):
        k = r + extra

        if verbose:
            print(f"    k = {k}  n_params = {_n_angles_stiefel(k, r)}", end="  ")

        _k_t0 = time.perf_counter()
        record = _optimise_single_k(
            k            = k,
            r            = r,
            eigvecs      = eigvecs_r,
            eigvals_sqrt = eigvals_sqrt,
            n            = n,
            n_restarts   = n_restarts_actual,
            max_iter     = max_iter,
            tol          = tol,
            use_de       = use_de,
            optimizer    = optimizer,
            measure      = measure,
        )
        
        record["elapsed_seconds"] = float(time.perf_counter() - _k_t0)
        k_records.append(record)
        current_C = record["C_GME"]
        k_history.append(current_C)
        # The convex roof is a MINIMUM over k, and each k is an independent
        # stochastic search: a later k can land on a worse local minimum.
        # Keep the best record seen, not the last one evaluated.
        if not best_record or current_C < best_record["C_GME"]:
            best_record = record
        if verbose:
            delta = abs(current_C - prev_C)
            print(f"C_GME = {current_C:.8f}  delta = {delta:.2e}")

        # convergence check — need at least one previous value
        if extra > 0 and abs(current_C - prev_C) < tol_k:
            if verbose:
                print(f"    Converged at k = {k}")
            break

        prev_C      = current_C
 
    else:
        if verbose:
            print(f"    Warning: did not converge within k = r + {k_max_extra}. "
                  f"Using best at k = {best_record['k']}.")
 
    _solver_elapsed = float(sum(float(rec.get("elapsed_seconds", 0.0)) for rec in k_records))
    return {
        "C_stiefel":   best_record["C_GME"],
        "rank":        r,
        "k_converged": best_record["k"],
        "k_records":   k_records,
        "k_history":   k_history,
        "best_record": best_record,
        "measure":      measure,
        "n_restarts_requested": int(n_restarts),
        "n_restarts_effective": int(n_restarts_actual),
        "k_max_extra": int(k_max_extra),
        "tol": float(tol),
        "tol_k": float(tol_k),
        "max_iter_input": max_iter,
        "max_iter_policy": "adaptive_by_k" if max_iter is None else "explicit",
        "max_iter_by_k": {int(rec["k"]): int(rec.get("max_iter_effective", rec.get("max_iter", 0)))
                          for rec in k_records},
        "maxfun_by_k": {int(rec["k"]): int(rec.get("maxfun_effective", 0))
                        for rec in k_records},
        "elapsed_seconds": _solver_elapsed,
        "elapsed_by_k": {int(rec["k"]): float(rec.get("elapsed_seconds", 0.0)) for rec in k_records},
        "converged": bool(len(k_records) > 1 and abs(k_records[-1]["C_GME"] - k_records[-2]["C_GME"]) < tol_k),
        "completed_successfully": bool(np.isfinite(best_record.get("C_GME", np.nan))),
        "success_by_k": {int(rec["k"]): bool(rec.get("completed_successfully", False)) for rec in k_records},
        "optimizer_success_by_k": {int(rec["k"]): bool(rec.get("optimizer_success", False)) for rec in k_records},
        "optimizer_message_by_k": {int(rec["k"]): rec.get("optimizer_message") for rec in k_records},
    }



#################
# ══════════════════════════════════════════════════════════════════════════════
# METHOD 6: ALGORITHM A — GENERALIZED CONJUGATE GRADIENT ON U(k)
# (Rothlisberger, Lehmann, Loss — Phys. Rev. A 80, 042301 (2009), Section III A)
#
# The cost function h(U) = Σ_i p_i C_GME(|ψ_i>) is minimised over U ∈ U(k),
# the full unitary group of dimension k. This over-parameterises the search
# space (St(k, r) has dimension 2kr − r²; U(k) has k²), but allows a clean
# Riemannian conjugate-gradient algorithm with parallel transport.
#
# Only the first r columns of U are used to build the decomposition; the
# remaining k − r columns are "padding". The cost depends only on the first r.
#
# Key ingredients:
#   1. Cost h(U)                        ← already defined via _cgme_pure_vector
#   2. Gradient ∂h/∂Re U, ∂h/∂Im U       ← Eqs. (10–11) of the paper
#   3. Skew-Hermitian gradient G(U)     ← Eqs. (22, 23, 26)
#   4. Geodesic step U → U exp(t X)     ← line search via Brent's method
#   5. Parallel transport               ← T = exp(-tX/2) G_old exp(+tX/2)
#      (body/left-trivialized coordinates; see the note in _gcg_optimise_single)
#   6. Modified Polak-Ribière           ← γ = ⟨G_new − T, G_new⟩ / ⟨G_old, G_old⟩
# ══════════════════════════════════════════════════════════════════════════════


def _cgme_pure_with_grad(psi: np.ndarray, n: int,
                         tie_tol: float = TIE_TOL) -> tuple:
    """
    Compute C_GME(|psi>) and a Clarke subgradient w.r.t. psi*.

    Near-tie policy (audit issue #7), shared with ``_cgme_pure_vector_robust``
    through the module-level ``TIE_TOL``:

    * VALUE    — the exact maximum purity; no averaging.
    * GRADIENT — the averaged Clarke subgradient over the active set (all
      bipartitions within ``tie_tol`` of the maximum). This is the centroid of
      the Clarke subdifferential and gives smooth behaviour through symmetric
      states (GHZ, W, ...), where the classical derivative does not exist.
    """
    psi_tensor = psi.reshape([2] * n)
    biparts    = get_bipartitions(n)

    # ---- Pass 1: store reshape data + purity for every bipartition ----
    bp_data = []   # (purity, M, perm, A, rho_A)
    for A in biparts:
        B    = tuple(i for i in range(n) if i not in A)
        perm = list(A) + list(B)
        M    = np.transpose(psi_tensor, perm).reshape(2**len(A), 2**len(B))
        rho_A = M @ M.conj().T
        pur_A = float(np.real(np.trace(rho_A @ rho_A)))
        bp_data.append((pur_A, M, perm, A, rho_A))

    max_purity = max(p[0] for p in bp_data)

    # ---- Identify active set (all bipartitions within tie_tol of max) ----
    active   = [bd for bd in bp_data if max_purity - bd[0] <= tie_tol]
    n_active = len(active)

    # ---- C_GME (depends only on max_purity, not on the active set size) ----
    C      = float(np.sqrt(max(0.0, 2.0 * (1.0 - max_purity))))
    # C = sqrt(2(1-P_max)) has a square-root cusp at C = 0 (separable states):
    # dC/dpsi* = -(1/C) dP_max/dpsi*, so the RAW gradient below diverges like
    # 1/C. The floor keeps it finite; it does NOT make C differentiable there.
    #
    # This is safe in practice because the divergence does not reach the
    # optimizer: _gcg_gradient multiplies this by the component probabilities,
    # sums over the k components and contracts with U, and the 1/C factor
    # cancels. Measured on |000> + eps|111> (see tests/test_separable_boundary.py):
    #
    #     C = 1.98e-01 -> ||grad_conj|| = 9.95e+00 , ||G|| = 2.18e-03
    #     C = 2.00e-05 -> ||grad_conj|| = 1.00e+05 , ||G|| = 1.30e-02
    #     C = 0        -> ||grad_conj|| = 2.00e+10 , ||G|| = 1.50e-02
    #
    # i.e. the final tangent gradient stays O(1e-2) while the raw one reaches
    # 2e10. test_separable_boundary.py asserts ||G|| < 1 so a future refactor
    # cannot silently break that cancellation.
    C_safe = max(C, 1e-10)

    # ---- Averaged Wirtinger subgradient over the active bipartitions ----
    grad_psi_conj = np.zeros(2**n, dtype=complex)
    for _, M, perm, A, rho_A in active:
        grad_M     = 2.0 * (rho_A @ M)
        grad_block = grad_M.reshape([2] * len(A) + [2] * (n - len(A)))
        grad_block = np.transpose(grad_block, np.argsort(perm))
        grad_psi_conj += grad_block.reshape(2**n)
    grad_psi_conj /= n_active

    grad_conj = -grad_psi_conj / C_safe
    return C, grad_conj

#convert U into pure states
def _decompose_full_unitary(
    U: np.ndarray, eigvecs: np.ndarray, eigvals_sqrt: np.ndarray
) -> tuple:
    """
    Build the k-component decomposition of ρ from a k×k unitary U.

    Only the first r = len(eigvals_sqrt) columns of U are used (the other
    columns are unused padding from the over-parameterisation).

    Returns
    -------
    probs       : np.ndarray (k,)
    pure_states : np.ndarray (k, 2^n)  — rows are normalised |ψ_i>
    """
    k = U.shape[0]
    r = eigvals_sqrt.shape[0]
    # Only first r columns of U contribute
    U_active = U[:, :r]                                   # k × r
    W = (eigvecs * eigvals_sqrt) @ U_active.T             # 2^n × k

    probs = np.zeros(k)
    pure_states = np.zeros((k, W.shape[0]), dtype=complex)
    for i in range(k):
        col = W[:, i]
        p_i = float(np.real(col.conj() @ col))
        p_i = max(p_i, 0.0)
        probs[i] = p_i
        if p_i > 1e-14:
            pure_states[i] = col / np.sqrt(p_i)

    return probs, pure_states


def _h_full_unitary(
    U: np.ndarray, eigvecs: np.ndarray, eigvals_sqrt: np.ndarray, n: int, measure: str = "cgme"
) -> float:
    """
    Cost h(U) = Σ_i p_i · C_GME(|ψ_i>) for U ∈ U(k).
    """
    m_grad_fn = _resolve_measure(measure)["grad"]
    probs, pure_states = _decompose_full_unitary(U, eigvecs, eigvals_sqrt)
    total = 0.0
    for i in range(probs.shape[0]):
        if probs[i] < 1e-14:
            continue
        C, _ = m_grad_fn(pure_states[i], n)
        total += probs[i] * C
    return total


def _gcg_gradient(
    U: np.ndarray, eigvecs: np.ndarray, eigvals_sqrt: np.ndarray, n: int, measure: str = "cgme"
) -> np.ndarray:
    """
    Compute the Riemannian gradient G of h(U) on U(k), as a skew-Hermitian
    k×k matrix. Implements Eqs. (10–11) and (22–23, 26) of the paper.

    The construction:
      1. From U, build the k pure states {|ψ_i>} and probabilities {p_i}.
      2. For each (k, l), compute ∂h/∂Re U_{kl} and ∂h/∂Im U_{kl} via Eqs. 10–11.
      3. Combine into matrices A and S (Eqs. 22–23).
      4. Form the skew-Hermitian gradient G via Eq. 26.

    Returns
    -------
    G : np.ndarray (k, k), complex, skew-Hermitian (G† = −G)
    """
    k = U.shape[0]
    r = eigvals_sqrt.shape[0]
    d = eigvecs.shape[0]                # 2^n, dimension of pure-state space
    m_grad_fn = _resolve_measure(measure)["grad"]   
    # Step 1: build decomposition
    probs, pure_states = _decompose_full_unitary(U, eigvecs, eigvals_sqrt)

    # Pre-compute m(|ψ_i>) and ∂m/∂ψ* for each i (vectorised gradient)
    m_vals = np.zeros(k)
    dm_dpsi_conj = np.zeros((k, d), dtype=complex)        # k × d
    for i in range(k):
        if probs[i] < 1e-14:
            continue
        C_i, grad_i = m_grad_fn(pure_states[i], n)
        m_vals[i] = C_i
        dm_dpsi_conj[i] = grad_i

    # Step 2: compute ∂h/∂Re U_{kl} and ∂h/∂Im U_{kl} for all (k, l)
    # Only l < r matters; columns l >= r contribute zero (the over-parameterisation
    # part). We still allocate the full k × k matrices to keep matrix algebra clean.
    dh_dRe_U = np.zeros((k, k))
    dh_dIm_U = np.zeros((k, k))

    sqrt_lambda = eigvals_sqrt                                # shape (r,)
    for kk in range(k):
        if probs[kk] < 1e-14:
            continue
        psi_k = pure_states[kk]
        # Recover ∂m/∂Re ψ_i and ∂m/∂Im ψ_i from Wirtinger:
        dm_dRe_psi =  2.0 * np.real(dm_dpsi_conj[kk])         # (d,)
        dm_dIm_psi = 2.0 * np.imag(dm_dpsi_conj[kk])         # (d,)

        for ll in range(r):
            U_kl = U[kk, ll]
            # |φ_R,kl> = sqrt(p_k λ_l) |χ_l> − λ_l Re(U_kl) |ψ_k>
            # |φ_I,kl> = i sqrt(p_k λ_l) |χ_l> − λ_l Im(U_kl) |ψ_k>
            chi_l = eigvecs[:, ll]                             # (d,)
            lam_l = sqrt_lambda[ll] ** 2
            sqrt_p_lam = np.sqrt(probs[kk] * lam_l)
            phi_R = sqrt_p_lam * chi_l - lam_l * U_kl.real * psi_k
            phi_I = 1j * sqrt_p_lam * chi_l - lam_l * U_kl.imag * psi_k

            # ∂h/∂Re U_kl = 2 λ_l Re(U_kl) m(|ψ_k>)
            #              + Σ_i [Re(φ_R,kl^(i)) ∂m/∂Re ψ^(i) + Im(φ_R,kl^(i)) ∂m/∂Im ψ^(i)]
            term1 = 2.0 * lam_l * U_kl.real * m_vals[kk]
            term2 = np.sum(np.real(phi_R) * dm_dRe_psi + np.imag(phi_R) * dm_dIm_psi)
            dh_dRe_U[kk, ll] = term1 + term2

            term1 = 2.0 * lam_l * U_kl.imag * m_vals[kk]
            term2 = np.sum(np.real(phi_I) * dm_dRe_psi + np.imag(phi_I) * dm_dIm_psi)
            dh_dIm_U[kk, ll] = term1 + term2

    # Step 3: assemble A and S matrices via Eqs. 22–23
    # A_kl = Σ_i [(∂h/∂Re U_il)(Re V_ik) + (∂h/∂Im U_il)(Im V_ik)]
    # S_kl = Σ_i [(∂h/∂Im U_il)(Re V_ik) − (∂h/∂Re U_il)(Im V_ik)]
    # Here V = U is the current point on U(k).
    V_re = np.real(U)
    V_im = np.imag(U)
    A = V_re.T @ dh_dRe_U + V_im.T @ dh_dIm_U           # (k, k), real
    S = V_re.T @ dh_dIm_U - V_im.T @ dh_dRe_U         # (k, k), real

    # Step 4: form skew-Hermitian gradient G via Eq. 26
    # G_kl = (A_kl − A_lk)/2 + i (S_kl + S_lk)/2
    G = 0.5 * (A - A.T) + 0.5j * (S + S.T)
    return G


def _gcg_inner(A: np.ndarray, B: np.ndarray) -> float:
    """Real inner product on the Lie algebra: <A, B> = Tr(A B†)."""
    return float(np.real(np.trace(A @ B.conj().T)))


def _gcg_optimise_single(
    eigvecs:      np.ndarray,
    eigvals_sqrt: np.ndarray,
    n:            int,
    k:            int,
    max_iter:     int   = 1000,
    tol:          float = 1e-9,
    seed:         int   = None,
    measure: str = "cgme",
) -> Dict:
    """
    Run one GCG optimisation from a random starting point in U(k).

    Implements the iteration:
      1. Line search along U exp(t X)
      2. Update U
      3. Compute new gradient G
      4. Parallel-transport old gradient: T = exp(tX/2) G_old exp(-tX/2)
      5. γ = <G − T, G> / <G_old, G_old>   (modified Polak-Ribière)
      6. New direction X = −G + γ X_old

    Returns
    -------
    dict:
        h_best       : best cost found
        U_best       : k×k unitary at best point
        n_iters      : iterations used
        history      : list of cost values per iteration
        converged    : whether tol was reached
    """
    # Audit issue #9: use a LOCAL Generator instead of np.random.seed(), which
    # mutates global RNG state and can silently perturb unrelated callers.
    # Behaviour is still fully reproducible via `seed`, which is echoed back in
    # the returned dict so the effective stream is recorded.
    rng = np.random.default_rng(seed)

    # Random starting point: random Hermitian matrix → exp(iH) gives a unitary
    H0 = rng.standard_normal((k, k)) + 1j * rng.standard_normal((k, k))
    H0 = 0.5 * (H0 + H0.conj().T)
    U = expm(1j * H0)

    # Initial gradient and search direction
    G = _gcg_gradient(U, eigvecs, eigvals_sqrt, n, measure)
    X = -G                                # initial direction is steepest descent

    # Must use the caller's measure: the gradient and every line-search
    # evaluation below use `measure`, so defaulting to cgme here would mix two
    # objectives (wrong history[0] and a wrong first accept/reject comparison).
    h_curr = _h_full_unitary(U, eigvecs, eigvals_sqrt, n, measure=measure)
    history = [h_curr]
    converged = False
    # Audit issue #8: `converged` alone does not describe how a restart ended.
    # Record the terminal path explicitly. Values:
    #   "gradient_and_step_tol" — grad_norm < tol AND improvement < tol
    #   "line_search_stalled"   — line search found no improving step
    #   "line_search_exception" — the scalar minimiser raised
    #   "max_iter"              — iteration budget exhausted
    termination = "max_iter"
    last_t = 1.0                          # for warm-starting line search bracket

    for it in range(max_iter):
        # ── 1. Line search along the geodesic U exp(t X) ──
        # X is skew-Hermitian, so U exp(t X) stays unitary for any real t.
        def line_h(t):
            return _h_full_unitary(U @ expm(t * X), eigvecs, eigvals_sqrt, n, measure)

        # Bracket: search t ∈ [0, t_max]. Use Brent's method.
        # Adaptive bracket: scale with last successful step.
        t_bracket = max(0.1, min(2.0, 4.0 * abs(last_t)))
        try:
            res = minimize_scalar(
                line_h, bracket=(0.0, t_bracket),
                method="brent", options={"xtol": 1e-8, "maxiter": 50},
            )
            t_new = float(res.x)
            h_new = float(res.fun)
        except Exception:
            # Fallback: small fixed step in case Brent fails
            t_new = 0.01
            h_new = line_h(t_new)

        # If t_new is negative or non-improving, reset to steepest descent
        if h_new >= h_curr - 1e-15 or t_new <= 0:
            X = -G
            try:
                res = minimize_scalar(
                    line_h, bracket=(0.0, 1.0),
                    method="brent", options={"xtol": 1e-8, "maxiter": 50},
                )
                t_new = float(res.x)
                h_new = float(res.fun)
            except Exception:
                termination = "line_search_exception"
                break
            if h_new >= h_curr - 1e-15 or t_new <= 0:
                # genuinely converged or stuck
                termination = "line_search_stalled"
                break

        # ── 2. Update U ──
        U_new = U @ expm(t_new * X)

        # ── 3. New gradient ──
        G_new = _gcg_gradient(U_new, eigvecs, eigvals_sqrt, n, measure  )

        # ── 4. Parallel transport old gradient: T = exp(−tX/2) G exp(+tX/2) ──
        # The step is U_new = U @ expm(tX), so X and G are expressed in
        # BODY (left-trivialized) coordinates — verified numerically: the
        # identity d/dt h(U exp(tY))|_0 = <G, Y> holds for arbitrary
        # skew-Hermitian Y, and G is exactly skew-Hermitian.
        #
        # In body coordinates the parallel transport along the geodesic
        # gamma(t) = U exp(tX) is  T(Y) = exp(−tX/2) Y exp(+tX/2).
        # (Eq. (16) of the paper is the AMBIENT form, exp(+tX/2) Y exp(−tX/2);
        # applying it to a body vector transports in the wrong direction and
        # spoils conjugacy.  See tests/test_gcg_transport.py, which measures
        # the covariant derivative: ~5e-11 for this form vs 1.75 for the other.)
        half_step = expm(0.5 * t_new * X)
        T_transp = half_step.conj().T @ G @ half_step

        # ── 5. Modified Polak-Ribière coefficient ──
        denom = _gcg_inner(G, G)
        if abs(denom) < 1e-20:
            gamma = 0.0
        else:
            gamma = _gcg_inner(G_new - T_transp, G_new) / denom
        # Standard practice: clamp γ ≥ 0 (PR+)
        gamma = max(0.0, gamma)

        # ── 6. New search direction ──
        # X generates this geodesic, so it is its own parallel transport
        # (exp(−tX/2) X exp(+tX/2) = X) and needs no correction here.
        X_new = -G_new + gamma * X
        # Descent safeguard: PR+ clamping alone does NOT guarantee a descent
        # direction (measured: 3/8 random trials were ascent both before and
        # after the transport fix). Fall back to steepest descent if needed.
        if _gcg_inner(G_new, X_new) >= 0.0:
            X_new = -G_new

        # ── Convergence check ──
        grad_norm = np.sqrt(_gcg_inner(G_new, G_new))
        improvement = h_curr - h_new
        if grad_norm < tol and improvement < tol:
            U, G, X = U_new, G_new, X_new
            h_curr = h_new
            history.append(h_curr)
            converged = True
            termination = "gradient_and_step_tol"
            break

        # ── Step accepted ──
        U, G, X = U_new, G_new, X_new
        h_curr = h_new
        last_t = t_new
        history.append(h_curr)

    return {
        "h_best":    max(0.0, h_curr),
        "U_best":    U,
        "n_iters":   len(history) - 1,
        "history":   history,
        "converged": converged,
        # audit #8: precise terminal path (see the definition above)
        "termination": termination,
        "seed":      seed,
    }


def convex_roof_gcg(
    rho:          np.ndarray,
    n:            int,
    n_restarts:   int   = 20,
    max_iter:     int   = 300,
    tol:          float = 1e-9,
    tol_k:        float = 1e-6,
    k_max_extra:  int   = 4,
    verbose:      bool  = True,
    auto_scale:   bool  = False,
    measure: str = "cgme",
) -> Dict:
    """
    Algorithm A from Rothlisberger, Lehmann, Loss (2009): Generalized
    Conjugate Gradient on the unitary group U(k).

    Parametrises U ∈ U(k) directly. Uses Riemannian conjugate gradient
    with parallel transport and modified Polak-Ribière. Loops k from
    rank(ρ) upward until consecutive results agree within tol_k.

    This is the analytical-gradient implementation: the gradient of
    h(U) is computed in closed form via Eqs. (10–11, 22–23, 26) of the
    paper; the gradient of C_GME(|ψ>) is computed in closed form via
    the standard ∂Tr(ρ_A²)/∂ψ* = 2[M M† M] identity for the argmax
    bipartition.

    Parameters
    ----------
    rho          : (2^n × 2^n) density matrix
    n            : number of spins
    n_restarts   : random restarts per k (auto-scaled by rank if auto_scale)
    max_iter     : max GCG iterations per restart
    tol          : per-iteration convergence tolerance (gradient norm or improvement)
    tol_k        : k-loop convergence tolerance
    k_max_extra  : max extra states beyond rank r to try
    auto_scale   : reduce restarts for high-rank ρ (matches Stiefel default)

    Returns
    -------
    dict:
        C_gcg        : final converged C_GME
        rank         : rank of ρ
        k_converged  : k at convergence
        k_records    : per-k records (h_best, history, ...)
        best_record  : record at converged k
    """
    # ── Eigendecomposition (done once) ────────────────────────────────────────
    eigvals, eigvecs = np.linalg.eigh(rho)
    mask         = eigvals > 1e-10
    eigvals_r    = eigvals[mask]
    eigvecs_r    = eigvecs[:, mask]
    eigvals_sqrt = np.sqrt(eigvals_r)
    r            = int(np.sum(mask))
    # n_restarts=30
    # ── Auto-scale n_restarts based on rank (same logic as Stiefel) ───────────
    n_restarts_actual = n_restarts
    if auto_scale:
        if   r <= 4:  n_restarts_actual = n_restarts
        elif r <= 8:  n_restarts_actual = max(10, n_restarts // 2)
        elif r <= 16: n_restarts_actual = max(6,  n_restarts // 3)
        elif r <= 24: n_restarts_actual = max(4,  n_restarts // 5)
        else:         n_restarts_actual = max(3,  n_restarts // 6)

    if verbose:
        scale_note = (f"  (auto-scaled from {n_restarts})"
                      if auto_scale and n_restarts_actual != n_restarts else "")
        print(f"    [GCG] rank r = {r}   "
              f"n_restarts = {n_restarts_actual}{scale_note}   "
              f"k_max_extra = {k_max_extra}")

    k_records:   List[Dict] = []
    prev_C:      float      = np.inf
    best_record: Dict       = {}

    # ── Outer k-loop ──────────────────────────────────────────────────────────
    for extra in range(k_max_extra + 1):
        k = r + extra

        if verbose:
            print(f"    k = {k}", end="  ")

        _k_t0 = time.perf_counter()
        # Try multiple random starts; keep the best
        best_h = np.inf
        best_inner = None
        restart_log: List[float] = []
        _termination_counts: Dict[str, int] = {}   # audit #8

        for restart in range(n_restarts_actual):
            seed = 31337 + 17 * extra + restart  # reproducible per (k, restart)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inner = _gcg_optimise_single(
                    eigvecs_r, eigvals_sqrt, n, k,
                    max_iter=max_iter, tol=tol, seed=seed, measure=measure
                )
            restart_log.append(inner["h_best"])
            # audit #8: how often each terminal path occurs across restarts,
            # so line-search stalls vs iteration-limit exits are measurable
            _t = inner.get("termination", "unknown")
            _termination_counts[_t] = _termination_counts.get(_t, 0) + 1
            if inner["h_best"] < best_h:
                best_h = inner["h_best"]
                best_inner = inner

        # Extract decomposition at best U
        U_best = best_inner["U_best"]
        probs, pure_states = _decompose_full_unitary(U_best, eigvecs_r, eigvals_sqrt)
        decomp = []
        for i in range(probs.shape[0]):
            if probs[i] < 1e-12:
                continue
            m_grad_fn = _resolve_measure(measure)["grad"]
            C_i, _ = m_grad_fn(pure_states[i], n)
            # C_i, _ = _cgme_pure_with_grad(pure_states[i], n)
            decomp.append(_component_decomp_entry(i, probs[i], C_i, pure_states[i]))

        record = {
            "k":           k,
            "C_GME":       float(best_h),
            "U":           U_best,
            "decomp":      decomp,
            "restart_log": restart_log,
            "n_iters":     best_inner["n_iters"],
            "history":     best_inner["history"],
            "converged":   bool(best_inner.get("converged", False)),
            # audit #8: how the winning restart actually ended
            "termination": best_inner.get("termination"),
            "termination_counts": _termination_counts,
            "completed_successfully": bool(np.isfinite(best_h)),
            "n_params":    k * k,
            "n_restarts":  n_restarts_actual,
            "max_iter":    int(max_iter),
            "max_iter_effective": int(max_iter),
            "tol":         float(tol),
            "optimizer":   "gcg",
            "measure":     measure,
        }
        record["elapsed_seconds"] = float(time.perf_counter() - _k_t0)
        k_records.append(record)

        # The convex roof is a MINIMUM over k, and each k is an independent
        # stochastic search: a later k can land on a worse local minimum.
        # Keep the best record seen, not the last one evaluated.
        if not best_record or record["C_GME"] < best_record["C_GME"]:
            best_record = record

        delta = abs(best_h - prev_C)
        if verbose:
            print(f"C_GME = {best_h:.8f}  Δ = {delta:.2e}")

        # k-convergence check
        if extra > 0 and delta < tol_k:
            if verbose:
                print(f"    Converged at k = {k}")
            break

        prev_C      = best_h

    else:
        if verbose:
            print(f"    [GCG] Warning: did not converge within k = r + {k_max_extra}.")

    _solver_elapsed = float(sum(float(rec.get("elapsed_seconds", 0.0)) for rec in k_records))
    return {
        "C_gcg":       best_record["C_GME"],
        "rank":        r,
        "k_converged": best_record["k"],
        "k_records":   k_records,
        "best_record": best_record,
        "measure":      measure,
        "n_restarts_requested": int(n_restarts),
        "n_restarts_effective": int(n_restarts_actual),
        "k_max_extra": int(k_max_extra),
        "tol": float(tol),
        "tol_k": float(tol_k),
        "max_iter": int(max_iter),
        "max_iter_effective": int(max_iter),
        "max_iter_by_k": {int(rec["k"]): int(rec.get("max_iter_effective", max_iter))
                          for rec in k_records},
        "elapsed_seconds": _solver_elapsed,
        "elapsed_by_k": {int(rec["k"]): float(rec.get("elapsed_seconds", 0.0)) for rec in k_records},
        "converged": bool(len(k_records) > 1 and abs(k_records[-1]["C_GME"] - k_records[-2]["C_GME"]) < tol_k),
        "completed_successfully": bool(np.isfinite(best_record.get("C_GME", np.nan))),
        "success_by_k": {int(rec["k"]): bool(rec.get("completed_successfully", False)) for rec in k_records},
        "optimizer_success_by_k": {int(rec["k"]): bool(rec.get("converged", False)) for rec in k_records},
    }
# ══════════════════════════════════════════════════════════════════════════════
# METHOD 7: STIEFEL MANIFOLD + RIEMANNIAN CONJUGATE GRADIENT  (hybrid B-on-A)
#
# Goal: minimise  h(U) = Σ_i p_i · m(|ψ_i⟩)  *directly* on the complex Stiefel
# manifold   St(k, r) = { U ∈ ℂ^{k×r} : U†U = I_r },  using the Riemannian
# conjugate-gradient engine of Algorithm A (line search along a retracted ray,
# vector transport, modified Polak–Ribière) — but on the MINIMAL 2kr − r²
# parametrisation of Algorithm B, instead of the over-parametrised U(k)
# (k² params, of which (k−r)² are redundant gauge directions).
#
# Why this is the best of both worlds
# ------------------------------------
#   • intrinsic dimension 2kr − r²        (no redundant gauge directions)
#   • smooth manifold optimisation        (no flat angle chart → no chart seams)
#   • the cost depends on U ONLY through  W = (eigvecs·√λ) @ U.T,  which is the
#     SAME object in the Stiefel and U(k) parametrisations.  Hence the closed-
#     form ambient gradient already derived for the U(k) GCG carries over
#     verbatim — we only change the *projection* and the *retraction*.
#
# Geometry (all standard; Edelman–Arias–Smith 1998; Absil–Mahony–Sepulchre 2008;
#           matches pymanopt's Stiefel defaults)
# ------------------------------------------------------------------------------
#   metric            : Euclidean (embedded)      ⟨A,B⟩ = Re Tr(A†B)
#   Riemannian grad   : grad = P_U(G^E),  P_U(Z) = Z − U·sym(U†Z), sym(X)=(X+X†)/2
#   retraction        : thin-QR,  R_U(ξ) = qf(U + ξ)        (O(k r²), robust)
#   vector transport  : projection,  T(ξ) = P_{U_new}(ξ)
#   β                 : PR+  (modified Polak–Ribière, clamped ≥ 0)
#
# Drop-in: relies only on helpers already present in this module
#   (_resolve_measure, _decompose_full_unitary, _h_full_unitary) plus numpy /
#   scipy.optimize.minimize_scalar / warnings, which are already imported above.
# ══════════════════════════════════════════════════════════════════════════════


# ── Stiefel geometry primitives ───────────────────────────────────────────────

def _stiefel_inner(A: np.ndarray, B: np.ndarray) -> float:
    """Real Euclidean inner product on ℂ^{k×r}:  ⟨A,B⟩ = Re Tr(A†B)."""
    return float(np.real(np.vdot(A, B)))      # vdot conjugates 1st arg → Re Tr(A†B)


def _stiefel_proj(U: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """
    Orthogonal projection of an ambient k×r matrix Z onto the tangent space
        T_U St(k,r) = { Z : U†Z + Z†U = 0 }    (U†Z skew-Hermitian)
    under the Euclidean metric:
        P_U(Z) = Z − U·sym(U†Z),   sym(X) = (X + X†)/2.
    The removed part  U·sym(U†Z)  lies in the normal space {U·H : H Hermitian}.
    """
    UZ  = U.conj().T @ Z                       # r×r
    sym = 0.5 * (UZ + UZ.conj().T)
    return Z - U @ sym


def _stiefel_retract_qr(U: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """
    Thin-QR retraction  R_U(ξ) = qf(U + ξ)  with the phase convention that makes
    R_U(0) = U exactly (positive-real diagonal of the triangular factor).
    Returns a k×r matrix with orthonormal columns.
    """
    Q, R = np.linalg.qr(U + xi)                # reduced: Q is k×r, R is r×r
    d  = np.diagonal(R)
    ph = np.where(np.abs(d) > 1e-300, d / np.abs(d), 1.0 + 0.0j)
    return Q * ph                              # column-wise phase fix == Q @ diag(ph)


def _stiefel_random(k: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """Random point on St(k,r): Q-factor of a complex Gaussian k×r matrix."""
    A = rng.standard_normal((k, r)) + 1j * rng.standard_normal((k, r))
    Q, _ = np.linalg.qr(A)
    return Q


# ── Ambient (Euclidean) gradient of h on ℂ^{k×r} ──────────────────────────────

def _stiefel_euclidean_grad(
    U:            np.ndarray,
    eigvecs:      np.ndarray,
    eigvals_sqrt: np.ndarray,
    n:            int,
    measure:      str = "cgme",
) -> tuple:
    """
    Ambient gradient  G^E ∈ ℂ^{k×r}  of  h(U) = Σ_i p_i m(|ψ_i⟩),  treating U as
    a free complex k×r matrix.  Returns (h, G^E).

    This reuses the EXACT closed-form gradient of the U(k) GCG (Eqs. 10–11 of
    Röthlisberger et al.); the cost depends on U only through W = W0 @ U.T, so
    the per-entry partials ∂h/∂ReU_{kl}, ∂h/∂ImU_{kl} are unchanged.  We then
    package them as  G^E_{kl} = ∂h/∂ReU_{kl} + i ∂h/∂ImU_{kl}, so that
        dh = Re Tr(G^E† dU).
    """
    k = U.shape[0]
    r = eigvals_sqrt.shape[0]
    d = eigvecs.shape[0]                       # 2^n
    m_grad_fn = _resolve_measure(measure)["grad"]

    # k pure states + probabilities (only first r cols of U used; here = all of U)
    probs, pure_states = _decompose_full_unitary(U, eigvecs, eigvals_sqrt)

    # per-state measure value and Wirtinger gradient ∂m/∂ψ*
    m_vals       = np.zeros(k)
    dm_dpsi_conj = np.zeros((k, d), dtype=complex)
    h            = 0.0
    for i in range(k):
        if probs[i] < 1e-14:
            continue
        C_i, grad_i     = m_grad_fn(pure_states[i], n)
        m_vals[i]       = C_i
        dm_dpsi_conj[i] = grad_i
        h              += probs[i] * C_i

    dh_dRe = np.zeros((k, r))
    dh_dIm = np.zeros((k, r))
    for kk in range(k):
        if probs[kk] < 1e-14:
            continue
        psi_k      = pure_states[kk]
        dm_dRe_psi = 2.0 * np.real(dm_dpsi_conj[kk])
        dm_dIm_psi = 2.0 * np.imag(dm_dpsi_conj[kk])
        for ll in range(r):
            U_kl       = U[kk, ll]
            chi_l      = eigvecs[:, ll]
            lam_l      = eigvals_sqrt[ll] ** 2
            sqrt_p_lam = np.sqrt(probs[kk] * lam_l)

            phi_R = sqrt_p_lam      * chi_l - lam_l * U_kl.real * psi_k
            phi_I = 1j * sqrt_p_lam * chi_l - lam_l * U_kl.imag * psi_k

            dh_dRe[kk, ll] = (2.0 * lam_l * U_kl.real * m_vals[kk]
                              + np.sum(np.real(phi_R) * dm_dRe_psi
                                       + np.imag(phi_R) * dm_dIm_psi))
            dh_dIm[kk, ll] = (2.0 * lam_l * U_kl.imag * m_vals[kk]
                              + np.sum(np.real(phi_I) * dm_dRe_psi
                                       + np.imag(phi_I) * dm_dIm_psi))

    G_euclid = dh_dRe + 1j * dh_dIm           # k×r ambient gradient
    return float(h), G_euclid


# ── One Riemannian-CG run on St(k,r) ──────────────────────────────────────────

def _stiefel_gcg_optimise_single(
    eigvecs:      np.ndarray,
    eigvals_sqrt: np.ndarray,
    n:            int,
    k:            int,
    max_iter:     int   = 300,
    tol:          float = 1e-9,
    seed:         int   = None,
    measure:      str   = "cgme",
) -> Dict:
    """
    Riemannian conjugate gradient on St(k,r) from one random start.

    Iteration:
      1. line search t along the retracted ray  t ↦ h(R_U(t d))   (Brent)
      2. U_new = R_U(t d)
      3. g_new = P_{U_new}(G^E(U_new))            (Riemannian gradient)
      4. transport old grad & old dir:  Tg = P_{U_new}(g),  Td = P_{U_new}(d)
      5. β = max(0, ⟨g_new − Tg, g_new⟩ / ⟨g, g⟩) (PR+)
      6. d_new = −g_new + β Td   (reset to −g_new if not a descent direction)
    """
    r   = eigvals_sqrt.shape[0]
    rng = np.random.default_rng(seed)

    U = _stiefel_random(k, r, rng)

    h_curr, G_e = _stiefel_euclidean_grad(U, eigvecs, eigvals_sqrt, n, measure)
    g = _stiefel_proj(U, G_e)                  # Riemannian gradient
    d = -g                                     # initial direction = steepest descent

    history   = [h_curr]
    converged = False
    # Audit issue #8: `converged` alone does not describe how a restart ended.
    # Record the terminal path explicitly. Values:
    #   "gradient_and_step_tol" — grad_norm < tol AND improvement < tol
    #   "line_search_stalled"   — line search found no improving step
    #   "line_search_exception" — the scalar minimiser raised
    #   "max_iter"              — iteration budget exhausted
    termination = "max_iter"
    last_t    = 1.0

    for _ in range(max_iter):
        # ── 1. line search along the retracted geodesic-like ray ──
        def line_h(t):
            return _h_full_unitary(_stiefel_retract_qr(U, t * d),
                                   eigvecs, eigvals_sqrt, n, measure)

        t_bracket = max(0.1, min(2.0, 4.0 * abs(last_t)))
        try:
            res = minimize_scalar(line_h, bracket=(0.0, t_bracket),
                                  method="brent",
                                  options={"xtol": 1e-8, "maxiter": 50})
            t_new, h_new = float(res.x), float(res.fun)
        except Exception:
            t_new, h_new = 0.01, line_h(0.01)

        # non-improving or backwards step → reset to steepest descent, retry once
        if h_new >= h_curr - 1e-15 or t_new <= 0:
            d = -g
            try:
                res = minimize_scalar(line_h, bracket=(0.0, 1.0),
                                      method="brent",
                                      options={"xtol": 1e-8, "maxiter": 50})
                t_new, h_new = float(res.x), float(res.fun)
            except Exception:
                break
            if h_new >= h_curr - 1e-15 or t_new <= 0:
                break                          # genuinely converged / stuck

        # ── 2. update point ──
        U_new = _stiefel_retract_qr(U, t_new * d)

        # ── 3. new Riemannian gradient ──
        _, G_e_new = _stiefel_euclidean_grad(U_new, eigvecs, eigvals_sqrt, n, measure)
        g_new = _stiefel_proj(U_new, G_e_new)

        # ── 4. vector transport (projection) of old grad & direction ──
        Tg = _stiefel_proj(U_new, g)
        Td = _stiefel_proj(U_new, d)

        # ── 5. PR+ coefficient ──
        denom = _stiefel_inner(g, g)
        beta  = 0.0 if abs(denom) < 1e-20 else \
                _stiefel_inner(g_new - Tg, g_new) / denom
        beta  = max(0.0, beta)

        # ── 6. new direction + descent safeguard ──
        d_new = -g_new + beta * Td
        if _stiefel_inner(d_new, g_new) >= 0.0:
            d_new = -g_new

        # ── convergence check ──
        grad_norm   = np.sqrt(_stiefel_inner(g_new, g_new))
        improvement = h_curr - h_new

        U, g, d, h_curr = U_new, g_new, d_new, h_new
        last_t = t_new
        history.append(h_curr)

        if grad_norm < tol and improvement < tol:
            converged = True
            termination = "gradient_and_step_tol"
            break

    return {
        "h_best":    max(0.0, h_curr),
        "U_best":    U,
        "n_iters":   len(history) - 1,
        "history":   history,
        "converged": converged,
        # audit #8: precise terminal path (see the definition above)
        "termination": termination,
        "seed":      seed,
    }


# ── Convex-roof driver (mirrors convex_roof_stiefel / convex_roof_gcg) ─────────

def convex_roof_stiefel_gcg(
    rho:          np.ndarray,
    n:            int,
    n_restarts:   int   = 20,
    max_iter:     int   = 300,
    tol:          float = 1e-9,
    tol_k:        float = 1e-6,
    k_max_extra:  int   = 4,
    verbose:      bool  = True,
    auto_scale:   bool  = False,
    measure:      str   = "cgme",
) -> Dict:
    """
    Hybrid Algorithm: Riemannian Conjugate Gradient on the complex Stiefel
    manifold St(k,r).  Same convex-roof outer loop as convex_roof_stiefel
    (k from rank(ρ) upward until consecutive C_GME agree within tol_k), but the
    inner minimiser is the manifold CG of Algorithm A restricted to the minimal
    2kr − r² parametrisation.

    Parameters mirror convex_roof_stiefel.  n_restarts default 20.

    Returns
    -------
    dict:
        C_stiefel_gcg : final converged C_GME
        rank          : rank r of ρ
        k_converged   : k at convergence
        k_records     : per-k records (history, restart_log, decomp, ...)
        k_history     : list of C_GME per k
        best_record   : record at converged k
    """
    # ── eigendecomposition (once) ──
    eigvals, eigvecs = np.linalg.eigh(rho)
    mask         = eigvals > 1e-10
    eigvals_r    = eigvals[mask]
    eigvecs_r    = eigvecs[:, mask]
    eigvals_sqrt = np.sqrt(eigvals_r)
    r            = int(np.sum(mask))

    # ── auto-scale restarts by rank (same bands as convex_roof_stiefel) ──
    n_restarts_actual = n_restarts
    if auto_scale:
        if   r <= 8:  n_restarts_actual = n_restarts
        elif r <= 16: n_restarts_actual = max(10, n_restarts // 2)
        elif r <= 24: n_restarts_actual = max(6,  n_restarts // 3)
        else:         n_restarts_actual = max(3,  n_restarts // 6)

    if verbose:
        scale_note = (f"  (auto-scaled from {n_restarts})"
                      if auto_scale and n_restarts_actual != n_restarts else "")
        print(f"    [Stiefel-GCG] rank r = {r}   "
              f"n_restarts = {n_restarts_actual}{scale_note}   "
              f"k_max_extra = {k_max_extra}")

    k_records:    List[Dict]  = []
    prev_C:       float       = np.inf
    best_record:  Dict        = {}
    k_history:    List[float] = []

    # ── outer k-loop ──
    for extra in range(k_max_extra + 1):
        k = r + extra
        if verbose:
            print(f"    k = {k}  n_params = {2 * k * r - r * r}", end="  ")

        _k_t0 = time.perf_counter()
        best_h      = np.inf
        best_inner  = None
        restart_log: List[float] = []
        _termination_counts: Dict[str, int] = {}   # audit #8

        for restart in range(n_restarts_actual):
            seed = 91011 + 17 * extra + restart       # reproducible per (k, restart)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                inner = _stiefel_gcg_optimise_single(
                    eigvecs_r, eigvals_sqrt, n, k,
                    max_iter=max_iter, tol=tol, seed=seed, measure=measure,
                )
            restart_log.append(inner["h_best"])
            # audit #8: how often each terminal path occurs across restarts,
            # so line-search stalls vs iteration-limit exits are measurable
            _t = inner.get("termination", "unknown")
            _termination_counts[_t] = _termination_counts.get(_t, 0) + 1
            if inner["h_best"] < best_h:
                best_h     = inner["h_best"]
                best_inner = inner

        # decomposition at best U
        U_best = best_inner["U_best"]
        probs, pure_states = _decompose_full_unitary(U_best, eigvecs_r, eigvals_sqrt)
        m_grad_fn = _resolve_measure(measure)["grad"]
        decomp = []
        for i in range(probs.shape[0]):
            if probs[i] < 1e-12:
                continue
            C_i, _ = m_grad_fn(pure_states[i], n)
            decomp.append(_component_decomp_entry(i, probs[i], C_i, pure_states[i]))

        record = {
            "k":           k,
            "C_GME":       float(best_h),
            "U":           U_best,
            "decomp":      decomp,
            "restart_log": restart_log,
            "n_iters":     best_inner["n_iters"],
            "history":     best_inner["history"],
            "converged":   bool(best_inner.get("converged", False)),
            # audit #8: how the winning restart actually ended
            "termination": best_inner.get("termination"),
            "termination_counts": _termination_counts,
            "completed_successfully": bool(np.isfinite(best_h)),
            "n_params":    2 * k * r - r * r,
            "n_restarts":  n_restarts_actual,
            "max_iter":    int(max_iter),
            "max_iter_effective": int(max_iter),
            "tol":         float(tol),
            "optimizer":   "stiefel-gcg",
            "measure":     measure,
        }
        record["elapsed_seconds"] = float(time.perf_counter() - _k_t0)
        k_records.append(record)
        if not best_record or record["C_GME"] < best_record["C_GME"]:
            best_record = record
        k_history.append(float(best_h))
        

        delta = abs(best_h - prev_C)
        if verbose:
            print(f"C_GME = {best_h:.8f}  Δ = {delta:.2e}")

        if extra > 0 and delta < tol_k:
            if verbose:
                print(f"    Converged at k = {k}")
            break

        prev_C      = best_h
    else:
        if verbose:
            print(f"    [Stiefel-GCG] Warning: did not converge within "
                  f"k = r + {k_max_extra}. Using best at k = {best_record['k']}.")

    _solver_elapsed = float(sum(float(rec.get("elapsed_seconds", 0.0)) for rec in k_records))
    return {
        "C_stiefel_gcg": best_record["C_GME"],
        "rank":          r,
        "k_converged":   best_record["k"],
        "k_records":     k_records,
        "k_history":     k_history,
        "best_record":   best_record,
        "measure":       measure,
        "n_restarts_requested": int(n_restarts),
        "n_restarts_effective": int(n_restarts_actual),
        "k_max_extra": int(k_max_extra),
        "tol": float(tol),
        "tol_k": float(tol_k),
        "max_iter": int(max_iter),
        "max_iter_effective": int(max_iter),
        "max_iter_by_k": {int(rec["k"]): int(rec.get("max_iter_effective", max_iter))
                          for rec in k_records},
        "elapsed_seconds": _solver_elapsed,
        "elapsed_by_k": {int(rec["k"]): float(rec.get("elapsed_seconds", 0.0)) for rec in k_records},
        "converged": bool(len(k_records) > 1 and abs(k_records[-1]["C_GME"] - k_records[-2]["C_GME"]) < tol_k),
        "completed_successfully": bool(np.isfinite(best_record.get("C_GME", np.nan))),
        "success_by_k": {int(rec["k"]): bool(rec.get("completed_successfully", False)) for rec in k_records},
        "optimizer_success_by_k": {int(rec["k"]): bool(rec.get("converged", False)) for rec in k_records},
    }
    
    

################## Literatre REsults reproduction
# ══════════════════════════════════════════════════════════════════════════════
# MEYER-WALLACH PURE-STATE MEASURE WITH ANALYTICAL GRADIENT
# (Rothlisberger, Lehmann, Loss — Phys. Rev. A 80, 042301 (2009), Eq. 45)
#
# Paste this block into convex_roof.py, alongside _cgme_pure_with_grad.
#
# This implements:
#
#   γ(|ψ⟩) = 2 [1 − (1/N) Σ_k Tr(ρ_k²)]
#
# where ρ_k = Tr_{all except k}(|ψ⟩⟨ψ|) is the single-qubit reduced density
# matrix at site k.
#
# The analytical gradient (Wirtinger) is:
#
#   ∂γ/∂ψ*_i = −(4/N) Σ_k [M_k M_k† M_k]_{at site (a_k, b_k) of i}
#
# where M_k is the bipartition reshape of ψ as (qubit k | rest).
#
# Unlike C_GME (which uses max over bipartitions and needs tie averaging),
# γ uses a SUM over single-qubit cuts → smooth, no argmax instability,
# no 1/C blow-up. Gradient is well-defined everywhere.
# ══════════════════════════════════════════════════════════════════════════════


def _meyerwallach_pure_vector(psi: np.ndarray, n: int) -> float:
    """
    Meyer-Wallach γ(|ψ⟩) = 2 [1 − (1/N) Σ_k Tr(ρ_k²)]   Eq. (45) of the paper.

    Parameters
    ----------
    psi : (2^n,) complex array, normalised pure state.
    n   : number of qubits.

    Returns
    -------
    gamma : float in [0, 1].
    """
    psi_tensor = psi.reshape([2] * n)
    purity_sum = 0.0

    for k in range(n):
        # Reshape ψ as M_k of shape (2, 2^(n-1)) with qubit k as row index.
        # Move axis k to position 0 (leading axis).
        axes = [k] + [i for i in range(n) if i != k]
        M_k = np.transpose(psi_tensor, axes).reshape(2, 2 ** (n - 1))

        # rho_k = M_k M_k†, then Tr(rho_k²) = Tr((M_k M_k†)(M_k M_k†))
        rho_k = M_k @ M_k.conj().T              # (2, 2)
        purity_k = float(np.real(np.trace(rho_k @ rho_k)))
        purity_sum += purity_k

    gamma = 2.0 * (1.0 - purity_sum / n)
    return float(max(0.0, gamma))


def _meyerwallach_pure_with_grad(psi: np.ndarray, n: int) -> tuple:
    """
    Compute γ(|ψ⟩) AND its complex Wirtinger gradient ∂γ/∂ψ*.

    Parameters
    ----------
    psi : (2^n,) complex array, normalised pure state.
    n   : number of qubits.

    Returns
    -------
    gamma     : float, the MW value.
    grad_conj : (2^n,) complex array. The Wirtinger gradient ∂γ/∂ψ*.
                Real and imaginary parts of the gradient w.r.t. Re(ψ_i) and
                Im(ψ_i) can be recovered as:
                    ∂γ/∂Re ψ_i = 2 Re(grad_conj[i])
                    ∂γ/∂Im ψ_i = 2 Im(grad_conj[i])
                (Same Wirtinger convention as _cgme_pure_with_grad.)

    Notes
    -----
    Uses the general formula Eq. (A.1, A.2) of the paper, valid for any
    pure state — does NOT assume rotational symmetry. (The simplified
    Eqs. A.4-A.7 are correct only for R-eigenstates.)

    The derivation:
      γ = 2 − (2/N) Σ_k Tr(ρ_k²)
      ∂γ/∂ψ* = −(2/N) Σ_k ∂Tr(ρ_k²)/∂ψ*
            = −(2/N) Σ_k · 2 · [M_k M_k† M_k]
            = −(4/N) Σ_k [M_k M_k† M_k]_{reshaped back to ψ}
    """
    psi_tensor = psi.reshape([2] * n)
    purity_sum = 0.0
    grad_psi_conj = np.zeros(2 ** n, dtype=complex)

    for k in range(n):
        # Build the permutation that puts qubit k first
        perm = [k] + [i for i in range(n) if i != k]

        # Reshape ψ as M_k of shape (2, 2^(n-1))
        M_k = np.transpose(psi_tensor, perm).reshape(2, 2 ** (n - 1))

        # Single-qubit reduced density matrix and its purity
        rho_k = M_k @ M_k.conj().T                          # (2, 2)
        purity_k = float(np.real(np.trace(rho_k @ rho_k)))
        purity_sum += purity_k

        # Gradient of Tr(ρ_k²) w.r.t. ψ*:
        # ∂Tr(ρ_k²)/∂ψ*  = 2 [M_k M_k† M_k] in (qubit k, rest) coordinates
        grad_M_k = 2.0 * (rho_k @ M_k)                      # (2, 2^(n-1))

        # Reshape gradient back into the original ψ ordering.
        # Step 1: unpack the rest-of-qubits axis into individual qubits.
        grad_block = grad_M_k.reshape([2] * n)
        # Step 2: invert the permutation
        inv_perm = np.argsort(perm)
        grad_block = np.transpose(grad_block, inv_perm)

        # Accumulate
        grad_psi_conj += grad_block.reshape(2 ** n)

    # γ = 2(1 − purity_sum/N), so dγ/dψ* = −(2/N) · d(purity_sum)/dψ*
    # And d(purity_sum)/dψ* = Σ_k 2 [M_k M_k† M_k] = grad_psi_conj as accumulated.
    # So dγ/dψ* = −(2/N) · grad_psi_conj.
    # But each [M_k M_k† M_k] term already had its factor of 2 built in above,
    # so the prefactor here is −(2/N), not −(4/N).
    grad_conj = -(2.0 / n) * grad_psi_conj

    gamma = 2.0 * (1.0 - purity_sum / n)
    return float(max(0.0, gamma)), grad_conj


# ──────────────────────────────────────────────────────────────────────────
# Measure registry — maps measure name to (value_fn, grad_fn) pair
# ──────────────────────────────────────────────────────────────────────────

MEASURES = {
    "cgme": {
        "value": _cgme_pure_vector_robust,
        "grad":  _cgme_pure_with_grad,
    },
    "meyerwallach": {
        "value": _meyerwallach_pure_vector,
        "grad":  _meyerwallach_pure_with_grad,
    },
}


def _resolve_measure(measure: str) -> dict:
    """Look up the (value_fn, grad_fn) pair for a measure name."""
    if measure not in MEASURES:
        raise ValueError(
            f"Unknown measure '{measure}'. "
            f"Available: {list(MEASURES.keys())}"
        )
    return MEASURES[measure]
#################



# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED INTERFACE
# ══════════════════════════════════════════════════════════════════════════════




SOLVER_METHODS = ("stiefel", "gcg", "stiefel_gcg")

# Effective defaults used when the CLI leaves these values as None.
# These constants make saved PKLs/PDF metadata explicit instead of showing
# ambiguous values such as "auto".
DEFAULT_SOLVER_TOL = 1e-9
DEFAULT_K_CONVERGENCE_TOL = 1e-6
DEFAULT_GCG_MAX_ITER = 300
DEFAULT_STIEFEL_GCG_MAX_ITER = 300


def _array_sha256(arr: Optional[np.ndarray]) -> Optional[str]:
    """Stable SHA256 hash for numpy arrays saved in metadata."""
    if arr is None:
        return None
    arr_c = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr_c.shape).encode("utf-8"))
    h.update(str(arr_c.dtype).encode("utf-8"))
    h.update(arr_c.view(np.uint8))
    return h.hexdigest()


def _rho_diagnostics(rho: np.ndarray, eig_threshold: float = 1e-10) -> Dict[str, object]:
    """Compact integrity/audit metadata for the input density matrix."""
    eigvals = np.linalg.eigvalsh(rho)
    eigvals_real = np.real(eigvals)
    rank = int(np.sum(eigvals_real > eig_threshold))
    return {
        "rho_shape": tuple(rho.shape),
        "rho_dtype": str(rho.dtype),
        "rho_hash_sha256": _array_sha256(rho),
        "trace": float(np.real(np.trace(rho))),
        "purity": float(np.real(np.trace(rho @ rho))),
        "rank_threshold": float(eig_threshold),
        "rank": rank,
        "eigvals": eigvals_real.tolist(),
        "eigvals_nonzero": eigvals_real[eigvals_real > eig_threshold].tolist(),
        "lambda_min": float(np.min(eigvals_real)) if eigvals_real.size else None,
        "lambda_max": float(np.max(eigvals_real)) if eigvals_real.size else None,
        "hermitian_max_error": float(np.max(np.abs(rho - rho.conj().T))),
        "trace_error": float(abs(np.real(np.trace(rho)) - 1.0)),
    }


def _component_decomp_entry(index: int, p: float, C_i: float, psi: np.ndarray) -> Dict[str, object]:
    """One pure-state component in a convex-roof decomposition."""
    psi_arr = np.asarray(psi, dtype=complex).copy()
    return {
        "component_index": int(index),
        "p": float(p),
        "C_GME": float(C_i),
        "psi": psi_arr,
        "psi_shape": tuple(psi_arr.shape),
        "psi_dtype": str(psi_arr.dtype),
        "psi_hash_sha256": _array_sha256(psi_arr),
        "psi_norm": float(np.real(np.vdot(psi_arr, psi_arr))),
    }


def _runtime_from_solver(solver_result: Optional[dict]) -> Optional[float]:
    if not isinstance(solver_result, dict):
        return None
    val = solver_result.get("elapsed_seconds")
    return None if val is None else float(val)


def _solver_success_summary(solver_result: Optional[dict]) -> Dict[str, object]:
    """Small human-readable success/convergence summary for one solver."""
    if not isinstance(solver_result, dict):
        return {"available": False, "completed_successfully": False, "converged": False}
    k_records = solver_result.get("k_records", []) or []
    finite = True
    for rec in k_records:
        try:
            finite = finite and np.isfinite(float(rec.get("C_GME", np.nan)))
        except Exception:
            finite = False
    return {
        "available": True,
        "completed_successfully": bool(len(k_records) > 0 and finite),
        "converged": bool(solver_result.get("converged", False)),
        "k_converged": solver_result.get("k_converged"),
        "n_k_records": len(k_records),
        "best_C_GME": solver_result.get("C_stiefel", solver_result.get("C_gcg", solver_result.get("C_stiefel_gcg"))),
        "elapsed_seconds": _runtime_from_solver(solver_result),
    }


def _empty_solver_status(status: str = "skipped") -> dict:
    return {key: status for key in SOLVER_METHODS}


def _solver_scalar(solver_result: Optional[dict], scalar_key: str) -> float:
    """Return a scalar value from one solver result, or NaN if that solver was not run."""
    if not isinstance(solver_result, dict):
        return float("nan")
    val = solver_result.get(scalar_key, solver_result.get("C_GME", None))
    if val is None:
        return float("nan")
    try:
        return float(val)
    except Exception:
        return float("nan")


def _solver_effective_settings(
    solver_key: str,
    solver_result: Optional[dict],
    *,
    n_restarts_requested: int,
    k_max_extra: int,
    max_iter_input: Optional[int],
    tol_effective: float,
    tol_k_effective: float,
) -> Dict[str, object]:
    """Summarise exact solver settings saved to PKL/PDF metadata.

    For Stiefel with max_iter_input=None, the exact maxiter is adaptive and
    depends on k/n_params, so it is saved per k in max_iter_by_k. GCG and
    Stiefel-GCG have a true default of 300 iterations when the CLI leaves
    max_iter unspecified.
    """
    summary: Dict[str, object] = {
        "solver": solver_key,
        "n_restarts_requested": int(n_restarts_requested),
        "k_max_extra": int(k_max_extra),
        "tol": float(tol_effective),
        "tol_k": float(tol_k_effective),
        "max_iter_input": None if max_iter_input is None else int(max_iter_input),
    }

    if not isinstance(solver_result, dict):
        summary.update({"status": "not_available"})
        return summary

    k_records = solver_result.get("k_records", []) or []
    k_values = [int(rec.get("k")) for rec in k_records if rec.get("k") is not None]
    max_iter_by_k = {
        int(rec.get("k")): int(rec.get("max_iter_effective", rec.get("max_iter", 0)))
        for rec in k_records
        if rec.get("k") is not None and rec.get("max_iter_effective", rec.get("max_iter", None)) is not None
    }
    n_restarts_by_k = {
        int(rec.get("k")): int(rec.get("n_restarts", len(rec.get("restart_log", []) or [])))
        for rec in k_records
        if rec.get("k") is not None
    }
    summary.update({
        "status": "available",
        "rank": solver_result.get("rank"),
        "k_values_tried": k_values,
        "k_converged": solver_result.get("k_converged"),
        "n_restarts_effective": solver_result.get("n_restarts_effective"),
        "n_restarts_by_k": n_restarts_by_k,
        "max_iter_by_k": max_iter_by_k,
        "measure": solver_result.get("measure"),
    })

    if solver_key == "stiefel" and max_iter_input is None:
        summary["max_iter_policy"] = "adaptive_by_k"
        summary["max_iter"] = max_iter_by_k
        summary["maxfun_by_k"] = solver_result.get("maxfun_by_k")
    else:
        eff = DEFAULT_GCG_MAX_ITER if max_iter_input is None else int(max_iter_input)
        if solver_key == "stiefel_gcg" and max_iter_input is None:
            eff = DEFAULT_STIEFEL_GCG_MAX_ITER
        summary["max_iter_policy"] = "default_300" if max_iter_input is None else "explicit"
        summary["max_iter"] = int(eff)

    return summary


def full_convex_roof_analysis(
    rho:        np.ndarray,
    sites:      tuple,
    n_restarts: int  = 20,
    k_max_extra: int   = 4,
    max_iter: int | None = None,
    tol: float | None = None,
    optimizer:    str       = "L-BFGS-B",
    method:       str       = "c_gme",
    measure:      str       = "cgme",
    use_de:     bool = False,
    use_cvxpy:  bool = False,
    verbose:    bool = True,
    active_solvers: Optional[tuple[str, ...]] = None,
    precomputed_solvers: Optional[Dict[str, Dict]] = None,
    solver_cache_status: Optional[Dict[str, object]] = None,
) -> Dict:
    """
    Run convex-roof analyses on a mixed n-spin RDM.

    The three numerical solvers can now be controlled independently:
      - stiefel
      - gcg
      - stiefel_gcg

    Parameters
    ----------
    active_solvers
        Tuple/list of solver keys that must be recomputed in this call.
        If None, all three solvers are computed (legacy behaviour).
        If empty, no solver is recomputed and only precomputed_solvers are used.
    precomputed_solvers
        Optional dict containing already-loaded solver results from PKL, e.g.
        {"stiefel": cached_result["stiefel"]}. These are inserted into the
        returned result without recomputation.
    solver_cache_status
        Optional per-solver metadata inserted into the returned result. The
        pipeline uses this to mark PDF rows as PKL-reused vs recomputed.

    Returns
    -------
    dict with keys:
        sites, n, global_purity
        C_LB_supervisor, C_LB_best, C_UB
        C_stiefel, C_gcg, C_stiefel_gcg
        stiefel, gcg, stiefel_gcg
        solver_cache_status / cache_status
    """
    _full_t0 = time.perf_counter()
    n  = len(sites)
    gp = purity(rho)
    rho_metadata = _rho_diagnostics(rho)

    if active_solvers is None:
        active_solver_set = set(SOLVER_METHODS)
    else:
        active_solver_set = {str(x) for x in active_solvers}
        invalid = active_solver_set.difference(SOLVER_METHODS)
        if invalid:
            raise ValueError(
                f"Unknown active_solvers={sorted(invalid)}. "
                f"Valid solver keys are {SOLVER_METHODS}."
            )

    precomputed_solvers = precomputed_solvers or {}
    solver_cache_status = dict(solver_cache_status or {})

    if verbose:
        site_str = ",".join(str(s) for s in sites)
        active_str = ", ".join(sorted(active_solver_set)) if active_solver_set else "none"
        reused_str = ", ".join(k for k in SOLVER_METHODS if k in precomputed_solvers) or "none"
        print(f"\n  [Convex roof]  sites=({site_str})  n={n}  γ={gp:.6f}")
        print(f"    solvers to compute : {active_str}")
        print(f"    solvers from PKL   : {reused_str}")

    # Standard lower/upper bounds. The paper witness is currently disabled in
    # this code path; keep its slot for backward-compatible result structure.
    _bounds_t0 = time.perf_counter()
    lb      = 0  # convex_roof_lower_bound(rho, n)
    lb_sup  = convex_roof_lower_bound_supervisor(rho, n)
    ub      = convex_roof_upper_bound(rho, n)
    bounds_elapsed = float(time.perf_counter() - _bounds_t0)
    num     = 0  # convex_roof_numerical(rho, n, n_restarts=n_restarts, use_de=use_de)
    num2    = 0  # unitary_convex_roof(rho, n, n_restarts)

    # ── Resolve the cost measure from `method` ───────────────────────────────
    _measure_aliases = {
        "c_gme": "cgme",
        "cgme": "cgme",
        "mw": "meyerwallach",
        "meyer_wallach": "meyerwallach",
        "meyerwallach": "meyerwallach",
    }
    if method in _measure_aliases:
        chosen_measure = _measure_aliases[method]
    elif method in MEASURES:
        chosen_measure = method
    else:
        chosen_measure = measure

    def _supported_kwargs(func, **candidate):
        """Keep only the kwargs that `func` actually declares."""
        params = inspect.signature(func).parameters
        return {k: v for k, v in candidate.items() if k in params}

    tol_effective = DEFAULT_SOLVER_TOL if tol is None else float(tol)
    tol_k_effective = DEFAULT_K_CONVERGENCE_TOL

    # Do not pass max_iter when the CLI leaves it unspecified for GCG-family
    # solvers: this preserves their code default (300). The exact effective
    # value is saved below in solver_effective_settings.
    _iter_kw = {} if max_iter is None else {"max_iter": int(max_iter)}

    # Start with cached/precomputed solver results, then recompute only the
    # selected active solvers. A recomputed solver overwrites the cached entry.
    stiefel = precomputed_solvers.get("stiefel")
    gcg = precomputed_solvers.get("gcg")
    stiefel_gcg = precomputed_solvers.get("stiefel_gcg")

    if "stiefel" in active_solver_set:
        _solver_t0 = time.perf_counter()
        stiefel = convex_roof_stiefel(
            rho, n,
            n_restarts=n_restarts, k_max_extra=k_max_extra,
            measure=chosen_measure, verbose=verbose,
            max_iter=max_iter, tol=tol_effective,
            **_supported_kwargs(convex_roof_stiefel, optimizer=optimizer),
        )
        stiefel["elapsed_seconds"] = float(stiefel.get("elapsed_seconds", 0.0)) or float(time.perf_counter() - _solver_t0)
        solver_cache_status.setdefault("stiefel", "recomputed")
    elif stiefel is not None:
        solver_cache_status.setdefault("stiefel", "reused_pkl")
    else:
        solver_cache_status.setdefault("stiefel", "skipped")

    if "gcg" in active_solver_set:
        _solver_t0 = time.perf_counter()
        gcg = convex_roof_gcg(
            rho, n,
            n_restarts=n_restarts, k_max_extra=k_max_extra,
            measure=chosen_measure, verbose=verbose, tol=tol_effective,
            **_iter_kw,
            **_supported_kwargs(convex_roof_gcg, optimizer=optimizer),
        )
        gcg["elapsed_seconds"] = float(gcg.get("elapsed_seconds", 0.0)) or float(time.perf_counter() - _solver_t0)
        solver_cache_status.setdefault("gcg", "recomputed")
    elif gcg is not None:
        solver_cache_status.setdefault("gcg", "reused_pkl")
    else:
        solver_cache_status.setdefault("gcg", "skipped")

    if "stiefel_gcg" in active_solver_set:
        _solver_t0 = time.perf_counter()
        stiefel_gcg = convex_roof_stiefel_gcg(
            rho, n,
            n_restarts=n_restarts, k_max_extra=k_max_extra,
            measure=chosen_measure, verbose=verbose, tol=tol_effective,
            **_iter_kw,
            **_supported_kwargs(convex_roof_stiefel_gcg, optimizer=optimizer),
        )
        stiefel_gcg["elapsed_seconds"] = float(stiefel_gcg.get("elapsed_seconds", 0.0)) or float(time.perf_counter() - _solver_t0)
        solver_cache_status.setdefault("stiefel_gcg", "recomputed")
    elif stiefel_gcg is not None:
        solver_cache_status.setdefault("stiefel_gcg", "reused_pkl")
    else:
        solver_cache_status.setdefault("stiefel_gcg", "skipped")

    C_stiefel = _solver_scalar(stiefel, "C_stiefel")
    C_gcg = _solver_scalar(gcg, "C_gcg")
    C_stiefel_gcg = _solver_scalar(stiefel_gcg, "C_stiefel_gcg")

    # Existing fields retained. C_LB paper is disabled above, so do not let the
    # disabled placeholder dominate C_LB_best.
    C_LB      = 0.0
    C_LB_sup  = lb_sup["C_LB"]
    # The pure-state formula is only valid for a PURE rho. Applying it to a
    # mixed rho saturates at ~1.0 and is not the convex roof (use C_gcg /
    # C_stiefel for that). Gate on purity, exactly as the solver dispatch does.
    _purity   = float(np.real(np.trace(rho @ rho)))
    _is_pure  = _purity > 1.0 - 1e-6
    C_pure    = float(C_GME_pure_dm(rho, n)) if _is_pure else float("nan")
    C_GME     = C_pure          # backward-compatible key; NaN when rho is mixed
    C_UB      = ub["C_UB"]
    C_expo    = 0.0
    C_cayl    = 0.0
    C_LB_best = max(C_LB, C_LB_sup)

    C_sdp = None
    sdp_result = None
    if use_cvxpy:
        sdp_result = convex_roof_sdp_relaxation(rho, n)
        C_sdp = sdp_result.get("C_sdp")

    # ── Numerical convex roof, bound consistency and GME status ──────────────
    # These three fields used to be hard-coded placeholders (`consistent = 0`,
    # `status = "none"`) even though the docstring, the tests and downstream
    # consumers (molecular_pipeline / *_visualization / molecular_audit) expect
    # real values. They are computed here from quantities already in hand.
    #
    # C_numerical is the best (lowest) value any convex-roof solver achieved:
    # each solver returns an UPPER bound on the true roof, so the minimum over
    # solvers is the tightest estimate. NaNs (skipped solvers) are ignored.
    _roof_candidates = [c for c in (C_stiefel, C_gcg, C_stiefel_gcg)
                        if c is not None and np.isfinite(c)]
    C_numerical = float(min(_roof_candidates)) if _roof_candidates else float("nan")

    # A consistent result satisfies   C_LB_best <= C_numerical <= C_UB
    # (tolerance absorbs solver noise). NaN => not verifiable => not consistent.
    _tol_consist = 1e-4
    consistent = int(
        np.isfinite(C_numerical)
        and (C_LB_best <= C_numerical + _tol_consist)
        and (C_numerical <= C_UB + _tol_consist)
    )

    # GME classification:
    #   no_GME       — the numerical roof is zero within tolerance
    #   GME          — a certified lower bound is strictly positive
    #   inconclusive — roof > 0 but no positive certified lower bound
    _tol_gme = 1e-5
    if not np.isfinite(C_numerical):
        status = "failed"
    elif C_numerical < _tol_gme:
        status = "no_GME"
    elif C_LB_best > _tol_gme:
        status = "GME"
    else:
        status = "inconclusive"

    solver_effective_settings = {
        "stiefel": _solver_effective_settings(
            "stiefel", stiefel,
            n_restarts_requested=n_restarts, k_max_extra=k_max_extra,
            max_iter_input=max_iter, tol_effective=tol_effective,
            tol_k_effective=tol_k_effective,
        ),
        "gcg": _solver_effective_settings(
            "gcg", gcg,
            n_restarts_requested=n_restarts, k_max_extra=k_max_extra,
            max_iter_input=max_iter, tol_effective=tol_effective,
            tol_k_effective=tol_k_effective,
        ),
        "stiefel_gcg": _solver_effective_settings(
            "stiefel_gcg", stiefel_gcg,
            n_restarts_requested=n_restarts, k_max_extra=k_max_extra,
            max_iter_input=max_iter, tol_effective=tol_effective,
            tol_k_effective=tol_k_effective,
        ),
    }


    solver_runtime = {
        "stiefel": _runtime_from_solver(stiefel),
        "gcg": _runtime_from_solver(gcg),
        "stiefel_gcg": _runtime_from_solver(stiefel_gcg),
    }
    solver_success = {
        "stiefel": _solver_success_summary(stiefel),
        "gcg": _solver_success_summary(gcg),
        "stiefel_gcg": _solver_success_summary(stiefel_gcg),
    }
    total_elapsed = float(time.perf_counter() - _full_t0)

    result = {
        "sites":              sites,
        "n":                  n,
        "global_purity":      float(gp),
        "rho_metadata":       rho_metadata,
        "rho_hash_sha256":    rho_metadata.get("rho_hash_sha256"),
        "rho_eigvals":        rho_metadata.get("eigvals"),
        "rho_eigvals_nonzero": rho_metadata.get("eigvals_nonzero"),
        "lower_bound":        lb,
        "lower_bound_super":  lb_sup,
        "upper_bound":        ub,
        "numerical":          num,
        "C_LB":               float(C_LB),
        "C_LB_supervisor":    float(C_LB_sup),
        "C_LB_best":          float(C_LB_best),
        # C_GME is retained for backward compatibility with existing PKLs and
        # consumers, but it is the PURE-state value: it is NaN unless rho is
        # pure. The mixed-state convex roof is C_gcg / C_stiefel.
        "C_GME":              C_GME,
        "C_pure":             C_pure,      # NaN when rho is mixed
        "rho_is_pure":        bool(_is_pure),
        "rho_purity":         _purity,
        "C_expo":             float(C_expo),
        "C_cayl":             float(C_cayl),
        "C_stiefel":          float(C_stiefel),
        "C_stiefel_gcg":      float(C_stiefel_gcg),
        "C_gcg":              float(C_gcg),
        "C_UB":               float(C_UB),
        "C_sdp":              C_sdp,
        # best (lowest) numerical convex roof over the solvers that ran
        "C_numerical":        float(C_numerical),
        "bounds_consistent":  consistent,
        "status":             status,
        "stiefel":            stiefel,
        "gcg":                gcg,
        "stiefel_gcg":        stiefel_gcg,
        "active_solvers":     tuple(sorted(active_solver_set)),
        "solver_cache_status": solver_cache_status,
        # Short alias used by visualization.py.
        "cache_status":       solver_cache_status,
        "solver_effective_settings": solver_effective_settings,
        "effective_settings": solver_effective_settings,
        "solver_runtime": solver_runtime,
        "solver_success": solver_success,
        "elapsed_seconds": total_elapsed,
        "elapsed_by_stage": {
            "bounds": bounds_elapsed,
            "stiefel": solver_runtime.get("stiefel"),
            "gcg": solver_runtime.get("gcg"),
            "stiefel_gcg": solver_runtime.get("stiefel_gcg"),
            "total": total_elapsed,
        },
        "tol": float(tol_effective),
        "tol_k": float(tol_k_effective),
        "max_iter_input": max_iter,
    }

    if sdp_result is not None:
        result["sdp_result"] = sdp_result

    return result

def convex_roof_analysis_complete(
    rho: np.ndarray,
    sites: tuple,
    n_restarts: int = 20,
    use_cvxpy: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Alias for full_convex_roof_analysis for backward compatibility.
    """
    return full_convex_roof_analysis(
        rho, sites, n_restarts=n_restarts, use_cvxpy=use_cvxpy, verbose=verbose
    )