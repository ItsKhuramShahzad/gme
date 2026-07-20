"""
cvxpy_convex_roof.py
--------------------
Tests all GME methods for both cases:
  Case 1: n=3 (mixed RDM, n < N)  — convex roof needed
  Case 2: n=4 (pure RDM,  n = N)  — exact C_GME known, test all bounds

Methods:
  1. Lower bound — paper (Ma et al.)
  2. Lower bound — supervisor eqs.(6-9)
  3. Convex roof — existing numerical (unitary optimization)
  4. Convex roof — CVXPY SDP relaxation (as suggested in paper)
     Variables: Y_i >= 0,  sum_i Y_i = rho
     Objective: minimize sum_i C_GME_relaxed(Y_i)
     The rank-1 constraint on Y_i is dropped (SDP relaxation)
     This gives a LOWER BOUND on the true convex roof

Run:
    cd ~/venvs/qchem/hubbard_gme
    pip install cvxpy
    python cvxpy_convex_roof.py
"""

import sys, os
import numpy as np
from scipy.optimize import minimize as sp_minimize

# Add project root to path — works regardless of where script is run from
_here = os.path.dirname(os.path.abspath(__file__))
# Walk up until we find the src directory
_root = _here
for _ in range(4):
    if os.path.isdir(os.path.join(_root, "src")):
        sys.path.insert(0, os.path.join(_root, "src"))
        break
    _root = os.path.dirname(_root)

from hubbard_gme.pipeline    import compute_ground_and_excited_states
from hubbard_gme.rdm         import compute_all_pure_rdms
from hubbard_gme.gme         import (
    compute_C_GME_pure, get_bipartitions,
    partial_trace_spins, purity,
    supervisor_gme_lower_bound,
)
from hubbard_gme.convex_roof import (
    full_convex_roof_analysis,
    convex_roof_lower_bound,
)

# ── Parameters ─────────────────────────────────────────────────────────
L        = 6; Ne = 6; t = 1.0; U = 4.0; mode = "uniform"
N_STARTS = 200
OUTPUT_PDF = f"outputs/cvxpy_convex_roof_L{L}_Ne{Ne}_U{U}.pdf"
os.makedirs("outputs", exist_ok=True)

# ── Check CVXPY ────────────────────────────────────────────────────────
try:
    import cvxpy as cp
    print(f"  CVXPY {cp.__version__}  OK")
except ImportError:
    print("  CVXPY not installed. Run:  pip install cvxpy")
    sys.exit(1)

# ── Step 1: ground state and RDMs ─────────────────────────────────────
print("=" * 60)
print(f"  L={L}  Ne={Ne}  t={t}  U={U}  mode={mode}")
print("=" * 60)
print("\nDiagonalising ...")
E0, psi0, E1, psi1, n_so = compute_ground_and_excited_states(
    L=L, t=t, U=U, mode=mode, n_electrons=Ne
)
print(f"  E0={E0:.8f}  E1={E1:.8f}  gap={E1-E0:.6f}")

print("\nComputing RDMs (n=3 and n=4) ...")
rdm_all, _, _ = compute_all_pure_rdms(
    L=L, psi0=psi0, psi1=psi1,
    n_spin_orbitals=2*L, n_electrons=Ne,
    min_n=6, max_n=6,
)

# ── Helper functions ───────────────────────────────────────────────────
def C_GME_pure_dm(rho_pure, n):
    """C_GME for a pure state given as density matrix."""
    purities = [purity(partial_trace_spins(rho_pure, n, A))
                for A in get_bipartitions(n)]
    return float(np.sqrt(max(2.0*(1.0 - max(purities)), 0.0)))

def compute_lower_bounds(rho, n):
    """Paper and supervisor lower bounds."""
    lb1 = convex_roof_lower_bound(rho, n)
    lb2 = supervisor_gme_lower_bound(rho, n)
    return {
        "paper":      lb1,
        "supervisor": lb2,
        "best":       max(lb1["C_LB"], lb2["C_LB"]),
    }

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
        np.random.seed(start)
        A0 = np.random.randn(rank, rank) * 0.3
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
        decomp.append({"p": p_i, "C_GME": C_i})

    return {"C_GME": best_C, "rank": rank, "decomp": decomp}


def cvxpy_sdp_convex_roof(rho, n):
    """
    SDP relaxation of the convex roof using CVXPY.

    The paper formulation:
        Minimize  Tr(C X)
        Subject to:  Tr(A_i X) = b_i  (i=1,...,p)
                     X >= 0

    For convex roof:
        Variables: Y_i (dim x dim PSD matrices, one per pure state)
                   t_A (scalar, upper bound on bipartition purity)

        Constraints:
            sum_i Y_i = rho          (decomposition of rho)
            Y_i >= 0                 (positive semidefinite)
            Tr(Y_i) = p_i >= 0       (valid probabilities)

        Objective:
            minimize sum_i sqrt(2*(1 - t_i))
            where t_i = max_A Tr(rho_A_i^2)

    Since Tr(rho_A^2) is quadratic in Y_i, we use the
    Schur complement to make it an SDP constraint:

        Tr(rho_A_i^2) >= s  iff
        [s*I    rho_A_i] >= 0   (Schur complement form)
        [rho_A_i   I   ]

    This gives a valid SDP lower bound on the convex roof.

    Parameters: A_1 = I, b_1 = 1 (trace preservation as in paper)
    """
    dim   = 2**n
    rank  = int(np.sum(np.linalg.eigvalsh(rho) > 1e-10))
    biparts = get_bipartitions(n)
    n_biparts = len(biparts)

    print(f"  dim={dim}  rank={rank}  bipartitions={n_biparts}")
    print(f"  Building SDP with CVXPY ...")

    # ── Variables ──────────────────────────────────────────────────────
    # Y_i: one PSD matrix per pure state in decomposition (rank matrices)
    Y = [cp.Variable((dim, dim), hermitian=True) for _ in range(rank)]

    # t_i_A: upper bound on Tr(rho_A_i^2) for each state i and bipartition A
    # We use s_i = max_A Tr(rho_A_i^2) per state
    # For simplicity: one variable t_i per state (the maximum over bipartitions)
    t = [cp.Variable(nonneg=True) for _ in range(rank)]

    # ── Constraints ────────────────────────────────────────────────────
    constraints = []

    # 1. sum_i Y_i = rho  (paper: Tr(A_1 X) = b_1 with A_1=I, b_1=1)
    constraints.append(cp.sum(Y) == rho)

    # 2. Y_i >= 0  (positive semidefinite)
    for i in range(rank):
        constraints.append(Y[i] >> 0)

    # 3. Tr(Y_i) >= 0  (probabilities — automatically satisfied by PSD)
    #    Tr(Y_i) = p_i  already implied by sum constraint

    # 4. For each state i and bipartition A:
    #    Tr(rho_A_i^2) <= t_i  (t_i bounds the max bipartition purity)
    #    Using Schur complement:
    #    Tr(rho_A^2) <= t  iff  [t*Tr(Y_i)  rho_A_i] >= 0
    #                            [rho_A_i†   I       ]
    #    where rho_A_i = Tr_B(Y_i) / Tr(Y_i)

    for i in range(rank):
        p_i = cp.real(cp.trace(Y[i]))
        for A in biparts:
            B    = tuple(j for j in range(n) if j not in A)
            dimA = 2**len(A)
            dimB = 2**len(B)

            # Partial trace of Y[i] over B
            rho_A_i = cp.Variable((dimA, dimA), hermitian=True)
            for a1 in range(dimA):
                for a2 in range(dimA):
                    constraints.append(
                        rho_A_i[a1, a2] ==
                        cp.sum([Y[i][a1*dimB+b, a2*dimB+b]
                                for b in range(dimB)])
                    )

            # Schur complement: Tr(rho_A_i^2) <= t_i * p_i^2
            # [t_i    rho_A_i/p_i] >= 0  — but p_i is variable
            # Simpler valid form:
            # Introduce s_i_A = Tr(rho_A_i @ rho_A_i)  (nuclear norm squared)
            # Use:  [t_i * I    rho_A_i] >= 0
            #        [rho_A_i†   I      ]
            # This means t_i >= max singular value of rho_A_i
            # which bounds the bipartition purity from above
            block = cp.bmat([
                [t[i] * np.eye(dimA),  rho_A_i],
                [rho_A_i.H,            np.eye(dimA)],
            ])
            constraints.append(block >> 0)

    # ── Objective ──────────────────────────────────────────────────────
    # minimize sum_i (1 - t_i)  — linear in t_i, valid DCP
    # This bounds C_GME from below since C_GME >= sqrt(2*(1-max_A purity_A))
    objective = cp.Minimize(cp.sum([1 - t[i] for i in range(rank)]))

    # ── Solve ──────────────────────────────────────────────────────────
    prob = cp.Problem(objective, constraints)
    print(f"  Solving SDP (solver=SCS) ...")
    try:
        prob.solve(solver=cp.SCS, verbose=False,
                   eps=1e-6, max_iters=10000)
        status = prob.status
        print(f"  SDP status: {status}")

        if prob.value is not None and status in ["optimal", "optimal_inaccurate"]:
            obj_val     = float(prob.value)
            # objective = sum_i (1 - t_i)
            # C_GME >= sqrt(2 * min_i t_i_effective)
            # Convert: obj = sum(1-t_i), so avg(t_i) = 1 - obj/rank
            avg_t       = 1.0 - obj_val / rank
            C_sdp_lower = float(np.sqrt(max(2.0 * (1.0 - avg_t), 0.0)))

            decomp_sdp = []
            for i in range(rank):
                p_i_val = float(np.real(np.trace(Y[i].value))) if Y[i].value is not None else 0.0
                t_i_val = float(t[i].value) if t[i].value is not None else 0.0
                C_i_val = float(np.sqrt(max(2.0*(1.0 - t_i_val), 0.0)))
                decomp_sdp.append({"p": p_i_val, "t": t_i_val, "C_GME_bound": C_i_val})

            return {
                "C_sdp":    C_sdp_lower,
                "obj_val":  obj_val,
                "status":   status,
                "decomp":   decomp_sdp,
                "rank":     rank,
            }
        else:
            print(f"  SDP did not converge: {status}")
            return {"C_sdp": None, "status": status, "rank": rank}

    except Exception as e:
        print(f"  SDP error: {e}")
        return {"C_sdp": None, "status": "error", "rank": rank}


# ══════════════════════════════════════════════════════════════════════
# CASE 1: n=3  MIXED STATE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  CASE 1: n=3  (mixed state, n < N)")
print("=" * 60)

sites3 = (0, 1, 2)
rho3   = rdm_all[sites3]
n3     = 3
pur3   = purity(rho3)
rank3  = int(np.sum(np.linalg.eigvalsh(rho3) > 1e-10))
print(f"\n  Tr(rho^2) = {pur3:.8f}   rank = {rank3}")

print("\n  [A] Lower bounds ...")
lbs3 = compute_lower_bounds(rho3, n3)
print(f"    C_LB paper (Ma et al.)      = {lbs3['paper']['C_LB']:.8f}")
print(f"    C_LB supervisor eqs.(6-9)   = {lbs3['supervisor']['C_LB']:.8f}")
print(f"    C_LB best                   = {lbs3['best']:.8f}")

print("\n  [B] Existing convex roof ...")
cr3    = full_convex_roof_analysis(rho3, sites3, n_restarts=20, verbose=False)
C_num3 = cr3["C_numerical"]
C_UB3  = cr3["C_UB"]
print(f"    C_UB  (eigen upper)         = {C_UB3:.8f}")
print(f"    C_num (existing numerical)  = {C_num3:.8f}")

print(f"\n  [C] Unitary convex roof ({N_STARTS} restarts) ...")
ur3 = unitary_convex_roof(rho3, n3)
print(f"    C_GME (unitary optim.)      = {ur3['C_GME']:.8f}")

print(f"\n  [D] CVXPY SDP relaxation ...")
sdp3 = cvxpy_sdp_convex_roof(rho3, n3)
if sdp3["C_sdp"] is not None:
    print(f"    C_SDP (lower bound)         = {sdp3['C_sdp']:.8f}")
else:
    print(f"    C_SDP: solver failed ({sdp3['status']})")

print(f"\n  SUMMARY n=3:")
print(f"    C_LB  (best lower bound)    = {lbs3['best']:.8f}")
print(f"    C_SDP (CVXPY relaxation)    = "
      f"{sdp3['C_sdp']:.8f}" if sdp3['C_sdp'] else "    C_SDP = N/A")
print(f"    C_num (existing numerical)  = {C_num3:.8f}")
print(f"    C_GME (unitary optim.)      = {ur3['C_GME']:.8f}")
print(f"    C_UB  (upper bound)         = {C_UB3:.8f}")

# ══════════════════════════════════════════════════════════════════════
# CASE 2: n=4  PURE STATE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  CASE 2: n=4  (pure state, n = N)")
print("=" * 60)

sites4  = (0, 1, 2, 3)
rho4    = rdm_all[sites4]
n4      = 4
pur4    = purity(rho4)
rank4   = int(np.sum(np.linalg.eigvalsh(rho4) > 1e-10))
C_exact = compute_C_GME_pure(rho4, n4)
print(f"\n  Tr(rho^2) = {pur4:.8f}   rank = {rank4}")
print(f"  C_GME exact (eq.1) = {C_exact:.8f}")

print("\n  [A] Lower bounds ...")
lbs4 = compute_lower_bounds(rho4, n4)
print(f"    C_LB paper (Ma et al.)      = {lbs4['paper']['C_LB']:.8f}")
print(f"    C_LB supervisor eqs.(6-9)   = {lbs4['supervisor']['C_LB']:.8f}")
print(f"    C_LB best                   = {lbs4['best']:.8f}")

print("\n  [B] Existing convex roof ...")
cr4    = full_convex_roof_analysis(rho4, sites4, n_restarts=20, verbose=False)
C_num4 = cr4["C_numerical"]
C_UB4  = cr4["C_UB"]
print(f"    C_UB  (eigen upper)         = {C_UB4:.8f}")
print(f"    C_num (existing numerical)  = {C_num4:.8f}")

print(f"\n  [C] Unitary convex roof ({N_STARTS} restarts) ...")
ur4 = unitary_convex_roof(rho4, n4)
print(f"    C_GME (unitary optim.)      = {ur4['C_GME']:.8f}")

print(f"\n  [D] CVXPY SDP relaxation ...")
sdp4 = cvxpy_sdp_convex_roof(rho4, n4)
if sdp4["C_sdp"] is not None:
    print(f"    C_SDP (lower bound)         = {sdp4['C_sdp']:.8f}")
else:
    print(f"    C_SDP: solver failed ({sdp4['status']})")

print(f"\n  SUMMARY n=4:")
print(f"    C_LB  (best lower bound)    = {lbs4['best']:.8f}")
print(f"    C_SDP (CVXPY relaxation)    = "
      f"{sdp4['C_sdp']:.8f}" if sdp4['C_sdp'] else "    C_SDP = N/A")
print(f"    C_num (existing numerical)  = {C_num4:.8f}")
print(f"    C_GME (unitary optim.)      = {ur4['C_GME']:.8f}")
print(f"    C_UB  (upper bound)         = {C_UB4:.8f}")
print(f"    C_exact (eq.1)              = {C_exact:.8f}")

# ══════════════════════════════════════════════════════════════════════
# PDF
# ══════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

print(f"\nGenerating PDF -> {OUTPUT_PDF}")

def make_page(fig, ax, title, lines):
    ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    ax.text(0.04, 0.97, "\n".join(lines), ha="left", va="top",
            fontsize=10, family="monospace", transform=ax.transAxes)

with PdfPages(OUTPUT_PDF) as pdf:

    # ── PAGE 1: n=3 ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10))
    c3 = sdp3["C_sdp"]
    lines3 = [
        f"System:  L={L}  Ne={Ne}  t={t}  U={U}  mode={mode}",
        f"RDM:  n=3  sites={sites3}  dim=8x8",
        f"Tr(rho) = {float(np.real(np.trace(rho3))):.8f}   "
        f"Tr(rho^2) = {pur3:.8f}   rank = {rank3}",
        f"",
        f"{'='*55}",
        f"  LOWER BOUNDS",
        f"{'='*55}",
        f"",
        # f"  C_LB  paper (Ma et al.)          = {lbs3['paper']['C_LB']:.8f}",
        # f"    best x = {lbs3['paper']['best_x']}",
        # f"    best y = {lbs3['paper']['best_y']}",
        f"",
        f"  C_LB  notes eqs.(6-9)       = {lbs3['supervisor']['C_LB']:.8f}",
        f"    best z1 = {lbs3['supervisor']['best_z1']}",
        f"    best z2 = {lbs3['supervisor']['best_z2']}",
        f"",
        f"  C_LB  best of all                = {lbs3['best']:.8f}",
        f"",
        f"{'='*55}",
        f"  CONVEX ROOF",
        f"{'='*55}",
        f"",
        # f"  C_UB  (eigendecomp upper bound)  = {C_UB3:.8f}",
        f"  C_num (existing numerical)       = {C_num3:.8f}",
        f"  C_GME (unitary optimization)     = {ur3['C_GME']:.8f}",
        f"    restarts = {N_STARTS}   optimizer = L-BFGS-B",
        f"",
        # f"  C_SDP (CVXPY SDP relaxation)     = "
        # f"{c3:.8f}  status={sdp3['status']}" if c3 else
        # f"  C_SDP (CVXPY SDP relaxation)     = N/A  status={sdp3['status']}",
        # f"    Variables: Y_i >= 0,  sum_i Y_i = rho",
        # f"    Objective: min sum_i p_i*(1-t_i)",
        # f"    Constraint: Tr(rho_A_i^2/p_i) <= t_i  (Schur complement)",
        # f"",
        f"  Decomposition from unitary optimization:",
    ] + [
        f"    i={i}: p={d['p']:.6f}  C_GME={d['C_GME']:.6f}  "
        f"contrib={d['p']*d['C_GME']:.6f}"
        for i, d in enumerate(ur3["decomp"])
    ] + [
        f"",
        f"{'='*55}",
        f"  C_LB={lbs3['best']:.6f} <= C_GME={ur3['C_GME']:.6f} <= C_UB={C_UB4:.6f}",
    ]
    make_page(fig, ax,
              f"GME Analysis — Hubbard L={L} Ne={Ne} t={t} U={U} — n=3 (mixed)",
              lines3)
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)

    # ── PAGE 2: n=4 ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10))
    c4 = sdp4["C_sdp"]
    lines4 = [
        f"System:  L={L}  Ne={Ne}  t={t}  U={U}  mode={mode}",
        f"RDM:  n=4  sites={sites4}  dim=16x16",
        f"Tr(rho) = {float(np.real(np.trace(rho4))):.8f}   "
        f"Tr(rho^2) = {pur4:.8f}   rank = {rank4}",
        f"",
        f"  C_GME exact (eq.1) = {C_exact:.8f}",
        f"",
        f"{'='*55}",
        f"  LOWER BOUNDS",
        f"{'='*55}",
        f"",
        # f"  C_LB  paper (Ma et al.)          = {lbs4['paper']['C_LB']:.8f}",
        # f"    best x = {lbs4['paper']['best_x']}",
        # f"    best y = {lbs4['paper']['best_y']}",
        f"",
        f"  C_LB  notes eqs.(6-9)       = {lbs4['supervisor']['C_LB']:.8f}",
        f"    best z1 = {lbs4['supervisor']['best_z1']}",
        f"    best z2 = {lbs4['supervisor']['best_z2']}",
        f"",
        f"  C_LB  best of all                = {lbs4['best']:.8f}",
        f"  Gap  C_exact - C_LB_best         = {C_exact - lbs4['best']:.8f}",
        f"",
        f"{'='*55}",
        f"  CONVEX ROOF  (pure state)",
        f"{'='*55}",
        f"",
        # f"  C_UB  (eigendecomp upper bound)  = {C_UB4:.8f}",
        f"  C_num (existing numerical)       = {C_num4:.8f}",
        f"  C_GME (unitary optimization)     = {ur4['C_GME']:.8f}",
        f"  C_exact (eq.1)                   = {C_exact:.8f}",
        f"",
        # f"  C_SDP (CVXPY SDP relaxation)     = "
        # f"{c4:.8f}  status={sdp4['status']}" if c4 else
        # f"  C_SDP (CVXPY SDP relaxation)     = N/A  status={sdp4['status']}",
        # f"",
        f"{'='*55}",
        f"  C_LB={lbs4['best']:.6f} <= C_GME={ur4['C_GME']:.6f} <= C_UB={C_UB4:.6f}",
        f"  C_exact = {C_exact:.6f}",
    ]
    make_page(fig, ax,
              f"GME Analysis — Hubbard L={L} Ne={Ne} t={t} U={U} — n=4 (pure)",
              lines4)
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)

print(f"  Done! PDF -> {OUTPUT_PDF}")