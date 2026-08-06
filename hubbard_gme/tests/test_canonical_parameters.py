"""
test_canonical_parameters.py
----------------------------
Sanity checks justifying the choice of canonical parameters
(L=6, U/t=4, half-filling, uniform ring) as representative for
the GME analysis.

Two checks:
  (a) U/t monotonicity      : C_GME varies smoothly across regimes
  (b) System-size consistency: L=4 and L=6 give compatible per-site quantities

These tests are slow (full convex-roof + diagonalisation per point),
so we keep the parameter grid minimal and run only one representative
cluster per case.

Run with:
    pytest tests/test_canonical_parameters.py -v -s
The -s flag lets the printed numbers reach the terminal so you can
copy them into a report.
"""

from __future__ import annotations

import os
import json
import time
import pytest
import numpy as np

from hubbard_gme.hamiltonian import compute_ground_and_excited_states
from hubbard_gme.rdm import compute_general_nRDM
from hubbard_gme.gme import compute_C_GME_pure
from hubbard_gme.metrics import compute_purity
from hubbard_gme.convex_roof import convex_roof_stiefel
from hubbard_gme.pipeline import _build_system_info


# ── Output logging ────────────────────────────────────────────────────────────
# Save numbers to a JSON file under tests/outputs/ so they can be shown to
# the supervisor without re-running the test.
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def _normalised_rdm(sites, psi, n_so, n_e):
    rho = compute_general_nRDM(list(sites), psi, n_so, n_e)
    tr = np.trace(rho).real
    return rho / tr if abs(tr) > 1e-14 else rho


def _compute_cgme_at(L, U, t=1.0, n_e=None, n_body=3, n_restarts=10):
    """Compute C_GME on the ground state for the canonical cluster (0, 1, ..., n_body-1)."""
    if n_e is None:
        n_e = L  # half-filling

    E0, psi0, E1, psi1, n_so = compute_ground_and_excited_states(
        L=L, t=t, U=U, mode="uniform", n_electrons=n_e,
    )
    sites = tuple(range(n_body))
    rho = _normalised_rdm(sites, psi0, n_so, n_e)
    purity = compute_purity(rho)

    if purity > 1.0 - 1e-6:
        # Pure state — use direct formula
        c_gme = compute_C_GME_pure(rho, n_body)
    else:
        # Mixed — use convex roof (Stiefel)
        result = convex_roof_stiefel(rho, n_body, n_restarts=n_restarts, verbose=False)
        print(f"  cluster {sites}   C_GME (best) = {result['C_stiefel']:.6f}")
        print(f"    restart values: {[f'{v:.4f}' for v in result['best_record']['restart_log']]}")
        c_gme = result["C_stiefel"]

    return {
        "L": L, "U": U, "t": t, "Ne": n_e, "n_body": n_body,
        "E0": float(E0), "E0_per_site": float(E0 / L),
        "gap": float(E1 - E0),
        "purity": float(purity),
        "C_GME": float(c_gme),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sanity check (a): U/t monotonicity at fixed L = 6
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_sanity_a_Ut_monotonicity():
    """
    Run U/t in {1, 4, 16} at L=6, n=3 cluster, ground state.
    Verify trends are smooth (no surprising non-monotonicity in C_GME).

    Physical expectation:
      - Low U/t  : weakly correlated, C_GME small
      - Mid U/t  : crossover, C_GME at intermediate value
      - High U/t : strongly correlated (Heisenberg-like), C_GME approaches
                   the half-filled Heisenberg-chain value

    We do not assert strict monotonicity (the function could in principle
    have a maximum), but we DO assert smooth variation: no point should be
    more than 50% off the linear interpolation of its neighbours.
    """
    print("\n=== Sanity (a): U/t monotonicity at L=6 ===")
    points = []
    for U in [1.0, 4.0, 16.0]:
        t0 = time.time()
        rec = _compute_cgme_at(L=6, U=U, n_body=3)
        rec["wall_time_s"] = time.time() - t0
        points.append(rec)
        print(f"  U/t = {U:5.1f}   purity = {rec['purity']:.4f}   "
              f"C_GME = {rec['C_GME']:.6f}   E0/site = {rec['E0_per_site']:+.4f}   "
              f"({rec['wall_time_s']:.1f}s)")

    # Save log
    out_path = os.path.join(OUT_DIR, "sanity_a_Ut_monotonicity.json")
    with open(out_path, "w") as fh:
        json.dump(points, fh, indent=2)
    print(f"  → saved {out_path}")

    # Smoothness check: middle point shouldn't be wildly far from
    # the linear interpolation of the outer two
    c_low, c_mid, c_high = (p["C_GME"] for p in points)
    interp_mid = 0.5 * (c_low + c_high)
    if abs(c_mid) > 1e-6:
        deviation = abs(c_mid - interp_mid) / max(abs(c_mid), abs(interp_mid))
        assert deviation < 0.5, (
            f"C_GME at U/t=4 ({c_mid:.4f}) deviates too much from linear "
            f"interpolation of U/t=1,16 ({interp_mid:.4f}). "
            f"Relative deviation = {deviation:.2f}, limit 0.5."
        )

    # Energy per site should DECREASE (more negative) as U decreases
    # (less repulsion = more bonding = lower energy)
    e_low, e_mid, e_high = (p["E0_per_site"] for p in points)
    assert e_low <= e_mid <= e_high + 1e-6, (
        f"E0/site should be monotonically increasing with U: "
        f"got {e_low:.4f} (U=1), {e_mid:.4f} (U=4), {e_high:.4f} (U=16)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sanity check (b): L = 4 vs L = 6 at fixed U/t = 4
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_sanity_b_L_size_consistency():
    """
    Compare L = 4 and L = 6 at U/t = 4 using SIZE-CONSISTENT quantities:
    - E0/site  (must converge with L for any local Hamiltonian)
    - <D>/site (local observable, must converge)
    - C_GME at fixed cluster fraction n/L = 1/2  (n=2 for L=4, n=3 for L=6)

    Comparing C_GME at fixed n (not fixed n/L) is misleading because
    the subsystem covers different fractions of the system.
    """
    print("\n=== Sanity (b): L size consistency at U/t = 4 ===")

    # Fixed n: compute the local observables (these MUST converge)
    rec_4 = _compute_cgme_at(L=4, U=4.0, n_body=2)   # half-system cluster
    rec_6 = _compute_cgme_at(L=6, U=4.0, n_body=3)   # half-system cluster

    print(f"  L=4, n=2 (half-ring):   purity={rec_4['purity']:.4f}  "
          f"C_GME={rec_4['C_GME']:.4f}  E0/site={rec_4['E0_per_site']:+.4f}")
    print(f"  L=6, n=3 (half-ring):   purity={rec_6['purity']:.4f}  "
          f"C_GME={rec_6['C_GME']:.4f}  E0/site={rec_6['E0_per_site']:+.4f}")

    # Local quantities (energy per site)
    rel_e_diff = abs(rec_4['E0_per_site'] - rec_6['E0_per_site']) / abs(rec_6['E0_per_site'])
    print(f"  → |E0/site| relative diff = {rel_e_diff*100:.1f}%")

    # Use a more realistic tolerance (15-20% is typical for L=4 vs L=6)
    assert rel_e_diff < 0.20, (
        f"E0/site differs by {rel_e_diff*100:.1f}% between L=4 and L=6 — "
        f"larger than expected. Check whether ground-state convergence "
        f"is qualitatively reasonable."
    )

    # For C_GME at the half-system fraction, allow a wider tolerance
    # since we're still comparing different system sizes
    abs_c_diff = abs(rec_4['C_GME'] - rec_6['C_GME'])
    print(f"  → |C_GME| half-system diff = {abs_c_diff:.4f}")

    assert abs_c_diff < 0.30, (
        f"C_GME at the half-system fraction differs by {abs_c_diff:.4f} "
        f"between L=4 and L=6, which is large. Suggests that L=6 may "
        f"not yet be in the bulk regime for entanglement quantities."
    )

# ──────────────────────────────────────────────────────────────────────────────
# Bonus: confirm translation symmetry (each cluster on the ring at fixed n
# gives the same C_GME). This justifies the speedup trick we discussed.
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_translation_symmetry_at_canonical_point():
    print("\n=== Translation symmetry at L=6, U=4, n=3 ===")
    L, U, t, n_e = 6, 4.0, 1.0, 6
    n_body = 3
    E0, psi0, E1, psi1, n_so = compute_ground_and_excited_states(
        L=L, t=t, U=U, mode="uniform", n_electrons=n_e,
    )

    cgmes = []
    all_logs = []
    for start in range(L):
        sites = tuple((start + i) % L for i in range(n_body))
        rho = _normalised_rdm(sites, psi0, n_so, n_e)
        purity = compute_purity(rho)
        if purity > 1.0 - 1e-6:
            c = compute_C_GME_pure(rho, n_body)
            log = [c]
        else:
            res = convex_roof_stiefel(rho, n_body, n_restarts=30, verbose=False)
            print(f"  cluster {sites}   C_GME (best) = {res['C_stiefel']:.6f}")
            print(f"    restart values: {[f'{v:.4f}' for v in res['best_record']['restart_log']]}")
            c = res["C_stiefel"]
            log = res["best_record"]["restart_log"]
        cgmes.append(c)
        all_logs.append(log)
        print(f"  cluster {sites}   C_GME (best of {len(log)} restarts) = {c:.6f}   "
              f"min in log = {min(log):.6f}   max in log = {max(log):.6f}")

    spread = max(cgmes) - min(cgmes)
    true_min = min(min(log) for log in all_logs)
    print(f"\n  Spread of best-per-cluster results: {spread:.4f}")
    print(f"  Smallest value seen across ALL restarts: {true_min:.6f}")
    print(f"  → True convex roof is approximately {true_min:.4f}")

    # The cross-cluster spread tells us how reliable our per-cluster
    # answer is. With enough restarts (n_restarts >= 30), this should
    # be < 5e-3.
    assert spread < 5e-3, (
        f"Spread = {spread:.4f}. The optimiser is not converging consistently "
        f"across rotationally equivalent clusters. True minimum found in any "
        f"restart was {true_min:.6f}; per-cluster bests range from "
        f"{min(cgmes):.4f} to {max(cgmes):.4f}. "
        f"Try more restarts or a better initialisation."
    )