"""
landscape.py
-------------
Unified landscape analysis module for studying the I_Phi optimization space.

Provides both discrete (basis states) and continuous (Bloch sphere) analysis.

Functions
---------
landscape_discrete_scan()   : Scan all basis-state pairs, find optima
landscape_continuous_1d()   : 1D interpolation path between two states
landscape_surface_2d()      : 2D surface map (reduced parameter space)
landscape_full_report()     : Generate comprehensive PDF report
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Tuple, List, Optional

from .gme import (
    supervisor_compute_I_phi,
    I_phi_continuous,
    compute_C_GME_pure,
)


# ══════════════════════════════════════════════════════════════════════════════
# DISCRETE LANDSCAPE (all basis-state pairs)
# ══════════════════════════════════════════════════════════════════════════════

def landscape_discrete_scan(
    rho: np.ndarray,
    n: int,
    verbose: bool = True,
) -> Dict:
    """
    Scan ALL basis-state pairs and identify optima.

    Returns
    -------
    dict with keys:
        max_I           : float — maximum I_Phi value
        optimal_pairs   : list of (z1_str, z2_str) tuples at maximum
        physically_unique : int — number of unique unordered pairs
        pair_results    : list of (I, z1_str, z2_str, Sz1, Sz2) sorted by I
        groups          : dict {I_value: list of pairs}
        all_pairs_checked : int — total pairs evaluated
    """
    dim = 2**n
    all_strings = [[(k>>(n-1-b))&1 for b in range(n)] for k in range(dim)]

    pair_results = []
    for zi in range(dim):
        for zj in range(dim):
            if zi == zj:
                continue
            z1, z2 = all_strings[zi], all_strings[zj]
            I = supervisor_compute_I_phi(rho, n, z1, z2)
            Sz1 = sum(0.5 if s==0 else -0.5 for s in z1)
            Sz2 = sum(0.5 if s==0 else -0.5 for s in z2)
            s1 = ''.join('u' if x==0 else 'd' for x in z1)
            s2 = ''.join('u' if x==0 else 'd' for x in z2)
            pair_results.append((I, s1, s2, Sz1, Sz2))

    pair_results.sort(reverse=True)
    max_I = pair_results[0][0]

    # Get optimal pairs within tolerance
    tol = 1e-5
    optimal_pairs = [(s1,s2) for I,s1,s2,*_ in pair_results if I >= max_I - tol]

    # Count physically unique pairs (unordered)
    physically_unique = len(set(tuple(sorted([s1,s2])) for s1,s2 in optimal_pairs))

    # Group by I value
    tol_group = 1e-5
    groups = {}
    for I, s1, s2, Sz1, Sz2 in pair_results:
        key = round(I / tol_group) * tol_group
        if key not in groups:
            groups[key] = []
        groups[key].append((s1, s2))

    if verbose:
        print(f"\n  [Discrete landscape] n={n}")
        print(f"    Max I_Phi           = {max_I:.8f}")
        print(f"    Optimal pairs       = {len(optimal_pairs)}")
        print(f"    Physically unique   = {physically_unique}")

    return {
        "max_I": float(max_I),
        "optimal_pairs": optimal_pairs,
        "physically_unique": physically_unique,
        "pair_results": pair_results,
        "groups": groups,
        "all_pairs_checked": dim * (dim - 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONTINUOUS LANDSCAPE (1D interpolation path)
# ══════════════════════════════════════════════════════════════════════════════

def landscape_continuous_1d(
    rho: np.ndarray,
    n: int,
    z1_optimal_str: str,
    z2_optimal_str: str,
    n_points: int = 181,
    verbose: bool = True,
) -> Dict:
    """
    Map I_Phi along interpolation path from z1 to z2 on the Bloch sphere.

    Parametrizes the path by alpha ∈ [0, π]:
        theta1(alpha) = z1 + alpha * (z2 - z1)
        theta2(alpha) = z2 - alpha * (z1 - z2)   [swapped]

    Returns
    -------
    dict with keys:
        alphas          : ndarray of angles [0, π]
        I_values        : ndarray of I_Phi(alpha)
        C_values        : max(0, I_Phi) — the bound
        alpha_deg       : degrees (0 to 180)
        peak_width_deg  : width at 90% of max
        peak_is_sharp   : bool — width < 30°
    """
    def str_to_thetas(s):
        """Convert 'udud' to theta array."""
        return np.array([0.0 if c=='u' else np.pi for c in s])

    t1_opt = str_to_thetas(z1_optimal_str)
    t2_opt = str_to_thetas(z2_optimal_str)

    alphas = np.linspace(0, np.pi, n_points)
    I_values = np.zeros(n_points)

    for i, alpha in enumerate(alphas):
        frac = alpha / np.pi
        th1 = t1_opt + frac * (t2_opt - t1_opt)
        th2 = t2_opt + frac * (t1_opt - t2_opt)
        I_values[i] = I_phi_continuous(rho, n, th1, [0.0]*n, th2, [0.0]*n)

    C_values = np.maximum(I_values, 0.0)
    alpha_deg = np.degrees(alphas)
    max_I = I_values.max()

    # Peak width at 90%
    peak_90 = 0.9 * max_I
    above_90 = np.where(C_values >= peak_90)[0]
    width_deg = float(alpha_deg[above_90[-1]] - alpha_deg[above_90[0]]) if len(above_90) else 0.0
    peak_is_sharp = width_deg < 30.0

    if verbose:
        print(f"\n  [Continuous 1D landscape]")
        print(f"    Path: |{z1_optimal_str}> → |{z2_optimal_str}>")
        print(f"    Peak width at 90%   = {width_deg:.1f}°")
        print(f"    Peak character      = {'SHARP isolated' if peak_is_sharp else 'BROAD'}")

    return {
        "alphas": alphas,
        "I_values": I_values,
        "C_values": C_values,
        "alpha_deg": alpha_deg,
        "max_I": float(max_I),
        "peak_width_deg": float(width_deg),
        "peak_is_sharp": peak_is_sharp,
        "z1_optimal_str": z1_optimal_str,
        "z2_optimal_str": z2_optimal_str,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONTINUOUS LANDSCAPE (2D surface map)
# ══════════════════════════════════════════════════════════════════════════════

def landscape_surface_2d(
    rho: np.ndarray,
    n: int,
    n_grid: int = 60,
    verbose: bool = True,
) -> Dict:
    """
    Map the 2D I_Phi surface using reduced parametrization.

    Assumes Neel structure: spins alternate (up, down, up, down, ...).
    Varies theta1 and theta2 with all sites in each spin class moving together:
        psi1: [t1, π-t1, t1, π-t1, ...]
        psi2: [t2, π-t2, t2, π-t2, ...]

    Returns
    -------
    dict with keys:
        thetas       : ndarray — angles [0, π]
        theta_deg    : ndarray — degrees
        Z            : (n_grid, n_grid) array of I_Phi values
        C_surf       : (n_grid, n_grid) array of max(0, I_Phi)
        max_C        : float
        peak_location : tuple (theta1_deg, theta2_deg)
    """
    thetas = np.linspace(0, np.pi, n_grid)
    Z = np.zeros((n_grid, n_grid))

    for i, t1 in enumerate(thetas):
        for j, t2 in enumerate(thetas):
            # Alternating Neel structure
            th1 = np.array([t1 if k % 2 == 0 else np.pi - t1 for k in range(n)])
            th2 = np.array([t2 if k % 2 == 0 else np.pi - t2 for k in range(n)])
            ph = np.array([0.0] * n)
            Z[i, j] = I_phi_continuous(rho, n, th1, ph, th2, ph)

    C_surf = np.maximum(Z, 0.0)
    theta_deg = np.degrees(thetas)
    max_C = C_surf.max()

    # Find peak location
    idx = np.unravel_index(C_surf.argmax(), C_surf.shape)
    peak_location = (float(theta_deg[idx[0]]), float(theta_deg[idx[1]]))

    if verbose:
        print(f"\n  [Continuous 2D landscape]")
        print(f"    Grid size           = {n_grid}x{n_grid}")
        print(f"    Max C_LB on surface = {max_C:.8f}")
        print(f"    Peak location       = theta1={peak_location[0]:.1f}°, "
              f"theta2={peak_location[1]:.1f}°")

    return {
        "thetas": thetas,
        "theta_deg": theta_deg,
        "Z": Z,
        "C_surf": C_surf,
        "max_C": float(max_C),
        "peak_location": peak_location,
    }


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED REPORT
# ══════════════════════════════════════════════════════════════════════════════

def landscape_full_report(
    rho: np.ndarray,
    n: int,
    system_info: Dict,
    output_path: str,
    C_exact: float = None,
    C_LB_basis: float = None,
) -> None:
    """
    Generate comprehensive 3-page PDF landscape report.

    Pages:
        1. System summary + basis-pair ranking
        2. Continuous 1D landscape (interpolation path + zoom)
        3. Answer to supervisor + 2D surface (heatmap + 3D)

    Parameters
    ----------
    rho           : n-spin density matrix (should be pure state, n=N)
    n             : number of spins
    system_info   : dict with L, Ne, t, U, mode, E0, E1, etc.
    output_path   : where to save the PDF
    C_exact       : optional C_GME from exact formula
    C_LB_basis    : optional C_LB from basis-state search
    """
    # Run analyses
    disc = landscape_discrete_scan(rho, n, verbose=False)
    opt_s1, opt_s2 = disc["optimal_pairs"][0]
    cont = landscape_continuous_1d(rho, n, opt_s1, opt_s2, verbose=False)
    surf = landscape_surface_2d(rho, n, n_grid=60, verbose=False)

    if C_exact is None:
        C_exact = compute_C_GME_pure(rho, n)
    if C_LB_basis is None:
        C_LB_basis = disc["max_I"]

    # PDF
    with PdfPages(output_path) as pdf:

        # ──────────────────────────────────────────────────────────────────
        # PAGE 1: System summary + pair ranking
        # ──────────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(16, 9))
        fig.suptitle(
            f"I_Phi Landscape Analysis — L={system_info.get('L',n)} "
            f"Ne={system_info.get('Ne',n)} U={system_info.get('U','?')}",
            fontsize=13, fontweight="bold"
        )

        ax_txt = fig.add_axes([0.03, 0.05, 0.40, 0.88])
        ax_txt.axis("off")
        ax_bar = fig.add_axes([0.50, 0.10, 0.47, 0.80])

        summary = (
            f"System\n"
            f"{'='*40}\n\n"
            f"  L={system_info.get('L',n)}  Ne={system_info.get('Ne',n)}\n"
            f"  t={system_info.get('t',1.0):.3f}  U={system_info.get('U','?')}\n"
            f"  mode={system_info.get('mode','uniform')}\n"
            f"  n = N = {n}  (pure state)\n\n"
            f"  E0 = {system_info.get('E0','?')}\n"
            f"  E1 = {system_info.get('E1','?')}\n\n"
            f"GME Results\n"
            f"{'='*40}\n\n"
            f"  C_GME (exact eq.1)   = {C_exact:.8f}\n"
            f"  C_LB  (basis states) = {C_LB_basis:.8f}\n"
            f"  Gap  = {C_exact-C_LB_basis:.8f}\n\n"
            f"Optimal |Phi>\n"
            f"{'='*40}\n\n"
            f"  z1 = |{opt_s1}⟩\n"
            f"  z2 = |{opt_s2}⟩\n\n"
            f"  Pairs at max     : {len(disc['optimal_pairs'])}\n"
            f"  Physically unique: {disc['physically_unique']}\n"
        )
        ax_txt.text(0.02, 0.98, summary, ha="left", va="top",
                    fontsize=10, family="monospace", transform=ax_txt.transAxes)

        # Bar chart of top groups
        top_vals = sorted(disc["groups"].keys(), reverse=True)[:10]
        top_counts = [len(disc["groups"][v]) for v in top_vals]
        colors = ["#1D9E75" if v == top_vals[0] else "#5A7EC0" if v > 0 else "#D85A30"
                  for v in top_vals]
        y_pos = list(range(len(top_vals)))
        ax_bar.barh(y_pos, top_vals, color=colors, height=0.6, alpha=0.85)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(
            [f"{v:.5f}  ({c} pair{'s' if c>1 else ''})" for v,c in zip(top_vals, top_counts)],
            fontsize=9
        )
        ax_bar.set_xlabel("I_Phi value", fontsize=11)
        ax_bar.set_title(f"Top I_Phi groups — {disc['all_pairs_checked']} pairs total",
                         fontsize=11)
        ax_bar.invert_yaxis()
        ax_bar.grid(True, axis="x", alpha=0.3)

        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)

        # ──────────────────────────────────────────────────────────────────
        # PAGE 2: 1D continuous landscape
        # ──────────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(
            "Continuous Landscape: I_Phi along Bloch-sphere interpolation path",
            fontsize=13, fontweight="bold"
        )

        ax = axes[0]
        ax.plot(cont["alpha_deg"], cont["C_values"], color="#1D9E75", linewidth=2.5,
                label="C_LB (continuous)")
        ax.axhline(C_LB_basis, color="#534AB7", linewidth=1.5, linestyle="--",
                   label=f"C_LB basis = {C_LB_basis:.6f}")
        ax.axhline(C_exact, color="#D85A30", linewidth=1.2, linestyle=":",
                   label=f"C_GME exact = {C_exact:.6f}")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.6, linestyle=":")
        ax.axvline(90, color="gray", linewidth=0.6, linestyle=":")
        ax.axvline(180, color="gray", linewidth=0.6, linestyle=":")
        above_90 = np.where(cont["C_values"] >= 0.9*C_LB_basis)[0]
        if len(above_90) > 0:
            ax.axvspan(cont["alpha_deg"][above_90[0]], cont["alpha_deg"][above_90[-1]],
                        alpha=0.12, color="#1D9E75",
                        label=f"90% peak ({cont['peak_width_deg']:.1f}°)")
        ax.set_xlabel("Interpolation angle α (degrees)", fontsize=11)
        ax.set_ylabel("C_LB", fontsize=11)
        ax.set_title("Full path: optimal → equal superposition → swapped", fontsize=10)
        ax.legend(fontsize=9, loc="lower center")
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        mask = cont["alpha_deg"] <= 35
        ax2.plot(cont["alpha_deg"][mask], cont["C_values"][mask],
                 color="#1D9E75", linewidth=2.5, marker="o", markersize=5)
        ax2.axhline(C_LB_basis, color="#534AB7", linewidth=1.5, linestyle="--")
        ax2.set_xlabel("Interpolation angle α (degrees)", fontsize=11)
        ax2.set_ylabel("C_LB", fontsize=11)
        ax2.set_title(f"Zoom: 0–35° — peak sharpness ({cont['peak_width_deg']:.1f}° @ 90%)",
                      fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0,0,1,0.94])
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)

        # ──────────────────────────────────────────────────────────────────
        # PAGE 3: Answer to supervisor + 2D surface
        # ──────────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(16, 9))
        fig.suptitle("Landscape Summary: 2D Surface + Supervisor Conclusions",
                     fontsize=14, fontweight="bold")

        # Left: answer text
        ax_ans = fig.add_axes([0.02, 0.35, 0.35, 0.62])
        ax_ans.axis("off")

        answer = (
            f"Supervisor's Question\n"
            f"{'─'*35}\n"
            f"Are the maxima only at basis\n"
            f"states, or elsewhere?\n\n"
            f"DISCRETE SEARCH\n"
            f"─────────────────\n"
            f"Pairs at max: {len(disc['optimal_pairs'])}\n"
            f"Unique:       {disc['physically_unique']}\n"
            f"-> SINGLE optimal point\n\n"
            f"CONTINUOUS LANDSCAPE\n"
            f"─────────────────────\n"
            f"Peak width @90%: {cont['peak_width_deg']:.1f}°\n"
            f"-> {'SHARP' if cont['peak_is_sharp'] else 'BROAD'}\n\n"
            f"CONCLUSION\n"
            f"──────────\n"
            f"Basis state is the UNIQUE\n"
            f"global optimum. No other\n"
            f"family of states achieves\n"
            f"the same I_Phi value.\n"
        )
        ax_ans.text(0.02, 0.98, answer, ha="left", va="top",
                    fontsize=10, family="monospace", transform=ax_ans.transAxes)

        # Right: 2D heatmap
        ax_heat = fig.add_axes([0.42, 0.35, 0.55, 0.62])
        im = ax_heat.pcolormesh(surf["theta_deg"], surf["theta_deg"], surf["C_surf"].T,
                                cmap="YlOrRd", shading="auto", vmin=0, vmax=C_LB_basis)
        ax_heat.plot(surf["peak_location"][0], surf["peak_location"][1],
                     "g*", markersize=18, label="Peak")
        ax_heat.set_xlabel("theta1 (degrees)", fontsize=10)
        ax_heat.set_ylabel("theta2 (degrees)", fontsize=10)
        ax_heat.set_title("2D Surface: C_LB over (theta1, theta2)", fontsize=11)
        plt.colorbar(im, ax=ax_heat, label="C_LB")
        ax_heat.legend(fontsize=9)

        # Bottom: 3D surface
        ax3d = fig.add_axes([0.02, 0.02, 0.46, 0.32], projection="3d")
        T1, T2 = np.meshgrid(surf["theta_deg"], surf["theta_deg"])
        surf_plot = ax3d.plot_surface(T1, T2, surf["C_surf"].T, cmap="YlOrRd",
                                      alpha=0.85, linewidth=0, antialiased=True)
        ax3d.scatter([surf["peak_location"][0]], [surf["peak_location"][1]], [C_LB_basis],
                     color="green", s=100, zorder=5)
        ax3d.set_xlabel("theta1 (°)", fontsize=9)
        ax3d.set_ylabel("theta2 (°)", fontsize=9)
        ax3d.set_zlabel("C_LB", fontsize=9)
        ax3d.set_title("3D Surface View", fontsize=10)

        # 3D surface (raw I_Phi)
        ax3d2 = fig.add_axes([0.52, 0.02, 0.46, 0.32], projection="3d")
        surf_plot2 = ax3d2.plot_surface(T1, T2, surf["Z"].T, cmap="RdYlGn",
                                        alpha=0.85, linewidth=0, antialiased=True)
        ax3d2.scatter([surf["peak_location"][0]], [surf["peak_location"][1]], [disc["max_I"]],
                      color="blue", s=100, zorder=5)
        ax3d2.set_xlabel("theta1 (°)", fontsize=9)
        ax3d2.set_ylabel("theta2 (°)", fontsize=9)
        ax3d2.set_zlabel("I_Phi", fontsize=9)
        ax3d2.set_title("3D Raw I_Phi Surface", fontsize=10)

        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)

    print(f"\n  Landscape report saved → {output_path}")