"""
multi_restart_analysis.py
--------------------------
Runs the optimizer many times from different random starting points
and records what each restart finds.

For each restart:
  - The best I_Phi value found
  - The theta and phi values at the optimum
  - Whether it converged to the basis state or somewhere else

Run:
    cd ~/venvs/qchem/hubbard_gme
    python multi_restart_analysis.py

Output:
    outputs/multi_restart_L4_Ne4_U4.pdf
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from hubbard_gme.pipeline import compute_ground_and_excited_states
from hubbard_gme.rdm import compute_all_pure_rdms
from hubbard_gme.gme import (
    I_phi_continuous,
    compute_C_GME_pure,
    supervisor_gme_lower_bound,
)

# ── Parameters ─────────────────────────────────────────────────────────
L    = 4; Ne = 4; t = 1.0; U = 4.0; mode = "uniform"; n = 4
N_RESTARTS = 50
OUTPUT_PDF = f"outputs/multi_restart_L{L}_Ne{Ne}_U{U}.pdf"
os.makedirs("outputs", exist_ok=True)

# ── Step 1: get real Hubbard RDM ───────────────────────────────────────
print("=" * 60)
print(f"  MULTI-RESTART ANALYSIS  L={L} Ne={Ne} t={t} U={U}")
print("=" * 60)
print("\nComputing Hubbard ground state ...")
E0, psi0, E1, psi1, n_so = compute_ground_and_excited_states(
    L=L, t=t, U=U, mode=mode, n_electrons=Ne
)
rdm_pure, _, _ = compute_all_pure_rdms(
    L=L, psi0=psi0, psi1=psi1,
    n_spin_orbitals=2*L, n_electrons=Ne,
    min_n=n, max_n=n,
)
rho   = rdm_pure[tuple(range(n))]
C_GME = compute_C_GME_pure(rho, n)
basis = supervisor_gme_lower_bound(rho, n)
C_LB_basis = basis["C_LB"]
print(f"  C_GME (exact)  = {C_GME:.8f}")
print(f"  C_LB  (basis)  = {C_LB_basis:.8f}")
print(f"  Best basis pair: z1={basis['best_z1']} z2={basis['best_z2']}")

# ── Step 2: define the optimizer objective ─────────────────────────────
bounds = ([(0.0, np.pi)] * n + [(0.0, 2*np.pi)] * n) * 2

def neg_I(params):
    t1 = list(params[0:n]);     p1 = list(params[n:2*n])
    t2 = list(params[2*n:3*n]); p2 = list(params[3*n:4*n])
    return -I_phi_continuous(rho, n, t1, p1, t2, p2)

# ── Step 3: run N_RESTARTS independent optimizations ──────────────────
print(f"\nRunning {N_RESTARTS} independent optimizer restarts ...")
print(f"(showing theta and phi in degrees for BOTH psi1 and psi2 per restart)")
print("-" * 60)

restart_results = []

for seed in range(N_RESTARTS):
    res = differential_evolution(
        neg_I,
        bounds,
        seed=seed,
        maxiter=500,
        tol=1e-9,
        popsize=15,
        workers=1,
        polish=True,
    )

    C_found   = max(0.0, -res.fun)
    params    = res.x
    t1_found  = params[0:n]
    p1_found  = params[n:2*n]
    t2_found  = params[2*n:3*n]
    p2_found  = params[3*n:4*n]

    is_pole = lambda t: (abs(t) < 0.15 or abs(t - np.pi) < 0.15)
    all_poles_t1 = all(is_pole(t) for t in t1_found)
    all_poles_t2 = all(is_pole(t) for t in t2_found)
    at_basis = all_poles_t1 and all_poles_t2

    diff = C_found - C_LB_basis
    conclusion = "basis state" if at_basis else "other point"
    if abs(diff) < 1e-4:
        quality = "= basis"
    elif diff > 1e-4:
        quality = f"+{diff:.6f} IMPROVEMENT!"
    else:
        quality = f"{diff:.6f} worse"

    # FIX: both thetas and phis converted to degrees
    t1_deg = [round(np.degrees(t), 1) for t in t1_found]
    t2_deg = [round(np.degrees(t), 1) for t in t2_found]
    p1_deg = [round(np.degrees(p), 1) for p in p1_found]
    p2_deg = [round(np.degrees(p), 1) for p in p2_found]

    print(f"  Restart {seed:3d}:")
    print(f"    C_LB = {C_found:.8f}   {quality}   {conclusion}")
    print(f"    psi1 thetas = {t1_deg}  degrees")
    print(f"    psi2 thetas = {t2_deg}  degrees")
    print(f"    psi1 phis   = {p1_deg}  degrees")
    print(f"    psi2 phis   = {p2_deg}  degrees")
    print()

    restart_results.append({
        "seed":     seed,
        "C_LB":     C_found,
        "t1":       t1_found,
        "p1":       p1_found,
        "t2":       t2_found,
        "p2":       p2_found,
        "at_basis": at_basis,
        "diff":     diff,
    })

# ── Step 4: summary statistics ─────────────────────────────────────────
C_values   = [r["C_LB"] for r in restart_results]
at_basis   = [r["at_basis"] for r in restart_results]
best_C     = max(C_values)
n_at_basis = sum(at_basis)
n_other    = N_RESTARTS - n_at_basis
n_improved = sum(1 for r in restart_results if r["diff"] > 1e-4)

print()
print("=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Total restarts        : {N_RESTARTS}")
print(f"  C_LB (basis)          : {C_LB_basis:.8f}")
print(f"  Best C_LB found       : {best_C:.8f}")
print(f"  Improvement over basis: {best_C - C_LB_basis:.2e}")
print()
print(f"  Restarts converging to basis state : {n_at_basis} / {N_RESTARTS}")
print(f"  Restarts converging elsewhere      : {n_other} / {N_RESTARTS}")
print(f"  Restarts with improvement          : {n_improved} / {N_RESTARTS}")
print()
if n_improved == 0:
    print("  All restarts converged to the same optimal point.")
else:
    print(f"  {n_improved} restarts found improvement over the basis state.")

# ── Step 5: generate PDF ───────────────────────────────────────────────
print(f"\nGenerating PDF -> {OUTPUT_PDF} ...")

with PdfPages(OUTPUT_PDF) as pdf:

    # ── PAGE 1: scatter plot ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"Multi-Restart Analysis:  Hubbard L={L} Ne={Ne} t={t} U={U}  "
        f"({N_RESTARTS} restarts)",
        fontsize=13, fontweight="bold"
    )

    ax = axes[0]
    seeds      = [r["seed"] for r in restart_results]
    clb_values = [r["C_LB"] for r in restart_results]
    colors     = ["#1D9E75" if r["at_basis"] else "#D85A30"
                  for r in restart_results]

    ax.scatter(seeds, clb_values, c=colors, s=60, zorder=5)
    ax.axhline(C_LB_basis, color="#534AB7", linewidth=2, linestyle="--",
               label=f"C_LB (basis) = {C_LB_basis:.6f}")
    ax.axhline(C_GME, color="#D85A30", linewidth=1.5, linestyle=":",
               label=f"C_GME (exact) = {C_GME:.6f}")

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#1D9E75",
               markersize=10, label=f"Converged to basis state ({n_at_basis})"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#D85A30",
               markersize=10, label=f"Converged elsewhere ({n_other})"),
        Line2D([0],[0], color="#534AB7", linewidth=2, linestyle="--",
               label=f"C_LB (basis) = {C_LB_basis:.4f}"),
        Line2D([0],[0], color="#D85A30", linewidth=1.5, linestyle=":",
               label=f"C_GME (exact) = {C_GME:.4f}"),
    ], fontsize=9)

    ax.set_xlabel("Restart number (different random seed)", fontsize=11)
    ax.set_ylabel("C_LB (continuous optimizer)", fontsize=11)
    ax.set_title("C_LB found by each independent restart", fontsize=11)
    ax.set_ylim(0, C_GME * 1.1)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(clb_values, bins=20, color="#5A7EC0", alpha=0.8, edgecolor="white")
    ax2.axvline(C_LB_basis, color="#534AB7", linewidth=2, linestyle="--",
                label=f"C_LB (basis) = {C_LB_basis:.6f}")
    ax2.axvline(C_GME, color="#D85A30", linewidth=1.5, linestyle=":",
                label=f"C_GME (exact) = {C_GME:.6f}")
    ax2.set_xlabel("C_LB value found", fontsize=11)
    ax2.set_ylabel("Number of restarts", fontsize=11)
    ax2.set_title("Distribution of C_LB across all restarts", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0,0,1,0.93])
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)

    # ── PAGE 2: theta values per restart ──────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Optimal theta values found by each restart  (degrees)\n"
        "theta = 0° or 180° corresponds to basis state poles",
        fontsize=13, fontweight="bold"
    )

    colors_r = ["#1D9E75" if r["at_basis"] else "#D85A30"
                for r in restart_results]

    for site in range(n):
        ax = axes[site // 2][site % 2]
        t1_site = [np.degrees(r["t1"][site]) for r in restart_results]
        t2_site = [np.degrees(r["t2"][site]) for r in restart_results]

        ax.scatter(seeds, t1_site, c=colors_r, s=40, alpha=0.8,
                   label="psi1 theta", marker="o")
        ax.scatter(seeds, t2_site, c=colors_r, s=40, alpha=0.8,
                   label="psi2 theta", marker="^")

        ax.axhline(0,   color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(180, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(90,  color="gray", linewidth=0.5, linestyle=":",  alpha=0.3)
        ax.text(N_RESTARTS*0.98,   3, "0° (up)",    ha="right", fontsize=8, color="gray")
        ax.text(N_RESTARTS*0.98, 177, "180° (down)", ha="right", fontsize=8, color="gray")
        ax.text(N_RESTARTS*0.98,  93, "90° (equator)", ha="right", fontsize=8, color="gray")

        ax.set_xlabel("Restart number", fontsize=10)
        ax.set_ylabel("theta (degrees)", fontsize=10)
        ax.set_title(f"Site {site}  optimal theta per restart", fontsize=10)
        ax.set_ylim(-10, 195)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0,0,1,0.94])
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)

    # ── PAGE 3: summary ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    fig.suptitle("Multi-Restart summary of Results",
                 fontsize=14, fontweight="bold")

    lines = [
        f"System:  L={L}  Ne={Ne}  t={t}  U={U}  mode={mode}",
        f"n = N = {n}   (pure state RDM,  Tr(rho^2) = 1)",
        f"",
        f"{'='*58}",
        f"",
        f"  C_GME (exact eq.1)       = {C_GME:.8f}",
        f"  C_LB  (basis states)     = {C_LB_basis:.8f}",
        f"  Best C_LB (continuous)   = {best_C:.8f}",
        f"  Improvement              = {best_C - C_LB_basis:.2e}",
        f"",
        f"{'='*58}",
        f"",
        f"  Optimizer  : differential_evolution (scipy)",
        f"  Parameters : 16  (4 thetas + 4 phis for each of psi1, psi2)",
        f"  Restarts   : {N_RESTARTS}  (each from a different random starting point)",
        f"",
        f"  Restarts converging to basis state : {n_at_basis} / {N_RESTARTS}",
        f"  Restarts converging elsewhere      : {n_other} / {N_RESTARTS}",
        f"  Restarts with improvement          : {n_improved} / {N_RESTARTS}",
        f"",
        f"{'='*58}",
        f"",
        f"  All {N_RESTARTS} restarts converged to the same optimal |Phi>:",
        f"  z1 = {basis['best_z1']}   z2 = {basis['best_z2']}",
        f"  (|udud> and |dudu> — the two Neel configurations)",
        f"",
        f"  The phi angles at the optimum are arbitrary",
        f"  (phi has no effect when theta = 0 or 180 degrees)",
    ] if n_improved == 0 else [
        f"System:  L={L}  Ne={Ne}  t={t}  U={U}  mode={mode}",
        f"  C_GME = {C_GME:.8f}   C_LB basis = {C_LB_basis:.8f}",
        f"  Best C_LB continuous = {best_C:.8f}",
        f"  Improvement = {best_C - C_LB_basis:.6f}",
        f"  {n_improved} restart(s) found improvement.",
    ]

    ax.text(0.05, 0.95, "\n".join(lines), ha="left", va="top",
            fontsize=11, family="monospace", transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)

    # ── PAGE 4: phi values per restart ─────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Optimal phi values found by each restart  (degrees)\n"
        "phi is arbitrary at the poles (theta = 0° or 180°)",
        fontsize=13, fontweight="bold"
    )
    colors_r = ["#1D9E75" if r["at_basis"] else "#D85A30"
                for r in restart_results]
    for site in range(n):
        ax = axes[site // 2][site % 2]
        p1_site = [np.degrees(r["p1"][site]) for r in restart_results]
        p2_site = [np.degrees(r["p2"][site]) for r in restart_results]
        ax.scatter(seeds, p1_site, c=colors_r, s=40, alpha=0.8,
                   label="psi1 phi", marker="o")
        ax.scatter(seeds, p2_site, c=colors_r, s=40, alpha=0.8,
                   label="psi2 phi", marker="^")
        ax.set_xlabel("Restart number", fontsize=10)
        ax.set_ylabel("phi (degrees)", fontsize=10)
        ax.set_title(f"Site {site}  —  optimal phi per restart", fontsize=10)
        ax.set_ylim(-10, 375)
        ax.axhline(0,   color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.axhline(360, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.text(N_RESTARTS*0.98, 5,   "0°",   ha="right", fontsize=8, color="gray")
        ax.text(N_RESTARTS*0.98, 355, "360°", ha="right", fontsize=8, color="gray")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0,0,1,0.94])
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)

    # ── PAGE 5: 2D heatmap ─────────────────────────────────────────────
    print("  Computing landscape surface ...")
    N_grid = 50
    thetas_g = np.linspace(0, np.pi, N_grid)
    Z = np.zeros((N_grid, N_grid))
    for i, t1 in enumerate(thetas_g):
        for j, t2 in enumerate(thetas_g):
            th1 = [t1, np.pi-t1, t1, np.pi-t1]
            th2 = [t2, np.pi-t2, t2, np.pi-t2]
            ph  = [0.0]*n
            Z[i, j] = I_phi_continuous(rho, n, th1, ph, th2, ph)
    C_surf    = np.maximum(Z, 0.0)
    theta_deg = np.degrees(thetas_g)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"I_Phi Landscape Surface  —  Hubbard L={L} Ne={Ne} t={t} U={U}\n"
        f"Restricted 2D slice:  phi=0,  psi1=[t1, 180-t1, t1, 180-t1],  "
        f"psi2=[t2, 180-t2, t2, 180-t2]",
        fontsize=12, fontweight="bold"
    )

    ax = axes[0]
    im = ax.pcolormesh(theta_deg, theta_deg, C_surf.T,
                       cmap="YlOrRd", shading="auto", vmin=0, vmax=C_LB_basis)
    plt.colorbar(im, ax=ax, label="C_LB = max(0, I_Phi)")
    ax.plot(0,   180, "g*", markersize=20,
            label=f"Optimal (discrete basis search)\nC_LB={C_LB_basis:.4f}")
    ax.plot(180, 0,   "g*", markersize=20)
    ax.set_xlabel("theta1 (degrees)  —  psi1", fontsize=11)
    ax.set_ylabel("theta2 (degrees)  —  psi2", fontsize=11)
    ax.set_title("C_LB heatmap  (top view)", fontsize=11)
    ax.legend(fontsize=9)
    ax.text(3,  174, "|udud>->|dudu>", fontsize=8, color="green")
    ax.text(128, 3,  "|dudu>->|udud>", fontsize=8, color="green")

    ax2 = axes[1]
    vmax = max(abs(Z.min()), abs(Z.max()))
    im2  = ax2.pcolormesh(theta_deg, theta_deg, Z.T,
                          cmap="RdYlGn", shading="auto",
                          vmin=-vmax, vmax=vmax)
    plt.colorbar(im2, ax=ax2, label="I_Phi (raw)")
    ax2.plot(0,   180, "b*", markersize=20)
    ax2.plot(180, 0,   "b*", markersize=20)
    ax2.set_xlabel("theta1 (degrees)  —  psi1", fontsize=11)
    ax2.set_ylabel("theta2 (degrees)  —  psi2", fontsize=11)
    ax2.set_title("Raw I_Phi  (green=positive, red=negative)", fontsize=11)

    plt.tight_layout(rect=[0,0,1,0.93])
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)

    # ── PAGE 6: 3D surface ─────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(
        f"3D Landscape Surface  —  restricted 2D slice  "
        f"(phi=0, Neel alternating structure)\n"
        f"Hubbard  L={L}  Ne={Ne}  t={t}  U={U}",
        fontsize=13, fontweight="bold"
    )
    T1, T2 = np.meshgrid(theta_deg, theta_deg)

    ax3d = fig.add_subplot(121, projection="3d")
    surf = ax3d.plot_surface(T1, T2, C_surf.T,
                             cmap="YlOrRd", alpha=0.85,
                             linewidth=0, antialiased=True)
    ax3d.scatter([0, 180], [180, 0], [C_surf.max(), C_surf.max()],
                 color="green", s=150, zorder=5,
                 label="Optimal |Phi>")
    ax3d.set_xlabel("theta1 (degrees)", fontsize=9)
    ax3d.set_ylabel("theta2 (degrees)", fontsize=9)
    ax3d.set_zlabel("C_LB", fontsize=9)
    ax3d.set_title("C_LB surface\n(negative clipped to 0)", fontsize=10)
    ax3d.legend(fontsize=8)
    fig.colorbar(surf, ax=ax3d, shrink=0.5, label="C_LB")

    ax3d2 = fig.add_subplot(122, projection="3d")
    surf2 = ax3d2.plot_surface(T1, T2, Z.T,
                               cmap="RdYlGn", alpha=0.85,
                               linewidth=0, antialiased=True)
    ax3d2.scatter([0, 180], [180, 0], [Z.max(), Z.max()],
                  color="blue", s=150, zorder=5, label="Global maximum")
    ax3d2.set_xlabel("theta1 (degrees)", fontsize=9)
    ax3d2.set_ylabel("theta2 (degrees)", fontsize=9)
    ax3d2.set_zlabel("I_Phi", fontsize=9)
    ax3d2.set_title("Raw I_Phi\n(includes negative values)", fontsize=10)
    ax3d2.legend(fontsize=8)
    fig.colorbar(surf2, ax=ax3d2, shrink=0.5, label="I_Phi")

    plt.tight_layout(rect=[0,0,1,0.94])
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)

print(f"\n  Done! PDF -> {OUTPUT_PDF}")