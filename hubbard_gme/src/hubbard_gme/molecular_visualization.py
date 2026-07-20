"""
molecular_visualization.py
--------------------------
PDF report generation for molecular n-body RDMs.

This module re-uses the existing hubbard_gme visualization primitives
(_write_rdm_page, _write_combined_page) and adds two molecule-aware pages:

    _write_molecular_title_page(pdf, mol_info, n_body)
    _write_molecular_summary_page(pdf, rdm_dict, metrics_dict, cr_results,
                                   gme_results, mol_info, n_body)

Public entry point:

    generate_molecular_gme_pdf(rdm_dict, metrics_dict, n, output_path,
                                mol_info, cr_results=None)

`mol_info` is a plain dict carrying molecular metadata — see
molecular_pipeline._build_molecular_info() for the expected keys.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Re-use existing heatmap + combined-page writers unchanged.
from .visualization import (
    generate_spin_labels,
    _fontsize_for_n,
    _IMAG_THRESHOLD,
    plot_rdm_three_view,
)
from .gme import analyze_one_rdm_for_GME


# ──────────────────────────────────────────────────────────────────────────────
# Molecular title page
# ──────────────────────────────────────────────────────────────────────────────

def _write_molecular_title_page(pdf, mol_info: dict, n_body: int) -> None:
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.patch.set_facecolor("#f7f9fc")

    ax.text(
        0.5, 0.97,
        "Genuine Multipartite Entanglement Analysis — Molecular System",
        ha="center", va="top", fontsize=16, fontweight="bold", color="#1a1a2e",
    )
    ax.plot(
        [0.03, 0.97], [0.93, 0.93],
        color="#1f6feb", linewidth=2,
        transform=ax.transAxes, clip_on=False,
    )

    def section(x, y, heading, body, hc, bc="#2c2c2c"):
        ax.text(x, y,       heading, ha="left", va="top",
                fontsize=10, fontweight="bold",
                color=hc, family="monospace")
        ax.text(x, y - 0.03, body, ha="left", va="top",
                fontsize=9, color=bc, family="monospace")

    section(
        0.03, 0.90, "MOLECULE",
        f"{mol_info['name']}  ({mol_info['formula']})\n"
        f"Basis set    : {mol_info['basis']}\n"
        f"Charge       : {mol_info['charge']}\n"
        f"Multiplicity : {mol_info['multiplicity']}\n"
        f"Total e-     : {mol_info['total_electrons']}\n"
        f"n-body order : {n_body}",
        "#1f6feb",
    )

    section(
        0.03, 0.60,
        f"ACTIVE SPACE  CAS({mol_info['nele_cas']},{mol_info['norb_cas']})",
        f"Frozen core orbs : {mol_info['ncore']}\n"
        f"Frozen e-        : {2 * mol_info['ncore']}\n"
        f"Active e-        : {mol_info['nele_cas']}\n"
        f"Active orbs      : {mol_info['norb_cas']}\n"
        f"Spin-orbitals    : {mol_info['n_so']}\n"
        f"Hilbert dim      : {mol_info['hilbert_dim']}",
        "#7c3aed",
    )

    section(
        0.36, 0.90, "GROUND STATE ENERGIES",
        f"E0 (Ha)  : {mol_info['E0']:.8f}\n"
        f"E1 (Ha)  : {mol_info['E1']:.8f}\n"
        f"Gap (Ha) : {mol_info['gap']:.8f}\n"
        f"HF  (Ha) : {mol_info['hf_energy']:.8f}",
        "#0ea472",
    )

    section(
        0.36, 0.62, "SPIN PROPERTIES",
        f"<S2>         : {mol_info['S2']:.6f}\n"
        f"S            : {mol_info['S']:.4f}\n"
        f"<Sz>         : {mol_info['Sz']:.6f}\n"
        f"Multiplicity : {mol_info['multiplicity']}\n"
        f"Spin state   : {mol_info['spin_label']}",
        "#dc2626",
    )

    section(
        0.36, 0.36, "DOUBLE OCCUPANCY",
        f"<D> total  : {mol_info['double_occ']:.6f}\n"
        f"<D> / site : {mol_info['double_occ_per_site']:.6f}",
        "#b45309",
    )

    cr_methods = mol_info.get("cr_methods", "expo, stiefel")
    section(
        0.69, 0.90, "CONVEX-ROOF METHODS",
        f"Methods run    : {cr_methods}\n"
        f"Restarts       : {mol_info.get('cr_restarts', '?')}\n"
        f"\nC_LB  : coherence witness bound\n"
        f"C_expo: unitary opt (exp map)\n"
        f"C_stf : Stiefel manifold (Alg. B)\n"
        f"C_UB  : eigendecomposition bound",
        "#0891b2",
    )

    section(
        0.69, 0.60, "GME CRITERIA",
        f"C_GME = sqrt(2 * min_A (1 - Tr(rho_A^2)))\n"
        f"\nConvex roof (mixed):\n"
        f"  C_GME(rho) = min Sum p_i C_GME(|psi_i>)\n"
        f"\nBounds ordering:\n"
        f"  C_LB <= C_GME <= C_UB",
        "#be185d",
    )

    n_clusters = mol_info.get(f"n_clusters_{n_body}", "?")
    section(
        0.69, 0.36, "THIS PDF",
        f"Body order : {n_body}\n"
        f"Clusters   : {n_clusters}\n"
        f"\nDate: {datetime.now().strftime('%Y-%m-%d  %H:%M')}",
        "#374151",
    )

    ax.plot(
        [0.03, 0.97], [0.04, 0.04],
        color="#1f6feb", linewidth=1.5, linestyle="--",
        transform=ax.transAxes, clip_on=False,
    )

    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Molecular summary page (all clusters at this n-body order)
# ──────────────────────────────────────────────────────────────────────────────

def _write_molecular_summary_page(
    pdf,
    rdm_dict: Dict,
    metrics_dict: Dict,
    cr_results: Optional[Dict],
    gme_results: Dict,
    mol_info: dict,
    n_body: int,
) -> None:
    """
    Single page with one row per cluster at this n-body order.
    Columns: Orbitals | Purity | Entropy | C_GME(pure) | C_LB | C_gcg | C_UB | State
    """
    # Collect clusters of requested body order
    clusters = sorted(s for s in rdm_dict if len(s) == n_body)

    fig, ax = plt.subplots(figsize=(16, max(5.0, len(clusters) * 0.35 + 3.0)))
    ax.axis("off")

    fig.suptitle(
        f"Summary — {n_body}-Body RDMs  ({mol_info['name']}, "
        f"CAS({mol_info['nele_cas']},{mol_info['norb_cas']}))",
        fontsize=13, fontweight="bold", y=0.98,
    )

    col_labels = [
        "Orbitals", "Purity", "Entropy", "C_GME (pure)", "C_LB",
        "C_gcg", "C_UB", "State",
    ]
    rows = []

    for sites in clusters:
        m   = metrics_dict.get(sites, {})
        cr  = (cr_results or {}).get(sites, {})
        gme = gme_results.get(sites, {})

        pur = m.get("purity", float("nan"))

        # Pure-state C_GME only reported if the RDM is pure and has GME
        if gme.get("status") == "GME EXISTS":
            c_gme = gme.get("C_GME", float("nan"))
            c_gme_str = f"{c_gme:.6f}"
        elif gme.get("status") == "no_GME":
            c_gme_str = "0.000000"
        else:
            c_gme_str = "—"

        def _fmt(x):
            if x is None: return "—"
            if isinstance(x, float) and np.isnan(x): return "—"
            return f"{x:.6f}"

        rows.append([
            ",".join(str(s) for s in sites),
            _fmt(pur),
            _fmt(m.get("entropy")),
            c_gme_str,
            _fmt(cr.get("C_LB_best", cr.get("C_LB"))),
            _fmt(cr.get("C_gcg", cr.get("C_expo"))),
            _fmt(cr.get("C_UB")),
            m.get("state_type", "—"),
        ])

    if not rows:
        ax.text(0.5, 0.5,
                f"No {n_body}-body clusters available for this active space.",
                ha="center", va="center", fontsize=12, color="gray",
                transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return

    table = ax.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    type_colors = {
        "PURE":            "#d4edda",
        "WEAKLY MIXED":    "#fff3cd",
        "MIXED":           "#f8d7da",
        "STRONGLY MIXED":  "#e2d9f3",
    }
    for i, row in enumerate(rows):
        for j in range(len(col_labels)):
            table[(i + 1, j)].set_facecolor(type_colors.get(row[-1], "white"))

    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#1f6feb")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def _write_rdm_combined_page(pdf, rho, metrics, sites, cr=None) -> None:
    """
    ONE page with the standard THREE views (shared with the Hubbard pipeline):
      (1) Re(rho) index order   (2) Re(rho) Sz-ordered   (3) |rho| Sz-ordered
    The footer carries the RDM metrics AND (when ``cr`` is given) the GME
    numbers — C_gcg with its lower/upper bounds — so no separate analysis page
    is needed per cluster.
    """
    # Shared three-view page (same as the Hubbard pipeline): Re index-order,
    # Re Sz-ordered (signed), |rho| Sz-ordered — values annotated for n<=4.
    fig, cross = plot_rdm_three_view(rho, sites)

    show_imag = np.max(np.abs(rho.imag)) > _IMAG_THRESHOLD
    imag_note = (f"Max|Im($\\rho$)|={np.max(np.abs(rho.imag)):.1e}"
                 if show_imag else r"Im($\rho$)<1e-6")
    fig.text(
        0.5, 0.040,
        f"Trace={metrics['trace']:.6f}   Purity={metrics['purity']:.6f}   "
        f"Entropy={metrics['entropy']:.6f}   ({imag_note})   |   "
        f"max |coherence between different $S_z$| = {cross:.1e}  "
        f"($\\Rightarrow S_z$ conserved)",
        ha="center", fontsize=9,
    )
    if cr:
        def _f(x):
            return "—" if x is None or (isinstance(x, float) and np.isnan(x)) \
                else f"{x:.6f}"
        c_lb = cr.get("C_LB_best", cr.get("C_LB"))
        c_val = cr.get("C_gcg", cr.get("C_expo"))
        # Verdict from C_gcg alone (the accurate convex-roof estimator); the
        # analytic lower bound is a weak witness and does not drive the label.
        if c_val is None or (isinstance(c_val, float) and np.isnan(c_val)):
            verdict = "—"
        else:
            verdict = "GME" if c_val > 1e-5 else "no GME"
        fig.text(
            0.5, 0.010,
            f"GME (GCG convex roof):   "
            f"$C_{{\\mathrm{{LB}}}}$ = {_f(c_lb)}   $\\leq$   "
            f"$C_{{\\mathrm{{GME}}}}$ = {_f(c_val)}   "
            f"$\\leq$   $C_{{\\mathrm{{UB}}}}$ = {_f(cr.get('C_UB'))}"
            f"      verdict: {verdict}",
            ha="center", fontsize=10, fontweight="bold", color="#1a5fb4",
        )
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


def generate_molecular_gme_pdf(
    rdm_dict: Dict,
    metrics_dict: Dict,
    n: int,
    output_path: str,
    mol_info: dict,
    cr_results: Optional[Dict] = None,
) -> None:
    """
    Generate a multi-page PDF for all RDMs of body order n.

    Pages produced (per RDM of the requested order):
        1. Molecular title page (system parameters & ground state)
        2. Summary table of all clusters at this n-body order
        3. For each cluster:
             (a) Heatmap of the RDM
             (b) Combined metrics + GME + convex-roof page (re-used from
                 hubbard_gme.visualization — the supervisor/continuous
                 sections simply remain unused here)

    Parameters
    ----------
    rdm_dict     : {sites: rho}
    metrics_dict : {sites: {trace, purity, entropy, eigvals, ...}}
    n            : body order to include in the PDF
    output_path  : destination .pdf path
    mol_info     : molecular system_info dict
    cr_results   : {sites: convex_roof_result} or None
    """
    # Pre-compute GME analysis dicts for the summary page
    gme_results: Dict = {}
    for sites, rho in rdm_dict.items():
        if len(sites) != n:
            continue
        gme_results[sites] = analyze_one_rdm_for_GME(rho, sites)

    with PdfPages(output_path) as pdf:
        _write_molecular_title_page(pdf, mol_info, n)

        _write_molecular_summary_page(
            pdf, rdm_dict, metrics_dict, cr_results,
            gme_results, mol_info, n,
        )

        for sites in sorted(s for s in rdm_dict if len(s) == n):
            rho  = rdm_dict[sites]
            info = metrics_dict[sites]
            cr   = (cr_results or {}).get(sites)

            # ONE page per cluster: standard heatmap (left) + Sz-ordered view
            # (right), with the GME numbers (C_LB <= C_gcg <= C_UB) in the
            # footer. The old per-cluster "Analysis Summary" page carried no
            # additional molecular information and is no longer written.
            _write_rdm_combined_page(pdf, rho, info, sites, cr=cr)

    print(f"  PDF saved → {output_path}")


def generate_molecular_rdm_only_pdf(
    rdm_dict: Dict,
    metrics_dict: Dict,
    n: int,
    output_path: str,
    mol_info: dict,
) -> None:
    """
    Generate an RDM-only PDF (no convex roof section).

    Useful when you only want to visualise the heatmaps and metrics without
    running the (expensive) convex-roof optimisation.
    """
    generate_molecular_gme_pdf(
        rdm_dict=rdm_dict,
        metrics_dict=metrics_dict,
        n=n,
        output_path=output_path,
        mol_info=mol_info,
        cr_results=None,
    )