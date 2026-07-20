"""
visualization.py
----------------
PDF report generation and heatmap plotting for n-body RDMs and GME analysis.

Each report page type
---------------------
title_page          : System parameters and ground-state properties
rdm_heatmap         : Real heatmap (+ imaginary only if max|Im| > 1e-6)
metrics_and_gme     : Metrics table + GME analysis combined on ONE page
convex_roof_page    : Convex-roof lower/upper/numerical bounds
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from typing import Dict, Optional

from .utils import generate_spin_labels
from .gme import analyze_one_rdm_for_GME

# Threshold below which imaginary part is considered zero and not shown
_IMAG_THRESHOLD = 1e-6


# ── Colour / size helpers ────────────────────────────────────────────────────

def _rdm_figsize(n_sites: int, two_panels: bool) -> tuple:
    base = {1: (6, 4.5), 2: (7, 5.5), 3: (9, 6.5), 4: (12, 8)}
    w, h = base.get(n_sites, (14, 10))
    return (w * 2 if two_panels else w, h)


def _fontsize_for_n(n_sites: int) -> int:
    return max(5, 10 - n_sites)


# ── Single-RDM heatmap ────────────────────────────────────────────────────────

def plot_rdm_heatmap(rho, sites, title_suffix=""):
    """
    Real heatmap always shown.
    Imaginary heatmap shown ONLY if max|Im(rho)| > 1e-6.
    """
    n_sites  = len(sites)
    dim      = 2 ** n_sites
    labels   = generate_spin_labels(n_sites)
    fsize    = _fontsize_for_n(n_sites)
    site_str = ",".join(str(s) for s in sites)

    show_imag = np.max(np.abs(rho.imag)) > _IMAG_THRESHOLD
    figsize   = _rdm_figsize(n_sites, two_panels=show_imag)

    if show_imag:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        panels = [(axes[0], rho.real, "Real Part"),
                  (axes[1], rho.imag, "Imaginary Part")]
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        panels  = [(ax, rho.real, "Real Part")]

    fig.suptitle(
        f"{n_sites}-Body RDM  (Sites {site_str})  {title_suffix}",
        fontsize=12, fontweight="bold",
    )

    for ax, data, title in panels:
        vmax = max(np.max(np.abs(data)) * 1.05, 1e-12)
        im = ax.matshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title, pad=10)
        ax.set_xticks(range(dim));  ax.set_xticklabels(labels, fontsize=fsize)
        ax.set_yticks(range(dim));  ax.set_yticklabels(labels, fontsize=fsize)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


# ── PDF page writers ──────────────────────────────────────────────────────────

def _write_title_page(pdf, system_info, n_body):
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")

    ax.text(0.5, 0.97,
            "Genuine Multipartite Entanglement Analysis",
            ha="center", va="top", fontsize=16, fontweight="bold")

    col1 = f"""
Model
-----
Hubbard ring (1-D, PBC)

System parameters
-----------------
Sites (L)         : {system_info['L']}
Electrons (Ne)    : {system_info['Ne']}
Hopping (t)       : {system_info['t']}
Interaction (U)   : {system_info['U']}
Mode              : {system_info['mode']}
n-body order      : {n_body}
"""

    col2 = f"""
Ground-state properties
-----------------------
E0                : {system_info['E0']:.8f}
E1                : {system_info['E1']:.8f}

<S2>              : {system_info['S2']:.6f}
S                 : {system_info['S']:.4f}
<Sz>              : {system_info['Sz']:.6f}
Multiplicity 2S+1 : {system_info['multiplicity']:.1f}
Spin state        : {system_info['spin_label']}

Double occupancy
----------------
<D> total         : {system_info['double_occ']:.6f}
<D> / site        : {system_info['double_occ_per_site']:.6f}
"""

    col3 = f"""
GME summary
-----------
Extracted n-spin RDM : n = {n_body}

Purity:
  gamma = Tr(rho^2)

Separable cut criterion:
  gamma_A = 1  <->  rho = rho_A x rho_B

C_GME (Troiani et al.):
  C_GME = sqrt(2 * min_A (1 - Tr(rho_A^2)))

Decision logic:
  gamma=1, all Tr(rho_A^2)<1  -> GME
  gamma=1, some Tr(rho_A^2)=1 -> No GME
  gamma < 1                   -> Mixed

Date: {datetime.now().strftime("%Y-%m-%d  %H:%M")}
"""

    ax.text(0.04, 0.86, col1, ha="left", va="top", fontsize=10, family="monospace")
    ax.text(0.37, 0.86, col2, ha="left", va="top", fontsize=10, family="monospace")
    ax.text(0.68, 0.86, col3, ha="left", va="top", fontsize=10, family="monospace")

    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _write_rdm_page(pdf, rho, metrics, sites):
    """RDM heatmap. Imaginary panel only shown when max|Im| > 1e-6."""
    fig = plot_rdm_heatmap(rho, sites)
    show_imag = np.max(np.abs(rho.imag)) > _IMAG_THRESHOLD
    note = "" if show_imag else "  (Imaginary part < 1e-6, not shown)"
    fig.text(
        0.5, 0.01,
        f"Trace={metrics['trace']:.6f}   "
        f"Purity={metrics['purity']:.6f}   "
        f"Entropy={metrics['entropy']:.6f}{note}",
        fontsize=9, ha="center",
    )
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _write_combined_page(pdf, rho, info, sites, cr_result=None):
    """
    Single page with 2×2 grid — no wasted space.
      Top-left    : RDM Metrics
      Top-right   : GME Analysis
      Bottom-left : Convex-roof result   (only if cr_result provided)
      Bottom-right: Convex-roof details  (only if cr_result provided)
    """
    import matplotlib.gridspec as gridspec

    site_str = ",".join(str(s) for s in sites)
    gme_info = analyze_one_rdm_for_GME(rho, sites)
    has_cr   = cr_result is not None

    fig = plt.figure(figsize=(16, 9 if has_cr else 5))
    fig.suptitle(
        f"Analysis Summary  -  {len(sites)}-Body RDM  (Sites {site_str})",
        fontsize=14, fontweight="bold",
    )

    nrows = 2 if has_cr else 1
    # Top row: ~8 lines of text. Bottom row: ~12 lines. Ratio 1:1.4
    height_ratios = [1, 1.4] if has_cr else [1]
    gs = gridspec.GridSpec(
        nrows, 2,
        figure=fig,
        hspace=0.25,
        wspace=0.05,
        height_ratios=height_ratios,
        top=0.92, bottom=0.04,
        left=0.03, right=0.97,
    )

    ax_tl = fig.add_subplot(gs[0, 0]); ax_tl.axis("off")
    ax_tr = fig.add_subplot(gs[0, 1]); ax_tr.axis("off")
    if has_cr:
        ax_bl = fig.add_subplot(gs[1, 0]); ax_bl.axis("off")
        ax_br = fig.add_subplot(gs[1, 1]); ax_br.axis("off")

    # ── TOP-LEFT: Metrics ──────────────────────────────────────────────────
    show_imag = np.max(np.abs(rho.imag)) > _IMAG_THRESHOLD
    imag_note = (
        f"  Max |Im(rho)|        : {np.max(np.abs(rho.imag)):.2e}\n"
        if show_imag else
        f"  Imaginary part       : < 1e-6 (not shown)\n"
    )
    metrics_text = (
        f"RDM Metrics\n"
        f"{'='*36}\n\n"
        f"  Trace(rho)           : {info['trace']:.10f}\n"
        f"  Purity Tr(rho^2)     : {info['purity']:.10f}\n"
        f"  Von Neumann entropy  : {info['entropy']:.10f}\n"
        f"  Eigenvalue min       : {info['eigvals'].min():.6e}\n"
        f"  Eigenvalue max       : {info['eigvals'].max():.6e}\n"
        f"{imag_note}"
    )
    ax_tl.text(0.02, 0.97, metrics_text, ha="left", va="top",
               fontsize=10, family="monospace", transform=ax_tl.transAxes)

    # ── TOP-RIGHT: GME bipartitions + conclusion ───────────────────────────
    bip_text = (
        f"GME Analysis\n"
        f"{'='*36}\n\n"
        f"  gamma = Tr(rho^2) = {gme_info['global_purity']:.8f}\n\n"
        f"  {'A':12s} | {'B':12s} | Tr(rho_A^2)\n"
        f"  {'─'*12} | {'─'*12} | {'─'*10}\n"
    )
    for entry in gme_info["bipartitions"]:
        sep_flag = "  <- sep." if abs(entry["purity_A"] - 1.0) < 1e-8 else ""
        bip_text += (
            f"  {str(entry['A']):12s} | {str(entry['B']):12s} "
            f"| {entry['purity_A']:.8f}{sep_flag}\n"
        )
    bip_text += "\n"

    if gme_info["status"] == "mixed_state":
        bip_text += (
            "STATUS: MIXED STATE\n"
            "─────────────────────────────────\n"
            "Tr(rho^2) < 1. Pure-state C_GME\n"
            "not applicable. See convex-roof\n"
            "result below."
        )
    elif gme_info["status"] == "no_GME":
        sep_list = "  ".join(f"A={a}|B={b}" for a, b in gme_info["separable_cuts"])
        bip_text += f"STATUS: NO GME\n─────────────────────────────────\nSeparable cuts: {sep_list}"
    else:
        bip_text += (
            f"STATUS: GME EXISTS\n"
            f"─────────────────────────────────\n"
            f"C_GME = {gme_info['C_GME']:.8f}\n"
            f"C_GME > 0: genuine entanglement\n"
            f"confirmed across ALL cuts."
        )

    ax_tr.text(0.02, 0.97, bip_text, ha="left", va="top",
               fontsize=10, family="monospace", transform=ax_tr.transAxes)

    # ── BOTTOM: Convex-roof ────────────────────────────────────────────────
    if has_cr:
        lb  = cr_result["lower_bound"]
        ub  = cr_result["upper_bound"]
        num = cr_result["numerical"]

        all_vals = num["all_values"]
        spread   = max(all_vals) - min(all_vals) if len(all_vals) > 1 else 0.0
        tight    = spread < 0.01 * max(cr_result["C_numerical"], 1e-8)
        status_sym = {"GME": "GME EXISTS", "no_GME": "No GME",
                      "inconclusive": "Inconclusive"}.get(cr_result["status"], "")

        cr_left = (
            f"Convex-Roof C_GME\n"
            f"{'='*36}\n\n"
            f"  Rank(rho)  = {num['rank']}\n\n"
            # f"  C_LB  (lower bound)  = {lb['C_LB']:.8f}\n"
            f"  C_num (convex roof)  = {num['C_numerical']:.8f}  <- best\n"
            # f"  C_UB  (upper bound)  = {ub['C_UB']:.8f}\n\n"
            f"  Bounds consistent : {'YES' if cr_result['bounds_consistent'] else 'CHECK'}\n"
            f"  Convergence       : {'TIGHT' if tight else 'LOOSE'}  (spread={spread:.1e})\n\n"
            f"  STATUS: {status_sym}\n"
        )

        contrib_lines = "   ".join(
            f"lam={c['eigenvalue']:.4f} C={c['C_GME_pure']:.4f}"
            for c in ub["contributions"]
        )
        cr_right = (
            f"Convex-Roof Details\n"
            f"{'='*36}\n\n"
            # f"  Lower bound witness:\n"
            # f"    x={lb['best_x']}, y={lb['best_y']}\n"
            # f"    I(rho) = {lb['best_I']:.6f}\n\n"
            # f"  Upper bound (eigendecomp):\n"
            # f"    {contrib_lines}\n\n"
            f"  Numerical (L-BFGS-B, {num['n_restarts']} restarts):\n"
            f"    Converged : {num['converged']}\n"
            f"    Values    : {[round(v,4) for v in num['all_values'][:6]]}\n"
        )

        # Divider line
        line = plt.Line2D(
            [0.03, 0.97], [gs[1, 0].get_position(fig).y1 + 0.01] * 2,
            transform=fig.transFigure,
            color="gray", linewidth=0.8, linestyle="--"
        )
        fig.add_artist(line)

        ax_bl.text(0.02, 0.97, cr_left,  ha="left", va="top",
                   fontsize=10, family="monospace", transform=ax_bl.transAxes)
        ax_br.text(0.02, 0.97, cr_right, ha="left", va="top",
                   fontsize=10, family="monospace", transform=ax_br.transAxes)

    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── Separate mixed-state analysis PDF ────────────────────────────────────────

def generate_mixed_state_pdf(rdm_dict, metrics_dict, cr_results, n, output_path, system_info):
    with PdfPages(output_path) as pdf:
        _write_title_page(pdf, system_info, n)
        for sites, rho in rdm_dict.items():
            if len(sites) != n:
                continue
            if sites not in cr_results:
                continue
            info = metrics_dict[sites]
            _write_rdm_page(pdf, rho, info, sites)
            _write_combined_page(pdf, rho, info, sites, cr_result=cr_results[sites])
    print(f"  Mixed-state PDF saved → {output_path}")


# ── Public entry point ────────────────────────────────────────────────────────

def generate_gme_pdf(rdm_dict, metrics_dict, n, output_path, system_info, cr_results=None):
    """
    Generate PDF for all n-body RDMs.

    Pages per cluster:
        1. RDM heatmap  (imaginary shown only if max|Im| > 1e-6)
        2. Metrics + GME + Convex-roof  (all on ONE combined page)
    """
    with PdfPages(output_path) as pdf:
        _write_title_page(pdf, system_info, n)
        for sites, rho in rdm_dict.items():
            if len(sites) != n:
                continue
            info = metrics_dict[sites]
            cr   = cr_results.get(sites) if cr_results else None
            _write_rdm_page(pdf, rho, info, sites)
            _write_combined_page(pdf, rho, info, sites, cr_result=cr)
    print(f"  PDF saved → {output_path}")
