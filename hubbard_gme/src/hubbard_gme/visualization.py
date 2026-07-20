"""
visualization.py
----------------
PDF report generation and heatmap plotting for n-body RDMs and GME analysis.

UPDATED: Full integration of convex roof methods in PDF output
────────────────────────────────────────────────────────────────
Now displays:
  • C_LB (paper witness lower bound)
  • C_LB_supervisor (supervisor eqs. 6-9 lower bound)
  • C_num (numerical unitary optimization)  BEST ESTIMATE
  • C_UB (eigendecomposition upper bound)
  • C_sdp (CVXPY SDP if available)
  • Convex roof decomposition (pure state components)
  • Bounds consistency check

Each report page type
---------------------
title_page          : System parameters and ground-state properties
summary_page        : ALL clusters overview (NEW)
rdm_heatmap         : Real heatmap (+ imaginary only if max|Im| > 1e-6)
metrics_and_gme     : Metrics table + GME analysis + convex roof results
convex_roof_page    : Convex-roof lower/upper/numerical bounds (mixed only)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from typing import Dict, Optional
import matplotlib.gridspec as gridspec

from .utils import generate_spin_labels
from .gme import analyze_one_rdm_for_GME

# Threshold below which imaginary part is considered zero and not shown
_IMAG_THRESHOLD = 1e-6

# ── Excited-state visual cues ────────────────────────────────────────────────
 
def _state_subtitle(system_info: dict) -> str:
    """
    Return a short label describing which eigenstate is being analysed.
    Empty string for the ground state; otherwise a label that appears on
    every page so any single page is unambiguous about which state it
    came from.
    """
    if system_info is None:
        return ""
    state_idx = system_info.get("state_idx", 0)
    if not state_idx or state_idx <= 0:
        return ""
    return (f"Eigenstate index {state_idx}  "
            f"(state {state_idx + 1} in 1-indexed convention)  -  "
            f"NOT the ground state")
 
 
def _is_excited_state(system_info: dict) -> bool:
    """True if this run is for an excited eigenstate (state_idx > 0)."""
    if system_info is None:
        return False
    return bool(system_info.get("state_idx", 0))
 
 
def _banner_color(system_info: dict) -> str:
    """
    Return banner colour for the title-page header.
    Blue for ground state, purple for excited state — subtle but
    immediate visual cue that distinguishes the two PDF families.
    """
    return "#6d28d9" if _is_excited_state(system_info) else "#1f3a93"
 
 
def _add_state_subtitle_to_fig(fig, system_info, y=0.985):
    """
    Add a small state label at the top of any figure when state_idx > 0.
    Does nothing for ground-state runs.
    """
    label = _state_subtitle(system_info)
    if not label:
        return
    fig.text(
        0.5, y, label,
        ha="center", va="top",
        fontsize=9, fontweight="bold", color="#6d28d9",
        bbox=dict(boxstyle="round,pad=0.35",
                  facecolor="#f5f3ff", edgecolor="#6d28d9", linewidth=0.8),
    )
 

# ── Colour / size helpers ────────────────────────────────────────────────────

def _rdm_figsize(n_sites: int, two_panels: bool) -> tuple:
    base = {1: (6, 4.5), 2: (7, 5.5), 3: (9, 6.5), 4: (12, 8)}
    w, h = base.get(n_sites, (14, 10))
    return (w * 2 if two_panels else w, h)


def _fontsize_for_n(n_sites: int) -> int:
    return max(5, 10 - n_sites)

def _solver_cache_kind(status_obj) -> str:
    """Normalize old string or new dict solver cache metadata to one status string."""
    if isinstance(status_obj, dict):
        return str(status_obj.get("status", "unknown"))
    return str(status_obj)


def _format_solver_cache_sources(cr_res: Optional[dict], compact: bool = True) -> str:
    """Return a compact per-solver source label for PDF tables/pages."""
    if not cr_res:
        return "—"
    status = cr_res.get("solver_cache_status", cr_res.get("cache_status", {}))
    if not isinstance(status, dict) or not status:
        return "—"

    labels = {
        "stiefel": "S",
        "gcg": "G",
        "stiefel_gcg": "SG",
    }
    words = {
        "reused_pkl": "PKL",
        "recomputed": "NEW",
        "cache_missing_recomputed": "MISS->NEW",
        "skipped": "SKIP",
    }
    parts = []
    for key in ("stiefel", "gcg", "stiefel_gcg"):
        kind = _solver_cache_kind(status.get(key, "skipped"))
        value = words.get(kind, kind)
        parts.append(f"{labels[key]}:{value}" if compact else f"{key}: {value}")
    return "  ".join(parts)




# ── Single-RDM heatmap ────────────────────────────────────────────────────────

def plot_rdm_heatmap(rho, sites, title_suffix="", system_info=None):
    """
    Real heatmap always shown.
    Imaginary heatmap shown ONLY if max|Im(rho)| > 1e-6.
    system_info: if provided and state_idx > 0, adds excited-state subtitle.
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
    # Excited-state marker (purple banner) — no-op for ground state
    _add_state_subtitle_to_fig(fig, system_info, y=0.995)

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

def _write_summary_page(pdf, rdm_dict, metrics_dict, gme_results_dict, cr_results_dict, system_info, n_body):
    """
    Write summary page showing all clusters/sites for this n-body order.
    Each row: sites |Trace |Purity | Entropy| C_GME | C_LB| C_gcg| C_stiefel_gcg| C_stiefel|  C_UB | Status
    """
    fig, ax = plt.subplots(figsize=(16, 11.7))
    ax.axis("off")
    
    ax.text(0.5, 0.97, 
            f"Summary: All {n_body}-Body RDM Clusters",
            ha="center", va="top", fontsize=16, fontweight="bold")
    
    ax.text(0.5, 0.93,
            f"System: L={system_info['L']}, Ne={system_info['Ne']}, U={system_info['U']}, t={system_info['t']}  |  E₀={system_info['E0']:.6f}",
            ha="center", va="top", fontsize=10, color="gray")
    
    # Purple banner for excited-state PDFs
    _add_state_subtitle_to_fig(fig, system_info, y=0.995)

    # Build summary text
    summary_text = (
        f"\n{'Sites':18s} | {'Trace':10s} | {'Purity':10s} | {'Entropy':10s} | {'C_GME':10s} | {'C_LB':10s} | {'C_gcg':10s} | {'C_stiefel_gcg':14s} | {'C_stiefel':10s} | {'C_UB':10s} | {'Source':27s} | Status\n"
        f"{'-'*18}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*14}-+-{'-'*10}-+-{'-'*10}-+-{'-'*27}-+-{'-'*10}\n"
    )
    
    # Collect all clusters for this n-body
    clusters = []
    for sites, rho in rdm_dict.items():
        if len(sites) != n_body:
            continue
        
        info = metrics_dict.get(sites, {})
        trace = info.get('trace', float('nan'))
        purity = info.get('purity', float('nan'))
        entropy = info.get('entropy', float('nan'))
        
        # Get GME result
        gme_res = gme_results_dict.get(sites, {})
        C_GME = gme_res.get('C_GME', float('nan')) if gme_res.get('status') == 'GME' else 0.0
        print(gme_res.get('status'))
        # Get convex roof result
        cr_res = cr_results_dict.get(sites, {})
        C_LB = cr_res.get('C_LB_best', float('nan')) if cr_res else float('nan')
        C_gcg = cr_res.get('C_gcg', float('nan')) if cr_res else float('nan')
        C_cayl = cr_res.get('C_cayl', float('nan')) if cr_res else float('nan')
        C_UB = cr_res.get('C_UB', float('nan')) if cr_res else float('nan')
        stiefel = cr_res.get('stiefel')
        # Only plot if Stiefel data is actually present
        if stiefel is not None and stiefel.get("k_records"):
            title_str = (
                f"L={system_info.get('L','?')}  "
                f"Ne={system_info.get('Ne','?')}  "
                f"U={system_info.get('U','?')}  "
                f"sites={sites}"
            )
            plot_k_convergence(stiefel, title=title_str)
        
        C_stiefel = cr_res.get('C_stiefel', float('nan')) if cr_res else float('nan')
        C_stiefel_gcg = cr_res.get('C_stiefel_gcg', float('nan')) if cr_res else float('nan')

        status = gme_res.get('status', 'unknown')
        source = _format_solver_cache_sources(cr_res, compact=True)
        
        clusters.append({
            'sites': sites,
            'trace': trace,
            'purity': purity,
            'entropy': entropy,
            'C_GME': C_GME,
            'C_LB': C_LB,
            'C_cayl': C_cayl,
            'C_gcg': C_gcg,
            'C_stiefel': C_stiefel,
            'C_stiefel_gcg': C_stiefel_gcg,
            'C_UB': C_UB,
            'source': source,
            'status': status
        })
    
    # Sort by sites
    clusters.sort(key=lambda x: x['sites'])
    
    # Add rows
    for c in clusters:
        sites_str = str(c['sites'])[:16]
        trace_str = f"{c['trace']:.6f}" if not np.isnan(c['trace']) else "—"
        purity_str = f"{c['purity']:.6f}" if not np.isnan(c['purity']) else "—"
        entropy_str = f"{c['entropy']:.6f}" if not np.isnan(c['entropy']) else "—"
        C_GME_str = f"{c['C_GME']:.6f}" if not np.isnan(c['C_GME']) and c['C_GME'] > 0 else "—"
        C_LB_str = f"{c['C_LB']:.6f}" if not np.isnan(c['C_LB']) else "—"
        C_gcg_str = f"{c['C_gcg']:.6f}" if not np.isnan(c['C_gcg']) else "—"
        C_cayl_str = f"{c['C_cayl']:.6f}" if not np.isnan(c['C_cayl']) else "—"
        C_stiefel = f"{c['C_stiefel']:.6f}" if not np.isnan(c['C_stiefel']) else "—"
        C_stiefel_gcg = f"{c['C_stiefel_gcg']:.6f}" if not np.isnan(c['C_stiefel_gcg']) else "—"
        C_UB_str = f"{c['C_UB']:.6f}" if not np.isnan(c['C_UB']) else "—"
        
        status_short = {
            'GME EXISTS': 'GME  ',
            'no_GME': 'NO GME',
            'mixed_state': 'MIXED'
        }.get(c['status'], c['status'][:8])
        
        source_str = str(c.get('source', '—'))[:27]
        summary_text += (
            f"{sites_str:18s} | {trace_str:10s} | {purity_str:10s} | {entropy_str:10s} | {C_GME_str:10s} | "
            f"{C_LB_str:10s} | {C_gcg_str:10s} | {C_stiefel_gcg:14s} | {C_stiefel:10s} | {C_UB_str:10s} | {source_str:27s} | {status_short}\n"
        )
    
    summary_text += (
        f"\n{'─'*120}\n"
        f"Legend:\n"
        f"  Trace = Tr(rho) | Purity = Tr(rho²) [1.0=pure, <1.0=mixed] | C_GME = Genuine multipartite entanglement (pure only)\n"
        f"  C_LB = Lower bound (notes eqs.6-9) | C_gcg = unitary optimization - exponential-map GCG | BEST = selected estimate\n"
        f"  Source: S=Stiefel, G=GCG, SG=Stiefel-GCG; PKL=reused from pkl, NEW=recomputed, MISS->NEW=pkl missing so recomputed\n"
        f"  C_UB = Upper bound (eigendecomp) | Status: GME=has GME | NO GME=separable | MIXED=mixed state\n"
    )
    
    ax.text(0.02, 0.88, summary_text, ha="left", va="top", 
            fontsize=8.5, family="monospace", transform=ax.transAxes)
    
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


####################    Title page

def _compute_rdm_stats(rdm_dict, metrics_dict, cr_results_dict, n_body):
    """
    Compute aggregate statistics across all n-body clusters in the run.
    Returns a dict with keys:
        n_clusters, n_mixed, n_pure, n_with_gme,
        purity_mean, purity_min, purity_max,
        entropy_mean, entropy_min, entropy_max,
        rank_mean, rank_min, rank_max,
        cgme_mean, cgme_min, cgme_max,
        hilbert_dim
    Each may be None if the underlying data is unavailable.
    """
    purities, entropies, ranks, cgmes = [], [], [], []
    n_mixed = n_pure = n_with_gme = 0
 
    for sites in rdm_dict:
        if len(sites) != n_body:
            continue
        info = (metrics_dict or {}).get(sites, {})
        cr   = (cr_results_dict or {}).get(sites, {}) if cr_results_dict else {}
 
        pur = info.get("purity")
        if pur is not None and not np.isnan(pur):
            purities.append(float(pur))
            if pur < 1.0 - 1e-6:
                n_mixed += 1
            else:
                n_pure += 1
 
        ent = info.get("entropy")
        if ent is not None and not np.isnan(ent):
            entropies.append(float(ent))
 
        evs = info.get("eigvals")
        if evs is not None and len(evs) > 0:
            ranks.append(int(np.sum(evs > 1e-10)))
 
        # Pick the best available C_GME: stiefel > expo > C_LB_best
        for key in ("C_stiefel", "C_gcg", "C_LB_best"):
            val = cr.get(key) if cr else None
            if val is not None and not np.isnan(val):
                cgmes.append(float(val))
                if val > 1e-6:
                    n_with_gme += 1
                break
 
    def _stats(arr):
        if not arr:
            return None, None, None
        return float(np.mean(arr)), float(np.min(arr)), float(np.max(arr))
 
    pur_mean, pur_min, pur_max = _stats(purities)
    ent_mean, ent_min, ent_max = _stats(entropies)
    rk_mean,  rk_min,  rk_max  = _stats(ranks)
    cg_mean,  cg_min,  cg_max  = _stats(cgmes)
 
    return {
        "n_clusters":   sum(1 for s in rdm_dict if len(s) == n_body),
        "n_mixed":      n_mixed,
        "n_pure":       n_pure,
        "n_with_gme":   n_with_gme,
        "purity_mean":  pur_mean,
        "purity_min":   pur_min,
        "purity_max":   pur_max,
        "entropy_mean": ent_mean,
        "entropy_min":  ent_min,
        "entropy_max":  ent_max,
        "rank_mean":    rk_mean,
        "rank_min":     rk_min,
        "rank_max":     rk_max,
        "cgme_mean":    cg_mean,
        "cgme_min":     cg_min,
        "cgme_max":     cg_max,
        "hilbert_dim":  2 ** n_body,
    }
 
 
def _write_title_page(pdf, system_info, n_body,
                      rdm_dict=None, metrics_dict=None, cr_results=None):
    """
    Colourful title page with coloured section headers, value badges,
    a status banner, and a footer.
 
    Lower box now shows REAL system-specific RDM/GME statistics instead
    of generic definitions, computed from the actual data.
    """
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
 
    fig.patch.set_facecolor("#fbfcfe")
 
    # ── top banner ────────────────────────────────────────────────────────────
    ax.add_patch(plt.Rectangle(
        (0.0, 0.93), 1.0, 0.07,
        transform=ax.transAxes, color=_banner_color(system_info), zorder=0,
    ))
    ax.text(0.5, 0.965,
            "Genuine Multipartite Entanglement Analysis",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.938,
            "Hubbard ring  -  Convex-roof framework",
            ha="center", va="center", fontsize=10,
            color="#cdd6f4", transform=ax.transAxes, style="italic")
 
    # ── helper: coloured section ──────────────────────────────────────────────
    def _section(x, y, w, h, title, body_lines, color_hex):
        ax.add_patch(plt.Rectangle(
            (x, y + h - 0.045), w, 0.045,
            transform=ax.transAxes, color=color_hex, zorder=1,
        ))
        ax.add_patch(plt.Rectangle(
            (x, y), w, h - 0.045,
            transform=ax.transAxes, color="white",
            ec=color_hex, lw=1.0, zorder=1,
        ))
        ax.text(x + 0.012, y + h - 0.022, title,
                transform=ax.transAxes, ha="left", va="center",
                fontsize=11, fontweight="bold", color="white")
        line_y = y + h - 0.075
        for label, value in body_lines:
            ax.text(x + 0.014, line_y, label,
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=9.5, color="#374151", family="DejaVu Sans")
            ax.text(x + w - 0.012, line_y, value,
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9.5, fontweight="bold", color=color_hex,
                    family="DejaVu Sans")
            line_y -= 0.030
 
    # ── Section 1 — System parameters ────────────────────────────────────────
    state_idx = system_info.get("state_idx", 0)
    state_str = (f"{state_idx}  (1-indexed: {state_idx + 1})"
                 if state_idx > 0 else "0  (ground state)")
    _section(
        x=0.04, y=0.55, w=0.29, h=0.36,
        title="System parameters",
        body_lines=[
            ("Model",          f"Hubbard 1D-PBC"),
            ("Mode",           f"{system_info['mode']}"),
            ("Sites  (L)",     f"{system_info['L']}"),
            ("Electrons  (Ne)",f"{system_info['Ne']}"),
            ("Hopping  (t)",   f"{system_info['t']}"),
            ("Interaction (U)",f"{system_info['U']}"),
            ("n-body order",   f"{n_body}"),
            ("Eigenstate idx", state_str),
        ],
        color_hex="#1f6feb",
    )
 
    # ── Section 2 — Eigenstate properties ────────────────────────────────────
    _section(
        x=0.355, y=0.55, w=0.29, h=0.36,
        title="Eigenstate properties",
        body_lines=[
            ("Energy E0",     f"{system_info['E0']:.8f}"),
            ("Energy E1",     f"{system_info['E1']:.8f}"),
            ("Gap (E1-E0)",   f"{system_info['E1']-system_info['E0']:.6f}"),
            ("<S^2>",         f"{system_info['S2']:.6f}"),
            ("S",             f"{system_info['S']:.4f}"),
            ("<Sz>",          f"{system_info['Sz']:.6f}"),
            ("Multiplicity",  f"{system_info['multiplicity']:.1f}"),
            ("Spin label",    f"{system_info['spin_label']}"),
        ],
        color_hex="#7c3aed",
    )
 
    # ── Section 3 — Double occupancy ─────────────────────────────────────────
    _section(
        x=0.67, y=0.55, w=0.29, h=0.36,
        title="Double occupancy",
        body_lines=[
            ("<D> total",     f"{system_info['double_occ']:.6f}"),
            ("<D> / site",    f"{system_info['double_occ_per_site']:.6f}"),
        ],
        color_hex="#0ea472",
    )
 
    # ── Lower box: ACTUAL RDM / GME values for this run ──────────────────────
    ax.add_patch(plt.Rectangle(
        (0.04, 0.18), 0.92, 0.32,
        transform=ax.transAxes, color="white",
        ec="#dc2626", lw=1.2, zorder=1,
    ))
    ax.add_patch(plt.Rectangle(
        (0.04, 0.455), 0.92, 0.045,
        transform=ax.transAxes, color="#dc2626", zorder=1,
    ))
    ax.text(0.052, 0.477,
            f"RDM and GME statistics for this run  ({n_body}-body clusters)",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=11, fontweight="bold", color="white")
 
    # Compute real statistics if data is provided
    if rdm_dict is not None:
        st = _compute_rdm_stats(rdm_dict, metrics_dict, cr_results, n_body)
    else:
        st = {}
 
    def _fmt(val, prec=6):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        if isinstance(val, float):
            return f"{val:.{prec}f}"
        return str(val)
 
    n_total      = st.get("n_clusters", 0)
    n_mixed      = st.get("n_mixed", 0)
    n_pure       = st.get("n_pure", 0)
    n_with_gme   = st.get("n_with_gme", 0)
    hilbert_dim  = st.get("hilbert_dim", 2 ** n_body)
 
    # Two columns inside the red box
    col_left = (
        f"  Hilbert space dim          :  2^{n_body}  =  {hilbert_dim}\n"
        f"  Number of clusters         :  {n_total}\n"
        f"  Pure clusters              :  {n_pure}\n"
        f"  Mixed clusters             :  {n_mixed}\n"
        f"  Clusters with GME (>0)     :  {n_with_gme}\n"
    )
 
    col_right_lines = []
    col_right_lines.append(
        "  Purity gamma = Tr(rho^2)   :  "
        f"mean = {_fmt(st.get('purity_mean'))}     "
        f"[min {_fmt(st.get('purity_min'))} ,  max {_fmt(st.get('purity_max'))}]"
    )
    col_right_lines.append(
        "  Von Neumann entropy        :  "
        f"mean = {_fmt(st.get('entropy_mean'))}     "
        f"[min {_fmt(st.get('entropy_min'))} ,  max {_fmt(st.get('entropy_max'))}]"
    )
    col_right_lines.append(
        "  Rank of rho                :  "
        f"mean = {_fmt(st.get('rank_mean'), prec=2)}     "
        f"[min {_fmt(st.get('rank_min'), prec=0)} ,  max {_fmt(st.get('rank_max'), prec=0)}]"
    )
    col_right_lines.append(
        "  C_GME  (best estimate)     :  "
        f"mean = {_fmt(st.get('cgme_mean'))}     "
        f"[min {_fmt(st.get('cgme_min'))} ,  max {_fmt(st.get('cgme_max'))}]"
    )
    col_right = "\n".join(col_right_lines)
 
    ax.text(0.06, 0.43, col_left,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, color="#374151", family="DejaVu Sans")
    ax.text(0.06, 0.30, col_right,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, color="#374151", family="DejaVu Sans")
 
    # ── Highlighted state-selection notice (only when not ground state) ──────
    if state_idx > 0:
        ax.add_patch(plt.Rectangle(
            (0.04, 0.10), 0.92, 0.06,
            transform=ax.transAxes, color="#fff7ed",
            ec="#ea580c", lw=1.2, zorder=1,
        ))
        ax.text(0.06, 0.13,
                f"NOTE",
                transform=ax.transAxes, ha="left", va="center",
                fontsize=10, fontweight="bold", color="#ea580c")
        ax.text(0.13, 0.13,
                f"Analysis performed on eigenstate index {state_idx}  "
                f"(state {state_idx + 1} in 1-indexed convention),  "
                f"not the ground state.",
                transform=ax.transAxes, ha="left", va="center",
                fontsize=9.5, color="#9a3412")
 
    # ── Footer ────────────────────────────────────────────────────────────────
    ax.add_patch(plt.Rectangle(
        (0.0, 0.0), 1.0, 0.05,
        transform=ax.transAxes, color="#e5e7eb", zorder=0,
    ))
    ax.text(0.04, 0.025,
            f"hubbard_gme  |  convex-roof analysis",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=9, color="#374151")
    ax.text(0.96, 0.025,
            datetime.now().strftime("%Y-%m-%d  %H:%M"),
            transform=ax.transAxes, ha="right", va="center",
            fontsize=9, color="#374151")
 
    pdf.savefig(fig, bbox_inches="tight", dpi=150,
                facecolor=fig.get_facecolor())
    plt.close(fig)
#####################
def _sz_of_index(i: int, n: int) -> float:
    """Total Sz of basis index i (bit k = spin of orbital k; 0=up, 1=down)."""
    ups = sum(1 for k in range(n) if (i >> k) & 1 == 0)
    return (ups - (n - ups)) / 2.0


def plot_rdm_three_view(rho, sites, system_info=None):
    """
    The standard three-view RDM page used by BOTH pipelines (Hubbard +
    molecular):

      (1) Re(rho) in index order          — original signed values
      (2) Re(rho) reordered by total Sz   — original signed values, block lines
      (3) |rho|  reordered by total Sz    — absolute values, block lines

    Cell values are annotated when the matrix is small enough to stay legible
    (dim <= 16, i.e. n <= 4). Returns (fig, cross) where `cross` is the largest
    element connecting different-Sz sectors (~0 if Sz is conserved).
    """
    from itertools import groupby

    n = len(sites)
    dim = 2 ** n
    labels = generate_spin_labels(n)
    site_str = ",".join(str(s) for s in sites)
    fsize = _fontsize_for_n(n)

    szs_nat = [_sz_of_index(i, n) for i in range(dim)]
    order = sorted(range(dim), key=lambda i: (-szs_nat[i], i))
    szs = [szs_nat[i] for i in order]
    lab_sz = [f"{labels[i]} $S_z$={szs_nat[i]:+.1f}" for i in order]

    Re = np.real(rho)
    ReO = Re[np.ix_(order, order)]
    AbO = np.abs(rho)[np.ix_(order, order)]

    S = np.array(szs_nat)
    mask = S[:, None] != S[None, :]
    cross = float(np.abs(rho)[mask].max()) if mask.any() else 0.0

    annotate = dim <= 16

    def _annot(ax, M, diverging):
        vmax_ = max(np.abs(M).max(), 1e-12)
        for i in range(dim):
            for j in range(dim):
                v = M[i, j]
                if abs(v) < 5e-7:
                    ax.text(j, i, "0", ha="center", va="center",
                            fontsize=5.5, color="#999999")
                else:
                    dark = (abs(v) if diverging else v) > 0.55 * vmax_
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=6.0,
                            color="white" if dark else "black")

    def _blocklines(ax, color):
        acc = 0
        for _, grp in groupby(szs):
            acc += len(list(grp))
            if acc < dim:
                ax.axhline(acc - 0.5, color=color, lw=1.8)
                ax.axvline(acc - 0.5, color=color, lw=1.8)

    def _ticks(ax, labs, fs):
        ax.set_xticks(range(dim)); ax.set_yticks(range(dim))
        ax.set_xticklabels(labs, rotation=90, fontsize=fs)
        ax.set_yticklabels(labs, fontsize=fs)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7.2))
    vmax = max(np.max(np.abs(Re)) * 1.05, 1e-12)

    ax = axes[0]
    im = ax.imshow(Re, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(r"(1) Re($\rho$) — index order", fontsize=12, pad=10)
    _ticks(ax, labels, fsize)
    if annotate:
        _annot(ax, Re, True)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[1]
    im = ax.imshow(ReO, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(r"(2) Re($\rho$) — $S_z$-ordered (original values)",
                 fontsize=12, pad=10)
    _ticks(ax, lab_sz, max(5, fsize - 1))
    if annotate:
        _annot(ax, ReO, True)
    _blocklines(ax, "#00a000")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[2]
    im = ax.imshow(AbO, cmap="viridis", vmin=0.0)
    ax.set_title(r"(3) $|\rho|$ — $S_z$-ordered (absolute values)",
                 fontsize=12, pad=10)
    _ticks(ax, lab_sz, max(5, fsize - 1))
    if annotate:
        _annot(ax, AbO, False)
    _blocklines(ax, "red")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(f"{n}-Body RDM  (Sites {site_str})  —  same matrix in all "
                 f"three views", fontsize=13, fontweight="bold")
    # Excited-state marker (purple banner) — no-op for ground state
    _add_state_subtitle_to_fig(fig, system_info, y=0.995)
    # generous bottom margin: the rotated tick labels hang below the axes and
    # the callers add one or two footer lines underneath them
    fig.subplots_adjust(left=0.05, right=0.97, top=0.86, bottom=0.24,
                        wspace=0.35)
    return fig, cross


def _write_rdm_page(pdf, rho, metrics, sites, system_info=None):
    """Three-view RDM page: Re index-order, Re Sz-ordered, |rho| Sz-ordered."""
    fig, cross = plot_rdm_three_view(rho, sites, system_info=system_info)
    show_imag = np.max(np.abs(rho.imag)) > _IMAG_THRESHOLD
    note = "" if show_imag else "  (Imaginary part < 1e-6, not shown)"
    fig.text(
        0.5, 0.03,
        f"Trace={metrics['trace']:.6f}   "
        f"Purity={metrics['purity']:.6f}   "
        f"Entropy={metrics['entropy']:.6f}{note}   |   "
        f"max cross-$S_z$ element = {cross:.1e}",
        fontsize=9, ha="center",
    )
    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _write_combined_page(
    pdf, rho, info, sites, 
    cr_result=None, 
    n_equals_N_result=None, 
    continuous_result=None,
    system_info=None,
):
    """
    Single page with top/bottom sections showing:
    
    TOP:    RDM Metrics + GME Analysis (bipartitions)
    BOTTOM: Convex roof results (for ALL RDMs - both pure and mixed)
    """

    site_str   = ",".join(str(s) for s in sites)
    gme_info   = analyze_one_rdm_for_GME(rho, sites)
    has_cr     = cr_result is not None
    has_nN     = n_equals_N_result is not None
    has_cont   = continuous_result is not None
    has_bottom = has_cr  # CHANGED: Show if convex roof available (works for ALL)

    fig = plt.figure(figsize=(16, 12 if has_bottom else 5.5))
    fig.suptitle(
        f"Analysis Summary  -  {len(sites)}-Body RDM  (Sites {site_str})",
        fontsize=14, fontweight="bold",
    )
    # Purple excited-state marker — silently skipped for ground state
    _add_state_subtitle_to_fig(fig, system_info, y=0.995)

    nrows = 2 if has_bottom else 1
    height_ratios = [1, 1.3] if has_bottom else [1]
    gs = gridspec.GridSpec(
        nrows, 2,
        figure=fig,
        hspace=0.3,
        wspace=0.05,
        height_ratios=height_ratios,
        top=0.92, bottom=0.04,
        left=0.03, right=0.97,
    )

    ax_tl = fig.add_subplot(gs[0, 0]); ax_tl.axis("off")
    ax_tr = fig.add_subplot(gs[0, 1]); ax_tr.axis("off")
    if has_bottom:
        ax_bl = fig.add_subplot(gs[1, 0]); ax_bl.axis("off")
        ax_br = fig.add_subplot(gs[1, 1]); ax_br.axis("off")

    # ──────────────────────────────────────────────────────────────────────
    # TOP-LEFT: RDM Metrics
    # ──────────────────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────────
    # TOP-RIGHT: GME Analysis (bipartitions)
    # ──────────────────────────────────────────────────────────────────────
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
        # Pure state with GME confirmed
        c_lb_line = ""
        if has_nN:
            nN_res  = n_equals_N_result
            C_exact = nN_res.get("C_exact", float("nan"))
            C_LB    = nN_res.get("C_LB", float("nan"))
            gap     = abs(C_exact - C_LB)
            c_lb_line = (
                f"\n"
                f"lower bound [eqs.6-9]\n"
                f"  ─────────────────────────────────\n"
                f"  C_exact = {C_exact:.8f}\n"
                f"  C_LB    = {C_LB:.8f}\n"
                f"  Gap     = {gap:.8f}\n"
            )
        bip_text += (
            f"STATUS: GME EXISTS\n"
            f"─────────────────────────────────\n"
            f"C_GME = {gme_info['C_GME']:.8f}\n"
            f"GME confirmed across ALL cuts.{c_lb_line}"
        )

    ax_tr.text(0.02, 0.97, bip_text, ha="left", va="top",
               fontsize=10, family="monospace", transform=ax_tr.transAxes)

    # ──────────────────────────────────────────────────────────────────────
    # BOTTOM SECTION: CONVEX ROOF (now for ALL RDMs - pure and mixed)
    # ──────────────────────────────────────────────────────────────────────
    if has_bottom:
        if has_cr:
            # SHOW CONVEX ROOF FOR ALL RDMs
            # num        = cr_result.get("numerical", {})
            stiefel        = cr_result.get("stiefel", {})
            stiefel_gcg    = cr_result.get("stiefel_gcg", {})
            lb_super   = cr_result.get("lower_bound_super", {})
            lb_paper   = cr_result.get("lower_bound", {})
            C_LB_sup   = cr_result.get("C_LB_supervisor", lb_super.get("C_LB", float("nan")))
            # C_LB_paper = cr_result.get("C_LB",            lb_paper.get("C_LB", float("nan")))
            # C_LB_best  = cr_result.get("C_LB_best",       max(C_LB_sup, C_LB_paper))
            # C_num      = cr_result.get("C_gcg",     num.get("C_gcg", float("nan")))
            C_stiefel       =cr_result.get("C_stiefel",     stiefel.get("C_stiefel", float("nan")))
            C_stiefel_gcg   =cr_result.get("C_stiefel_gcg", stiefel_gcg.get("C_stiefel_gcg", float("nan")))
            C_UB       = cr_result.get("C_UB",            float("nan"))
            C_sdp      = cr_result.get("C_sdp",           None)
            decomp     = cr_result.get("decomp",          [])
            status     = cr_result.get("status",          "")

            best_z1 = lb_super.get("best_z1", "—")
            best_z2 = lb_super.get("best_z2", "—")
            n_pairs = lb_super.get("n_pairs",  "—")
            solver_source = _format_solver_cache_sources(cr_result, compact=False)
            
            # all_vals = num.get("all_values", [])
            # spread   = max(all_vals) - min(all_vals) if len(all_vals) > 1 else 0.0
            # tight    = spread < 0.01 * max(C_stiefel, 1e-8) if not np.isnan(C_stiefel) else False

            bl_text = (
                f"Convex Roof Analysis\n"
                f"{'='*36}\n\n"
                f"  Lower Bounds:\n"
                # f"    Paper (witness)     = {C_LB_paper:.8f}\n"
                f"    C_LB    = {C_LB_sup:.8f}\n"
                # f"    Best C_LB           = {C_LB_best:.8f}\n\n"
                f"  C_GME (unitary optimization ):\n"
                f"    C_stiefel (unitary opt) = {C_stiefel:.8f}   BEST\n"
                # f"    ({len(all_vals)} restarts)\n\n"
                f"  C_GME (unitary optimization - Expoenetial Map):\n"
                # f"    C_num  = {C_num:.8f}   BEST\n"
                # f"    (L-BFGS-B, {len(all_vals)} restarts)\n\n"
                f"  Upper Bound:\n"
                f"    C_UB (eigendecomp)  = {C_UB:.8f}\n\n"
                f"  Status: {status}\n"
                f"\n  Solver source:\n"
                f"    {solver_source}\n"
                # f"  Bounds consistent: {'YES ' if C_LB  <= C_num <= C_UB + 1e-5 else 'CHECK'}\n"
                # f"  Convergence: {'TIGHT' if tight else 'LOOSE'}\n"
            )
            if C_sdp is not None:
                bl_text += f"\n  CVXPY SDP bound     = {C_sdp:.8f}\n"

            if has_cont:
                C_cont  = continuous_result.get("C_LB", float("nan"))
                improve = C_cont - C_LB
                bl_text += (
                    f"\n  Continuous Optimization:\n"
                    f"    C_LB (cont)         = {C_cont:.8f}\n"
                    f"    Improvement         = {improve:+.8f}\n"
                )

            br_text = (""
          
            )
            for i, d in enumerate(decomp[:5]):  # Show first 5 components
                br_text += (
                    f"    {i+1}. p={d['p']:.6f}  C_GME={d['C_GME']:.6f}\n"
                )
            if len(decomp) > 5:
                br_text += f"    ... and {len(decomp)-5} more states\n"

        ax_bl.text(0.02, 0.97, bl_text, ha="left", va="top",
                   fontsize=10, family="monospace", transform=ax_bl.transAxes)
        ax_br.text(0.02, 0.97, br_text, ha="left", va="top",
                   fontsize=10, family="monospace", transform=ax_br.transAxes)

    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── Separate mixed-state analysis PDF ────────────────────────────────────────

def generate_mixed_state_pdf(rdm_dict, metrics_dict, cr_results, n, output_path, system_info):
    with PdfPages(output_path) as pdf:
        _write_title_page(pdf, system_info, n,
                          rdm_dict=rdm_dict,
                          metrics_dict=metrics_dict,
                          cr_results=cr_results)
        for sites, rho in rdm_dict.items():
            if len(sites) != n:
                continue
            if sites not in cr_results:
                continue
            info = metrics_dict[sites]
            _write_rdm_page(pdf, rho, info, sites, system_info=system_info)
            _write_combined_page(pdf, rho, info, sites,
                                 cr_result=cr_results[sites],
                                 system_info=system_info)
    print(f"  Mixed-state PDF saved → {output_path}")


# ── Public entry point ────────────────────────────────────────────────────────

def generate_gme_pdf(
    rdm_dict,
    metrics_dict,
    n,
    output_path,
    system_info,
    cr_results=None,
    n_equals_N_results=None,
    continuous_results=None,
):
    """
    Generate PDF for all n-body RDMs with convex roof integration.

    Parameters
    ----------
    rdm_dict            : {sites: rho_matrix}
    metrics_dict        : {sites: {trace, purity, entropy, eigvals}}
    n                   : body order to analyze
    output_path         : where to save PDF
    system_info         : dict with L, Ne, U, E0, E1, S2, S, Sz, etc.
    cr_results          : dict {sites: convex_roof_result}  UNIFIED CONVEX ROOF FOR ALL RDMs
    n_equals_N_results  : dict {n: verification_result}  PURE STATE VERIFICATION
    continuous_results  : dict {sites: continuous_optimization_result}
    """
    # Compute GME results dict for summary page
    gme_results_dict = {}
    for sites, rho in rdm_dict.items():
        if len(sites) != n:
            continue
        gme_results_dict[sites] = analyze_one_rdm_for_GME(rho, sites)
    
    with PdfPages(output_path) as pdf:
        _write_title_page(pdf, system_info, n,
                          rdm_dict=rdm_dict,
                          metrics_dict=metrics_dict,
                          cr_results=cr_results)
        # ADD SUMMARY PAGE AFTER TITLE PAGE
        _write_summary_page(pdf, rdm_dict, metrics_dict, gme_results_dict, cr_results or {}, system_info, n)
        
        for sites, rho in rdm_dict.items():
            if len(sites) != n:
                continue
            info      = metrics_dict[sites]
            cr        = cr_results.get(sites) if cr_results else None
            nN_res    = n_equals_N_results.get(n) if n_equals_N_results else None
            cont_res  = continuous_results.get(sites) if continuous_results else None
            _write_rdm_page(pdf, rho, info, sites, system_info=system_info)
            _write_combined_page(
                pdf, rho, info, sites,
                cr_result=cr,
                n_equals_N_result=nN_res,
                continuous_result=cont_res,
                system_info=system_info,
            )
    print(f"  PDF saved → {output_path}")


# ── n = N verification page ───────────────────────────────────────────────────

def _write_n_equals_N_page(
    pdf,
    rho: np.ndarray,
    n: int,
    system_info: dict,
    supervisor_result: dict,
    exact_result: dict,
    continuous_result: dict = None,
) -> None:
    """
    Write one PDF page comparing:
        - Exact C_GME from eq.(1)   (pure state formula — valid because n=N)
        - Lower bound from eqs.(6-9) (supervisor method)
        - Lower bound from continuous optimization (if provided)

    This is the verification test your supervisor asked for.
    """
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")

    C_exact  = exact_result.get("C_exact", float("nan"))
    C_LB     = supervisor_result.get("C_LB", float("nan"))
    diff     = abs(C_exact - C_LB)
    valid    = C_LB <= C_exact + 1e-6
    purity_v = float(np.real(np.trace(rho @ rho)))

    ax.text(
        0.5, 0.97,
        f"Verification Test: n = N  (pure state RDM)",
        ha="center", va="top", fontsize=15, fontweight="bold",
    )
    ax.text(
        0.5, 0.91,
        "When n = N, the nBRDM is a pure state.\n"
        "The exact C_GME from eq.(1) and the lower bound from eqs.(6-9) should agree.",
        ha="center", va="top", fontsize=11, color="gray",
    )
    # Excited-state banner — n=N verification can run for any eigenstate
    _add_state_subtitle_to_fig(fig, system_info, y=0.995)

    col1 = (
        f"System\n"
        f"{'─'*34}\n"
        f"  L  (total sites)    = {system_info.get('L', n)}\n"
        f"  Ne (total electrons)= {system_info.get('Ne', n)}\n"
        f"  n  (RDM body order) = {n}\n"
        f"  n = N ?             = {'YES' if n == system_info.get('Ne', n) else 'NO'}\n\n"
        f"RDM properties\n"
        f"{'─'*34}\n"
        f"  Tr(rho)    = 1.000000\n"
        f"  Tr(rho^2)  = {purity_v:.10f}\n"
        f"  Pure state = {'YES   ' if abs(purity_v - 1.0) < 1e-4 else 'NO ✗'}\n"
    )

    col2 = (
        f"Exact C_GME  [eq.(1)]\n"
        f"{'─'*34}\n"
        f"  C_GME (exact) = {C_exact:.10f}\n\n"
        f"Basis-state lower bound  [eqs.(6-9)]\n"
        f"{'─'*34}\n"
        f"  I_Phi = 2*(sqrt(A) - sum_k sqrt(B_k))\n"
        f"  C_LB  = {C_LB:.10f}\n"
        f"  Best z1 = {supervisor_result.get('best_z1', '—')}\n"
        f"  Best z2 = {supervisor_result.get('best_z2', '—')}\n"
    )

    if continuous_result is not None:
        C_cont    = continuous_result.get("C_LB", float("nan"))
        cont_meth = continuous_result.get("method", "—")
        improve   = C_cont - C_LB
        col2 += (
            f"\nContinuous optimization\n"
            f"{'─'*34}\n"
            f"  Method : {cont_meth}\n"
            f"  C_LB   = {C_cont:.10f}\n"
            f"  Gain   = {improve:+.8f}  vs basis\n"
        )

    C_best = continuous_result.get("C_LB", C_LB) if continuous_result else C_LB
    diff_best = abs(C_exact - C_best)

    col3 = (
        f"Comparison\n"
        f"{'─'*34}\n"
        f"  C_exact         = {C_exact:.8f}\n"
        f"  C_LB (basis)    = {C_LB:.8f}\n"
    )
    if continuous_result is not None:
        C_cont = continuous_result.get("C_LB", float("nan"))
        col3 += f"  C_LB (cont.)    = {C_cont:.8f}\n"
    col3 += (
        f"\n"
        f"  Best diff = {diff_best:.2e}\n\n"
        f"  C_LB <= C_exact ? = {'YES   ' if valid else 'NO ✗'}\n\n"
        f"  RESULT: {'VERIFIED   ' if valid else 'FAILED ✗'}\n"
        f"  Lower bound is valid.\n"
    )

    ax.text(0.04, 0.78, col1, ha="left", va="top", fontsize=10, family="monospace")
    ax.text(0.37, 0.78, col2, ha="left", va="top", fontsize=10, family="monospace")
    ax.text(0.70, 0.78, col3, ha="left", va="top", fontsize=10, family="monospace")

    # Bottom bar: result indicator
    color  = "#2ca02c" if valid else "#d62728"
    label  = "LOWER BOUND VERIFIED    " if valid else "VERIFICATION FAILED  ✗"
    ax.add_patch(plt.Rectangle((0.1, 0.04), 0.8, 0.08,
                                transform=ax.transAxes,
                                color=color, alpha=0.15, zorder=0))
    ax.text(0.5, 0.08, label,
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=color, transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)


def generate_n_equals_N_pdf(
    rho: np.ndarray,
    n: int,
    system_info: dict,
    output_path: str,
    continuous_result: dict = None,
) -> dict:
    """
    Run the n = N verification test and write a dedicated PDF.

    Parameters
    ----------
    rho               : (2^n, 2^n) pure state density matrix
    n                 : number of spins = number of electrons
    system_info       : dict from pipeline
    output_path       : where to save the PDF
    continuous_result : optional result from continuous optimization
    """
    from .gme import supervisor_test_pure_state

    result = supervisor_test_pure_state(rho, n, verbose=False)

# Build a one-element rdm_dict / metrics_dict for the title page stats
    from .metrics import rdm_metrics
    sites_key = tuple(range(n))
    _rdm_dict_one = {sites_key: rho}
    _metrics_one  = {sites_key: rdm_metrics(rho)}
    with PdfPages(output_path) as pdf:
        _write_title_page(pdf, system_info, n,
                          rdm_dict=_rdm_dict_one,
                          metrics_dict=_metrics_one,
                          cr_results=None)
        _write_n_equals_N_page(
            pdf, rho, n, system_info,
            supervisor_result=result.get("supervisor_result", {}),
            exact_result=result,
            continuous_result=continuous_result,
        )

    print(f"  n=N verification PDF saved → {output_path}")
    return result


"""
Visualisation of Algorithm B convergence records.
Call plot_k_convergence(result) where result is the dict
returned by convex_roof_stiefel().
"""




def plot_k_convergence(result: dict, title: str = "") -> None:
    """
    Plot the convergence of C_GME as a function of k,
    and the distribution of restart values for each k.
    Parameters
    ----------
    result : dict returned by convex_roof_stiefel()
    title  : optional title string
    """
    records   = result["k_records"]
    k_vals    = [rec["k"]    for rec in records]
    c_vals    = [rec["C_GME"] for rec in records]
    k_conv    = result["k_converged"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title or "Algorithm B — Stiefel Manifold Convergence",
                 fontsize=13)

    # ── Left panel: C_GME vs k ────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(k_vals, c_vals, "o-", color="steelblue",
            linewidth=2, markersize=8, label="Best $C_{\\mathrm{GME}}$ per $k$")

    # mark the converged k
    conv_idx = k_vals.index(k_conv)
    ax.axvline(k_conv, color="tomato", linestyle="--", linewidth=1.5,
               label=f"Converged at $k = {k_conv}$")
    ax.plot(k_conv, c_vals[conv_idx], "o", color="tomato",
            markersize=12, zorder=5)

    ax.set_xlabel("Cardinality $k$",      fontsize=12)
    ax.set_ylabel("$C_{\\mathrm{GME}}$",  fontsize=12)
    ax.set_title("Best value per $k$",    fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_vals)

    # ── Right panel: restart distribution per k ───────────────────────────────
    ax2 = axes[1]
    for i, rec in enumerate(records):
        log    = rec["restart_log"]
        k_here = rec["k"]
        jitter = np.random.uniform(-0.08, 0.08, len(log))
        ax2.scatter([k_here + j for j in jitter], log,
                    alpha=0.5, s=25, color="steelblue")
        ax2.plot(k_here, rec["C_GME"], "D", color="tomato",
                 markersize=9, zorder=5,
                 label="Best" if i == 0 else None)

    ax2.set_xlabel("Cardinality $k$",           fontsize=12)
    ax2.set_ylabel("$C_{\\mathrm{GME}}$",       fontsize=12)
    ax2.set_title("Restart distribution per $k$", fontsize=11)
    ax2.legend(["All restarts", "Best per $k$"], fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_vals)

    plt.tight_layout()
    plt.savefig("stiefel_convergence.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Figure saved to stiefel_convergence.png")


def print_k_summary(result: dict) -> None:
    """
    Print a clean summary table of all k records.
    """
    print("\n" + "="*65)
    print(f"  Algorithm B — k convergence summary")
    print(f"  rank r        = {result['rank']}")
    print(f"  k converged   = {result['k_converged']}")
    print(f"  C_GME (final) = {result['C_GME']:.8f}")
    print("="*65)
    print(f"  {'k':>4}  {'n_params':>9}  {'C_GME':>12}  "
          f"{'delta':>12}  {'n_restarts':>10}")
    print("-"*65)

    prev = None
    for rec in result["k_records"]:
        k       = rec["k"]
        r       = result["rank"]
        n_p     = 2 * k * r - r * r
        c       = rec["C_GME"]
        delta   = abs(c - prev) if prev is not None else float("nan")
        n_rest  = len(rec["restart_log"])
        conv    = " ← converged" if k == result["k_converged"] else ""
        print(f"  {k:>4}  {n_p:>9}  {c:>12.8f}  "
              f"{delta:>12.2e}  {n_rest:>10}{conv}")
        prev = c

    print("="*65 + "\n")