"""
plots.py
--------
Publication-quality comparison plots for parameter scan results.

All plot functions accept the aggregated DataFrame from scan.py and
return matplotlib Figure objects (or save directly to PDF/PNG).

Figures produced
----------------
plot_cgme_vs_U(df_agg)
    5-panel figure:
        Panel 1 : C_GME  for each eigenstate  vs U
        Panel 2 : C_GME  for each eigenstate  vs U/t
        Panel 3 : Energy gap Δ = Eₙ - E₀      vs U  and vs U/t
        Panel 4 : Double occupancy ⟨D⟩/site    vs U  and vs U/t
        Panel 5 : RDM purity                   vs U  and vs U/t

plot_cgme_panel(df_agg, x="U")
    Single panel C_GME vs x for all eigenstates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from typing import Optional


# ── Colour / style helpers ────────────────────────────────────────────────────

# One colour and marker per eigenstate (supports up to 8 states)
_STATE_COLORS  = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
                  "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
_STATE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_STATE_LS      = ["-", "--", "-.", ":", "-", "--", "-.", ":"]


def _state_style(state_idx: int) -> dict:
    i = state_idx % len(_STATE_COLORS)
    return dict(
        color=_STATE_COLORS[i],
        marker=_STATE_MARKERS[i],
        linestyle=_STATE_LS[i],
        linewidth=1.8,
        markersize=6,
    )


def _state_label(state_idx: int) -> str:
    labels = ["ψ₀ (ground)", "ψ₁ (1st excited)", "ψ₂ (2nd excited)",
              "ψ₃", "ψ₄", "ψ₅", "ψ₆", "ψ₇"]
    return labels[state_idx] if state_idx < len(labels) else f"ψ{state_idx}"


# ── Single-panel helper ───────────────────────────────────────────────────────

def _plot_quantity_vs_x(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    ylabel: str,
    title: str,
    per_state: bool = True,
    hline: Optional[float] = None,
) -> None:
    """
    Draw one panel. If per_state=True, one line per eigenstate.
    If False, single line (e.g. gap only depends on system, not state).
    """
    if per_state:
        for idx, grp in df.groupby("state_idx"):
            grp_sorted = grp.sort_values(x_col)
            ax.plot(
                grp_sorted[x_col], grp_sorted[y_col],
                label=_state_label(int(idx)),
                **_state_style(int(idx)),
            )
    else:
        # Use state_idx == 1 for gap (gap of ψ₁ relative to ψ₀)
        grp = df[df["state_idx"] == 1].sort_values(x_col)
        ax.plot(grp[x_col], grp[y_col],
                label="Δ = E₁ − E₀",
                color="#2ca02c", marker="s", linewidth=1.8, markersize=6)

    if hline is not None:
        ax.axhline(hline, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_xlabel(x_col.replace("_over_", "/"), fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=9)


# ── Main 5-quantity figure ────────────────────────────────────────────────────

def plot_cgme_vs_U(
    df_agg: pd.DataFrame,
    L: int,
    n_body: int,
    mode: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    5-row × 2-column figure.

    Left column  : x = U
    Right column : x = U/t

    Rows
    ----
    1. C_GME (mean over clusters) per eigenstate
    2. C_GME (max over clusters)  per eigenstate
    3. Energy gap  Δ = E₁ − E₀
    4. Double occupancy ⟨D⟩/site
    5. RDM purity (mean over clusters)

    Parameters
    ----------
    df_agg    : Aggregated DataFrame from scan.run_U_scan / run_Ut_scan.
    L, n_body : For the figure title.
    save_path : If given, saves to PDF.
    """
    fig = plt.figure(figsize=(14, 20))
    fig.suptitle(
        f"GME Parameter Scan  —  L={L}, {n_body}-body RDM, mode={mode}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(5, 2, figure=fig,
                           hspace=0.45, wspace=0.32,
                           top=0.94, bottom=0.04)

    panels = [
        # (row, y_col, ylabel, title, per_state, hline)
        (0, "C_GME_mean", "C_GME (cluster avg)", "C_GME mean vs U",      True,  None),
        (1, "C_GME_max",  "C_GME (cluster max)", "C_GME max vs U",       True,  None),
        (2, "gap",        "Δ = E₁ − E₀",         "Energy gap vs U",      False, 0.0),
        (3, "d_occ_site", "⟨D⟩ / site",           "Double occupancy vs U",True,  None),
        (4, "purity_mean","Purity Tr(ρ²)",         "RDM purity vs U",     True,  1.0),
    ]

    for row, y_col, ylabel, title_base, per_state, hline in panels:
        # Left: vs U
        ax_l = fig.add_subplot(gs[row, 0])
        _plot_quantity_vs_x(
            ax_l, df_agg, "U", y_col, ylabel,
            title_base, per_state, hline
        )
        # Right: vs U/t
        ax_r = fig.add_subplot(gs[row, 1])
        _plot_quantity_vs_x(
            ax_r, df_agg, "U_over_t", y_col, ylabel,
            title_base.replace("vs U", "vs U/t"), per_state, hline
        )

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")

    return fig


# ── C_GME only, single panel ──────────────────────────────────────────────────

def plot_cgme_panel(
    df_agg:    pd.DataFrame,
    x:         str  = "U",
    title:     str  = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compact single-panel C_GME vs x for all eigenstates.

    Parameters
    ----------
    x : 'U' or 'U_over_t'
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for idx, grp in df_agg.groupby("state_idx"):
        grp_sorted = grp.sort_values(x)
        ax.plot(
            grp_sorted[x], grp_sorted["C_GME_mean"],
            label=_state_label(int(idx)),
            **_state_style(int(idx)),
        )
        # Shaded band between min and max
        ax.fill_between(
            grp_sorted[x],
            grp_sorted["C_GME_min"],
            grp_sorted["C_GME_max"],
            alpha=0.12,
            color=_STATE_COLORS[int(idx) % len(_STATE_COLORS)],
        )

    xlabel = "U" if x == "U" else "U/t"
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("C_GME", fontsize=12)
    ax.set_title(title or f"C_GME vs {xlabel}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")

    return fig


# ── Excited-state heatmap ─────────────────────────────────────────────────────

def plot_excited_state_heatmap(
    df_agg:    pd.DataFrame,
    quantity:  str  = "C_GME_mean",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    2D heatmap: rows = eigenstates, columns = U values.
    Colour = chosen quantity (C_GME_mean, purity_mean, entropy_mean).

    Useful for seeing which states are entangled at which U.
    """
    pivot = df_agg.pivot_table(
        index="state_idx", columns="U", values=quantity
    )

    ylabel_map = {
        "C_GME_mean":  "C_GME (cluster avg)",
        "C_GME_max":   "C_GME (cluster max)",
        "purity_mean": "Purity",
        "entropy_mean":"Entropy",
    }

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 0.8 + 2), 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis",
                   origin="lower", interpolation="nearest",
                   extent=[-0.5, len(pivot.columns) - 0.5,
                           -0.5, len(pivot.index) - 0.5])

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns], fontsize=8, rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([_state_label(int(i)) for i in pivot.index], fontsize=9)

    ax.set_xlabel("U", fontsize=11)
    ax.set_ylabel("Eigenstate", fontsize=11)
    ax.set_title(
        f"{ylabel_map.get(quantity, quantity)} — Eigenstate vs U",
        fontsize=12, fontweight="bold"
    )
    plt.colorbar(im, ax=ax, label=ylabel_map.get(quantity, quantity))
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Heatmap saved → {save_path}")

    return fig


# ── Full scan PDF ─────────────────────────────────────────────────────────────

def generate_scan_pdf(
    df_agg:    pd.DataFrame,
    L:         int,
    n_body:    int,
    mode:      str,
    output_path: str,
) -> None:
    """
    Combine all scan plots into one PDF report.

    Pages
    -----
    1. C_GME panel (U x-axis)
    2. C_GME panel (U/t x-axis)
    3. Full 5-quantity comparison figure
    4. C_GME heatmap (eigenstates × U)
    5. Purity heatmap (eigenstates × U)
    """
    with PdfPages(output_path) as pdf:

        # Page 1: C_GME vs U
        fig = plot_cgme_panel(df_agg, x="U",
                              title=f"C_GME vs U  (L={L}, {n_body}-body, {mode})")
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)

        # Page 2: C_GME vs U/t
        fig = plot_cgme_panel(df_agg, x="U_over_t",
                              title=f"C_GME vs U/t  (L={L}, {n_body}-body, {mode})")
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)

        # Page 3: Full 5-quantity comparison
        fig = plot_cgme_vs_U(df_agg, L=L, n_body=n_body, mode=mode)
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)

        # Page 4: C_GME heatmap
        fig = plot_excited_state_heatmap(df_agg, quantity="C_GME_mean")
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)

        # Page 5: Purity heatmap
        fig = plot_excited_state_heatmap(df_agg, quantity="purity_mean")
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)

    print(f"  Scan PDF saved → {output_path}")
