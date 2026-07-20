#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_all_methods_full.py
========================

Reproduce the FULL figure suite of generate_report.py, but reading the
`all_method_results` block of a convergence_benchmark.py PKL, so EVERY method
appears: GCG, all Method-II (Stiefel) sub-optimisers, AND the Stiefel-GCG hybrid.

Differences vs generate_report.py:
  * data source is `all_method_results` (not the gcg_*/st_* prefixes), so the
    hybrid and every Stiefel optimiser show up — not just GCG + the primary one.
  * every output filename carries a YYYYMMDD_HHMMSS stamp, so reruns never
    overwrite previous PDFs.
  * the cost-vs-iteration plots show the full cost equation on the y-axis.

Figures produced (mirroring the originals):
  convergence_n{n}_{stamp}.pdf        cost vs iter (left) + log error (right)
  convergence_combined_{stamp}.pdf    all n stacked
  convergence_normalized_{stamp}.pdf  cost vs fraction of iterations, all n
  final_cost_vs_n_{stamp}.pdf         final C_GME vs n (line)
  iterations_vs_n_{stamp}.pdf         recorded iterations (grouped bar)
  runtime_vs_n_{stamp}.pdf            wall-clock time per job (grouped bar)
  final_cost_vs_time_{stamp}.pdf      cost vs time, faceted by n
  outer_k_{method}_n{n}_{stamp}.pdf   outer-k convergence (where recorded)
  final_costs_and_timings_{stamp}.csv all numbers

Usage
-----
  python plot_all_methods_full.py                        # newest conv_*.pkl in default dir
  python plot_all_methods_full.py --pkl <file> --outdir report_out
  python plot_all_methods_full.py --input <dir>
"""

from __future__ import annotations

import argparse
import csv
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 8,
    "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.4,
    "lines.linewidth": 1.4, "lines.markersize": 3.2,
})

EPS = 1e-12
XLABEL = "Iteration (callback index)"
# Full cost equation for the convergence-plot y-axes.
YLABEL_COST = r"Cost $h(U)=\sum_i p_i\, C_{\mathrm{GME}}(|\psi_i\rangle)$"

# tag (key in all_method_results) -> (legend label, colour, marker, sort order)
STYLE = {
    "gcg":                   ("GCG (Method I)",            "#1f77b4", "o", 0),
    "methodII_L-BFGS-B":     ("Stiefel L-BFGS-B (II)",     "#ff7f0e", "s", 1),
    "methodII_Powell":       ("Stiefel Powell (II)",       "#2ca02c", "^", 2),
    "methodII_SLSQP":        ("Stiefel SLSQP (II)",        "#9467bd", "D", 3),
    "methodII_TNC":          ("Stiefel TNC (II)",          "#d62728", "v", 4),
    "methodII_trust-constr": ("Stiefel trust-constr (II)", "#8c564b", "P", 5),
    "methodII_gcg":          ("Stiefel-GCG (Method III)",  "#111111", "X", 6),
}
_FB_COL = ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
_FB_MK = ["*", "X", "p", "h"]
_cache = {}


def style_for(tag):
    if tag in STYLE:
        lbl, col, mk, _ = STYLE[tag]
        return lbl, col, mk
    if tag in _cache:
        return _cache[tag]
    i = len(_cache)
    out = (str(tag), _FB_COL[i % len(_FB_COL)], _FB_MK[i % len(_FB_MK)])
    _cache[tag] = out
    return out


def order_val(tag):
    return STYLE[tag][3] if tag in STYLE else 99


def label_for(tag):
    return style_for(tag)[0]


def safe_tag(x):
    return (str(x).replace(" ", "").replace("/", "-").replace("\\", "-")
            .replace(".", "p").replace("+", ""))


def find_pkl(input_dir, explicit):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"PKL not found: {p}")
        return p
    d = Path(input_dir)
    cands = sorted(d.glob("conv_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise SystemExit(f"No conv_*.pkl in {d} (pass --pkl).")
    return cands[0]


def rec_from_entry(entry):
    hist = np.asarray(entry.get("history", []) or [], dtype=float).ravel()
    raw = entry.get("raw", {}) or {}
    rank = raw.get("rank") if isinstance(raw, dict) else None
    kc = raw.get("k_converged") if isinstance(raw, dict) else None
    return {
        "hist": hist,
        "final": float(entry.get("final", np.nan)),
        "time": float(entry.get("elapsed", 0.0) or 0.0),
        "iters": max(int(hist.size) - 1, 0),
        "n_points": int(hist.size),
        "raw": raw,
        "rank": rank,
        "k": kc,
    }


def build_data(results_by_n):
    out = {}
    for n, d in results_by_n.items():
        if not isinstance(d, dict):
            continue
        amr = d.get("all_method_results", {}) or {}
        if not amr:
            continue
        out[int(n)] = {tag: rec_from_entry(e) for tag, e in amr.items()}
    return out


def ordered_tags(per):
    return sorted(per.keys(), key=order_val)


# ── figures ───────────────────────────────────────────────────────────────────

def fig_single_n(per, n, path):
    tags = ordered_tags(per)
    h_min = min((per[t]["final"] for t in tags), default=0.0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for t in tags:
        h = per[t]["hist"]
        lbl, col, mk = style_for(t)
        if h.size:
            ax.plot(np.arange(h.size), h, marker=mk, ls="-", color=col, label=lbl)
    ax.set_xlabel(XLABEL); ax.set_ylabel(YLABEL_COST)
    ax.set_title(f"n={n}: cost vs iteration"); ax.legend(loc="best")
    ax = axes[1]
    for t in tags:
        h = per[t]["hist"]
        lbl, col, mk = style_for(t)
        if h.size:
            err = np.maximum(h - h_min, EPS)
            ax.semilogy(np.arange(err.size), err, marker=mk, ls="-", color=col, label=lbl)
    ax.set_xlabel(XLABEL); ax.set_ylabel(r"$h(U)-h_{\min}$")
    ax.set_title(f"n={n}: log error to best minimum"); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print("   saved ->", path)


def fig_combined(data_by_n, path):
    ns = sorted(data_by_n)
    if not ns:
        return None
    fig, axes = plt.subplots(len(ns), 2, figsize=(12, 4.2 * len(ns)))
    if len(ns) == 1:
        axes = np.array(axes).reshape(1, -1)
    for row, n in enumerate(ns):
        per = data_by_n[n]; tags = ordered_tags(per)
        h_min = min((per[t]["final"] for t in tags), default=0.0)
        a0, a1 = axes[row, 0], axes[row, 1]
        for t in tags:
            h = per[t]["hist"]; lbl, col, mk = style_for(t)
            if h.size:
                a0.plot(np.arange(h.size), h, marker=mk, ls="-", color=col, label=lbl)
                err = np.maximum(h - h_min, EPS)
                a1.semilogy(np.arange(err.size), err, marker=mk, ls="-", color=col, label=lbl)
        a0.set_xlabel(XLABEL); a0.set_ylabel(YLABEL_COST)
        a0.set_title(f"n={n}: cost vs iteration"); a0.legend(loc="best")
        a1.set_xlabel(XLABEL); a1.set_ylabel(r"$h(U)-h_{\min}$")
        a1.set_title(f"n={n}: log error"); a1.legend(loc="best")
    fig.suptitle("Convergence comparison (all methods)", fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print("   saved ->", path); return path


def fig_normalized(data_by_n, path):
    ns = sorted(data_by_n)
    if not ns:
        return None
    fig, axes = plt.subplots(len(ns), 1, figsize=(7.6, 3.7 * len(ns)), squeeze=False)
    axes = axes[:, 0]
    for row, n in enumerate(ns):
        per = data_by_n[n]; ax = axes[row]
        for t in ordered_tags(per):
            h = per[t]["hist"]; lbl, col, mk = style_for(t)
            if h.size:
                x = np.linspace(0.0, 1.0, h.size) if h.size >= 2 else np.array([0.0])
                ax.plot(x, h, marker=mk, ls="-", color=col,
                        markevery=max(1, h.size // 20), label=lbl)
        ax.set_xlabel("Fraction of recorded iterations"); ax.set_ylabel(YLABEL_COST)
        ax.set_title(f"n={n}: convergence shape (normalized iteration)"); ax.legend(loc="best")
    fig.suptitle("Convergence vs normalized iteration (all methods)", fontweight="bold", y=1.0)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print("   saved ->", path); return path


def fig_final_vs_n(data_by_n, path):
    ns = sorted(data_by_n)
    tags = sorted({t for per in data_by_n.values() for t in per}, key=order_val)
    if not ns:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for t in tags:
        xs, ys = [], []
        for n in ns:
            if t in data_by_n[n]:
                xs.append(n); ys.append(data_by_n[n][t]["final"])
        if xs:
            lbl, col, mk = style_for(t)
            ax.plot(xs, ys, marker=mk, ls="-", color=col, label=lbl)
    ax.set_xlabel(r"Cluster size $n$ (sites)")
    ax.set_ylabel(r"$h_{\mathrm{final}} \equiv C_{\mathrm{GME}}$ estimate")
    ax.set_title("Final cost vs cluster size"); ax.set_xticks(ns); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print("   saved ->", path); return path


def grouped_bar(data_by_n, value_key, ylabel, title, path):
    ns = sorted(data_by_n)
    tags = sorted({t for per in data_by_n.values() for t in per}, key=order_val)
    if not ns or not tags:
        return None
    x = np.arange(len(ns), dtype=float); width = 0.8 / max(len(tags), 1)
    fig, ax = plt.subplots(figsize=(max(7.4, 1.6 * len(ns)), 4.8))
    for i, t in enumerate(tags):
        vals = [data_by_n[n].get(t, {}).get(value_key, np.nan) for n in ns]
        off = (i - (len(tags) - 1) / 2.0) * width
        lbl, col, _ = style_for(t)
        ax.bar(x + off, vals, width=width, color=col, label=lbl)
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel(r"Cluster size $n$ (sites)"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(loc="best"); ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print("   saved ->", path); return path


def fig_cost_vs_time(data_by_n, path):
    ns = sorted(data_by_n)
    if not ns:
        return None
    all_times = [per[t]["time"] for per in data_by_n.values() for t in per]
    if not all_times or max(all_times) <= 0.0:
        print("   (skipped final_cost_vs_time: no wall-clock times stored)")
        return None
    fig, axes = plt.subplots(1, len(ns), figsize=(4.7 * len(ns), 4.4), squeeze=False)
    axes = axes[0]
    for j, n in enumerate(ns):
        ax = axes[j]; per = data_by_n[n]
        for t in ordered_tags(per):
            lbl, col, mk = style_for(t)
            ax.scatter([per[t]["time"]], [per[t]["final"]], color=col, marker=mk, s=70,
                       edgecolor="k", linewidth=0.4, zorder=3, label=lbl)
        ax.set_xlabel("Wall-clock time per job [s]")
        ax.set_ylabel(r"$h_{\mathrm{final}} \equiv C_{\mathrm{GME}}$")
        ax.set_title(f"n={n}"); ax.legend(loc="best", fontsize=7)
    fig.suptitle("Final cost vs wall-clock time (lower-left is better)", fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print("   saved ->", path); return path


def fig_outer_k(data_by_n, fig_dir, stamp):
    out = []
    for n in sorted(data_by_n):
        per = data_by_n[n]
        for t in ordered_tags(per):
            raw = per[t]["raw"]
            kh = raw.get("k_history") if isinstance(raw, dict) else None
            if kh is None and isinstance(raw, dict):
                kr = raw.get("k_records")
                if kr:
                    try:
                        kh = [float(r["C_GME"]) for r in kr]
                    except Exception:
                        kh = None
            if not kh:
                continue
            kh = np.asarray(kh, dtype=float)
            if kh.size < 2:
                continue
            lbl, col, mk = style_for(t)
            fig, ax = plt.subplots(figsize=(6.6, 4.4))
            ax.plot(np.arange(kh.size), kh, marker=mk, ls="-", color=col)
            ax.set_xlabel(r"Outer step $k-r$"); ax.set_ylabel(r"$C_{\mathrm{GME}}$")
            ax.set_title(f"{lbl} outer-k convergence, n={n}")
            fig.tight_layout()
            p = fig_dir / f"outer_k_{safe_tag(t)}_n{n}_{stamp}.pdf"
            fig.savefig(p, bbox_inches="tight"); plt.close(fig)
            print("   saved ->", p); out.append(p)
    return out


def write_csv(data_by_n, path):
    cols = ["n", "method", "tag", "final_cost", "callback_points",
            "time_s", "time_min", "rank", "k"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols)
        for n in sorted(data_by_n):
            for t in ordered_tags(data_by_n[n]):
                rec = data_by_n[n][t]
                w.writerow([n, label_for(t), t, f"{rec['final']:.10f}", rec["iters"],
                            f"{rec['time']:.4f}", f"{rec['time'] / 60.0:.4f}",
                            rec["rank"], rec["k"]])
    print("   wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", default=None,
                    help="Explicit conv_*.pkl. If omitted, newest in --input is used.")
    ap.add_argument("--input", default="../results/rdm_pkl_results/convergence_runs",
                    help="Folder to scan for the newest conv_*.pkl.")
    ap.add_argument("--outdir", default="report_out", help="Output directory.")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pkl = find_pkl(args.input, args.pkl)
    print("Loading:", pkl)
    with open(pkl, "rb") as f:
        data = pickle.load(f)

    data_by_n = build_data(data.get("results_by_n", {}) or {})
    if not data_by_n:
        raise SystemExit("No 'all_method_results' in this PKL — was it written by the "
                         "updated convergence_benchmark.py?")

    fig_dir = Path(args.outdir) / "figs"
    data_dir = Path(args.outdir) / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStamp: {stamp}")
    for n in sorted(data_by_n):
        names = ", ".join(label_for(t) for t in ordered_tags(data_by_n[n]))
        print(f"n={n}: {len(data_by_n[n])} methods -> {names}")
    print(f"\nFigures -> {fig_dir}")

    for n in sorted(data_by_n):
        fig_single_n(data_by_n[n], n, fig_dir / f"convergence_n{n}_{stamp}.pdf")
    fig_combined(data_by_n, fig_dir / f"convergence_combined_{stamp}.pdf")
    fig_normalized(data_by_n, fig_dir / f"convergence_normalized_{stamp}.pdf")
    fig_final_vs_n(data_by_n, fig_dir / f"final_cost_vs_n_{stamp}.pdf")
    grouped_bar(data_by_n, "iters", "Recorded iterations",
                "Iterations vs cluster size", fig_dir / f"iterations_vs_n_{stamp}.pdf")
    grouped_bar(data_by_n, "time", "Wall-clock time per job [s]",
                "Runtime vs cluster size", fig_dir / f"runtime_vs_n_{stamp}.pdf")
    fig_cost_vs_time(data_by_n, fig_dir / f"final_cost_vs_time_{stamp}.pdf")
    fig_outer_k(data_by_n, fig_dir, stamp)

    print(f"\nCSV -> {data_dir}")
    write_csv(data_by_n, data_dir / f"final_costs_and_timings_{stamp}.csv")

    print("\nDone. Outputs in:", Path(args.outdir).resolve())


if __name__ == "__main__":
    main()