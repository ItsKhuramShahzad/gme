"""
report_multimethod.py
---------------------
Plain tabular PDF report + comprehensive .pkl file for multi-method
convex-roof comparison results.

PDF pages
---------
  Page 1  -- System parameters + ground-state properties
  Page 2  -- CLUSTER OVERVIEW: one row per cluster, one column per method
             Cluster | Purity | C_LB | C_UB | C_GME_L-BFGS-B | C_GME_Nelder-M |
             C_GME_Powell | C_GME_COBYLA | C_GME_Diff-Evo | C_GME_Sim-Ann | BestMethod | Status
  Page 3  -- METHOD SUMMARY: mean/std/best/worst/time/rank per method
  Page 4  -- TIMING: total/avg time + fraction per method
  Page 5+ -- Per-cluster detail: bipartitions + all method results + decomposition

PKL keys
--------
  system, ground_state, rdm, metrics, gme, convex_roof, summary

Public API
----------
    save_pkl(data, path)
    load_pkl(path) -> dict
    generate_comparison_pdf(results, system_info, output_path, ...)
    save_full_pkl(results, system_info, output_path, ...)
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ─────────────────────────────────────────────────────────────────────────────
# PKL helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_pkl(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  PKL saved  -> {path}")


def load_pkl(path: str) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _build_summary(results: Dict) -> Dict:
    if not results:
        return {}
    methods = list(next(iter(results.values()))["methods"].keys())
    agg: Dict[str, Dict] = {m: {"vals": [], "times": []} for m in methods}
    for sr in results.values():
        for mname, mres in sr["methods"].items():
            agg[mname]["vals"].append(mres["C_GME"])
            agg[mname]["times"].append(mres["time_s"])

    stats = {}
    for mname, data in agg.items():
        vals = data["vals"]
        stats[mname] = {
            "mean_C_GME":   float(np.mean(vals)),
            "std_C_GME":    float(np.std(vals)),
            "best_C_GME":   float(np.min(vals)),
            "worst_C_GME":  float(np.max(vals)),
            "total_time_s": float(sum(data["times"])),
            "avg_time_s":   float(np.mean(data["times"])),
        }

    ranked = sorted(stats.keys(), key=lambda m: stats[m]["mean_C_GME"])
    for rank_i, mname in enumerate(ranked, 1):
        stats[mname]["rank"] = rank_i
    return stats


def save_full_pkl(
    results:      Dict,
    system_info:  dict,
    output_path:  str,
    ground_state: Optional[Dict] = None,
    rdm:          Optional[Dict] = None,
    metrics:      Optional[Dict] = None,
    gme:          Optional[Dict] = None,
) -> dict:
    def _str_keys(d):
        if d is None:
            return {}
        return {str(k): v for k, v in d.items()}

    summary = _build_summary(results)
    data = {
        "system":       {**system_info, "timestamp": datetime.now().isoformat()},
        "ground_state": ground_state or {},
        "rdm":          _str_keys(rdm),
        "metrics":      _str_keys(metrics),
        "gme":          _str_keys(gme),
        "convex_roof":  _str_keys(results),
        "summary":      summary,
    }
    save_pkl(data, output_path)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# PDF rendering
# ─────────────────────────────────────────────────────────────────────────────

_MONO  = "monospace"
_SMALL = 7
_SEP2  = "=" * 130


def _new_page(pdf: PdfPages, text: str) -> None:
    fig, ax = plt.subplots(figsize=(23, 11.7))
    ax.axis("off")
    ax.text(0.01, 0.99, text, ha="left", va="top",
            fontsize=_SMALL, family=_MONO,
            transform=ax.transAxes, wrap=False)
    pdf.savefig(fig, bbox_inches="tight", dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — System info
# ─────────────────────────────────────────────────────────────────────────────

def _page_system(si: dict, n_clusters: int, methods: list, n_restarts: int) -> str:
    now = datetime.now().strftime("%Y-%m-%d  %H:%M")
    # desc = {
    #     "L-BFGS-B":        "Gradient-based  (limited-memory BFGS, local)",
    #     "Nelder-Mead":      "Simplex  (derivative-free, local)",
    #     "Powell":           "Direction-set  (derivative-free, local)",
    #     "COBYLA":           "Linear-approx constrained  (local)",
    #     "Differential-Evo": "Global stochastic  (most thorough, slowest)",
    #     "Simulated-Ann":    "Basin hopping  (global, moderate speed)",
    # }
    desc = {
        "L-BFGS-B",
        "Nelder-Mead",
        "Powell",
        "COBYLA",
        "Differential-Evo",
        "Simulated-Ann",
    }
    E0 = si.get("E0", float("nan"))
    E1 = si.get("E1", float("nan"))
    try:
        gap_str = f"{E1 - E0:.10f}"
    except Exception:
        gap_str = "--"
    lines = [
        _SEP2,
        "  CONVEX-ROOF C_GME  -- METHOD-2 (Exponential Map)",
        _SEP2,
        "",
        f"  Generated          : {now}",
        f"  System             : 1D Hubbard ring  (mode: {si.get('mode','uniform')})",
        "",
        "  SYSTEM PARAMETERS",
        "  " + "-" * 50,
        f"  Sites          L   = {si.get('L','--')}",
        f"  Electrons     Ne   = {si.get('Ne','--')}",
        f"  Hopping        t   = {si.get('t','--')}",
        f"  Interaction    U   = {si.get('U','--')}",
        "",
        "  GROUND STATE",
        "  " + "-" * 50,
        f"  E0                 = {E0:.10f}",
        f"  E1                 = {E1:.10f}",
        f"  Gap  E1-E0         = {gap_str}",
        f"  <S^2>              = {si.get('S2',  float('nan')):.6f}",
        f"  S                  = {si.get('S',   float('nan')):.4f}",
        f"  <Sz>               = {si.get('Sz',  float('nan')):.6f}",
        f"  Multiplicity       = {si.get('multiplicity', float('nan')):.1f}",
        f"  Spin state         = {si.get('spin_label','--')}",
        f"  <D> total          = {si.get('double_occ', float('nan')):.6f}",
        f"  <D> / site         = {si.get('double_occ_per_site', float('nan')):.6f}",
        "",
        "  ANALYSIS PARAMETERS",
        "  " + "-" * 50,
        f"  Clusters analysed  = {n_clusters}",
        f"  Optimization Methods used       = {len(methods)}",
        f"  Restarts / method  = {n_restarts}",
        "",
        "  OPTIMIZATION METHODS",
        "  " + "-" * 50,
    ]
    for i, m in enumerate(methods, 1):
        lines.append(f"  {i}. {m:<18}")
    lines += [
        "",
        "  NOTATION",
        "  " + "-" * 50,
        "  C_LB     = Lower bound  (paper coherence-witness, analytical)",
        "  C_GME    = GME for pure rdm",
        "  C_UB     = Upper bound  (eigendecomposition, analytical)",
        "  C_GME_X  = Convex-roof estimate from optimiser X",
        "",
        "  True C_GME satisfies:  C_LB <= C_GME <= C_UB",
        # "  Status: GME=entangled | SEP=separable | MIX=mixed/inconclusive",
        "",
        _SEP2,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — MAIN OVERVIEW TABLE  (one column per method)
# ─────────────────────────────────────────────────────────────────────────────

def _page_cluster_overview(results: Dict) -> str:
    if not results:
        return ""

    methods = list(next(iter(results.values()))["methods"].keys())

    short = {
        "L-BFGS-B":        "L-BFGS-B",
        "Nelder-Mead":      "Nelder-M",
        "Powell":           "Powell",
        "COBYLA":           "COBYLA",
        "Differential-Evo": "Diff-Evo",
        "Simulated-Ann":    "Sim-Ann",
    }
    labels = [short.get(m, m[:10]) for m in methods]

    CL  = 16   # Cluster
    CF  = 12   # Purity / C_LB / C_UB
    CW  = 14   # per-method C_GME
    CBM = 12   # BestMethod
    CS  =  6   # Status

    # Header with | separators
    hdr = (
        f"  | {'Cluster':<{CL}} "
        f"| {'Purity':>{CF}} "
        f"| {'C_LB':>{CF}} "
        f"| {'C_GME':>{CF}} "
        f"| {'C_UB':>{CF}} "
    )
    for lb in labels:
        hdr += f"| {'C_GME_' + lb:>{CW}} "
    hdr += f"| {'BestMethod':<{CBM}} | {'Status':>{CS}} |"

    sep = "  +" + "-" * (len(hdr) - 4) + "+"

    lines = [
        _SEP2,
        "  CLUSTER OVERVIEW  --  All Optimizers",
        # "  Sorted by: n (body order) ascending, then sites tuple",
        # "  * marks the best (lowest) C_GME per cluster",
        # "  Status: GME=entangled | SEP=separable | MIX=inconclusive",
        _SEP2,
        "",
        hdr,
        sep,
    ]

    # Sort by (n_sites, sites_tuple)
    sorted_items = sorted(results.items(), key=lambda x: (len(x[1]["n"]) if isinstance(x[1]["n"], tuple) else x[1]["n"], x[0]))

    for sites, sr in sorted_items:
        pur    = sr["purity"]
        C_LB   = sr["C_LB"]
        C_GME   = sr["C_GME"]
        C_UB   = sr["C_UB"]
        
        bm     = sr.get("best_method", "--")
        best_C = sr.get("best_C_GME", float("nan"))

        if C_LB > 1e-8:
            status = "GME"
        elif best_C < 1e-8:
            status = "SEP"
        else:
            status = "MIX"

        bm_short = short.get(bm, bm[:CBM])

        row = (
            f"  | {str(sites):<{CL}} "
            f"| {pur:>{CF}.8f} "
            f"| {C_LB:>{CF}.8f} "
            f"| {C_GME:>{CF}.8f} "
            f"| {C_UB:>{CF}.8f} "
        )
        for mname in methods:
            c      = sr["methods"][mname]["C_GME"]
            marker = "*" if mname == bm else " "
            row   += f"| {marker}{c:>{CW - 1}.8f} "
        row += f"| {bm_short:<{CBM}} | {status:>{CS}} |"
        lines.append(row)

    lines += [
        sep,
        "  * = best (lowest) C_GME for that cluster",
        "",
        _SEP2,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — Method summary
# ─────────────────────────────────────────────────────────────────────────────

def _page_method_summary(summary: Dict, results: Dict) -> str:
    methods = list(summary.keys())
    lines = [
        _SEP2,
        "  Optimizers SUMMARY  (aggregated across all clusters)",
        _SEP2,
        "",
        f"  {'Optimizer':<18} {'Mean C_GME':>12} {'Std Dev':>10} "
        f"{'Best':>12} {'Worst':>12} {'Total(s)':>10} {'Rank':>5}",
        "  " + "-" * 84,
    ]
    for mname in sorted(methods, key=lambda m: summary[m]["rank"]):
        s = summary[mname]
        lines.append(
            f"  {mname:<18} {s['mean_C_GME']:>12.8f} {s['std_C_GME']:>10.6f} "
            f"{s['best_C_GME']:>12.8f} {s['worst_C_GME']:>12.8f} "
            f"{s['total_time_s']:>10.2f} {s['rank']:>5}"
        )
    lines += ["", _SEP2]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 — Timing
# ─────────────────────────────────────────────────────────────────────────────

def _page_timing(summary: Dict, results: Dict) -> str:
    total_all = sum(s["total_time_s"] for s in summary.values())
    methods   = list(summary.keys())
    lines = [
        _SEP2,
        "  TIMING ANALYSIS",
        _SEP2,
        "",
        f"  {'Optimizer':<18} {'Total(s)':>10} {'Avg/cluster(s)':>16} "
        f"{'Fraction(%)':>12} {'Rank(quality)':>14}",
        "  " + "-" * 75,
    ]
    for mname in sorted(methods, key=lambda m: -summary[m]["total_time_s"]):
        s    = summary[mname]
        frac = 100.0 * s["total_time_s"] / total_all if total_all > 0 else 0.0
        lines.append(
            f"  {mname:<18} {s['total_time_s']:>10.2f} "
            f"{s['avg_time_s']:>16.2f} "
            f"{frac:>12.1f} {s['rank']:>14}"
        )
    lines += [
        "  " + "-" * 75,
        f"  {'TOTAL':<18} {total_all:>10.2f}",
        "",
        # "  Rank 1 = best quality (lowest mean C_GME)",
        "",
        _SEP2,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Pages 5+ — Per-cluster detail
# ─────────────────────────────────────────────────────────────────────────────

def _page_cluster_detail(sites: tuple, sr: Dict) -> str:
    n       = sr["n"]
    purity  = sr["purity"]
    rank    = sr["rank"]
    C_LB    = sr["C_LB"]
    C_GME    = sr["C_GME"]
    C_UB    = sr["C_UB"]
    best_m  = sr["best_method"]
    best_C  = sr["best_C_GME"]
    methods = sr["methods"]
    bips    = sr.get("bipartitions", [])

    lines = [
        _SEP2,
        f"  CLUSTER DETAIL  --  Sites {sites}",
        _SEP2,
        "",
        f"  n (body order) = {n}",
        f"  Purity Tr(r^2) = {purity:.10f}",
        f"  Rank           = {rank}",
        f"  C_LB (lower)   = {C_LB:.10f}",
        f"  C_GME   = {C_GME:.10f}",
        f"  C_UB (upper)   = {C_UB:.10f}",
        f"  Best C_GME (optimizer)     = {best_C:.10f}  [{best_m}]",
        f"  Bounds : C_LB <= C_GME <= C_UB?     = {'YES' if C_LB <= best_C <= C_UB + 1e-6 else 'CHECK'}",
        "",
        "  Bipartition purities",
        "  " + "-" * 55,
        f"  {'A':<20} {'B':<20} {'Tr(rho_A^2)':>14}",
        "  " + "-" * 55,
    ]
    for b in sorted(bips, key=lambda x: (len(x["A"]), x["A"])):
        lines.append(
            f"  {str(b['A']):<20} {str(b['B']):<20} {b['purity_A']:>14.10f}"
        )

    lines += [
        "",
        "  Method results",
        "  " + "-" * 80,
        f"  {'Method':<18} {'C_GME':>14} {'Time(s)':>9} {'Restarts':>9} "
        f"{'Evals':>7} {'Spread':>12}",
        "  " + "-" * 72,
    ]
    for mname, mres in methods.items():
        marker = " " if mname == best_m else ""
        lines.append(
            f"  {mname:<18} {mres['C_GME']:>14.10f} {mres['time_s']:>9.2f} "
            f"{mres['n_restarts']:>9} {mres['n_evals']:>7} "
            f"{mres['spread']:>12.2e}{marker}"
        )

    decomp = methods.get(best_m, {}).get("decomp", [])
    if decomp:
        lines += [
            "",
            f"  Pure-state decomposition  [{best_m}]",
            "  " + "-" * 60,
            f"  {'#':>4} {'weight p_i':>14} {'C_GME(|psi_i>)':>16} {'p_i*C_GME':>14}",
            "  " + "-" * 52,
        ]
        for d in decomp:
            lines.append(
                f"  {d['component']:>4} {d['weight_p']:>14.10f} "
                f"{d['C_GME_i']:>16.10f} {d['contribution']:>14.10f}"
            )
        total_sum = sum(d["contribution"] for d in decomp)
        lines += [
            "  " + "-" * 52,
            f"  {'Sum':>4} {'':>14} {'':>16} {total_sum:>14.10f}",
            f"  ( best C_GME = {best_C:.10f})",
        ]

    lines += ["", _SEP2]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_comparison_pdf(
    results:      Dict,
    system_info:  dict,
    output_path:  str            = "comparison_report.pdf",
    n_restarts:   int            = 50,
    ground_state: Optional[Dict] = None,
    rdm:          Optional[Dict] = None,
    metrics:      Optional[Dict] = None,
    gme:          Optional[Dict] = None,
) -> None:
    if not results:
        print("  [WARN] No results -- nothing to write.")
        return

    methods    = list(next(iter(results.values()))["methods"].keys())
    n_clusters = len(results)
    summary    = _build_summary(results)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"\n  Generating PDF -> {output_path}")
    with PdfPages(output_path) as pdf:
        _new_page(pdf, _page_system(system_info, n_clusters, methods, n_restarts))
        _new_page(pdf, _page_cluster_overview(results))
        _new_page(pdf, _page_method_summary(summary, results))
        _new_page(pdf, _page_timing(summary, results))
        for sites, sr in sorted(results.items(), key=lambda x: (x[1]["n"], x[0])):
            _new_page(pdf, _page_cluster_detail(sites, sr))

    total_pages = 4 + n_clusters
    print(f"  PDF saved  -> {output_path}  ({total_pages} pages)")