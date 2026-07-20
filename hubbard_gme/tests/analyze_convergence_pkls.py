#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_report.py  (merged, all-methods edition)
==================================================

Read ALL cached convergence PKLs in a folder (plus optional system / RDM PKLs),
then generate -- WITHOUT recomputing anything:

  1. A printed SANITY REPORT (what is inside, whether a comparison is possible,
     warnings such as missing GCG, non-finite or placeholder histories,
     cross-file disagreements).
  2. PDF FIGURES comparing every available method as a function of cost h(U) vs
     iteration: GCG (Method I), each Stiefel sub-optimizer (Method II), AND the
     Stiefel-GCG hybrid (Method III) overlaid, per cluster size n, plus summary
     figures (final cost / iterations / runtime vs n) and outer-k convergence.
  3. A CSV of final costs, timings, decomposition rank and k.
  4. A self-contained LaTeX report (report.tex) plus standalone, \\input-able
     booktabs table snippets in tables/.

This script only reads cached PKL files; it never re-runs the optimizers.

What changed vs the original
----------------------------------------------------------------------------------
* Data is read from each PKL's `all_method_results` block, so EVERY method shows
  up -- GCG, all Stiefel sub-optimisers, and the Stiefel-GCG hybrid -- not just
  GCG + the single primary Stiefel optimiser. (Old PKLs without that block fall
  back to the legacy gcg_*/st_* prefixes automatically.)
* Stiefel-GCG is a first-class series ("Method III") with its own colour/label.
* Every output filename carries a YYYYMMDD_HHMMSS stamp, so reruns never
  overwrite previous PDFs / tables / reports.
* The cost-vs-iteration plots show the full cost equation on the y-axis.

----------------------------------------------------------------------------------
Usage
----------------------------------------------------------------------------------
    # discover every conv_*.pkl / convergence_full_run_*.pkl in a folder
    python generate_report.py --input  results/.../convergence_runs \
                              --system-pkl results/.../system_v1_....pkl \
                              --rdm-pkl    results/.../rdm_v1_*_.pkl \
                              --outdir report_out --src ../src --compile

    # or pass specific files explicitly (can combine with --input)
    python generate_report.py --conv-pkl runA.pkl runB.pkl --outdir report_out

Only convergence PKLs are required. --system-pkl / --rdm-pkl just add metadata.
--src points at the dir containing the 'hubbard_gme' package; it is inserted on
sys.path BEFORE unpickling so any objects stored in the *_raw fields resolve.
--compile runs pdflatex (via latexmk if present) on the generated report.tex.
"""

from __future__ import annotations

import os
import sys
import csv
import argparse
import pickle
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------
# Matplotlib defaults
# ----------------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "lines.linewidth": 1.4,
    "lines.markersize": 3.2,
})

EPS = 1e-12
XLABEL = "Iteration (callback index)"
# Full cost equation for the convergence-plot y-axes.
YLABEL_COST = r"Cost $h(U)=\sum_i p_i\, C_{\mathrm{GME}}(|\psi_i\rangle)$"

# Known series: stable colours / markers.
# GCG = Method I; Stiefel optimisers = Method II; Stiefel-GCG hybrid = Method III.
KNOWN_SERIES = [
    ("gcg",                   "GCG (Method\\,I)",                     "#1f77b4", "o"),
    ("stiefel:L-BFGS-B",      "Stiefel L-BFGS-B (Method\\,II)",       "#ff7f0e", "s"),
    ("stiefel:Powell",        "Stiefel Powell (Method\\,II)",         "#2ca02c", "^"),
    ("stiefel:SLSQP",         "Stiefel SLSQP (Method\\,II)",          "#9467bd", "D"),
    ("stiefel:TNC",           "Stiefel TNC (Method\\,II)",            "#d62728", "v"),
    ("stiefel:trust-constr",  "Stiefel trust-constr (Method\\,II)",   "#8c564b", "P"),
    ("stiefel_gcg",           "Stiefel-GCG (Method\\,III)",           "#111111", "X"),
]
_KNOWN_KEYS = [k for (k, _, _, _) in KNOWN_SERIES]
_FALLBACK_COLORS = ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
_FALLBACK_MARKERS = ["*", "X", "p", "h"]
_style_cache = {}

STIEFEL_OPTS = ["L-BFGS-B", "Powell", "SLSQP", "TNC", "trust-constr"]


def series_style(key):
    """Return (display_name_latex, color, marker) for a series key."""
    for (k, disp, col, mk) in KNOWN_SERIES:
        if k == key:
            return disp, col, mk
    if key in _style_cache:
        return _style_cache[key]
    opt = key.split(":", 1)[1] if ":" in key else key
    idx = len(_style_cache)
    disp = f"Stiefel {opt} (Method\\,II)"
    col = _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]
    mk = _FALLBACK_MARKERS[idx % len(_FALLBACK_MARKERS)]
    _style_cache[key] = (disp, col, mk)
    return _style_cache[key]


def display_name(key):
    return series_style(key)[0]


def legend_label(key):
    """Short label for plot legends."""
    if key == "gcg":
        return "GCG"
    if key == "stiefel_gcg":
        return "Stiefel-GCG"
    if key.startswith("stiefel:"):
        return "Stiefel " + key.split(":", 1)[1]
    return key


def order_series_keys(keys):
    keys = set(keys)
    present_known = [k for k in _KNOWN_KEYS if k in keys]
    unknown = sorted(k for k in keys if k not in _KNOWN_KEYS)
    return present_known + unknown


# ==================================================================================
# Loading helpers
# ==================================================================================
def load_pickle(path: Path, src_dirs):
    for s in src_dirs:
        if s and os.path.isdir(s) and os.path.abspath(s) not in sys.path:
            sys.path.insert(0, os.path.abspath(s))
    with open(path, "rb") as f:
        return pickle.load(f)


def as_int_key_dict(d):
    if not isinstance(d, dict):
        return {}
    out = {}
    for k, v in d.items():
        try:
            ik = int(k)
        except (ValueError, TypeError):
            ik = k
        out[ik] = v
    return out


def infer_optimizer(path: Path, data: dict) -> str:
    """config.stiefel_optimizer -> filename token -> 'L-BFGS-B' (legacy default)."""
    cfg = data.get("config", {}) or {}
    opt = cfg.get("stiefel_optimizer")
    if opt:
        return str(opt)
    name = path.name
    for cand in STIEFEL_OPTS:
        if cand in name:
            return cand
    return "L-BFGS-B"


def _raw_get(raw, *keys):
    for key in keys:
        if isinstance(raw, dict):
            if key in raw and raw[key] is not None:
                return raw[key]
        else:
            val = getattr(raw, key, None)
            if val is not None:
                return val
    return None


def extract_rank(raw):
    return _raw_get(raw, "rank")


def extract_k(raw):
    k = _raw_get(raw, "k_converged")
    if k is not None:
        return k
    br = _raw_get(raw, "best_record")
    if isinstance(br, dict):
        return br.get("k")
    if br is not None:
        return getattr(br, "k", None)
    return None


def n_monotonic_increases(hist: np.ndarray) -> int:
    if hist.size < 2:
        return 0
    return int(np.sum(np.diff(hist) > 1e-10))


def extract_record(per_n: dict, prefix: str, source_file: str, optimizer=None):
    """Pull one method's data from a results_by_n[n] entry; None if absent.

    Legacy path: reads the gcg_*/st_* prefixes used by older PKLs.
    """
    hk, fk = f"{prefix}_hist", f"{prefix}_final"
    if hk not in per_n or fk not in per_n:
        return None
    if per_n[hk] is None or per_n[fk] is None:
        return None

    hist = np.asarray(per_n[hk], dtype=float).ravel()
    final = float(per_n[fk])
    time_s = float(per_n.get(f"{prefix}_time", 0.0) or 0.0)
    raw = per_n.get(f"{prefix}_raw", {})

    if per_n.get(f"{prefix}_loaded_from_job_cache", False):
        source = "job_cache"
    elif per_n.get(f"{prefix}_loaded_from_full_pkl", False):
        source = "full_pkl"
    else:
        source = "computed"

    return {
        "hist": hist,
        "final": final,
        "time": time_s,
        "iters": max(int(hist.size) - 1, 0),
        "n_points": int(hist.size),
        "source": source,
        "source_file": source_file,
        "optimizer": optimizer,
        "rank": extract_rank(raw),
        "k": extract_k(raw),
        "raw": raw,
    }


def _amr_tag_to_key(tag):
    """Map an all_method_results tag to a report series key."""
    if tag == "gcg":
        return "gcg"
    if tag == "methodII_gcg":
        return "stiefel_gcg"
    if tag.startswith("methodII_"):
        return "stiefel:" + tag.split("_", 1)[1]
    return str(tag)


def extract_record_from_amr(tag, entry, source_file):
    """Build a record (same schema as extract_record) from an all_method_results entry.

    Returns (series_key, record) or (None, None) if the entry is unusable.
    """
    if not isinstance(entry, dict):
        return None, None
    if entry.get("history") is None or entry.get("final") is None:
        return None, None

    hist = np.asarray(entry.get("history", []), dtype=float).ravel()
    raw = entry.get("raw", {}) or {}

    source = entry.get("source")
    if not source:
        if entry.get("loaded_from_job_cache"):
            source = "job_cache"
        elif entry.get("loaded_from_full_pkl"):
            source = "full_pkl"
        else:
            source = "computed"

    rec = {
        "hist": hist,
        "final": float(entry.get("final")),
        "time": float(entry.get("elapsed", 0.0) or 0.0),
        "iters": max(int(hist.size) - 1, 0),
        "n_points": int(hist.size),
        "source": source,
        "source_file": source_file,
        "optimizer": entry.get("stiefel_optimizer"),
        "rank": extract_rank(raw),
        "k": extract_k(raw),
        "raw": raw,
    }
    return _amr_tag_to_key(tag), rec


def discover_conv_files(input_dir, explicit):
    files = []
    if input_dir:
        d = Path(input_dir)
        if d.is_dir():
            files += sorted(d.glob("conv_*.pkl"))
            files += sorted(d.glob("convergence_full_run_*.pkl"))
    if explicit:
        for e in explicit:
            files.append(Path(e))
    # dedup preserving order
    seen, uniq = set(), []
    for f in files:
        rp = str(f.resolve()) if f.exists() else str(f)
        if rp not in seen:
            seen.add(rp)
            uniq.append(f)
    return uniq


# ==================================================================================
# Build merged dataset
# ==================================================================================
def build_dataset(conv_files, src_dirs):
    """
    Returns dict with:
        data_by_n : {n: {series_key: record}}
        n_list    : sorted list of n
        series_present : ordered list of series keys
        system_meta : first system_state_ref carrying E0
        rdm_meta : {n: {sites, metrics}}
        config : representative config (first file) + merged fields
        files : list of Path actually used
        warnings : list[str]
    """
    data_by_n = {}
    rdm_meta = {}
    system_meta = {}
    configs = []
    files_used = []
    warnings = []

    def merge_record(n, key, rec):
        data_by_n.setdefault(n, {})
        if key in data_by_n[n]:
            old = data_by_n[n][key]
            if abs(old["final"] - rec["final"]) > 1e-6:
                warnings.append(
                    f"n={n} {key}: final differs across files "
                    f"({old['final']:.6f} in {old['source_file']} vs "
                    f"{rec['final']:.6f} in {rec['source_file']}); keeping first.")
            return
        data_by_n[n][key] = rec

    for path in conv_files:
        try:
            data = load_pickle(path, src_dirs)
        except ModuleNotFoundError as e:
            warnings.append(f"{path.name}: needs missing module {e!r}; "
                            f"set --src to the dir with 'hubbard_gme'. Skipped.")
            continue
        except Exception as e:
            warnings.append(f"{path.name}: could not load ({e!r}); skipped.")
            continue

        if not isinstance(data, dict) or "results_by_n" not in data:
            warnings.append(f"{path.name}: not a convergence PKL (no results_by_n); skipped.")
            continue

        files_used.append(path)
        cfg = data.get("config", {}) or {}
        configs.append((path, cfg))
        opt = infer_optimizer(path, data)

        ssr = data.get("system_state_ref", {}) or {}
        if not system_meta and ssr.get("E0") is not None:
            system_meta = ssr

        for n, ref in as_int_key_dict(data.get("rdm_refs", {}) or {}).items():
            if n not in rdm_meta and isinstance(ref, dict):
                rdm_meta[n] = {"sites": ref.get("sites"),
                               "metrics": ref.get("metrics", {}) or {}}

        for n, per_n in as_int_key_dict(data.get("results_by_n", {}) or {}).items():
            if not isinstance(per_n, dict):
                continue
            amr = per_n.get("all_method_results", {}) or {}
            if amr:
                # Preferred path: read every method (GCG, all Stiefel optimisers,
                # and the Stiefel-GCG hybrid) from the rich all_method_results block.
                for tag, entry in amr.items():
                    key, rec = extract_record_from_amr(tag, entry, path.name)
                    if rec is not None:
                        merge_record(n, key, rec)
            else:
                # Back-compat: older PKLs only carry the gcg_*/st_* prefixes.
                rec_g = extract_record(per_n, "gcg", path.name)
                if rec_g is not None:
                    merge_record(n, "gcg", rec_g)
                rec_s = extract_record(per_n, "st", path.name, optimizer=opt)
                if rec_s is not None:
                    merge_record(n, f"stiefel:{opt}", rec_s)

    # representative config + consistency checks
    ref_cfg = dict(configs[0][1]) if configs else {}
    for path, cfg in configs[1:]:
        for k in ["L", "Ne", "U", "t", "mode", "n_so"]:
            if k in cfg and k in ref_cfg and cfg[k] != ref_cfg[k]:
                warnings.append(f"config mismatch '{k}': {ref_cfg[k]} vs {cfg[k]} in {path.name}")
    # merge restarts/max_iter -> value if consistent else 'varies'
    for k in ["n_restarts", "max_iter"]:
        vals = {cfg.get(k) for _, cfg in configs if cfg.get(k) is not None}
        if len(vals) == 1:
            ref_cfg[k] = next(iter(vals))
        elif len(vals) > 1:
            ref_cfg[k] = "varies"
    n_list = sorted(data_by_n.keys(), key=lambda x: (isinstance(x, str), x))
    keys_present = order_series_keys(
        {key for n in data_by_n for key in data_by_n[n]})

    # optimizers present (Stiefel) + hybrid flag
    opts_present = [k.split(":", 1)[1] for k in keys_present if k.startswith("stiefel:")]
    ref_cfg["_stiefel_optimizers_present"] = opts_present
    ref_cfg["_has_hybrid"] = "stiefel_gcg" in keys_present
    ref_cfg["_n_list_union"] = n_list

    return {
        "data_by_n": data_by_n,
        "n_list": n_list,
        "series_present": keys_present,
        "system_meta": system_meta,
        "rdm_meta": rdm_meta,
        "config": ref_cfg,
        "files": files_used,
        "warnings": warnings,
    }


# ==================================================================================
# Sanity report
# ==================================================================================
def sanity_report(ds: dict):
    line = "=" * 90
    print(line)
    print("SANITY REPORT")
    print(line)
    print("Convergence files merged:", len(ds["files"]))
    for p in ds["files"]:
        print("   -", p.name)
    cfg = ds["config"]
    print("-" * 90)
    print("System            : L=%s  Ne=%s  U=%s  t=%s  mode=%s  n_so=%s"
          % (cfg.get("L"), cfg.get("Ne"), cfg.get("U"), cfg.get("t"),
             cfg.get("mode"), cfg.get("n_so")))
    print("Restarts / max_iter: %s / %s" % (cfg.get("n_restarts"), cfg.get("max_iter")))
    print("Stiefel optimisers : %s"
          % (", ".join(cfg.get("_stiefel_optimizers_present", [])) or "none"))
    print("Stiefel-GCG hybrid : %s" % ("present" if cfg.get("_has_hybrid") else "absent"))
    ssr = ds["system_meta"]
    if ssr.get("E0") is not None:
        print("Ground state E0   : %.10f" % float(ssr["E0"]))
    if ssr.get("E1") is not None:
        print("First excited E1  : %.10f" % float(ssr["E1"]))
    print("-" * 90)

    hdr = "%-4s | %-26s | %-7s | %-8s | %-14s | %-9s | %s"
    print(hdr % ("n", "series", "points", "iters", "h_final", "source", "file"))
    print("-" * 90)
    for n in ds["n_list"]:
        for key in order_series_keys(ds["data_by_n"][n].keys()):
            rec = ds["data_by_n"][n][key]
            warn = ""
            if not np.all(np.isfinite(rec["hist"])):
                warn += " [NON-FINITE!]"
            ups = n_monotonic_increases(rec["hist"])
            if ups > 0:
                warn += f" [{ups} up-steps]"
            if rec["n_points"] <= 1:
                warn += " [placeholder hist]"
            print((hdr % (n, legend_label(key), rec["n_points"], rec["iters"],
                          f"{rec['final']:.8f}", rec["source"],
                          rec["source_file"])) + warn)
    print("-" * 90)

    if ds["warnings"]:
        print("WARNINGS:")
        for w in ds["warnings"]:
            print("   !", w)
        print("-" * 90)

    keys = ds["series_present"]
    has_gcg = "gcg" in keys
    has_hybrid = "stiefel_gcg" in keys
    stiefel_keys = [k for k in keys if k.startswith("stiefel:")]
    paired_n = [n for n in ds["n_list"]
                if "gcg" in ds["data_by_n"][n]
                and any(k.startswith("stiefel:") for k in ds["data_by_n"][n])]

    print("=" * 90)
    if has_gcg and stiefel_keys and paired_n:
        print("VERDICT: OK. GCG and Stiefel are both present for n =", paired_n)
        print("         -> GCG (Method I) vs Stiefel (Method II) comparison can be produced.")
        if len(stiefel_keys) > 1:
            print("         Bonus: %d Stiefel optimisers present (%s) -> cross-optimiser "
                  "comparison too."
                  % (len(stiefel_keys), ", ".join(k.split(':')[1] for k in stiefel_keys)))
        if has_hybrid:
            print("         Plus: Stiefel-GCG hybrid (Method III) present -> full "
                  "three-way comparison.")
    elif stiefel_keys and not has_gcg:
        print("VERDICT: NO GCG. Only Stiefel results are present "
              "(consistent with 'methods-stiefel' filenames).")
        if len(stiefel_keys) > 1:
            print("         A Stiefel-vs-Stiefel (cross-optimiser) comparison IS available.")
        print("         For Method I vs Method II, also run GCG, e.g.:")
        print("            python convergence_benchmark.py --methods gcg "
              "--n-restarts 10 --max-iter 5000")
        print("         then re-run this report (GCG is picked up from cache).")
    elif has_gcg and not stiefel_keys:
        print("VERDICT: GCG ONLY. No Stiefel results present; run --methods stiefel.")
    else:
        print("VERDICT: NO USABLE METHOD DATA FOUND.")
    print("=" * 90)
    print()

    ds["has_gcg"] = has_gcg
    ds["has_hybrid"] = has_hybrid
    ds["stiefel_keys"] = stiefel_keys
    ds["paired_n"] = paired_n
    ds["comparison_possible"] = bool(has_gcg and stiefel_keys and paired_n)
    return ds


# ==================================================================================
# Figures
# ==================================================================================
def fig_single_n(per_n: dict, n, save_path, suffix=""):
    keys = order_series_keys(per_n.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    finals = [per_n[k]["final"] for k in keys]
    h_min = min(finals) if finals else 0.0

    ax = axes[0]
    for key in keys:
        rec = per_n[key]
        _, col, mk = series_style(key)
        ax.plot(np.arange(rec["hist"].size), rec["hist"],
                marker=mk, linestyle="-", color=col, label=legend_label(key))
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL_COST)
    ax.set_title(f"n={n}: cost vs iteration{suffix}")
    ax.legend(loc="best")

    ax = axes[1]
    for key in keys:
        rec = per_n[key]
        _, col, mk = series_style(key)
        err = np.maximum(rec["hist"] - h_min, EPS)
        ax.semilogy(np.arange(err.size), err,
                    marker=mk, linestyle="-", color=col, label=legend_label(key))
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(r"$h(U)-h_{\min}$")
    ax.set_title(f"n={n}: log error to best minimum{suffix}")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print("    saved figure ->", save_path)


def fig_combined(ds, save_path, suffix=""):
    data_by_n = ds["data_by_n"]
    n_values = [n for n in ds["n_list"] if data_by_n[n]]
    if not n_values:
        return None
    nrows = len(n_values)
    fig, axes = plt.subplots(nrows, 2, figsize=(12, 4.2 * nrows))
    if nrows == 1:
        axes = np.array(axes).reshape(1, -1)

    for row, n in enumerate(n_values):
        per_n = data_by_n[n]
        keys = order_series_keys(per_n.keys())
        finals = [per_n[k]["final"] for k in keys]
        h_min = min(finals) if finals else 0.0
        ax0, ax1 = axes[row, 0], axes[row, 1]
        for key in keys:
            rec = per_n[key]
            _, col, mk = series_style(key)
            ax0.plot(np.arange(rec["hist"].size), rec["hist"],
                     marker=mk, linestyle="-", color=col, label=legend_label(key))
            err = np.maximum(rec["hist"] - h_min, EPS)
            ax1.semilogy(np.arange(err.size), err,
                         marker=mk, linestyle="-", color=col, label=legend_label(key))
        ax0.set_xlabel(XLABEL); ax0.set_ylabel(YLABEL_COST)
        ax0.set_title(f"n={n}: cost vs iteration"); ax0.legend(loc="best")
        ax1.set_xlabel(XLABEL); ax1.set_ylabel(r"$h(U)-h_{\min}$")
        ax1.set_title(f"n={n}: log error"); ax1.legend(loc="best")

    fig.suptitle(f"Convergence comparison{suffix}", fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print("    saved figure ->", save_path)
    return save_path


def fig_final_vs_n(ds, save_path):
    data_by_n = ds["data_by_n"]
    ns = [n for n in ds["n_list"] if isinstance(n, int)]
    if not ns:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    plotted = False
    for key in ds["series_present"]:
        xs, ys = [], []
        for n in ns:
            rec = data_by_n[n].get(key)
            if rec is not None:
                xs.append(n); ys.append(rec["final"])
        if xs:
            plotted = True
            _, col, mk = series_style(key)
            ax.plot(xs, ys, marker=mk, linestyle="-", color=col, label=legend_label(key))
    if not plotted:
        plt.close(fig); return None
    ax.set_xlabel(r"Cluster size $n$ (sites)")
    ax.set_ylabel(r"$h_{\mathrm{final}} \equiv C_{\mathrm{GME}}$ estimate")
    ax.set_title("Final cost vs cluster size")
    ax.set_xticks(ns); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    print("    saved figure ->", save_path)
    return save_path


def _grouped_bar(ds, value_key, ylabel, title, save_path):
    data_by_n = ds["data_by_n"]
    ns = [n for n in ds["n_list"] if isinstance(n, int)]
    keys = ds["series_present"]
    if not ns or not keys:
        return None
    x = np.arange(len(ns), dtype=float)
    width = 0.8 / max(len(keys), 1)
    fig, ax = plt.subplots(figsize=(max(7.4, 1.6 * len(ns)), 4.8))
    for i, key in enumerate(keys):
        vals = [data_by_n[n].get(key, {}).get(value_key, np.nan) for n in ns]
        offset = (i - (len(keys) - 1) / 2.0) * width
        _, col, _ = series_style(key)
        ax.bar(x + offset, vals, width=width, color=col, label=legend_label(key))
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel(r"Cluster size $n$ (sites)")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(loc="best"); ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)
    print("    saved figure ->", save_path)
    return save_path


def fig_cost_vs_time(ds, save_path):
    """
    Fair cross-method efficiency view: final cost vs wall-clock time per job,
    one point per series, faceted by cluster size n. Lower-left is better
    (low cost, low time). Uses only the stored per-job elapsed times.
    """
    data_by_n = ds["data_by_n"]
    ns = [n for n in ds["n_list"] if data_by_n[n]]
    if not ns:
        return None
    # need at least some non-zero timing to be meaningful
    all_times = [rec["time"] for n in ns for rec in data_by_n[n].values()]
    if not all_times or max(all_times) <= 0.0:
        print("    (skipped final_cost_vs_time: no wall-clock times stored)")
        return None

    ncols = len(ns)
    fig, axes = plt.subplots(1, ncols, figsize=(4.7 * ncols, 4.4), squeeze=False)
    axes = axes[0]
    for j, n in enumerate(ns):
        ax = axes[j]
        for key in order_series_keys(data_by_n[n].keys()):
            rec = data_by_n[n][key]
            _, col, mk = series_style(key)
            ax.scatter([rec["time"]], [rec["final"]], color=col, marker=mk, s=70,
                       edgecolor="k", linewidth=0.4, zorder=3, label=legend_label(key))
        ax.set_xlabel("Wall-clock time per job [s]")
        ax.set_ylabel(r"$h_{\mathrm{final}} \equiv C_{\mathrm{GME}}$")
        ax.set_title(f"n={n}")
        ax.legend(loc="best", fontsize=7)
    fig.suptitle("Final cost vs wall-clock time (lower-left is better)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print("    saved figure ->", save_path)
    return save_path


def fig_normalized_combined(ds, save_path, suffix=""):
    """
    Compare convergence SHAPE on a common x-axis: cost vs fraction of recorded
    iterations (0..1). This makes curves comparable even when their iteration
    counts differ (they are method-/optimiser-specific units of work).
    """
    data_by_n = ds["data_by_n"]
    n_values = [n for n in ds["n_list"] if data_by_n[n]]
    if not n_values:
        return None
    nrows = len(n_values)
    fig, axes = plt.subplots(nrows, 1, figsize=(7.6, 3.7 * nrows), squeeze=False)
    axes = axes[:, 0]
    for row, n in enumerate(n_values):
        ax = axes[row]
        for key in order_series_keys(data_by_n[n].keys()):
            rec = data_by_n[n][key]
            h = rec["hist"]
            x = np.linspace(0.0, 1.0, h.size) if h.size >= 2 else np.array([0.0])
            _, col, mk = series_style(key)
            ax.plot(x, h, marker=mk, linestyle="-", color=col,
                    markevery=max(1, h.size // 20), label=legend_label(key))
        ax.set_xlabel("Fraction of recorded iterations")
        ax.set_ylabel(YLABEL_COST)
        ax.set_title(f"n={n}: convergence shape (normalized iteration)")
        ax.legend(loc="best")
    fig.suptitle(f"Convergence vs normalized iteration{suffix}",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print("    saved figure ->", save_path)
    return save_path


def fig_outer_k(ds, fig_dir, stamp=""):
    """Outer-k convergence for any method that records a k-loop (Stiefel
    optimisers, the Stiefel-GCG hybrid, and GCG)."""
    out = []
    sfx = f"_{stamp}" if stamp else ""
    for n in ds["n_list"]:
        for key in order_series_keys(ds["data_by_n"][n].keys()):
            rec = ds["data_by_n"][n][key]
            raw = rec.get("raw")
            k_history = _raw_get(raw, "k_history")
            if k_history is None:
                kr = _raw_get(raw, "k_records")
                if kr is not None:
                    try:
                        k_history = [float(r["C_GME"]) for r in kr]
                    except Exception:
                        k_history = None
            if not k_history:
                continue
            k_history = np.asarray(k_history, dtype=float)
            if k_history.size < 2:
                continue
            label = legend_label(key)
            fig, ax = plt.subplots(figsize=(6.6, 4.4))
            _, col, mk = series_style(key)
            ax.plot(np.arange(k_history.size), k_history, marker=mk, linestyle="-", color=col)
            ax.set_xlabel(r"Outer step $k-r$"); ax.set_ylabel(r"$C_{\mathrm{GME}}$")
            ax.set_title(f"{label} outer-k convergence, n={n}")
            fig.tight_layout()
            p = fig_dir / f"outer_k_{safe_tag(key)}_n{n}{sfx}.pdf"
            fig.savefig(p, bbox_inches="tight"); plt.close(fig)
            print("    saved figure ->", p)
            out.append((n, label, p))
    return out


def safe_tag(x):
    return (str(x).replace(" ", "").replace("/", "-").replace("\\", "-")
            .replace(".", "p").replace("+", "").replace(":", "-"))


# ==================================================================================
# CSV
# ==================================================================================
def write_csv(ds, path):
    cols = ["n", "method", "series_key", "optimizer", "final_cost",
            "callback_points", "time_s", "time_min", "rank", "k",
            "source", "source_file"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for n in ds["n_list"]:
            for key in order_series_keys(ds["data_by_n"][n].keys()):
                rec = ds["data_by_n"][n][key]
                disp = display_name(key).replace("\\,", " ")
                w.writerow([n, disp, key, rec.get("optimizer") or "",
                            f"{rec['final']:.10f}", rec["iters"],
                            f"{rec['time']:.4f}", f"{rec['time']/60.0:.4f}",
                            rec.get("rank"), rec.get("k"),
                            rec["source"], rec["source_file"]])
    print("    wrote", path)


# ==================================================================================
# LaTeX helpers
# ==================================================================================
def latex_escape(s) -> str:
    s = str(s)
    # NOTE: substitute backslash FIRST, then the rest, so we don't double-escape
    # the backslashes introduced by later replacements.
    s = s.replace("\\", r"\textbackslash{}")
    for k, v in {
        "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }.items():
        s = s.replace(k, v)
    return s


def fmt_g(x):
    try:
        return f"{float(x):g}"
    except (ValueError, TypeError):
        return latex_escape(x)


def fmt_cost(x):
    return f"{float(x):.6f}"


def fmt_time(x):
    return f"{float(x):.2f}"


def fmt_sci_latex(x):
    x = float(x)
    if x <= 0 or x < 1e-300:
        return "$0$"
    exp = int(np.floor(np.log10(x)))
    mant = x / (10.0 ** exp)
    return f"${mant:.2f}\\times 10^{{{exp}}}$"


def build_system_table(ds):
    cfg = ds["config"]; ssr = ds["system_meta"]
    rows = [
        (r"Lattice sites $L$", fmt_g(cfg.get("L"))),
        (r"Electrons $N_e$", fmt_g(cfg.get("Ne"))),
        (r"Interaction $U$", fmt_g(cfg.get("U"))),
        (r"Hopping $t$", fmt_g(cfg.get("t"))),
        (r"Mode", latex_escape(cfg.get("mode"))),
        (r"Spin-orbitals $n_{\mathrm{so}}$", fmt_g(cfg.get("n_so"))),
    ]
    nl = cfg.get("_n_list_union")
    rows.append((r"Cluster sizes $n$",
                 latex_escape(", ".join(str(x) for x in nl)) if nl else "--"))
    opts = cfg.get("_stiefel_optimizers_present", [])
    rows.append((r"Stiefel optimisers compared",
                 latex_escape(", ".join(opts)) if opts else "--"))
    rows.append((r"GCG (Method I) present", "yes" if ds.get("has_gcg") else "no"))
    rows.append((r"Stiefel-GCG (Method III) present",
                 "yes" if cfg.get("_has_hybrid") else "no"))
    rows.append((r"Restarts per method", latex_escape(cfg.get("n_restarts"))))
    rows.append((r"Max iterations", latex_escape(cfg.get("max_iter"))))
    if ssr.get("E0") is not None:
        rows.append((r"Ground-state energy $E_0$", f"{float(ssr['E0']):.8f}"))
    if ssr.get("E1") is not None:
        rows.append((r"First-excited energy $E_1$", f"{float(ssr['E1']):.8f}"))

    lines = [r"\begin{tabular}{ll}", r"\toprule", r"Quantity & Value \\", r"\midrule"]
    for k, v in rows:
        lines.append(f"{k} & {v} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def build_rdm_table(ds):
    rdm = ds["rdm_meta"]
    if not rdm:
        return None
    lines = [r"\begin{tabular}{cccc c}", r"\toprule",
             r"$n$ & Sites & Purity $\mathrm{Tr}\,\rho^2$ & Entropy $S$ & Num. rank \\",
             r"\midrule"]
    any_row = False
    for n in ds["n_list"]:
        ref = rdm.get(n)
        if not isinstance(ref, dict):
            continue
        m = ref.get("metrics", {}) or {}
        sites = ref.get("sites")
        sites_str = (latex_escape("(" + ",".join(str(s) for s in sites) + ")")
                     if sites is not None else "--")
        purity, entropy, rank = m.get("purity"), m.get("entropy"), m.get("rank_numerical")
        if purity is None and entropy is None and rank is None:
            continue
        any_row = True
        lines.append("%s & %s & %s & %s & %s \\\\" % (
            n, sites_str,
            f"{float(purity):.6f}" if purity is not None else "--",
            f"{float(entropy):.6f}" if entropy is not None else "--",
            fmt_g(rank) if rank is not None else "--"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) if any_row else None


def build_summary_table(ds):
    lines = [r"\begin{tabular}{c l r r r r}", r"\toprule",
             r"$n$ & Method & $h_{\mathrm{final}}$ & Iter. & Time [s] & Gap to best \\",
             r"\midrule"]
    for n in ds["n_list"]:
        per_n = ds["data_by_n"][n]
        keys = order_series_keys(per_n.keys())
        finals = [per_n[k]["final"] for k in keys]
        h_best = min(finals) if finals else None
        block = []
        for key in keys:
            rec = per_n[key]
            gap = (rec["final"] - h_best) if h_best is not None else 0.0
            gap_str = fmt_sci_latex(gap) if gap > EPS else "$0$"
            block.append("%s & %s & %s & %d & %s & %s \\\\" % (
                str(n), display_name(key), fmt_cost(rec["final"]),
                rec["iters"], fmt_time(rec["time"]), gap_str))
        if block:
            lines.extend(block)
            lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def build_provenance_table(ds):
    lines = [r"\begin{tabular}{c l l l}", r"\toprule",
             r"$n$ & Method & Source & File \\", r"\midrule"]
    for n in ds["n_list"]:
        for key in order_series_keys(ds["data_by_n"][n].keys()):
            rec = ds["data_by_n"][n][key]
            lines.append("%s & %s & %s & {\\scriptsize\\texttt{%s}} \\\\" % (
                str(n), display_name(key), latex_escape(rec["source"]),
                latex_escape(rec["source_file"])))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def build_observations(ds):
    data_by_n = ds["data_by_n"]
    keys = ds["series_present"]
    has_gcg = ds.get("has_gcg")
    has_hybrid = ds.get("has_hybrid")
    stiefel_keys = ds.get("stiefel_keys", [])
    sents = []

    # spreads where >= 2 series exist
    spreads = []
    for n in ds["n_list"]:
        finals = [r["final"] for r in data_by_n[n].values()]
        if len(finals) >= 2:
            spreads.append(max(finals) - min(finals))
    max_spread = max(spreads) if spreads else None

    if has_gcg and stiefel_keys:
        opt_names = [k.split(":", 1)[1] for k in stiefel_keys]
        sents.append(
            "All series minimise the same convex-roof cost "
            "$h(U)=\\sum_i p_i\\,C_{\\mathrm{GME}}(|\\psi_i\\rangle)$. "
            "Method\\,I is the GCG scheme; Method\\,II is the Stiefel-manifold "
            "optimiser, here run with sub-optimiser" +
            ("s " if len(opt_names) > 1 else " ") +
            latex_escape(", ".join(opt_names)) + ".")
        if has_hybrid:
            sents.append(
                "Method\\,III is the Stiefel-GCG hybrid: Riemannian conjugate "
                "gradient performed directly on the Stiefel manifold, included here "
                "as a distinct series.")
        if max_spread is not None:
            if max_spread <= 1e-6:
                sents.append(
                    "Across all cluster sizes the methods reach the same minimum: "
                    f"the largest spread of final costs is {fmt_sci_latex(max_spread)}, "
                    "i.e.\\ agreement to numerical precision.")
            else:
                sents.append(
                    "The largest spread of final costs across methods and sizes is "
                    f"{fmt_sci_latex(max_spread)}.")
        # per-series iteration ranges
        rng = []
        for key in keys:
            its = [data_by_n[n][key]["iters"] for n in ds["n_list"] if key in data_by_n[n]]
            if its:
                rng.append(f"{legend_label(key)} {min(its)}--{max(its)}")
        if rng:
            sents.append("Recorded iteration counts (history length minus one) span: "
                         + latex_escape("; ".join(rng)) + ".")
    elif stiefel_keys and not has_gcg:
        opt_names = [k.split(":", 1)[1] for k in stiefel_keys]
        if len(stiefel_keys) > 1:
            sents.append(
                "This dataset contains \\emph{Stiefel (Method\\,II) results only}, but "
                "for several sub-optimisers (" + latex_escape(", ".join(opt_names)) +
                "), so a cross-optimiser comparison is shown below. No GCG "
                "(Method\\,I) history is stored, so a Method\\,I vs Method\\,II "
                "comparison is not yet possible from these files.")
        else:
            sents.append(
                "This dataset contains \\emph{Stiefel (Method\\,II) results only} "
                f"(sub-optimiser {latex_escape(opt_names[0])}); no GCG (Method\\,I) "
                "history is stored, so a head-to-head comparison is not yet possible "
                "from these files. To obtain it, run the GCG optimiser for the same "
                "system and regenerate this report.")
        if max_spread is not None:
            sents.append("The largest spread of final costs across the available "
                         f"series is {fmt_sci_latex(max_spread)}.")
    else:
        sents.append("Only a single method is available, so no comparison is possible.")
    return " ".join(sents)


# ==================================================================================
# LaTeX document
# ==================================================================================
LATEX_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{caption}
\usepackage[hidelinks]{hyperref}

\graphicspath{{figs/}{./}}
\setlength{\parskip}{0.5em}
\setlength{\parindent}{0pt}
"""


def assemble_report(ds, fig_paths, table_files, stamp=""):
    sfx = f"_{stamp}" if stamp else ""
    out = [LATEX_PREAMBLE]
    out.append(r"\title{Convergence comparison of convex-roof optimisers:\\"
               r" GCG (I), Stiefel (II) and Stiefel-GCG (III)}")
    out.append(r"\author{}")
    out.append(r"\date{\today}")
    out.append(r"\begin{document}")
    out.append(r"\maketitle")

    out.append(r"\section{Setup}")
    out.append(
        "This report compares algorithms for evaluating the genuine multipartite "
        "entanglement (GME) of a reduced density matrix $\\rho$ via the convex roof. "
        "Each algorithm minimises the same objective over pure-state decompositions "
        "$\\{p_i,|\\psi_i\\rangle\\}$ parameterised by an isometry $U$,")
    out.append(r"\begin{equation}"
               r"h(U)=\sum_i p_i\, C_{\mathrm{GME}}\big(|\psi_i\rangle\big),"
               r"\qquad C_{\mathrm{GME}}(\rho)=\min_{U} h(U)."
               r"\end{equation}")
    nfiles = len(ds["files"])
    out.append(
        f"Results are aggregated from {nfiles} cached file"
        f"{'s' if nfiles != 1 else ''}; nothing is recomputed. The quantity of "
        "interest is how the cost $h(U)$ decreases with iteration, and whether the "
        "methods reach the same minimum.")

    out.append(r"\begin{table}[H]\centering")
    out.append(r"\caption{System and optimiser configuration.}")
    out.append(r"\input{tables/system_table%s.tex}" % sfx)
    out.append(r"\end{table}")

    if table_files.get("rdm"):
        out.append(r"\begin{table}[H]\centering")
        out.append(r"\caption{Properties of the reduced density matrices used as input.}")
        out.append(r"\input{tables/rdm_table%s.tex}" % sfx)
        out.append(r"\end{table}")

    if not ds["comparison_possible"]:
        out.append(r"\section*{Note}")
        out.append(r"\fbox{\parbox{\linewidth}{%")
        out.append(build_observations(ds))
        out.append(r"}}")

    out.append(r"\section{Cost versus iteration}")
    out.append(build_observations(ds) if ds["comparison_possible"]
               else "The per-size convergence of the available method(s) is shown below.")
    out.append(
        r"\textbf{Note on the iteration axis.} The horizontal axis counts recorded "
        r"optimiser steps (callback points). This is a method- and optimiser-specific "
        r"unit of work: one GCG (Method\,I) step embeds an internal line search, one "
        r"Stiefel (Method\,II) step is a single iteration of the chosen sub-optimiser "
        r"(L-BFGS-B, TNC, Powell, \dots), and one Stiefel-GCG (Method\,III) step is a "
        r"single Riemannian conjugate-gradient iteration. Iteration \emph{counts} are "
        r"therefore not directly comparable across methods; the curves below show "
        r"per-method convergence behaviour, while \emph{efficiency} is compared via "
        r"final cost and wall-clock time (Section~\ref{sec:summary}). The plotted "
        r"history is the best restart at the converged cardinality $k$ for each method.")

    if fig_paths.get("combined"):
        out.append(r"\begin{figure}[H]\centering")
        out.append(r"\includegraphics[width=\linewidth]{%s}" % fig_paths["combined"].name)
        out.append(r"\caption{Cost $h(U)$ versus iteration (left) and log error "
                   r"$h(U)-h_{\min}$ (right), per cluster size $n$. All available "
                   r"methods are overlaid.}")
        out.append(r"\label{fig:combined}")
        out.append(r"\end{figure}")

    for n in ds["n_list"]:
        p = fig_paths["per_n"].get(n)
        if p is None:
            continue
        out.append(r"\begin{figure}[H]\centering")
        out.append(r"\includegraphics[width=\linewidth]{%s}" % p.name)
        out.append(r"\caption{$n=%s$: cost versus iteration (left) and log error "
                   r"to the best minimum (right).}" % str(n))
        out.append(r"\label{fig:n%s}" % str(n))
        out.append(r"\end{figure}")

    if fig_paths.get("normalized"):
        out.append(r"\subsection{Convergence shape on a common axis}")
        out.append("Because the iteration counts differ between methods and "
                   "sub-optimisers, the same histories are replotted against the "
                   "fraction of recorded iterations $\\in[0,1]$, which makes the "
                   "convergence \\emph{shape} comparable independently of length.")
        out.append(r"\begin{figure}[H]\centering")
        out.append(r"\includegraphics[width=\linewidth]{%s}" % fig_paths["normalized"].name)
        out.append(r"\caption{Cost $h(U)$ versus the normalized iteration fraction, "
                   r"per cluster size $n$.}")
        out.append(r"\label{fig:normalized}")
        out.append(r"\end{figure}")

    out.append(r"\section{Summary across cluster size}\label{sec:summary}")
    out.append("Final cost (the estimated $C_{\\mathrm{GME}}$), recorded iterations, "
               "and wall-clock time per job are summarised below and plotted against $n$.")
    out.append(r"\begin{table}[H]\centering")
    out.append(r"\caption{Per-method convergence summary. ``Gap to best'' is the "
               r"difference between a method's final cost and the lowest final cost at "
               r"that $n$.}")
    out.append(r"\input{tables/summary_table%s.tex}" % sfx)
    out.append(r"\end{table}")

    for key, cap in [
        ("final_vs_n", "Estimated $C_{\\mathrm{GME}}$ (final cost) versus cluster size $n$."),
        ("iters_vs_n", "Recorded iteration count versus cluster size $n$."),
        ("runtime_vs_n", "Wall-clock time per job versus cluster size $n$."),
    ]:
        p = fig_paths.get(key)
        if p is None:
            continue
        out.append(r"\begin{figure}[H]\centering")
        out.append(r"\includegraphics[width=0.78\linewidth]{%s}" % p.name)
        out.append(r"\caption{%s}" % cap)
        out.append(r"\end{figure}")

    if fig_paths.get("cost_vs_time"):
        out.append(r"\subsection{Cross-method efficiency: cost vs wall-clock time}")
        out.append("Since iteration counts are not a common currency, the fair "
                   "efficiency comparison is the final cost reached against the "
                   "wall-clock time spent. Points toward the lower-left (low cost, "
                   "low time) are preferable; points at the same height indicate "
                   "methods that reach the same minimum at different cost in time.")
        out.append(r"\begin{figure}[H]\centering")
        out.append(r"\includegraphics[width=\linewidth]{%s}" % fig_paths["cost_vs_time"].name)
        out.append(r"\caption{Final cost $h_{\mathrm{final}}$ versus wall-clock time "
                   r"per job, faceted by cluster size $n$.}")
        out.append(r"\label{fig:cost_vs_time}")
        out.append(r"\end{figure}")

    if fig_paths.get("outer_k"):
        out.append(r"\section{Outer-loop ($k$) convergence}")
        out.append("Where a method records an outer $k$-loop, its convergence in the "
                   "decomposition cardinality $k$ is shown below.")
        for n, label, p in fig_paths["outer_k"]:
            out.append(r"\begin{figure}[H]\centering")
            out.append(r"\includegraphics[width=0.58\linewidth]{%s}" % p.name)
            out.append(r"\caption{%s outer-$k$ convergence, $n=%s$.}"
                       % (latex_escape(label), str(n)))
            out.append(r"\end{figure}")

    out.append(r"\section*{Appendix: data provenance}")
    out.append("Each result is tagged by how it entered its cached file "
               "(\\texttt{computed}, \\texttt{job\\_cache}, or \\texttt{full\\_pkl}) "
               "and by the file it came from.")
    out.append(r"\begin{table}[H]\centering")
    out.append(r"\caption{Provenance of each stored result.}")
    out.append(r"\input{tables/provenance_table%s.tex}" % sfx)
    out.append(r"\end{table}")

    out.append(r"\end{document}")
    return "\n".join(out)


# ==================================================================================
# Compilation
# ==================================================================================
def compile_latex(report_tex: Path):
    workdir = report_tex.parent
    name = report_tex.name
    if shutil.which("latexmk"):
        cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", name]
        passes = 1
    elif shutil.which("pdflatex"):
        cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name]
        passes = 2
    else:
        print("No LaTeX engine found on PATH; skipping compilation.")
        return None
    print("Compiling:", " ".join(cmd), "(cwd=%s)" % workdir)
    try:
        for _ in range(passes):
            subprocess.run(cmd, cwd=str(workdir), check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        pdf = report_tex.with_suffix(".pdf")
        if pdf.exists():
            print("PDF built ->", pdf)
            return pdf
    except subprocess.CalledProcessError as e:
        log = (e.stdout or b"").decode("utf-8", "replace")
        print("LaTeX compilation FAILED. Tail of log:\n", log[-2500:])
    return None


# ==================================================================================
# Main
# ==================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=".",
                    help="Directory to scan for conv_*.pkl / convergence_full_run_*.pkl.")
    ap.add_argument("--conv-pkl", nargs="*", default=None,
                    help="Explicit convergence PKL path(s); combined with --input.")
    ap.add_argument("--system-pkl", default=None, help="Optional system_*.pkl (metadata).")
    ap.add_argument("--rdm-pkl", nargs="*", default=None, help="Optional rdm_*.pkl file(s).")
    ap.add_argument("--outdir", "--out", default="report_out", help="Output directory.")
    ap.add_argument("--src", default="../src",
                    help="Dir containing the hubbard_gme package (added to sys.path).")
    ap.add_argument("--compile", action="store_true", help="Compile report.tex with pdflatex.")
    args = ap.parse_args(argv)

    src_dirs = [args.src, "src", "../src", "./"]

    conv_files = discover_conv_files(args.input, args.conv_pkl)
    conv_files = [p for p in conv_files if p.exists()]
    if not conv_files:
        ap.error("No convergence PKL files found (looked in --input and --conv-pkl). "
                 "Pass --input <dir> or --conv-pkl <file> ...")
    print("Discovered convergence files:")
    for p in conv_files:
        print("   -", p)

    ds = build_dataset(conv_files, src_dirs)
    if not ds["data_by_n"]:
        print("\nERROR: no usable method results in the discovered files.")
        for w in ds["warnings"]:
            print("   !", w)
        return 2

    # optional standalone system / rdm enrichment
    if args.system_pkl and not ds["system_meta"]:
        try:
            sd = load_pickle(Path(args.system_pkl), src_dirs)
            if isinstance(sd, dict):
                ds["system_meta"] = sd.get("system_state_ref", sd)
        except Exception as e:
            print("(warning) could not read system pkl:", repr(e))
    if args.rdm_pkl:
        for rp in args.rdm_pkl:
            try:
                rd = load_pickle(Path(rp), src_dirs)
            except Exception as e:
                print("(warning) could not read rdm pkl", rp, ":", repr(e)); continue
            meta = rd.get("meta", {}) if isinstance(rd, dict) else {}
            metrics = rd.get("metrics", {}) if isinstance(rd, dict) else {}
            sites = meta.get("sites") if isinstance(meta, dict) else None
            if sites is not None:
                n = len(sites)
                ds["rdm_meta"].setdefault(n, {"sites": tuple(sites), "metrics": metrics or {}})

    ds = sanity_report(ds)

    cfg = ds["config"]
    suffix = f"  (L={cfg.get('L')}, U={cfg.get('U')})"

    # Timestamp so every rerun produces fresh, non-overwriting outputs.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    outdir = Path(args.outdir)
    fig_dir = outdir / "figs"
    tab_dir = outdir / "tables"
    data_dir = outdir / "data"
    for d in (outdir, fig_dir, tab_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures ->", fig_dir, " (stamp:", stamp + ")")
    fig_paths = {"per_n": {}}
    for n in ds["n_list"]:
        if not ds["data_by_n"][n]:
            continue
        p = fig_dir / f"convergence_n{n}_{stamp}.pdf"
        fig_single_n(ds["data_by_n"][n], n, p, suffix=suffix)
        fig_paths["per_n"][n] = p
    fig_paths["combined"] = fig_combined(
        ds, fig_dir / f"convergence_combined_{stamp}.pdf", suffix=suffix)
    fig_paths["final_vs_n"] = fig_final_vs_n(ds, fig_dir / f"final_cost_vs_n_{stamp}.pdf")
    fig_paths["iters_vs_n"] = _grouped_bar(
        ds, "iters", "Recorded iterations", "Iterations vs cluster size",
        fig_dir / f"iterations_vs_n_{stamp}.pdf")
    fig_paths["runtime_vs_n"] = _grouped_bar(
        ds, "time", "Wall-clock time per job [s]", "Runtime vs cluster size",
        fig_dir / f"runtime_vs_n_{stamp}.pdf")
    fig_paths["cost_vs_time"] = fig_cost_vs_time(
        ds, fig_dir / f"final_cost_vs_time_{stamp}.pdf")
    fig_paths["normalized"] = fig_normalized_combined(
        ds, fig_dir / f"convergence_normalized_{stamp}.pdf", suffix=suffix)
    fig_paths["outer_k"] = fig_outer_k(ds, fig_dir, stamp)

    print("\nWriting CSV ->", data_dir)
    write_csv(ds, data_dir / f"final_costs_and_timings_{stamp}.csv")

    print("\nWriting LaTeX tables ->", tab_dir)
    table_files = {}
    (tab_dir / f"system_table_{stamp}.tex").write_text(build_system_table(ds), encoding="utf-8")
    print("    wrote tables/system_table_%s.tex" % stamp)
    rdm_tab = build_rdm_table(ds)
    if rdm_tab:
        (tab_dir / f"rdm_table_{stamp}.tex").write_text(rdm_tab, encoding="utf-8")
        table_files["rdm"] = True
        print("    wrote tables/rdm_table_%s.tex" % stamp)
    (tab_dir / f"summary_table_{stamp}.tex").write_text(build_summary_table(ds), encoding="utf-8")
    print("    wrote tables/summary_table_%s.tex" % stamp)
    (tab_dir / f"provenance_table_{stamp}.tex").write_text(
        build_provenance_table(ds), encoding="utf-8")
    print("    wrote tables/provenance_table_%s.tex" % stamp)

    report_tex = outdir / f"report_{stamp}.tex"
    report_tex.write_text(assemble_report(ds, fig_paths, table_files, stamp), encoding="utf-8")
    print("\nWrote LaTeX report ->", report_tex)

    if args.compile:
        compile_latex(report_tex)

    print("\nAll done. Outputs in:", outdir.resolve())
    print("  Figures :", fig_dir)
    print("  Tables  :", tab_dir)
    print("  CSV     :", data_dir / f"final_costs_and_timings_{stamp}.csv")
    print("  Report  :", report_tex, "  (compile with: latexmk -pdf %s)" % report_tex.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())