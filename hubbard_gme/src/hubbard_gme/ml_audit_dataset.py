#!/usr/bin/env python3
"""
audit_dataset.py
----------------
Run a battery of sanity checks on the dataset produced by
generate_ml_dataset.py.  Outputs a verdict: PASS, WARN, or FAIL.

Usage:
    python audit_dataset.py                     # uses ./data_for_ML
    python audit_dataset.py --outdir my_data    # custom path
    python audit_dataset.py --plots             # also save diagnostic plots

Reports:
    1. Label completeness (NaN scan)
    2. Bracket consistency (LB ≤ C_target ≤ UB)
    3. C_target distribution (histogram, quantiles)
    4. Coverage by (L, n_body, U/t bin)
    5. Stiefel label-noise (when n_outer_restarts > 1)
    6. Pure vs mixed split

Designed to be reread before EVERY training run as a quick health check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Tolerance for the bracket check (matches generate_ml_dataset.py)
BRACKET_TOL = 1e-3

# Severity icons
ICON = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def _verdict(name: str, status: str, details: str = "") -> None:
    """Print one check result. status ∈ {PASS, WARN, FAIL}."""
    icon = ICON.get(status, "?")
    print(f"  [{icon}] {status:<5} {name}")
    if details:
        for line in details.splitlines():
            print(f"           {line}")


def _q(arr, qs=(0.0, 0.25, 0.5, 0.75, 1.0)) -> str:
    """Format quantile summary."""
    if len(arr) == 0:
        return "  (empty)"
    vals = np.quantile(arr, qs)
    parts = [f"q{int(q*100)}={v:.4f}" for q, v in zip(qs, vals)]
    return "  " + "  ".join(parts)


# ────────────────────────────────────────────────────────────────────────────
# The six checks
# ────────────────────────────────────────────────────────────────────────────

def check_completeness(df: pd.DataFrame) -> str:
    """Return PASS / WARN / FAIL based on NaN labels."""
    _print_section("1. Label completeness (NaN scan)")
    n = len(df)

    cols_to_check = ["C_target", "C_pure", "lower_bound_super", "C_UB",
                     "C_stiefel_best", "purity"]

    worst_status = "PASS"
    for col in cols_to_check:
        if col not in df.columns:
            _verdict(f"column '{col}' present", "WARN", "  column missing entirely")
            worst_status = "WARN" if worst_status == "PASS" else worst_status
            continue
        n_nan = df[col].isna().sum()
        pct = 100 * n_nan / n if n > 0 else 0
        if n_nan == 0:
            _verdict(f"{col}: 0 NaNs out of {n}", "PASS")
        elif pct < 1.0:
            _verdict(f"{col}: {n_nan} NaNs out of {n}", "WARN",
                     f"  {pct:.2f}% — small, but check which configs failed")
            worst_status = "WARN" if worst_status == "PASS" else worst_status
        else:
            _verdict(f"{col}: {n_nan} NaNs out of {n}", "FAIL",
                     f"  {pct:.1f}% — significant labeling failure")
            worst_status = "FAIL"

    # Where do C_target NaNs cluster?
    if "C_target" in df.columns and df["C_target"].isna().any():
        bad = df[df["C_target"].isna()]
        if len(bad) > 0 and "n_body" in bad.columns:
            print()
            print("    NaN C_target distribution:")
            for col in ["L", "n_body", "mode"]:
                if col in bad.columns:
                    vc = bad[col].value_counts().to_dict()
                    print(f"      by {col}: {vc}")

    return worst_status


def check_bracket(df: pd.DataFrame) -> str:
    """Verify lower_bound_super ≤ C_target ≤ C_UB."""
    _print_section("2. Bracket consistency  (LB ≤ C_target ≤ UB)")

    needed = {"C_target", "lower_bound_super", "C_UB"}
    if not needed.issubset(df.columns):
        _verdict("required columns present", "FAIL",
                 f"  missing: {needed - set(df.columns)}")
        return "FAIL"

    valid = df.dropna(subset=list(needed))
    if "target_in_bracket" in df.columns:
        # Use the precomputed flag — fast path
        n_violations = (~df["target_in_bracket"].fillna(False)).sum()
    else:
        below = valid["C_target"] < (valid["lower_bound_super"] - BRACKET_TOL)
        above = valid["C_target"] > (valid["C_UB"] + BRACKET_TOL)
        n_violations = int((below | above).sum())

    n = len(df)
    pct = 100 * n_violations / n if n > 0 else 0

    if n_violations == 0:
        _verdict(f"all {n} records inside bracket", "PASS")
        return "PASS"
    elif pct < 1.0:
        _verdict(f"{n_violations} bracket violations", "WARN",
                 f"  {pct:.2f}% — likely numerical noise. Check magnitudes:")
        # Show the size of the worst violations
        if n_violations > 0:
            dev_below = (valid["lower_bound_super"] - valid["C_target"]).clip(lower=0)
            dev_above = (valid["C_target"] - valid["C_UB"]).clip(lower=0)
            dev_max = max(dev_below.max(), dev_above.max())
            print(f"           max deviation outside bracket: {dev_max:.2e}")
        return "WARN"
    else:
        _verdict(f"{n_violations} bracket violations", "FAIL",
                 f"  {pct:.1f}% — one of LB/UB/Stiefel has a bug")
        return "FAIL"


def check_target_distribution(df: pd.DataFrame) -> str:
    """Inspect C_target distribution: regression-friendly or pathological?"""
    _print_section("3. C_target distribution")

    if "C_target" not in df.columns:
        _verdict("C_target present", "FAIL")
        return "FAIL"

    vals = df["C_target"].dropna().values
    if len(vals) == 0:
        _verdict("non-empty C_target", "FAIL")
        return "FAIL"

    print(f"  N = {len(vals)} records with valid labels")
    print(f"  mean = {vals.mean():.4f}   std = {vals.std():.4f}")
    print(f"  min  = {vals.min():.4f}   max = {vals.max():.4f}")
    print("  quantiles:" + _q(vals))

    # Pathology checks
    near_zero = (vals < 1e-4).sum()
    near_one  = (vals > 0.9999).sum()
    pct_zero  = 100 * near_zero / len(vals)
    pct_one   = 100 * near_one  / len(vals)
    print(f"  fraction near 0 (<1e-4): {pct_zero:.1f}%")
    print(f"  fraction near 1 (>0.9999): {pct_one:.1f}%")

    if pct_zero > 80:
        _verdict("distribution shape", "WARN",
                 "  >80% of labels ~ 0. Most clusters are unentangled.\n"
                 "  This will train a 'predict zero' model unless you\n"
                 "  rebalance or filter to entangled regions.")
        return "WARN"
    if pct_one > 80:
        _verdict("distribution shape", "WARN",
                 "  >80% of labels ~ 1. Mostly maximally-entangled — unusual.")
        return "WARN"

    spread = vals.std()
    if spread < 0.01:
        _verdict("distribution spread", "FAIL",
                 f"  std={spread:.4f} — labels are nearly constant.\n"
                 f"  Nothing for the model to learn.")
        return "FAIL"

    _verdict("distribution looks regression-friendly", "PASS")
    return "PASS"


def check_coverage(df: pd.DataFrame) -> str:
    """Count records per (L, n_body) and per U/t bin to spot gaps."""
    _print_section("4. Coverage by physics regime")

    if {"L", "n_body"}.issubset(df.columns):
        ct = df.groupby(["L", "n_body"]).size().unstack(fill_value=0)
        print("  Records per (L, n_body):")
        for line in ct.to_string().splitlines():
            print(f"    {line}")
    else:
        _verdict("L and n_body columns", "FAIL")
        return "FAIL"

    # U/t coverage
    if "U_over_t" in df.columns:
        bins = [0, 0.5, 1, 2, 4, 8, 16, np.inf]
        labels = ["0-0.5", "0.5-1", "1-2", "2-4", "4-8", "8-16", "16+"]
        df = df.copy()
        df["ut_bin"] = pd.cut(df["U_over_t"], bins=bins, labels=labels,
                              include_lowest=True)
        print("\n  Records per U/t bin:")
        ut_ct = df["ut_bin"].value_counts().sort_index()
        for label, count in ut_ct.items():
            print(f"    U/t {label:>8s}: {count}")

    # Verdict
    counts = ct.values.flatten()
    counts = counts[counts > 0]
    if len(counts) == 0:
        return "FAIL"
    ratio = counts.max() / max(counts.min(), 1)
    if ratio > 50:
        _verdict("balance across (L, n_body) cells", "WARN",
                 f"  imbalance ratio max/min = {ratio:.1f}\n"
                 f"  rare cells will be poorly predicted")
        return "WARN"

    _verdict("coverage balanced enough", "PASS",
             f"  imbalance ratio max/min = {ratio:.1f}")
    return "PASS"


def check_label_noise(df: pd.DataFrame) -> str:
    """If C_stiefel_std is available, quantify label noise."""
    _print_section("5. Stiefel label noise  (only if --n-outer-restarts > 1)")

    if "C_stiefel_std" not in df.columns:
        _verdict("C_stiefel_std present", "WARN",
                 "  column missing — were outer restarts disabled?")
        return "WARN"

    stds = df["C_stiefel_std"].dropna().values
    if len(stds) == 0 or np.allclose(stds, 0):
        _verdict("Stiefel std", "WARN",
                 "  all zeros — likely n_outer_restarts=1.\n"
                 "  Cannot estimate label noise. Consider re-running\n"
                 "  with --n-outer-restarts 3 on a sample for diagnostics.")
        return "WARN"

    print(f"  Stiefel std distribution:")
    print("  " + _q(stds))
    print(f"  fraction with std > 0.01: {100*(stds > 0.01).mean():.1f}%")
    print(f"  fraction with std > 0.05: {100*(stds > 0.05).mean():.1f}%")

    if (stds > 0.05).mean() > 0.10:
        _verdict("label noise level", "WARN",
                 "  >10% of labels have std > 0.05 — model accuracy will\n"
                 "  be capped near this noise floor. Consider more restarts.")
        return "WARN"

    _verdict("label noise acceptable", "PASS")
    return "PASS"


def check_pure_mixed(df: pd.DataFrame) -> str:
    """Report pure vs mixed split."""
    _print_section("6. Pure vs mixed split")

    if "is_pure" not in df.columns:
        _verdict("is_pure column", "FAIL")
        return "FAIL"

    n = len(df)
    n_pure = int(df["is_pure"].sum())
    n_mixed = n - n_pure
    print(f"  pure  ρ (n=N or near-pure):  {n_pure}  ({100*n_pure/n:.1f}%)")
    print(f"  mixed ρ (n<N):              {n_mixed}  ({100*n_mixed/n:.1f}%)")

    if n_pure == 0:
        _verdict("at least some pure RDMs", "WARN",
                 "  no pure-state records — pure formula label never used.\n"
                 "  Fine if you only care about mixed RDMs.")
        return "WARN"
    if n_mixed == 0:
        _verdict("at least some mixed RDMs", "FAIL",
                 "  no mixed-state records — Stiefel labels never used.\n"
                 "  This is the regime where ML provides actual value.")
        return "FAIL"

    if n_pure / n < 0.05 or n_mixed / n < 0.05:
        _verdict("balance", "WARN",
                 f"  one class makes up <5% of the data.\n"
                 f"  Consider training separate models per class.")
        return "WARN"

    _verdict("both classes well-represented", "PASS")
    return "PASS"


# ────────────────────────────────────────────────────────────────────────────
# Optional plots (require matplotlib)
# ────────────────────────────────────────────────────────────────────────────

def make_plots(df: pd.DataFrame, outdir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [plots] matplotlib not installed — skipping")
        return

    plot_dir = outdir / "audit_plots"
    plot_dir.mkdir(exist_ok=True)

    # 1. C_target histogram
    valid = df["C_target"].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valid, bins=60, edgecolor="black", alpha=0.7)
    ax.set_xlabel("C_target")
    ax.set_ylabel("count")
    ax.set_title(f"C_target distribution  (N={len(valid)})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "01_target_histogram.png", dpi=120)
    plt.close(fig)

    # 2. C_target vs U/t (per L)
    if {"U_over_t", "L"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(9, 5))
        for L_val, sub in df.groupby("L"):
            ax.scatter(sub["U_over_t"], sub["C_target"],
                       s=8, alpha=0.4, label=f"L={L_val}")
        ax.set_xscale("log")
        ax.set_xlabel("U / t  (log scale)")
        ax.set_ylabel("C_target")
        ax.set_title("C_target vs U/t")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "02_target_vs_ut.png", dpi=120)
        plt.close(fig)

    # 3. Bracket diagnostic
    if {"lower_bound_super", "C_UB", "C_target"}.issubset(df.columns):
        sub = df[["lower_bound_super", "C_target", "C_UB"]].dropna()
        bracket_width = sub["C_UB"] - sub["lower_bound_super"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(bracket_width, bins=60, edgecolor="black", alpha=0.7)
        axes[0].set_xlabel("bracket width  (UB - LB)")
        axes[0].set_ylabel("count")
        axes[0].set_title("How tight is the [LB, UB] bracket?")
        axes[0].grid(alpha=0.3)

        # C_target position within bracket  (0 = at LB, 1 = at UB)
        pos = (sub["C_target"] - sub["lower_bound_super"]) / bracket_width.clip(1e-12)
        axes[1].hist(pos.clip(-0.1, 1.1), bins=60, edgecolor="black", alpha=0.7)
        axes[1].axvline(0, color="red",  ls="--", label="at LB")
        axes[1].axvline(1, color="blue", ls="--", label="at UB")
        axes[1].set_xlabel("position of C_target inside [LB, UB]")
        axes[1].set_ylabel("count")
        axes[1].set_title("Where does C_target sit inside the bracket?")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "03_bracket_diagnostic.png", dpi=120)
        plt.close(fig)

    print(f"\n  [plots] saved to {plot_dir}")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Audit the dataset built by generate_ml_dataset.py.",
    )
    parser.add_argument("--outdir", type=str, default="data_for_ML",
                        help="Dataset directory (default: ./data_for_ML)")
    parser.add_argument("--plots", action="store_true",
                        help="Save diagnostic PNGs to <outdir>/audit_plots/")
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    manifest = outdir / "manifest.csv"
    if not manifest.exists():
        print(f"ERROR: no manifest at {manifest}")
        sys.exit(1)

    print(f"\nAuditing dataset at: {outdir}")
    df = pd.read_csv(manifest)
    print(f"  Loaded {len(df)} records from manifest")

    # Run all checks
    statuses = []
    statuses.append(("completeness",  check_completeness(df)))
    statuses.append(("bracket",       check_bracket(df)))
    statuses.append(("distribution",  check_target_distribution(df)))
    statuses.append(("coverage",      check_coverage(df)))
    statuses.append(("label_noise",   check_label_noise(df)))
    statuses.append(("pure_mixed",    check_pure_mixed(df)))

    if args.plots:
        make_plots(df, outdir)

    # Final verdict
    print(f"  FINAL VERDICT")
    print(f"{'═' * 70}")
    for name, st in statuses:
        print(f"  [{ICON[st]}] {st:<5}  {name}")

    n_fail = sum(1 for _, s in statuses if s == "FAIL")
    n_warn = sum(1 for _, s in statuses if s == "WARN")

    print()
    if n_fail > 0:
        print(f"  ✗ FAIL — {n_fail} blocking issue(s). Fix before training.")
        sys.exit(2)
    elif n_warn > 0:
        print(f"  ! WARN — {n_warn} concern(s). Training can proceed but")
        print(f"          read the warnings above and decide whether to mitigate.")
        sys.exit(1)
    else:
        print(f"  ✓ PASS — dataset is clean. Proceed to training.")
        sys.exit(0)


if __name__ == "__main__":
    main()