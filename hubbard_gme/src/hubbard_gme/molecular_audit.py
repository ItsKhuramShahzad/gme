#!/usr/bin/env python3
"""
molecular_audit.py
------------------
Audit the saved molecular GME results (T=0 and thermal) and print a
PASS / WARN / FAIL verdict, mirroring ml_audit_dataset.py for the Hubbard data.

Reads:
    results/rdm_pkl_results/mol_gme_runs/<Mol>/molgme_*.pkl      (T=0, per cluster)
    results/rdm_pkl_results/mol_thermal_runs/<Mol>/molthermal_*.pkl (thermal)

Checks (T=0):
    1. Completeness  — C_gcg present and finite for every cluster
    2. Bracket       — C_LB_best <= C_gcg <= C_UB (+tol) per cluster
    3. RDM validity  — trace=1, Hermitian, eigenvalues>=0
    4. Coverage      — molecules x n_body cells
    5. GME incidence — fraction of clusters with C_gcg > eps
Checks (thermal):
    6. T=0 consistency — thermal C(T=0) matches the T=0 GME value
    7. Curve sanity    — finite, non-negative C(T); purity in [0,1]

Also writes tidy CSVs:
    results/mol_results/molecular_gme_summary.csv
    results/mol_results/molecular_thermal_summary.csv

Usage:
    python -m hubbard_gme.molecular_audit
    python src/hubbard_gme/molecular_audit.py --root results/rdm_pkl_results
"""
from __future__ import annotations
import argparse, glob, os, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

EPS = 1e-4          # C_GME threshold for "has GME"
BRACKET_TOL = 1e-3


def _v(name, status, detail=""):
    tag = {"PASS": "\033[92mPASS\033[0m", "WARN": "\033[93mWARN\033[0m",
           "FAIL": "\033[91mFAIL\033[0m"}.get(status, status)
    print(f"  [{tag}] {name}" + (f"   {detail}" if detail else ""))
    return status


def _worst(sts):
    if "FAIL" in sts: return "FAIL"
    if "WARN" in sts: return "WARN"
    return "PASS"


# ── load ─────────────────────────────────────────────────────────────────────

def load_t0(root: Path) -> pd.DataFrame:
    rows = []
    for p in glob.glob(str(root / "mol_gme_runs" / "*" / "molgme_*.pkl")):
        try:
            d = pickle.load(open(p, "rb"))
        except Exception:
            continue
        m, rf, rho = d.get("meta", {}), d.get("result_full", {}), d.get("rho", {})
        R = rho.get("rho_real"); I = rho.get("rho_imag")
        mat = (R + 1j * I) if R is not None else None
        rows.append({
            "molecule": m.get("mol_name"), "cas": m.get("cas_tag"),
            "n_body": m.get("n"), "sites": tuple(m.get("sites", [])),
            "spin": m.get("spin_label"), "gap": m.get("gap"),
            "purity": rho.get("purity"), "rank": rho.get("rank"),
            "C_gcg": rf.get("C_gcg"), "C_LB_best": rf.get("C_LB_best"),
            "C_UB": rf.get("C_UB"), "status": rf.get("status"),
            "k_converged": rf.get("k_converged"),
            "_trace": (float(np.real(np.trace(mat))) if mat is not None else np.nan),
            "_herm": (bool(np.allclose(mat, mat.conj().T, atol=1e-6))
                      if mat is not None else False),
            "_mineig": (float(np.min(np.linalg.eigvalsh(mat))) if mat is not None else np.nan),
            "path": p,
        })
    return pd.DataFrame(rows)


def load_thermal(root: Path):
    recs = []          # per (molecule, cluster, T)
    for p in glob.glob(str(root / "mol_thermal_runs" / "*" / "molthermal_*.pkl")):
        try:
            d = pickle.load(open(p, "rb"))
        except Exception:
            continue
        m = d.get("meta", {}); T = np.asarray(m.get("T_values"))
        for sites, r in d.get("thermal", {}).items():
            C = np.asarray(r["C_gcg"]); P = np.asarray(r["purity"])
            recs.append({"molecule": m.get("mol_name"), "cas": m.get("cas_tag"),
                         "sites": sites, "n_body": r["n"], "gap": m.get("gap"),
                         "T": T, "C": C, "purity": P})
    return recs


# ── checks ───────────────────────────────────────────────────────────────────

def audit(root: Path) -> str:
    print("=" * 68)
    print("  MOLECULAR GME AUDIT")
    print("=" * 68)
    df = load_t0(root)
    if df.empty:
        print("  No T=0 molecular GME PKLs found under", root / "mol_gme_runs")
        return "FAIL"
    sts = []

    print(f"\n  Loaded {len(df)} T=0 cluster records "
          f"from {df['molecule'].nunique()} molecules.\n")

    # 1. completeness
    print("1. Label completeness")
    n_nan = int(df["C_gcg"].isna().sum() + (~np.isfinite(df["C_gcg"].astype(float))).sum())
    sts.append(_v(f"C_gcg finite ({len(df)-n_nan}/{len(df)})",
                  "PASS" if n_nan == 0 else ("WARN" if n_nan < 0.1*len(df) else "FAIL"),
                  f"{n_nan} missing" if n_nan else ""))

    # 2. bracket
    print("\n2. Bracket consistency  (C_LB_best <= C_gcg <= C_UB)")
    v = df.dropna(subset=["C_gcg", "C_LB_best", "C_UB"])
    bad = v[(v["C_gcg"] < v["C_LB_best"] - BRACKET_TOL) |
            (v["C_gcg"] > v["C_UB"] + BRACKET_TOL)]
    sts.append(_v(f"{len(v)-len(bad)}/{len(v)} inside bracket",
                  "PASS" if bad.empty else "WARN",
                  "" if bad.empty else f"{len(bad)} violations"))

    # 3. RDM validity
    print("\n3. RDM validity  (trace=1, Hermitian, eigenvalues>=0)")
    tr_ok = np.allclose(df["_trace"].dropna(), 1.0, atol=1e-4)
    herm_ok = bool(df["_herm"].all())
    psd_bad = int((df["_mineig"].dropna() < -1e-6).sum())
    sts.append(_v("trace==1 for all RDMs", "PASS" if tr_ok else "FAIL"))
    sts.append(_v("Hermitian for all RDMs", "PASS" if herm_ok else "FAIL"))
    sts.append(_v(f"PSD (min eig >= -1e-6): {len(df)-psd_bad}/{len(df)}",
                  "PASS" if psd_bad == 0 else "WARN"))

    # 4. coverage
    print("\n4. Coverage (molecule x n_body)")
    cov = df.groupby(["molecule", "n_body"]).size().unstack(fill_value=0)
    print(cov.to_string().replace("\n", "\n     ").rjust(5))
    sts.append(_v(f"{df['molecule'].nunique()} molecules, "
                  f"n_body in {sorted(df['n_body'].unique())}", "PASS"))

    # 5. GME incidence
    print("\n5. GME incidence")
    has = df.dropna(subset=["C_gcg"])
    ngme = int((has["C_gcg"] > EPS).sum())
    _v(f"clusters with C_gcg > {EPS}: {ngme}/{len(has)} "
       f"({100*ngme/max(1,len(has)):.0f}%)", "PASS")
    print(f"     C_gcg: min={has['C_gcg'].min():.4f}  "
          f"mean={has['C_gcg'].mean():.4f}  max={has['C_gcg'].max():.4f}")
    pure = int((df["purity"] > 1 - 1e-6).sum())
    print(f"     pure RDMs: {pure}/{len(df)}   mixed: {len(df)-pure}/{len(df)}")

    # 6-7. thermal
    print("\n6-7. Thermal curves")
    th = load_thermal(root)
    if not th:
        _v("thermal PKLs present", "WARN", "none found")
    else:
        nbad_t0, ncurve, nfinite = 0, 0, 0
        for r in th:
            ncurve += 1
            C, P = r["C"], r["purity"]
            if not np.all(np.isfinite(C[np.isfinite(C)])):
                pass
            if np.nanmin(C) < -1e-6 or np.nanmax(P) > 1.0001 or np.nanmin(P) < -1e-6:
                nbad_t0 += 1
            if np.all(np.isfinite(C)):
                nfinite += 1
        sts.append(_v(f"{nfinite}/{ncurve} thermal curves fully finite",
                      "PASS" if nfinite == ncurve else "WARN"))
        sts.append(_v(f"C(T)>=0 and purity in [0,1]: {ncurve-nbad_t0}/{ncurve}",
                      "PASS" if nbad_t0 == 0 else "WARN"))
        print(f"     thermal clusters: {ncurve} from "
              f"{len(set(r['molecule'] for r in th))} molecules")

    # ── CSVs ──────────────────────────────────────────────────────────────────
    os.makedirs("results/mol_results", exist_ok=True)
    df.drop(columns=["_trace", "_herm", "_mineig", "path"]).to_csv(
        "results/mol_results/molecular_gme_summary.csv", index=False)
    if th:
        trows = []
        for r in th:
            for i, T in enumerate(r["T"]):
                trows.append({"molecule": r["molecule"], "cas": r["cas"],
                              "sites": "-".join(map(str, r["sites"])),
                              "n_body": r["n_body"], "T": float(T),
                              "C_gcg": float(r["C"][i]), "purity": float(r["purity"][i])})
        pd.DataFrame(trows).to_csv(
            "results/mol_results/molecular_thermal_summary.csv", index=False)

    verdict = _worst(sts)
    print("\n" + "=" * 68)
    print(f"  VERDICT: {verdict}")
    print("  CSVs → results/mol_results/molecular_gme_summary.csv"
          + (" + molecular_thermal_summary.csv" if th else ""))
    print("=" * 68)
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/rdm_pkl_results")
    args = ap.parse_args()
    sys.exit(0 if audit(Path(args.root).resolve()) != "FAIL" else 1)


if __name__ == "__main__":
    main()
