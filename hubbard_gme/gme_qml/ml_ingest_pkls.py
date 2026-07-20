#!/usr/bin/env python3
"""
ml_ingest_pkls.py
-----------------
Turn already-computed GME / surface PKL caches into ML dataset records in the
SAME .npz format as ml_generate_ml_dataset.py — so the expensive results you
have already produced become training data, with no recomputation.

Two sources are linked:

GCG ONLY. The training target `C_target` is always the GCG convex-roof value.
Stiefel results are never ingested and no Stiefel-named field is written — even
though the standalone GME PKLs also contain Stiefel/stiefel_gcg columns, those
are ignored here. (The native dataset in ../records/ is Stiefel-based; keep it
separate and do not mix.)

Two sources are linked:

  1. Standalone GME  (results/rdm_pkl_results/gme_runs/job_cache/*.pkl)
       `result_full` holds C_gcg (used) plus the supervisor LB and C_UB. Links
       to its RDM in rdm_cache/ via integrity.rho_cache_path. Records without a
       GCG value are skipped.

  2. Surface (thermal) (results/rdm_pkl_results/thermal_runs/tTsurface_*.pkl)
       Whole-result surface PKLs give the GCG C_GME(t/U, T) per (t, T, n). The
       thermal RDM Γ⁽ⁿ⁾(T) is NOT stored, but is reconstructed deterministically
       from the cached RDM bank (rdm_bank/) + Boltzmann weights at T. We pair
       each reconstructed ρ with its GCG C_GME label and emit one record.

Output (kept SEPARATE from the native Stiefel dataset so it is never touched):

    data_for_ML/ingested/
        manifest.csv
        records/<name>.npz

npz keys: rho_real, rho_imag, eigvals, config_json,
C_target (= GCG), C_gcg, C_pure, lower_bound_super, C_UB, purity, is_pure,
target_in_bracket, source, solver (= "gcg"), T, thermal.

Usage (from the repo root):
    python gme_qml/ml_ingest_pkls.py --source both
    python gme_qml/ml_ingest_pkls.py --source gme
    python gme_qml/ml_ingest_pkls.py --source surface
    python gme_qml/ml_ingest_pkls.py --source both \
        --results-root results/rdm_pkl_results --outdir gme_qml/data_for_ML
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ── make hubbard_gme importable regardless of CWD ────────────────────────────
# This script lives in <repo>/gme_qml/; the package lives in <repo>/src/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # <repo>/gme_qml
_REPO_ROOT = os.path.dirname(_THIS_DIR)                     # <repo>
_SRC_DIR = os.path.join(_REPO_ROOT, "src")                  # <repo>/src
for _p in (_SRC_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hubbard_gme.metrics import compute_entropy, compute_purity
from ml_generate_ml_dataset import record_filename, config_hash
from hubbard_gme.rdm import normalize_density_matrix
from hubbard_gme.thermal_pipeline import (
    _rdm_bank_cache_path,
    _load_thermal_rdm_bank_pkl,
    _thermal_active_probabilities,
    _gme_lower_bound,
    _sector_tag,
    _cgme_analytical_n2,
    _wootters_concurrence,
)


def _n2_werner_labels(rho: np.ndarray) -> Dict[str, Any]:
    """
    n=2 label WITHOUT the convex roof: the analytic Werner-state GME
    sqrt(1 - sqrt(1 - C^2)) (C = Wootters concurrence). Both are stored;
    C_target is the Werner GME, NOT the bare concurrence.
    """
    c_wootters = float(_wootters_concurrence(rho))
    c_werner = float(_cgme_analytical_n2(rho))
    return {
        "C_target": c_werner, "C_gcg": float("nan"),
        "C_pure": float("nan"), "lower_bound_super": float("nan"),
        "C_UB": float("nan"), "C_wootters": c_wootters, "status": "",
        "source": "werner_analytic", "solver": "werner_analytic",
    }

PURE_THRESHOLD = 1.0 - 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Record construction (shared by both sources)
# ─────────────────────────────────────────────────────────────────────────────

def _derived_features(rho: np.ndarray) -> Dict[str, Any]:
    eigvals = np.linalg.eigvalsh(rho)
    purity = float(compute_purity(rho))
    return {
        "eigvals": eigvals,
        "purity": purity,
        "is_pure": bool(purity > PURE_THRESHOLD),
        "entropy": float(compute_entropy(rho)),
        "tr_rho2": float(np.real(np.trace(rho @ rho))),
        "tr_rho3": float(np.real(np.trace(rho @ rho @ rho))),
        "rank": int(np.sum(eigvals > 1e-10)),
    }


def _save_convergence(detail: Optional[dict], config_hash_str: str,
                      outdir: Path) -> bool:
    """
    Persist the FULL GCG convergence history for one record: every k, every
    restart's final value (restart_log), the per-iteration cost history of the
    best restart, the decomposition, timings and effective settings — exactly
    the `detail` dict returned by the solver. Written as a sidecar pickle keyed
    by the record's config hash, so the .npz training file stays light.
    """
    if not detail or detail.get("k_records") is None:
        return False
    conv_dir = outdir / "convergence"
    conv_dir.mkdir(parents=True, exist_ok=True)
    with open(conv_dir / f"{config_hash_str}.pkl", "wb") as fh:
        pickle.dump(detail, fh)
    return True


def _write_record(
    *,
    rho: np.ndarray,
    cfg: Dict[str, Any],
    labels: Dict[str, Any],
    records_dir: Path,
    outdir: Path,
    detail: Optional[dict] = None,
) -> Optional[Dict[str, Any]]:
    """Normalise ρ, derive features, write the .npz, return the manifest row."""
    tr = float(np.real(np.trace(rho)))
    if abs(tr) < 1e-12:
        return None
    rho = rho / tr

    feat = _derived_features(rho)
    cfg = dict(cfg)
    cfg["config_hash"] = config_hash(cfg)
    npz_path = records_dir / record_filename(cfg)
    has_conv = _save_convergence(detail, cfg["config_hash"], outdir)

    C_target = labels.get("C_target", float("nan"))
    np.savez_compressed(
        npz_path,
        rho_real=rho.real.astype(np.float32),
        rho_imag=rho.imag.astype(np.float32),
        eigvals=feat["eigvals"].astype(np.float64),
        config_json=np.array(json.dumps(
            {k: v for k, v in cfg.items() if k != "config_hash"})),
        # labels — GCG only (no Stiefel data, no Stiefel-named fields)
        C_target=np.float64(C_target),
        C_gcg=np.float64(labels.get("C_gcg", float("nan"))),
        C_pure=np.float64(labels.get("C_pure", float("nan"))),
        lower_bound_super=np.float64(labels.get("lower_bound_super", float("nan"))),
        C_UB=np.float64(labels.get("C_UB", float("nan"))),
        C_wootters=np.float64(labels.get("C_wootters", float("nan"))),
        purity=np.float64(feat["purity"]),
        is_pure=np.bool_(feat["is_pure"]),
        target_in_bracket=np.bool_(labels.get("target_in_bracket", True)),
        source=np.array(labels.get("source", "")),
        solver=np.array(labels.get("solver", "gcg")),
        T=np.float64(cfg.get("T", float("nan"))),
        thermal=np.bool_(bool(cfg.get("thermal", False))),
        has_convergence=np.bool_(has_conv),
    )

    U = float(cfg["U"]); t = float(cfg["t"])
    row = {
        "L": cfg["L"], "t": t, "U": U,
        "U_over_t": U / t if abs(t) > 1e-14 else float("inf"),
        "mode": cfg["mode"], "n_electrons": cfg["n_electrons"],
        "state_idx": cfg["state_idx"], "sites": cfg["sites"],
        "n_body": cfg["n_body"], "sweep_tag": cfg["sweep_tag"],
        "config_hash": cfg["config_hash"],
        "source": labels.get("source", ""), "solver": labels.get("solver", ""),
        "T": cfg.get("T", float("nan")), "thermal": bool(cfg.get("thermal", False)),
        "purity": feat["purity"], "is_pure": feat["is_pure"],
        "entropy": feat["entropy"],
        "C_target": C_target, "C_gcg": labels.get("C_gcg", float("nan")),
        "C_pure": labels.get("C_pure", float("nan")),
        "lower_bound_super": labels.get("lower_bound_super", float("nan")),
        "C_UB": labels.get("C_UB", float("nan")),
        "status": labels.get("status", ""),
        "tr_rho2": feat["tr_rho2"], "tr_rho3": feat["tr_rho3"],
        "rank": feat["rank"], "rdm_eigvals": feat["eigvals"].tolist(),
        "has_convergence": has_conv,
        "npz_path": str(npz_path.relative_to(outdir)),
    }
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Source 1 — standalone GME job caches
# ─────────────────────────────────────────────────────────────────────────────

def ingest_gme_runs(results_root: Path, records_dir: Path, outdir: Path,
                    done: set, verbose: bool = True) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    files = sorted(glob.glob(str(results_root / "gme_runs" / "job_cache" / "*.pkl")))
    if verbose:
        print(f"[gme] {len(files)} job caches found")
    for p in files:
        try:
            d = pickle.load(open(p, "rb"))
        except Exception as e:
            if verbose: print(f"  [skip] {os.path.basename(p)}: {e}")
            continue
        meta = d.get("meta", {})
        rf = d.get("result_full", {})
        rho_path = d.get("integrity", {}).get("rho_cache_path")
        if not rho_path or not os.path.exists(rho_path):
            if verbose: print(f"  [skip] {os.path.basename(p)}: linked ρ missing")
            continue
        try:
            rho = pickle.load(open(rho_path, "rb"))["rho"]
        except Exception as e:
            if verbose: print(f"  [skip] {os.path.basename(p)}: ρ load failed {e}")
            continue

        sites = list(meta.get("sites", range(int(meta.get("n", 0)))))
        cfg = {
            "L": int(meta["L"]), "t": float(meta["t"]), "U": float(meta["U"]),
            "mode": str(meta.get("mode", "uniform")),
            "n_electrons": int(meta.get("Ne", meta["L"])),
            "state_idx": 0, "sites": sites, "n_body": int(meta["n"]),
            "sweep_tag": "ingested_gme",
        }
        ch = config_hash({**cfg, "config_hash": ""})
        if ch in done:
            continue
        # GCG ONLY — the convex-roof value from the GCG solver. Do NOT fall back
        # to Stiefel or to C_LB_best (which is the best across solvers, i.e. may
        # be a Stiefel value). Skip records where GCG was not run.
        c_gcg = rf.get("C_gcg")
        if c_gcg is None or (isinstance(c_gcg, float) and np.isnan(c_gcg)):
            if verbose:
                print(f"  [skip] {os.path.basename(p)}: no GCG value (C_gcg missing)")
            continue
        labels = {
            "C_target": float(c_gcg),
            "C_gcg": float(c_gcg),
            "C_pure": rf.get("C_GME", float("nan")),
            "lower_bound_super": rf.get("C_LB_supervisor",
                                        rf.get("lower_bound_super", float("nan"))),
            "C_UB": rf.get("C_UB", float("nan")),
            "status": rf.get("status", ""),
            "source": "gme_runs", "solver": "gcg",
        }
        row = _write_record(rho=np.asarray(rho, dtype=complex), cfg=cfg,
                            labels=labels, records_dir=records_dir, outdir=outdir)
        if row:
            rows.append(row)
            done.add(row["config_hash"])
    if verbose:
        print(f"[gme] ingested {len(rows)} new records")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Source 2 — surface (thermal) whole-result PKLs, ρ reconstructed from the bank
# ─────────────────────────────────────────────────────────────────────────────

def _reconstruct_gamma(bank_payload: dict, n: int, T: float,
                       cutoff: Optional[float]) -> Optional[np.ndarray]:
    energies = np.asarray(bank_payload["energies"])
    bank = bank_payload["bank"]
    if n not in bank:
        return None
    E0 = float(energies[0])
    if float(T) <= 1e-14:
        active = np.where(np.abs(energies - E0) < 1e-10)[0]
        probs = np.full(len(active), 1.0 / len(active))
    else:
        active, probs = _thermal_active_probabilities(energies, float(T), E0, cutoff)
    gamma = np.einsum("m,mij->ij", probs, bank[n][active])
    return normalize_density_matrix(gamma)


def ingest_surface(results_root: Path, records_dir: Path, outdir: Path,
                   done: set, verbose: bool = True) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    files = sorted(glob.glob(str(results_root / "thermal_runs" / "tTsurface_*.pkl")))
    if verbose:
        print(f"[surface] {len(files)} whole-result PKLs found")
    for p in files:
        try:
            payload = pickle.load(open(p, "rb"))
        except Exception as e:
            if verbose: print(f"  [skip] {os.path.basename(p)}: {e}")
            continue
        meta = payload.get("meta", {})
        res = payload.get("result", {})
        L = int(meta["L"]); U = float(meta["U"]); mode = str(meta.get("mode", "uniform"))
        Ne = int(meta.get("Ne", L)); n_min = int(meta.get("n_min", 2))
        n_max = meta.get("n_max")           # per-n capped banks (L=8 array jobs)
        sp = bool(meta.get("spin_preserving", True))
        cutoff = meta.get("thermal_energy_cutoff", 10.0)
        t_values = np.asarray(res["t_values"]); T_values = np.asarray(res["T_values"])
        n_values = list(res["n_values"]); C_surface = res["C_surface"]
        purity_surface = res.get("purity_surface", {})

        bank_cache: Dict[float, Optional[dict]] = {}
        made = 0
        for ti, t in enumerate(t_values):
            t = float(t)
            if t not in bank_cache:
                bp = _rdm_bank_cache_path(
                    L=L, t=t, U=U, t_prime=None, t2=None, t_perp=None, mode=mode,
                    n_electrons=Ne, n_min=n_min,
                    n_max=(int(n_max) if n_max is not None else None),
                    average_site_clusters=True,
                    cache_version=meta.get("cache_version", "v1"), spin_preserving=sp)
                bank_cache[t] = _load_thermal_rdm_bank_pkl(bp)
            bank_payload = bank_cache[t]
            if bank_payload is None:
                if verbose:
                    print(f"  [skip] t={t:.4g}: bank not found (build it first)")
                continue
            for Ti, T in enumerate(T_values):
                T = float(T)
                for n in n_values:
                    cfg = {
                        "L": L, "t": t, "U": U, "mode": mode, "n_electrons": Ne,
                        "state_idx": f"T{T:.3f}", "sites": list(range(n)),
                        "n_body": int(n), "sweep_tag": "surface_thermal",
                        "T": T, "thermal": True, "cluster_averaged": True,
                        "sector": _sector_tag(sp),
                    }
                    ch = config_hash({**cfg, "config_hash": ""})
                    if ch in done:
                        continue
                    gamma = _reconstruct_gamma(bank_payload, int(n), T, cutoff)
                    if gamma is None:
                        continue
                    if int(n) == 2:
                        # n=2 bipartite: analytic Werner GME, not the stored
                        # convex-roof surface value.
                        labels = _n2_werner_labels(gamma)
                    else:
                        C = float(np.asarray(C_surface[n])[ti, Ti])
                        labels = {       # surface thermal path is GCG-only
                            "C_target": C, "C_gcg": C,
                            "C_pure": float("nan"),
                            "lower_bound_super": float("nan"),
                            "C_UB": float("nan"),
                            "status": "", "source": "surface", "solver": "gcg",
                        }
                    row = _write_record(rho=gamma, cfg=cfg, labels=labels,
                                        records_dir=records_dir, outdir=outdir)
                    if row:
                        rows.append(row); done.add(row["config_hash"]); made += 1
        if verbose:
            print(f"  {os.path.basename(p)}: +{made} records")
    if verbose:
        print(f"[surface] ingested {len(rows)} new records")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Manifest + CLI
# ─────────────────────────────────────────────────────────────────────────────

def _load_done(manifest: Path) -> set:
    if not manifest.exists():
        return set()
    try:
        return set(pd.read_csv(manifest, usecols=["config_hash"])["config_hash"].astype(str))
    except Exception:
        return set()


def _append_manifest(rows: List[Dict[str, Any]], manifest: Path) -> None:
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if manifest.exists():
        df_old = pd.read_csv(manifest)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["config_hash"], keep="last")
    else:
        df = df_new
    df.to_csv(manifest, index=False)


def ingest_banks(results_root: Path, records_dir: Path, outdir: Path, done: set,
                 *, T_list: List[float], cutoff: Optional[float],
                 n_restarts: int, max_iter: int, tol: float, k_max_extra: int,
                 verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Generate labeled GCG records DIRECTLY from cached RDM banks.

    A bank stores ρ (the RDMs) but NOT the label. For each bank (L, t, U,
    sector) and body order n we extract the RDM at each requested temperature
    (T=0 → ground state; T>0 → Boltzmann-mixed), run the GCG convex roof, and
    emit a record. This is how banks become ML data — the bank supplies ρ
    (cheap, no diagonalization), GCG supplies the C_GME label.
    """
    rows: List[Dict[str, Any]] = []
    files = sorted(glob.glob(str(results_root / "rdm_bank" / "*.pkl")))
    if verbose:
        print(f"[banks] {len(files)} bank files; T grid = {list(T_list)}")
    for p in files:
        payload = _load_thermal_rdm_bank_pkl(Path(p))
        if payload is None:
            if verbose: print(f"  [skip] {os.path.basename(p)}: not a valid bank")
            continue
        L = int(payload["L"]); t = float(payload["t"]); U = float(payload["U"])
        mode = str(payload.get("mode", "uniform")); Ne = int(payload.get("Ne", L))
        sp = bool(payload.get("spin_preserving", True))
        energies = np.asarray(payload["energies"]); bank = payload["bank"]
        E0 = float(energies[0])
        made = 0
        for n in payload["n_values"]:
            for T in T_list:
                T = float(T)
                if T <= 1e-14:
                    active = np.where(np.abs(energies - E0) < 1e-10)[0]
                    probs = np.full(len(active), 1.0 / len(active))
                    state_idx, thermal = 0, False
                else:
                    active, probs = _thermal_active_probabilities(energies, T, E0, cutoff)
                    state_idx, thermal = f"T{T:.3f}", True
                rho = normalize_density_matrix(
                    np.einsum("m,mij->ij", probs, bank[int(n)][active]))
                cfg = {
                    "L": L, "t": t, "U": U, "mode": mode, "n_electrons": Ne,
                    "state_idx": state_idx, "sites": list(range(int(n))),
                    "n_body": int(n), "sweep_tag": "bank", "T": T,
                    "thermal": thermal, "cluster_averaged": True,
                    "sector": _sector_tag(sp),
                }
                ch = config_hash({**cfg, "config_hash": ""})
                if ch in done:
                    continue
                detail = None
                if int(n) == 2:
                    # n=2 is bipartite: use the exact analytic Werner GME,
                    # never the convex roof.
                    labels = _n2_werner_labels(rho)
                else:
                    C, _pur, detail = _gme_lower_bound(
                        rho, int(n), solver="gcg", n_restarts=n_restarts,
                        max_iter=max_iter, tol=tol, k_max_extra=k_max_extra)
                    labels = {
                        "C_target": float(C), "C_gcg": float(C),
                        "C_pure": float("nan"), "lower_bound_super": float("nan"),
                        "C_UB": float("nan"), "status": "",
                        "source": "bank", "solver": "gcg",
                    }
                row = _write_record(rho=rho, cfg=cfg, labels=labels,
                                    records_dir=records_dir, outdir=outdir,
                                    detail=detail)
                if row:
                    rows.append(row); done.add(row["config_hash"]); made += 1
        if verbose:
            print(f"  {os.path.basename(p)}: +{made} records "
                  f"(L={L}, t={t:.4g}, {_sector_tag(sp)})")
    if verbose:
        print(f"[banks] generated {len(rows)} new records")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["gme", "surface", "banks", "both", "all"],
                    default="both",
                    help="gme/surface = convert existing labeled PKLs; "
                         "banks = generate NEW GCG labels from cached RDM banks; "
                         "both = gme+surface; all = gme+surface+banks")
    ap.add_argument("--results-root",
                    default=os.path.join(_REPO_ROOT, "results", "rdm_pkl_results"),
                    help="root of the PKL caches (default: <repo>/results/rdm_pkl_results)")
    ap.add_argument("--outdir",
                    default=os.path.join(_THIS_DIR, "data_for_ML"),
                    help="native dataset dir; ingested records go to <outdir>/ingested/ "
                         "(default: <gme_qml>/data_for_ML)")
    # options for --source banks (running GCG to label bank RDMs)
    ap.add_argument("--bank-T", type=float, nargs="+", default=[0.0],
                    help="temperatures to label from each bank (default: 0.0 = "
                         "ground state only). e.g. --bank-T 0 0.5 1 2")
    ap.add_argument("--bank-cutoff", type=float, default=10.0,
                    help="thermal energy cutoff for T>0 bank labeling (default 10)")
    ap.add_argument("--n-restarts", type=int, default=10, help="GCG restarts")
    ap.add_argument("--max-iter", type=int, default=200, help="GCG max iterations")
    ap.add_argument("--tol", type=float, default=1e-9, help="GCG tolerance")
    ap.add_argument("--k-max-extra", type=int, default=4, help="GCG extra ranks")
    args = ap.parse_args()

    results_root = Path(args.results_root).resolve()
    outdir = Path(args.outdir).resolve() / "ingested"
    records_dir = outdir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    manifest = outdir / "manifest.csv"

    done = _load_done(manifest)
    print(f"Ingesting into {outdir}  (already have {len(done)} records)")

    rows: List[Dict[str, Any]] = []
    if args.source in ("gme", "both", "all"):
        rows += ingest_gme_runs(results_root, records_dir, outdir, done)
    if args.source in ("surface", "both", "all"):
        rows += ingest_surface(results_root, records_dir, outdir, done)
    if args.source in ("banks", "all"):
        rows += ingest_banks(
            results_root, records_dir, outdir, done,
            T_list=args.bank_T, cutoff=args.bank_cutoff,
            n_restarts=args.n_restarts, max_iter=args.max_iter,
            tol=args.tol, k_max_extra=args.k_max_extra)

    _append_manifest(rows, manifest)
    total = len(_load_done(manifest))
    print(f"\nDone. +{len(rows)} new records → {total} total in {manifest}")


if __name__ == "__main__":
    main()
