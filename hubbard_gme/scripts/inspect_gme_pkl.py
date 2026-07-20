#!/usr/bin/env python3
"""
scripts/inspect_gme_pkl.py

Print a compact summary of one GME job-cache PKL.
Usage:
  python scripts/inspect_gme_pkl.py results/rdm_pkl_results/gme_runs/job_cache/<file>.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def fmt(x):
    if isinstance(x, float):
        return f"{x:.10g}"
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pkl", help="Path to a GME job-cache PKL")
    args = ap.parse_args()

    path = Path(args.pkl)
    with path.open("rb") as f:
        data = pickle.load(f)

    meta = data.get("meta", {})
    result = data.get("result_full", {})
    integrity = data.get("integrity", {})
    print("=" * 80)
    print(f"GME PKL: {path}")
    print("=" * 80)
    print("Meta")
    for key in ["L", "Ne", "n_so", "U", "t", "mode", "state_label", "sites", "n",
                "method", "optimizer", "n_restarts", "k_max_extra", "tol", "tol_k",
                "cache_version", "saved_at"]:
        if key in meta:
            print(f"  {key:22s}: {meta[key]}")

    print("\nRDM integrity")
    for key in ["rho_hash_sha256", "trace", "purity", "rank", "lambda_min", "lambda_max",
                "hermitian_max_error", "trace_error"]:
        if key in integrity:
            print(f"  {key:22s}: {fmt(integrity[key])}")

    print("\nScalar C values")
    for key in ["C_LB_best", "C_UB", "C_stiefel", "C_gcg", "C_stiefel_gcg"]:
        if key in result:
            print(f"  {key:22s}: {fmt(result[key])}")

    print("\nSolver runtime")
    for k, v in (result.get("solver_runtime") or {}).items():
        print(f"  {k:22s}: {fmt(v)} sec")

    print("\nSolver success")
    for k, v in (result.get("solver_success") or {}).items():
        print(f"  {k:22s}: {v}")

    print("\nSolver cache status")
    for k, v in (result.get("solver_cache_status") or {}).items():
        print(f"  {k:22s}: {v}")

    print("\nSolver k information")
    for solver in ["stiefel", "gcg", "stiefel_gcg"]:
        sr = result.get(solver) or {}
        if not sr:
            continue
        print(f"  {solver}:")
        print(f"    rank          : {sr.get('rank')}")
        print(f"    k_converged   : {sr.get('k_converged')}")
        print(f"    k_values_tried: {sr.get('k_values_tried')}")
        print(f"    n_records     : {len(sr.get('k_records') or [])}")
        print(f"    tol           : {sr.get('tol')}")
        print(f"    tol_k         : {sr.get('tol_k')}")
        print(f"    max_iter_eff  : {sr.get('max_iter_effective', sr.get('max_iter'))}")

    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
