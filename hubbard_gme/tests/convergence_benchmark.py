"""
convergence_benchmark.py

Benchmark GCG vs Method-II using the central RDM/state PKL cache defined in:

    src/hubbard_gme/rdm.py

Central cache structure:

    <project_root>/results/rdm_pkl_results/
    ├── state_cache/
    ├── rdm_cache/
    └── convergence_runs/
        └── job_cache/

This script:
  1. Loads or computes the system state through rdm.py helpers.
  2. Loads or computes RDMs through rdm.py helpers.
  3. Loads old optimizer job results from job_cache if available.
  4. Optionally imports matching results from old full convergence PKLs.
  5. Runs only missing requested GCG/Method-II jobs.
  6. Saves every completed optimizer job immediately to job_cache.
  7. Saves a full convergence-run PKL with informative filename.
  8. Stores explicit links to system-state PKL and RDM PKLs.
  9. Saves PDF convergence plots.

Naming note:
  "Method-II" is the optimiser referred to as Stiefel-manifold optimisation in the
  code/literature. It is shown to readers as "Method-II" (Stiefel is only the name
  of the manifold, not the method). For backwards compatibility with existing
  caches and the report reader, the internal data keys remain st_hist/st_final/
  st_raw and the config key remains "stiefel_optimizer".

Example:

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=../src \
python convergence_benchmark.py --workers 6 --n-restarts 10 --max-iter 5000

Run only Method-II and reuse cached GCG if available:

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=../src \
python convergence_benchmark.py --workers 6 --n-restarts 10 --max-iter 5000 --methods method-II
"""

from __future__ import annotations

# Important: set thread limits before NumPy/SciPy-heavy imports.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
import pickle
import argparse
import platform
import inspect
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

# Make ../src importable when running from tests/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hubbard_gme.hamiltonian import compute_ground_and_excited_states
from hubbard_gme.convex_roof import convex_roof_gcg, convex_roof_stiefel, convex_roof_stiefel_gcg
from hubbard_gme.rdm import (
    load_or_compute_system_state_pkl,
    load_or_compute_rdm_pkl,
    make_system_cache_key,
    get_project_root,
    get_rdm_pkl_results_dir,
    get_state_cache_dir,
    get_rdm_cache_dir,
)


# ─────────────────────────────────────────────────────────────────────────────
# General helpers
# ─────────────────────────────────────────────────────────────────────────────

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | os.PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_tag(x: Any) -> str:
    """
    Filename-safe string tag.

    Keeps '-' because it is useful for names like L-BFGS-B.
    Replaces decimal point with p.
    """
    s = str(x)
    s = s.replace(" ", "")
    s = s.replace("/", "-")
    s = s.replace("\\", "-")
    s = s.replace(".", "p")
    s = s.replace("+", "")
    return s


def float_tag(x: float) -> str:
    """
    Compact float tag for filenames.
    Examples:
        4.0  -> 4
        1.5  -> 1p5
        -1.5 -> m1p5
    """
    s = f"{float(x):.12g}"
    s = s.replace("-", "m")
    return safe_tag(s)


def n_list_tag(n_list: List[int]) -> str:
    return "n" + "-".join(str(int(n)) for n in n_list)


def format_iters(history) -> str:
    if history is None:
        return "NA"
    try:
        n = len(history)
    except TypeError:
        return "NA"
    if n == 0:
        return "NA"
    return str(n - 1)


def safe_history_from_record(record: Dict[str, Any]) -> List[float]:
    """
    Get optimizer history safely.

    If a method returns empty history, use a one-point history containing C_GME
    so plotting does not silently fail.
    """
    hist = record.get("history", None)
    if hist is None:
        hist = []

    hist = list(hist)

    if len(hist) == 0 and "C_GME" in record:
        hist = [float(record["C_GME"])]

    return hist


# ─────────────────────────────────────────────────────────────────────────────
# Extended all-method helpers (added without removing old plotting functions)
# ─────────────────────────────────────────────────────────────────────────────

STIEFEL_OPTIMIZERS = ["L-BFGS-B", "Powell", "SLSQP", "TNC", "trust-constr"]


def parse_stiefel_optimizers(text: Optional[str], fallback: str) -> List[str]:
    """Parse --method-II-optimizers.  Accepts comma list or 'all'."""
    if text is None or str(text).strip() == "":
        return [fallback]
    text = str(text).strip()
    if text.lower() == "all":
        return STIEFEL_OPTIMIZERS[:]
    opts = [x.strip() for x in text.split(",") if x.strip()]
    bad = [x for x in opts if x not in STIEFEL_OPTIMIZERS]
    if bad:
        raise ValueError(f"Unknown Method-II optimizer(s): {bad}. Valid: {STIEFEL_OPTIMIZERS}")
    return opts


def method_label(method: str, stiefel_optimizer: Optional[str] = None) -> str:
    if method == "gcg":
        return "GCG (Algorithm A)"
    if method == "method-II":
        return f"Method-II (Algorithm B, {stiefel_optimizer or 'L-BFGS-B'})"
    if method == "method-II-gcg":
        return "Stiefel-GCG / Method-II-GCG hybrid"
    return str(method)


def method_cache_tag(method: str, stiefel_optimizer: Optional[str] = None) -> str:
    if method == "gcg":
        return "gcg"
    if method == "method-II":
        return f"methodII_{safe_tag(stiefel_optimizer or 'L-BFGS-B')}"
    if method == "method-II-gcg":
        return "methodII_gcg"
    return safe_tag(method)


def build_method_jobs(methods: str, stiefel_optimizers: List[str]) -> List[tuple[str, Optional[str]]]:
    """Return a list of (method, optimizer) jobs to compute/load."""
    methods_norm = methods.strip().lower().replace("_", "-")
    jobs: List[tuple[str, Optional[str]]] = []

    def add(method: str, opt: Optional[str] = None):
        item = (method, opt)
        if item not in jobs:
            jobs.append(item)

    if methods_norm == "gcg":
        add("gcg", None)
    elif methods_norm in {"method-ii", "stiefel"}:
        for opt in stiefel_optimizers:
            add("method-II", opt)
    elif methods_norm in {"method-ii-gcg", "stiefel-gcg"}:
        add("method-II-gcg", None)
    elif methods_norm == "both":
        add("gcg", None)
        for opt in stiefel_optimizers:
            add("method-II", opt)
    elif methods_norm in {"all3", "three"}:
        add("gcg", None)
        add("method-II", stiefel_optimizers[0])
        add("method-II-gcg", None)
    elif methods_norm in {"stiefel-all-optimizers", "method-ii-all-optimizers"}:
        for opt in STIEFEL_OPTIMIZERS:
            add("method-II", opt)
    elif methods_norm == "all":
        add("gcg", None)
        for opt in STIEFEL_OPTIMIZERS:
            add("method-II", opt)
        add("method-II-gcg", None)
    else:
        raise ValueError(f"Unknown --methods={methods}")
    return jobs


def append_all_method_entry(results_by_n: Dict[int, Dict[str, Any]], r: Dict[str, Any]) -> None:
    """Store every method/optimizer in a generic all-method block while preserving old keys."""
    n = r["n"]
    d = results_by_n.setdefault(n, {})
    d.setdefault("all_method_results", {})
    tag = method_cache_tag(r["method"], r.get("stiefel_optimizer"))
    d["all_method_results"][tag] = {
        "method": r["method"],
        "stiefel_optimizer": r.get("stiefel_optimizer"),
        "label": method_label(r["method"], r.get("stiefel_optimizer")),
        "history": r.get("history", []),
        "final": float(r.get("final", np.nan)),
        "elapsed": float(r.get("elapsed", 0.0)),
        "raw": r.get("raw"),
        "job_cache_path": r.get("job_cache_path"),
        "loaded_from_job_cache": r.get("loaded_from_job_cache", False),
        "loaded_from_full_pkl": r.get("loaded_from_full_pkl", False),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Informative filenames
# ─────────────────────────────────────────────────────────────────────────────

def make_full_convergence_filename(
    *,
    L: int,
    Ne: int,
    n_so: int,
    U: float,
    t: float,
    mode: str,
    n_list: List[int],
    methods: str,
    stiefel_optimizer: str,
    n_restarts: int,
    max_iter: int,
    cache_version: str,
    stamp: str,
) -> str:
    return (
        f"conv_{cache_version}_"
        f"L{L}_Ne{Ne}_nso{n_so}_"
        f"U{float_tag(U)}_t{float_tag(t)}_"
        f"mode-{safe_tag(mode)}_"
        f"{n_list_tag(n_list)}_"
        f"methods-{safe_tag(methods)}_"
        f"method-II-{safe_tag(stiefel_optimizer)}_"
        f"restarts{n_restarts}_"
        f"iter{max_iter}_"
        f"{stamp}.pkl"
    )


def make_job_convergence_filename(
    *,
    L: int,
    Ne: int,
    n_so: int,
    U: float,
    t: float,
    mode: str,
    n: int,
    method: str,
    stiefel_optimizer: Optional[str],
    n_restarts: int,
    max_iter: int,
    cache_version: str,
    k_max_extra: int = 2,
) -> str:
    opt_part = ""
    if method == "method-II":
        opt = stiefel_optimizer or "L-BFGS-B"
        opt_part = f"_opt-{safe_tag(opt)}"

    return (
        f"job_{cache_version}_"
        f"L{L}_Ne{Ne}_nso{n_so}_"
        f"U{float_tag(U)}_t{float_tag(t)}_"
        f"mode-{safe_tag(mode)}_"
        f"n{n}_"
        f"method-{safe_tag(method)}{opt_part}_"
        f"restarts{n_restarts}_"
        f"kmax{k_max_extra}_"
        f"iter{max_iter}.pkl"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-job convergence cache
# ─────────────────────────────────────────────────────────────────────────────

def build_job_meta(
    *,
    L: int,
    Ne: int,
    n_so: int,
    U: float,
    t: float,
    mode: str,
    n: int,
    method: str,
    n_restarts: int,
    max_iter: int,
    cache_version: str,
    stiefel_optimizer: Optional[str],
    k_max_extra: int = 2,
) -> Dict[str, Any]:
    """
    Metadata used to uniquely identify one optimizer job.

    For GCG, stiefel_optimizer is kept in metadata but ignored in filename.
    """
    return {
        "L": L,
        "Ne": Ne,
        "n_so": n_so,
        "U": U,
        "t": t,
        "mode": mode,
        "n": n,
        "method": method,
        "n_restarts": n_restarts,
        "max_iter": max_iter,
        "k_max_extra": k_max_extra,
        "cache_version": cache_version,
        "stiefel_optimizer": stiefel_optimizer,
    }


def save_job_result(job_cache_dir: Path, result: Dict[str, Any], job_meta: Dict[str, Any]) -> Path:
    """
    Save one finished GCG/Method-II job immediately.

    This makes long runs resumable.
    """
    job_cache_dir = ensure_dir(job_cache_dir)

    filename = make_job_convergence_filename(**job_meta)
    path = job_cache_dir / filename

    payload = {
        "meta": {
            **job_meta,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        "result": result,
    }

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)

    return path


def load_job_result(job_cache_dir: Path, job_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Load an exact cached job result if present.

    Returns None if no matching job cache exists.
    """
    filename = make_job_convergence_filename(**job_meta)
    path = Path(job_cache_dir) / filename

    if not path.exists():
        return None

    with open(path, "rb") as f:
        payload = pickle.load(f)

    result = payload["result"]
    result.setdefault("method", job_meta.get("method"))
    result.setdefault("stiefel_optimizer", job_meta.get("stiefel_optimizer"))
    result.setdefault("k_max_extra", job_meta.get("k_max_extra"))
    result["loaded_from_job_cache"] = True
    result["loaded_from_full_pkl"] = False
    result["job_cache_path"] = str(path)
    return result


def _config_matches_full_pkl(config: Dict[str, Any], job_meta: Dict[str, Any]) -> bool:
    """
    Check whether an old full convergence PKL has compatible settings.
    """
    keys = ["L", "Ne", "U", "t", "mode", "n_so", "n_restarts", "max_iter", "cache_version"]
    for key in keys:
        if key in config and config.get(key) != job_meta.get(key):
            return False

    # If config lacks n_so or cache_version because it came from older code,
    # do not reject only for missing keys.
    if "n_list" in config and job_meta["n"] not in config["n_list"]:
        return False

    method = job_meta["method"]
    if method == "method-II":
        requested_opt = job_meta.get("stiefel_optimizer") or "L-BFGS-B"
        old_opt = config.get("stiefel_optimizer", "L-BFGS-B")
        if old_opt != requested_opt:
            return False

    return True


def load_result_from_old_full_pkls(
    *,
    convergence_pkl_dir: Path,
    job_meta: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Try to recover a job result from older full convergence_run PKLs.

    This is useful if you previously saved a full run before adding job_cache.
    It scans newest full PKLs first and extracts GCG/Method-II for the requested n.

    Returns None if no compatible full PKL contains the requested result.
    """
    convergence_pkl_dir = Path(convergence_pkl_dir)
    candidates = sorted(
        [p for p in convergence_pkl_dir.glob("*.pkl") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    method = job_meta["method"]
    n = job_meta["n"]

    for path in candidates:
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            config = data.get("config", {})
            if not _config_matches_full_pkl(config, job_meta):
                continue

            results_by_n = data.get("results_by_n", {})
            if n not in results_by_n:
                # Some pickle files may store n as string keys.
                if str(n) in results_by_n:
                    d = results_by_n[str(n)]
                else:
                    continue
            else:
                d = results_by_n[n]

            if method == "gcg":
                required = ["gcg_final", "gcg_hist", "gcg_raw"]
                if not all(k in d for k in required):
                    continue
                result = {
                    "job_id": -1,
                    "n": n,
                    "method": "gcg",
                    "stiefel_optimizer": None,
                    "k_max_extra": job_meta.get("k_max_extra"),
                    "history": d["gcg_hist"],
                    "final": float(d["gcg_final"]),
                    "elapsed": float(d.get("gcg_time", 0.0)),
                    "raw": d["gcg_raw"],
                    "loaded_from_job_cache": False,
                    "loaded_from_full_pkl": True,
                    "source_full_pkl": str(path),
                    "job_cache_path": None,
                }
                return result

            if method == "method-II":
                required = ["st_final", "st_hist", "st_raw"]
                if not all(k in d for k in required):
                    continue
                result = {
                    "job_id": -1,
                    "n": n,
                    "method": "method-II",
                    "stiefel_optimizer": job_meta.get("stiefel_optimizer"),
                    "k_max_extra": job_meta.get("k_max_extra"),
                    "history": d["st_hist"],
                    "final": float(d["st_final"]),
                    "elapsed": float(d.get("st_time", 0.0)),
                    "raw": d["st_raw"],
                    "loaded_from_job_cache": False,
                    "loaded_from_full_pkl": True,
                    "source_full_pkl": str(path),
                    "job_cache_path": None,
                }
                return result

        except Exception:
            # Bad/partial/old file: ignore and continue.
            continue

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Worker: one optimization job
# ─────────────────────────────────────────────────────────────────────────────

def run_one_job(job):
    """
    Run one optimization job in a subprocess.

    job = (job_id, n, method, rho, n_restarts, k_max_extra, max_iter, stiefel_optimizer)
    """
    job_id, n, method, rho, n_restarts, k_max_extra, max_iter, stiefel_optimizer = job

    t0 = time.time()

    if method == "gcg":
        res = convex_roof_gcg(
            rho,
            n,
            n_restarts=n_restarts,
            k_max_extra=k_max_extra,
            max_iter=max_iter,
            verbose=False,
        )

        best_record = res["best_record"]
        history = safe_history_from_record(best_record)
        final = float(res["C_gcg"])

    elif method == "method-II":
        # Compatible with older convex_roof_stiefel versions.
        stiefel_kwargs = {
            "n_restarts": n_restarts,
            "verbose": False,
        }

        sig = inspect.signature(convex_roof_stiefel).parameters

        if "k_max_extra" in sig:
            stiefel_kwargs["k_max_extra"] = k_max_extra

        if "max_iter" in sig:
            stiefel_kwargs["max_iter"] = max_iter

        if "optimizer" in sig and stiefel_optimizer is not None:
            stiefel_kwargs["optimizer"] = stiefel_optimizer

        res = convex_roof_stiefel(
            rho,
            n,
            **stiefel_kwargs,
        )

        best_record = res["best_record"]
        history = safe_history_from_record(best_record)
        final = float(res["C_stiefel"])

    elif method == "method-II-gcg":
        # Hybrid Stiefel + Riemannian conjugate-gradient solver.
        sg_kwargs = {
            "n_restarts": n_restarts,
            "verbose": False,
        }

        sig = inspect.signature(convex_roof_stiefel_gcg).parameters

        if "k_max_extra" in sig:
            sg_kwargs["k_max_extra"] = k_max_extra

        if "max_iter" in sig:
            sg_kwargs["max_iter"] = max_iter

        res = convex_roof_stiefel_gcg(
            rho,
            n,
            **sg_kwargs,
        )

        best_record = res["best_record"]
        history = safe_history_from_record(best_record)
        final = float(res["C_stiefel_gcg"])

    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "job_id": job_id,
        "n": n,
        "method": method,
        "stiefel_optimizer": stiefel_optimizer,
        "k_max_extra": k_max_extra,
        "history": history,
        "final": final,
        "elapsed": time.time() - t0,
        "raw": res,
        "loaded_from_job_cache": False,
        "loaded_from_full_pkl": False,
        "source_full_pkl": None,
        "job_cache_path": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_method_only(hist, final, method_label, n, save_path, suffix=""):
    hist = np.asarray(hist, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    if len(hist) > 0:
        ax.plot(range(len(hist)), hist, "o-", markersize=3, linewidth=1.2)
    else:
        ax.text(0.5, 0.5, "No history recorded",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$h(U)$")
    ax.set_title(f"{method_label}: n={n} cost vs iteration{suffix}")
    ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes[1]
    if len(hist) > 0:
        eps = 1e-12
        err = np.maximum(hist - final, eps)
        ax.semilogy(range(len(err)), err, "o-", markersize=3, linewidth=1.2)
    else:
        ax.text(0.5, 0.5, "No history recorded",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$h(U)-h_{\mathrm{final}}$")
    ax.set_title(f"{method_label}: n={n} log error{suffix}")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"    saved → {save_path}", flush=True)


def plot_individual_convergence(results_by_n, plot_dir, stamp, method_ii_label, suffix=""):
    """
    Individual convergence: ONE figure per method (GCG and Method-II), each showing
    that method's own descent for every cluster size n. Left panel is the cost
    h(U) vs iteration; right panel is the log error to that method's OWN final.
    One curve per n, coloured by n. This lets you read how each method goes down
    on its own, without the other method overlaid.
    """
    # method key prefix -> (display label, marker)
    specs = [
        ("gcg", "GCG (Algorithm A)", "o"),
        ("st",  f"Method-II (Algorithm B, {method_ii_label})", "s"),
    ]

    ns = sorted(results_by_n.keys())
    cmap = plt.get_cmap("viridis")
    n_color = {n: cmap(i / max(1, len(ns) - 1)) for i, n in enumerate(ns)}
    eps = 1e-12

    for prefix, label, marker in specs:
        items = [
            (n, results_by_n[n])
            for n in ns
            if f"{prefix}_hist" in results_by_n[n] and f"{prefix}_final" in results_by_n[n]
        ]
        if not items:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        for n, d in items:
            h = np.asarray(d[f"{prefix}_hist"], dtype=float)
            if h.size:
                ax.plot(range(h.size), h, marker=marker, markersize=3, linewidth=1.2,
                        color=n_color[n], label=f"n={n}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"Cost $h(U)=\sum_i p_i C_{\mathrm{GME}}(|\psi_i\rangle)$")
        ax.set_title(f"{label}: cost vs iteration{suffix}")
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", alpha=0.4)

        ax = axes[1]
        for n, d in items:
            h = np.asarray(d[f"{prefix}_hist"], dtype=float)
            f_final = float(d[f"{prefix}_final"])
            if h.size:
                err = np.maximum(h - f_final, eps)
                ax.semilogy(range(err.size), err, marker=marker, markersize=3,
                            linewidth=1.2, color=n_color[n], label=f"n={n}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$h(U)-h_{\mathrm{final}}$")
        ax.set_title(f"{label}: log error to its own final{suffix}")
        ax.legend(loc="best")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

        plt.tight_layout()
        tag = "gcg" if prefix == "gcg" else "methodII"
        save_path = plot_dir / f"individual_{tag}_{stamp}.pdf"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"    saved → {save_path}", flush=True)


def plot_single_n(gcg_hist, st_hist, gcg_final, st_final, n, save_path, suffix=""):
    gcg_hist = np.asarray(gcg_hist, dtype=float)
    st_hist = np.asarray(st_hist, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]

    if len(gcg_hist) > 0:
        ax.plot(range(len(gcg_hist)), gcg_hist,
                "o-", markersize=3, linewidth=1.2,
                label="GCG (Algorithm A)")
    else:
        ax.text(0.5, 0.6, "GCG history missing",
                ha="center", va="center", transform=ax.transAxes)

    if len(st_hist) > 0:
        ax.plot(range(len(st_hist)), st_hist,
                "s-", markersize=3, linewidth=1.2,
                label="Method-II (Algorithm B)")
    else:
        ax.text(0.5, 0.4, "Method-II history missing",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Cost $h(U)=\sum_i p_i C_{\mathrm{GME}}(|\psi_i\rangle)$")
    ax.set_title(f"n={n}: cost vs iteration{suffix}")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes[1]
    h_min = min(gcg_final, st_final)
    eps = 1e-12

    if len(gcg_hist) > 0:
        err_g = np.maximum(gcg_hist - h_min, eps)
        ax.semilogy(range(len(err_g)), err_g,
                    "o-", markersize=3, linewidth=1.2,
                    label="GCG (Algorithm A)")

    if len(st_hist) > 0:
        err_s = np.maximum(st_hist - h_min, eps)
        ax.semilogy(range(len(err_s)), err_s,
                    "s-", markersize=3, linewidth=1.2,
                    label="Method-II (Algorithm B)")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$h(U)-h_{\min}$")
    ax.set_title(f"n={n}: log error to best minimum{suffix}")
    ax.legend(loc="best")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"    saved → {save_path}", flush=True)


def plot_combined(results_by_n, save_path, suffix=""):
    n_values = sorted(results_by_n.keys())

    fig, axes = plt.subplots(len(n_values), 2, figsize=(13, 4.2 * len(n_values)))
    if len(n_values) == 1:
        axes = axes.reshape(1, -1)

    for row, n in enumerate(n_values):
        d = results_by_n[n]
        gcg_hist = np.asarray(d["gcg_hist"], dtype=float)
        st_hist = np.asarray(d["st_hist"], dtype=float)

        h_min = min(d["gcg_final"], d["st_final"])
        eps = 1e-12

        ax = axes[row, 0]

        if len(gcg_hist) > 0:
            ax.plot(range(len(gcg_hist)), gcg_hist,
                    "o-", markersize=3, linewidth=1.2,
                    label="GCG (Algorithm A)")

        if len(st_hist) > 0:
            ax.plot(range(len(st_hist)), st_hist,
                    "s-", markersize=3, linewidth=1.2,
                    label="Method-II (Algorithm B)")

        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$h(U)$")
        ax.set_title(f"n={n}: cost vs iteration")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)

        ax = axes[row, 1]

        if len(gcg_hist) > 0:
            err_g = np.maximum(gcg_hist - h_min, eps)
            ax.semilogy(range(len(err_g)), err_g,
                        "o-", markersize=3, linewidth=1.2,
                        label="GCG (Algorithm A)")

        if len(st_hist) > 0:
            err_s = np.maximum(st_hist - h_min, eps)
            ax.semilogy(range(len(err_s)), err_s,
                        "s-", markersize=3, linewidth=1.2,
                        label="Method-II (Algorithm B)")

        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$h(U)-h_{\min}$")
        ax.set_title(f"n={n}: log error to best minimum")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    fig.suptitle(
        f"GCG vs Method-II convergence{suffix}",
        fontsize=14,
        fontweight="bold",
        y=1.0,
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"    saved → {save_path}", flush=True)


def plot_stiefel_outer_k(results_by_n, plot_dir, stamp):
    """
    Plot Method-II outer k-loop convergence if k_records or k_history exist.
    """
    for n in sorted(results_by_n.keys()):
        raw = results_by_n[n].get("st_raw", {})
        k_history = raw.get("k_history", None)

        if k_history is None:
            k_records = raw.get("k_records", None)
            if k_records is not None:
                try:
                    k_history = [float(rec["C_GME"]) for rec in k_records]
                except Exception:
                    k_history = None

        if not k_history:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(range(len(k_history)), k_history, "o-", markersize=4, linewidth=1.4)
        ax.set_xlabel(r"Outer step $k-r$")
        ax.set_ylabel(r"$C_{\mathrm{GME}}$")
        ax.set_title(f"Method-II outer k convergence, n={n}")
        ax.grid(True, linestyle="--", alpha=0.4)

        save_path = plot_dir / f"methodII_outer_k_n{n}_{stamp}.pdf"
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"    saved → {save_path}", flush=True)


def plot_job_durations(all_results, plot_dir, stamp):
    if not all_results:
        return

    ordered = sorted(all_results, key=lambda r: (r["n"], r["method"]))
    labels = [f"n={r['n']} {r['method']}" for r in ordered]
    times = [float(r.get("elapsed", 0.0)) for r in ordered]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(labels))))
    y = np.arange(len(labels))
    ax.barh(y, times)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Elapsed time per job (s)")
    ax.set_title("GCG/Method-II job durations")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    save_path = plot_dir / f"job_durations_{stamp}.pdf"
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"    saved → {save_path}", flush=True)




def plot_single_n_all_methods(all_method_results, n, save_path, suffix=""):
    """Overlay all available methods/optimizers for one n."""
    if not all_method_results:
        return

    entries = list(all_method_results.items())
    h_min = min(float(v["final"]) for _, v in entries)
    eps = 1e-12

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    for tag, v in entries:
        h = np.asarray(v.get("history", []), dtype=float)
        if h.size:
            ax.plot(range(h.size), h, marker="o", markersize=3, linewidth=1.2,
                    label=v.get("label", tag))
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Cost $h(U)=\sum_i p_i C_{\mathrm{GME}}(|\psi_i\rangle)$")
    ax.set_title(f"n={n}: all methods cost vs iteration{suffix}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    ax = axes[1]
    for tag, v in entries:
        h = np.asarray(v.get("history", []), dtype=float)
        if h.size:
            err = np.maximum(h - h_min, eps)
            ax.semilogy(range(err.size), err, marker="o", markersize=3, linewidth=1.2,
                        label=v.get("label", tag))
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$h(U)-h_{\min}$")
    ax.set_title(f"n={n}: all methods log error to best minimum")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"    saved → {save_path}", flush=True)


def plot_individual_all_method_convergence(results_by_n, plot_dir, stamp, suffix=""):
    """One figure per method/optimizer, with one curve per n."""
    tags = []
    labels = {}
    for n, d in results_by_n.items():
        for tag, v in d.get("all_method_results", {}).items():
            if tag not in tags:
                tags.append(tag)
                labels[tag] = v.get("label", tag)

    ns = sorted(results_by_n.keys())
    if not tags or not ns:
        return

    cmap = plt.get_cmap("viridis")
    n_color = {n: cmap(i / max(1, len(ns) - 1)) for i, n in enumerate(ns)}
    eps = 1e-12

    for tag in tags:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        any_curve = False
        for n in ns:
            entry = results_by_n[n].get("all_method_results", {}).get(tag)
            if entry is None:
                continue
            h = np.asarray(entry.get("history", []), dtype=float)
            if h.size:
                any_curve = True
                ax.plot(range(h.size), h, marker="o", markersize=3, linewidth=1.2,
                        color=n_color[n], label=f"n={n}")
        if not any_curve:
            ax.text(0.5, 0.5, "No history recorded", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"Cost $h(U)$")
        ax.set_title(f"{labels[tag]}: cost vs iteration{suffix}")
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", alpha=0.4)

        ax = axes[1]
        for n in ns:
            entry = results_by_n[n].get("all_method_results", {}).get(tag)
            if entry is None:
                continue
            h = np.asarray(entry.get("history", []), dtype=float)
            if h.size:
                err = np.maximum(h - float(entry["final"]), eps)
                ax.semilogy(range(err.size), err, marker="o", markersize=3, linewidth=1.2,
                            color=n_color[n], label=f"n={n}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$h(U)-h_{\mathrm{final}}$")
        ax.set_title(f"{labels[tag]}: log error to own final")
        ax.legend(loc="best")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

        plt.tight_layout()
        save_path = plot_dir / f"individual_all_{safe_tag(tag)}_{stamp}.pdf"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"    saved → {save_path}", flush=True)


def plot_all_method_final_values(results_by_n, plot_dir, stamp):
    rows = []
    for n in sorted(results_by_n.keys()):
        for tag, v in results_by_n[n].get("all_method_results", {}).items():
            rows.append((n, v.get("label", tag), float(v.get("final", np.nan)), float(v.get("elapsed", 0.0))))
    if not rows:
        return

    labels = [f"n={n} {label}" for n, label, _, _ in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(11, max(5, 0.42 * len(rows))))
    ax.barh(y, [r[2] for r in rows])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(r"Final $C_{\mathrm{GME}}$")
    ax.set_title("Final values: all methods/optimizers")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    save_path = plot_dir / f"final_values_all_methods_{stamp}.pdf"
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"    saved → {save_path}", flush=True)

    fig, ax = plt.subplots(figsize=(11, max(5, 0.42 * len(rows))))
    ax.barh(y, [r[3] for r in rows])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Elapsed time per job (s)")
    ax.set_title("Runtime: all methods/optimizers")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    save_path = plot_dir / f"runtime_all_methods_{stamp}.pdf"
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"    saved → {save_path}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--n-restarts", type=int, default=1)
    parser.add_argument("--k-max-extra", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--n-list", type=str, default="3,4,5")

    parser.add_argument(
        "--methods",
        type=str,
        default="both",
        choices=[
            "both", "gcg", "method-II", "stiefel",
            "method-II-gcg", "stiefel-gcg",
            "all3", "all",
            "stiefel-all-optimizers", "method-II-all-optimizers",
        ],
        help=(
            "Which methods to compute if not already cached. "
            "both=GCG+one Method-II optimizer; all3=GCG+Method-II+Stiefel-GCG; "
            "all=GCG+Stiefel with all optimizers+Stiefel-GCG."
        ),
    )

    parser.add_argument(
        "--method-II-optimizer",
        dest="method_II_optimizer",
        type=str,
        default="L-BFGS-B",
        choices=["L-BFGS-B", "Powell", "SLSQP", "TNC", "trust-constr"],
        help="Primary bounded optimizer used inside Method-II. Old plots keep using this name.",
    )

    parser.add_argument(
        "--method-II-optimizers",
        dest="method_II_optimizers",
        type=str,
        default=None,
        help="Comma list of Method-II optimizers or 'all'. Overrides --method-II-optimizer for all-method runs.",
    )

    parser.add_argument("--force-state-recompute", action="store_true")
    parser.add_argument("--force-rdm-recompute", action="store_true")
    parser.add_argument("--force-job-recompute", action="store_true")
    parser.add_argument("--force-gcg-recompute", action="store_true")
    parser.add_argument("--force-method-II-recompute", dest="force_method_II_recompute",
                        action="store_true")
    parser.add_argument("--no-load-from-full-runs", action="store_true")
    parser.add_argument("--cache-version", type=str, default="v1")
    parser.add_argument("--no-save-full-pkl", action="store_true")

    args = parser.parse_args()

    # System setup
    L = 6
    Ne = 6
    U = 4.0
    t = 1.0
    mode = "uniform"

    n_list = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
    stiefel_optimizer_list = parse_stiefel_optimizers(args.method_II_optimizers, args.method_II_optimizer)
    requested_method_jobs = build_method_jobs(args.methods, stiefel_optimizer_list)

    # Central project paths from rdm.py
    project_root = get_project_root()
    rdm_pkl_root = get_rdm_pkl_results_dir()
    state_cache_dir = get_state_cache_dir()
    rdm_cache_dir = get_rdm_cache_dir()

    convergence_pkl_dir = ensure_dir(rdm_pkl_root / "convergence_runs")
    job_cache_dir = ensure_dir(convergence_pkl_dir / "job_cache")
    plot_dir = ensure_dir(project_root / "results" / "plots" / "convergence")

    stamp = timestamp()

    print("\n" + "─" * 80, flush=True)
    print("Convergence benchmark using central RDM PKL cache + job resume cache", flush=True)
    print("─" * 80, flush=True)
    print(f"Project root:        {project_root}", flush=True)
    print(f"RDM PKL root:        {rdm_pkl_root}", flush=True)
    print(f"State cache dir:     {state_cache_dir}", flush=True)
    print(f"RDM cache dir:       {rdm_cache_dir}", flush=True)
    print(f"Convergence PKL dir: {convergence_pkl_dir}", flush=True)
    print(f"Job cache dir:       {job_cache_dir}", flush=True)
    print(f"PDF plot dir:        {plot_dir}", flush=True)
    print("─" * 80, flush=True)
    print(f"System: L={L}, Ne={Ne}, U={U}, t={t}, mode={mode}", flush=True)
    print(f"Cluster sizes:       {n_list}", flush=True)
    print(f"Methods:             {args.methods}", flush=True)
    print(f"Workers:             {args.workers}", flush=True)
    print(f"Restarts/method:     {args.n_restarts}", flush=True)
    print(f"k_max_extra:         {args.k_max_extra}", flush=True)
    print(f"Max iter:            {args.max_iter}", flush=True)
    print(f"Primary Method-II optimizer: {args.method_II_optimizer}", flush=True)
    print(f"Method-II optimizer list:    {stiefel_optimizer_list}", flush=True)
    print("Requested optimizer jobs:", flush=True)
    for _m, _opt in requested_method_jobs:
        print(f"  - {method_label(_m, _opt)}", flush=True)
    print("─" * 80 + "\n", flush=True)

    # ── Load or compute system state ─────────────────────────────────────────
    print("Loading/computing system state ...", flush=True)
    t0 = time.time()

    state_result = load_or_compute_system_state_pkl(
        compute_fn=compute_ground_and_excited_states,
        compute_kwargs={
            "L": L,
            "t": t,
            "U": U,
            "mode": mode,
            "n_electrons": Ne,
        },
        L=L,
        Ne=Ne,
        U=U,
        t=t,
        mode=mode,
        cache_version=args.cache_version,
        force_recompute=args.force_state_recompute,
        basis_meta={
            "basis_sector": "fixed particle number",
            "spin_orbital_convention": "2 spin orbitals per site",
            "n_so_expected": 2 * L,
        },
        operator_meta={
            "hamiltonian": "Hubbard model",
            "boundary_condition_or_mode": mode,
        },
        sigma_meta={
            "pauli_matrices": "standard convention, not stored explicitly",
        },
    )

    E0 = state_result["E0"]
    psi0 = state_result["psi0"]
    E1 = state_result["E1"]
    psi1 = state_result["psi1"]
    n_so = state_result["n_so"]

    state_elapsed = time.time() - t0

    print(f"  State cache: {state_result['cache_path']}", flush=True)
    print(f"  Loaded state from cache: {state_result['loaded_from_cache']}", flush=True)
    print(f"  E0 = {E0:.12f}", flush=True)
    if E1 is not None:
        print(f"  E1 = {E1:.12f}", flush=True)
    print(f"  n_so = {n_so}", flush=True)
    print(f"  state step time = {state_elapsed:.2f} s\n", flush=True)

    parent_state_key = make_system_cache_key(
        L=L,
        Ne=Ne,
        U=U,
        t=t,
        mode=mode,
        cache_version=args.cache_version,
    )

    parent_state_hash = None
    try:
        parent_state_hash = state_result["raw"]["states"]["gs"]["psi_hash"]
    except Exception:
        pass

    # ── Load or compute RDMs ─────────────────────────────────────────────────
    print("Loading/computing RDMs ...", flush=True)
    rdms_by_n: Dict[int, np.ndarray] = {}
    rdm_records_by_n: Dict[int, Dict[str, Any]] = {}

    for n in n_list:
        sites = tuple(range(n))

        rdm_result = load_or_compute_rdm_pkl(
            psi=psi0,
            L=L,
            Ne=Ne,
            U=U,
            t=t,
            mode=mode,
            sites=sites,
            n_so=n_so,
            n_electrons=Ne,
            state_label="gs",
            state_energy=E0,
            parent_state_cache_path=state_result["cache_path"],
            parent_state_cache_key=parent_state_key,
            parent_state_hash=parent_state_hash,
            cache_version=args.cache_version,
            force_recompute=args.force_rdm_recompute,
            basis_meta={
                "basis_sector": "fixed particle number",
                "spin_orbital_convention": "2 spin orbitals per site",
            },
            operator_meta={
                "rdm_source": "compute_general_nRDM",
            },
            sigma_meta={
                "pauli_matrices": "standard convention, not stored explicitly",
            },
        )

        rho = rdm_result["rho"]
        metrics = rdm_result["metrics"]

        rdms_by_n[n] = rho
        rdm_records_by_n[n] = rdm_result

        status = "loaded" if rdm_result["loaded_from_cache"] else "computed"
        print(
            f"  n={n}, sites={sites}, {status}, "
            f"purity={metrics['purity']:.8f}, "
            f"entropy={metrics['entropy']:.8f}, "
            f"rank={metrics['rank_numerical']}",
            flush=True,
        )
        print(f"    RDM cache: {rdm_result['cache_path']}", flush=True)

    print("", flush=True)

    # ── Load cached job results and build missing jobs ───────────────────────
    all_results: List[Dict[str, Any]] = []
    jobs = []
    job_meta_by_id: Dict[int, Dict[str, Any]] = {}
    job_id = 0

    methods_to_compute = requested_method_jobs

    # Load all requested jobs. Also load GCG + primary Method-II when available,
    # so the original combined plots still work exactly like before.
    methods_to_load = list(requested_method_jobs)
    for _extra in [("gcg", None), ("method-II", args.method_II_optimizer)]:
        if _extra not in methods_to_load:
            methods_to_load.append(_extra)

    print("Checking optimizer job cache ...", flush=True)

    for n in n_list:
        for method, stiefel_opt in methods_to_load:
            job_meta = build_job_meta(
                L=L,
                Ne=Ne,
                n_so=n_so,
                U=U,
                t=t,
                mode=mode,
                n=n,
                method=method,
                n_restarts=args.n_restarts,
                max_iter=args.max_iter,
                cache_version=args.cache_version,
                stiefel_optimizer=stiefel_opt,
                k_max_extra=args.k_max_extra,
            )

            cached = load_job_result(job_cache_dir, job_meta)

            if cached is None and not args.no_load_from_full_runs:
                cached = load_result_from_old_full_pkls(
                    convergence_pkl_dir=convergence_pkl_dir,
                    job_meta=job_meta,
                )

                # If recovered from old full PKL, immediately convert to job_cache.
                if cached is not None:
                    recovered_path = save_job_result(job_cache_dir, cached, job_meta)
                    cached["job_cache_path"] = str(recovered_path)

            if cached is not None:
                src = "job_cache" if cached.get("loaded_from_job_cache") else "full_pkl"
                print(
                    f"  [cache:{src}] n={n} method={method:<8} "
                    f"final={cached['final']:.8f} "
                    f"iters={format_iters(cached['history']):>4} "
                    f"path={cached.get('job_cache_path') or cached.get('source_full_pkl')}",
                    flush=True,
                )
                all_results.append(cached)

    cached_keys = {(r["n"], r["method"], r.get("stiefel_optimizer")) for r in all_results}

    for n in n_list:
        rho = rdms_by_n[n]

        for method, stiefel_opt in methods_to_compute:
            already_cached = (n, method, stiefel_opt) in cached_keys

            force_this = args.force_job_recompute
            if method == "gcg" and args.force_gcg_recompute:
                force_this = True
            if method == "method-II" and args.force_method_II_recompute:
                force_this = True

            if already_cached and not force_this:
                continue

            job_meta = build_job_meta(
                L=L,
                Ne=Ne,
                n_so=n_so,
                U=U,
                t=t,
                mode=mode,
                n=n,
                method=method,
                n_restarts=args.n_restarts,
                max_iter=args.max_iter,
                cache_version=args.cache_version,
                stiefel_optimizer=stiefel_opt,
                k_max_extra=args.k_max_extra,
            )

            stiefel_opt_for_job = stiefel_opt if method == "method-II" else None
            jobs.append((job_id, n, method, rho, args.n_restarts, args.k_max_extra, args.max_iter, stiefel_opt_for_job))
            job_meta_by_id[job_id] = job_meta
            job_id += 1

    # ── Run missing jobs in parallel ─────────────────────────────────────────
    if jobs:
        print(f"\nDispatching {len(jobs)} new jobs to {args.workers} workers ...", flush=True)
        print("Each completed job is saved immediately to job_cache.\n", flush=True)

        pool_t0 = time.time()

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {executor.submit(run_one_job, job): job for job in jobs}

            for future in as_completed(future_to_job):
                jid, n, method, _, _, _, _, _ = future_to_job[future]

                try:
                    result = future.result()

                    job_meta = job_meta_by_id[jid]
                    job_cache_path = save_job_result(job_cache_dir, result, job_meta)
                    result["job_cache_path"] = str(job_cache_path)

                    all_results.append(result)

                    print(
                        f"  [done] job={result['job_id']:>2} "
                        f"n={result['n']} method={result['method']:<8} "
                        f"final={result['final']:.8f} "
                        f"iters={format_iters(result['history']):>4} "
                        f"time={result['elapsed']:.2f} s",
                        flush=True,
                    )
                    print(f"         saved job cache → {job_cache_path}", flush=True)

                except Exception as e:
                    print(
                        f"  [FAIL] job={jid} n={n} method={method}: {repr(e)}",
                        flush=True,
                    )

        pool_elapsed = time.time() - pool_t0
    else:
        print("\nNo new optimizer jobs needed. Requested results were loaded from cache.\n", flush=True)
        pool_elapsed = 0.0

    new_results = [
        r for r in all_results
        if not r.get("loaded_from_job_cache", False) and not r.get("loaded_from_full_pkl", False)
    ]
    sum_job_times = sum(float(r.get("elapsed", 0.0)) for r in new_results)
    longest_job_time = max((float(r.get("elapsed", 0.0)) for r in new_results), default=0.0)

    print("\nTiming for newly computed jobs:", flush=True)
    print(f"  pool wall time       = {pool_elapsed:.2f} s", flush=True)
    print(f"  sum individual times = {sum_job_times:.2f} s", flush=True)
    print(f"  longest job time     = {longest_job_time:.2f} s", flush=True)
    if pool_elapsed > 0:
        print(f"  rough speedup        = {sum_job_times / pool_elapsed:.2f}x", flush=True)
    print("", flush=True)

    # ── Organise results ─────────────────────────────────────────────────────
    # If duplicates exist due to recompute, newly computed results should overwrite cached ones.
    # all_results order is cached first, computed later, so simple assignment works.
    results_by_n: Dict[int, Dict[str, Any]] = {}

    for r in all_results:
        n = r["n"]

        if n not in results_by_n:
            results_by_n[n] = {}

        append_all_method_entry(results_by_n, r)

        if r["method"] == "gcg":
            results_by_n[n]["gcg_hist"] = r["history"]
            results_by_n[n]["gcg_final"] = r["final"]
            results_by_n[n]["gcg_time"] = float(r.get("elapsed", 0.0))
            results_by_n[n]["gcg_raw"] = r["raw"]
            results_by_n[n]["gcg_job_cache_path"] = r.get("job_cache_path")
            results_by_n[n]["gcg_loaded_from_job_cache"] = r.get("loaded_from_job_cache", False)
            results_by_n[n]["gcg_loaded_from_full_pkl"] = r.get("loaded_from_full_pkl", False)

        elif r["method"] == "method-II":
            # Preserve the original st_hist/st_final keys for the primary optimizer.
            if r.get("stiefel_optimizer") == args.method_II_optimizer:
                results_by_n[n]["st_hist"] = r["history"]
                results_by_n[n]["st_final"] = r["final"]
                results_by_n[n]["st_time"] = float(r.get("elapsed", 0.0))
                results_by_n[n]["st_raw"] = r["raw"]
                results_by_n[n]["st_job_cache_path"] = r.get("job_cache_path")
                results_by_n[n]["st_loaded_from_job_cache"] = r.get("loaded_from_job_cache", False)
                results_by_n[n]["st_loaded_from_full_pkl"] = r.get("loaded_from_full_pkl", False)

        elif r["method"] == "method-II-gcg":
            results_by_n[n]["stgcg_hist"] = r["history"]
            results_by_n[n]["stgcg_final"] = r["final"]
            results_by_n[n]["stgcg_time"] = float(r.get("elapsed", 0.0))
            results_by_n[n]["stgcg_raw"] = r["raw"]
            results_by_n[n]["stgcg_job_cache_path"] = r.get("job_cache_path")
            results_by_n[n]["stgcg_loaded_from_job_cache"] = r.get("loaded_from_job_cache", False)
            results_by_n[n]["stgcg_loaded_from_full_pkl"] = r.get("loaded_from_full_pkl", False)

    complete_n = []
    incomplete_n = []

    for n in n_list:
        d = results_by_n.get(n, {})
        if all(k in d for k in ["gcg_hist", "gcg_final", "st_hist", "st_final"]):
            complete_n.append(n)
        else:
            incomplete_n.append(n)

    if incomplete_n:
        print("Some n values do not have both GCG and Method-II:", flush=True)
        for n in incomplete_n:
            print(f"  n={n}, available keys={sorted(results_by_n.get(n, {}).keys())}", flush=True)
        print("Combined plots will be made only for complete n values.\n", flush=True)

    # ── Save full benchmark pkl ──────────────────────────────────────────────
    if not args.no_save_full_pkl:
        full_data = {
            "meta": {
                "script": os.path.basename(__file__),
                "created_at": stamp,
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "platform": platform.platform(),
            },
            "config": {
                "L": L,
                "Ne": Ne,
                "U": U,
                "t": t,
                "mode": mode,
                "n_so": n_so,
                "n_list": n_list,
                "methods": args.methods,
                "workers": args.workers,
                "n_restarts": args.n_restarts,
                "k_max_extra": args.k_max_extra,
                "max_iter": args.max_iter,
                # Config key kept as "stiefel_optimizer" for cache/report compatibility.
                "stiefel_optimizer": args.method_II_optimizer,
                "stiefel_optimizers": stiefel_optimizer_list,
                "requested_method_jobs": requested_method_jobs,
                "cache_version": args.cache_version,
                "force_state_recompute": args.force_state_recompute,
                "force_rdm_recompute": args.force_rdm_recompute,
                "force_job_recompute": args.force_job_recompute,
                "project_root": str(project_root),
                "rdm_pkl_root": str(rdm_pkl_root),
                "state_cache_dir": str(state_cache_dir),
                "rdm_cache_dir": str(rdm_cache_dir),
                "convergence_pkl_dir": str(convergence_pkl_dir),
                "job_cache_dir": str(job_cache_dir),
                "plot_dir": str(plot_dir),
            },

            # Explicit link to the system/state PKL used to generate the RDMs.
            "system_state_ref": {
                "cache_path": state_result["cache_path"],
                "cache_key": parent_state_key,
                "state_label": "gs",
                "state_hash": parent_state_hash,
                "E0": E0,
                "E1": E1,
                "n_so": n_so,
                "loaded_from_cache": state_result["loaded_from_cache"],
            },

            # Explicit links to the exact RDM PKLs used for each n.
            # We do NOT duplicate rho here; rho already lives in the RDM cache file.
            "rdm_refs": {
                n: {
                    "sites": tuple(range(n)),
                    "cache_path": rdm_records_by_n[n]["cache_path"],
                    "state_label": "gs",
                    "parent_state_cache_path": state_result["cache_path"],
                    "parent_state_cache_key": parent_state_key,
                    "parent_state_hash": parent_state_hash,
                    "meta": rdm_records_by_n[n]["meta"],
                    "metrics": rdm_records_by_n[n]["metrics"],
                    "loaded_from_cache": rdm_records_by_n[n]["loaded_from_cache"],
                    "rho_saved_in_this_file": False,
                }
                for n in n_list
            },

            "all_results": all_results,
            "results_by_n": results_by_n,
            "complete_n_for_combined_plots": complete_n,
            "timing": {
                "state_elapsed": state_elapsed,
                "pool_wall_time_new_jobs": pool_elapsed,
                "sum_new_job_times": sum_job_times,
                "longest_new_job_time": longest_job_time,
                "rough_speedup_new_jobs": (sum_job_times / pool_elapsed) if pool_elapsed > 0 else None,
            },
        }

        full_pkl_name = make_full_convergence_filename(
            L=L,
            Ne=Ne,
            n_so=n_so,
            U=U,
            t=t,
            mode=mode,
            n_list=n_list,
            methods=args.methods,
            stiefel_optimizer=args.method_II_optimizer,
            n_restarts=args.n_restarts,
            max_iter=args.max_iter,
            cache_version=args.cache_version,
            stamp=stamp,
        )

        pkl_path = convergence_pkl_dir / full_pkl_name
        with open(pkl_path, "wb") as f:
            pickle.dump(full_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved full benchmark PKL → {pkl_path}\n", flush=True)

    # ── Plots ────────────────────────────────────────────────────────────────
    print("Generating PDF plots ...", flush=True)

    # Method-only plots whenever that method is available (one per method per n).
    for n in sorted(results_by_n.keys()):
        d = results_by_n[n]
        sites = tuple(range(n))
        suffix = f"  (L={L}, U={U}, sites={sites})"

        if "gcg_hist" in d:
            plot_method_only(
                d["gcg_hist"],
                d["gcg_final"],
                "GCG Algorithm A",
                n,
                plot_dir / f"convergence_gcg_n{n}_{stamp}.pdf",
                suffix=suffix,
            )

        if "st_hist" in d:
            plot_method_only(
                d["st_hist"],
                d["st_final"],
                f"Method-II Algorithm B ({args.method_II_optimizer})",
                n,
                plot_dir / f"convergence_methodII_n{n}_{stamp}.pdf",
                suffix=suffix,
            )

    # Individual convergence: one figure per method showing its descent for every n.
    plot_individual_convergence(
        results_by_n,
        plot_dir,
        stamp,
        args.method_II_optimizer,
        suffix=f"  (L={L}, U={U})",
    )

    # Combined plots only where both methods exist.
    combined_results_by_n = {n: results_by_n[n] for n in complete_n}

    for n in sorted(combined_results_by_n.keys()):
        d = combined_results_by_n[n]
        sites = tuple(range(n))
        suffix = f"  (L={L}, U={U}, sites={sites})"

        print(
            f"  DEBUG n={n}: "
            f"GCG hist len={len(d['gcg_hist'])}, "
            f"Method-II hist len={len(d['st_hist'])}, "
            f"GCG final={d['gcg_final']:.8f}, "
            f"Method-II final={d['st_final']:.8f}",
            flush=True,
        )

        plot_single_n(
            d["gcg_hist"],
            d["st_hist"],
            d["gcg_final"],
            d["st_final"],
            n,
            plot_dir / f"convergence_n{n}_{stamp}.pdf",
            suffix=suffix,
        )

    if combined_results_by_n:
        plot_combined(
            combined_results_by_n,
            plot_dir / f"convergence_combined_{stamp}.pdf",
            suffix=f"  (L={L}, U={U})",
        )
        plot_stiefel_outer_k(combined_results_by_n, plot_dir, stamp)

    # New all-method plots: includes GCG, primary/all Method-II optimizers, and Method-II-GCG.
    for n in sorted(results_by_n.keys()):
        all_block = results_by_n[n].get("all_method_results", {})
        if all_block:
            sites = tuple(range(n))
            plot_single_n_all_methods(
                all_block,
                n,
                plot_dir / f"convergence_all_methods_n{n}_{stamp}.pdf",
                suffix=f"  (L={L}, U={U}, sites={sites})",
            )

    plot_individual_all_method_convergence(
        results_by_n,
        plot_dir,
        stamp,
        suffix=f"  (L={L}, U={U})",
    )

    plot_all_method_final_values(results_by_n, plot_dir, stamp)

    plot_job_durations(all_results, plot_dir, stamp)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\nSummary:", flush=True)
    print(
        f"{'n':>3} {'method':<28} {'h_final':>14} "
        f"{'iters':>8} {'time(s)':>10} {'source':>12}",
        flush=True,
    )
    print("-" * 92, flush=True)

    for n in sorted(results_by_n.keys()):
        d = results_by_n[n]

        if "gcg_final" in d:
            if d.get("gcg_loaded_from_job_cache"):
                source = "job_cache"
            elif d.get("gcg_loaded_from_full_pkl"):
                source = "full_pkl"
            else:
                source = "computed"

            print(
                f"{n:>3} {'GCG (Algorithm A)':<28} "
                f"{d['gcg_final']:>14.8f} "
                f"{format_iters(d['gcg_hist']):>8} "
                f"{d['gcg_time']:>10.2f} "
                f"{source:>12}",
                flush=True,
            )

        if "st_final" in d:
            if d.get("st_loaded_from_job_cache"):
                source = "job_cache"
            elif d.get("st_loaded_from_full_pkl"):
                source = "full_pkl"
            else:
                source = "computed"

            print(
                f"{n:>3} {('Method-II ' + args.method_II_optimizer):<28} "
                f"{d['st_final']:>14.8f} "
                f"{format_iters(d['st_hist']):>8} "
                f"{d['st_time']:>10.2f} "
                f"{source:>12}",
                flush=True,
            )

    print("\nAll-method / all-optimizer summary:", flush=True)
    print(f"{'n':>3} {'method/optimizer':<42} {'h_final':>14} {'iters':>8} {'time(s)':>10} {'source':>12}", flush=True)
    print("-" * 100, flush=True)
    for n in sorted(results_by_n.keys()):
        for tag, v in results_by_n[n].get("all_method_results", {}).items():
            if v.get("loaded_from_job_cache"):
                source = "job_cache"
            elif v.get("loaded_from_full_pkl"):
                source = "full_pkl"
            else:
                source = "computed"
            print(
                f"{n:>3} {v.get('label', tag):<42} "
                f"{float(v.get('final', np.nan)):>14.8f} "
                f"{format_iters(v.get('history', [])):>8} "
                f"{float(v.get('elapsed', 0.0)):>10.2f} "
                f"{source:>12}",
                flush=True,
            )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()