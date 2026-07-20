#!/usr/bin/env python3
"""
fig4_full_all_methods.py
------------------------

Reproduce Fig. 4 of Rothlisberger, Lehmann, Loss, PRA 80, 042301 (2009)
for multiple convex-roof solvers:

  1. gcg             : Algorithm A
  2. stiefel         : Algorithm B, with selectable/all SciPy optimizers
  3. stiefel_gcg     : hybrid Stiefel + Riemannian CG

This script can produce:
  - one Fig. 4 style plot per method/optimizer
  - one comparison plot at a selected temperature

WARNING:
The full all-method run is very expensive.  With default full grid:
  N=2,3,4,5; T=4 values; b=50 values
one method = 800 optimization jobs.
All three methods plus all five Stiefel optimizers can be thousands of jobs.
Use --coarse first.

Examples
--------

Fast debug, all 3 main methods:
python fig4_full_all_methods.py --methods all3 --coarse --n-restarts 1 --k-max-extra 0 --workers 8 --format pdf

All Stiefel optimizers only:
python fig4_full_all_methods.py --methods stiefel-all-optimizers --coarse --format pdf

Everything:
python fig4_full_all_methods.py --methods all --stiefel-optimizers all --format pdf

Reload:
python fig4_full_all_methods.py --reload --input-npz outputs/fig4_all_methods_coarse.npz --format pdf
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
for candidate in [HERE / ".." / "src", HERE / "src", HERE.parent / "src"]:
    candidate = candidate.resolve()
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from hubbard_gme.convex_roof import (
    _meyerwallach_pure_vector,
    convex_roof_stiefel,
    convex_roof_gcg,
    convex_roof_stiefel_gcg,
)

STIEFEL_OPTIMIZERS = ["L-BFGS-B", "Powell", "SLSQP", "TNC", "trust-constr"]


sigma_x = np.array([[0, 1],  [1, 0]],   dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0],  [0, -1]],  dtype=complex)
I2 = np.eye(2, dtype=complex)

Sx_single = sigma_x / 2
Sy_single = sigma_y / 2
Sz_single = sigma_z / 2
#!/usr/bin/env python3
"""
fig4_full_all_methods.py
------------------------

Reproduce Fig. 4 of Rothlisberger, Lehmann, Loss, PRA 80, 042301 (2009)
for multiple convex-roof solvers:

  1. gcg             : Algorithm A
  2. stiefel         : Algorithm B, with selectable/all SciPy optimizers
  3. stiefel_gcg     : hybrid Stiefel + Riemannian CG

This script can produce:
  - one Fig. 4 style plot per method/optimizer
  - one comparison plot at a selected temperature

WARNING:
The full all-method run is very expensive.  With default full grid:
  N=2,3,4,5; T=4 values; b=50 values
one method = 800 optimization jobs.
All three methods plus all five Stiefel optimizers can be thousands of jobs.
Use --coarse first.

Examples
--------

Fast debug, all 3 main methods:
python fig4_full_all_methods.py --methods all3 --coarse --n-restarts 1 --k-max-extra 0 --workers 8 --format pdf

All Stiefel optimizers only:
python fig4_full_all_methods.py --methods stiefel-all-optimizers --coarse --format pdf

Everything:
python fig4_full_all_methods.py --methods all --stiefel-optimizers all --format pdf

Reload:
python fig4_full_all_methods.py --reload --input-npz outputs/fig4_all_methods_coarse.npz --format pdf
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
for candidate in [HERE / ".." / "src", HERE / "src", HERE.parent / "src"]:
    candidate = candidate.resolve()
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from hubbard_gme.convex_roof import (
    _meyerwallach_pure_vector,
    convex_roof_stiefel,
    convex_roof_gcg,
    convex_roof_stiefel_gcg,
)

STIEFEL_OPTIMIZERS = ["L-BFGS-B", "Powell", "SLSQP", "TNC", "trust-constr"]

def _meyerwallach_pure_vector(psi: np.ndarray, n: int) -> float:
    """
    Meyer-Wallach γ(|ψ⟩) = 2 [1 − (1/N) Σ_k Tr(ρ_k²)]   Eq. (45) of the paper.

    Parameters
    ----------
    psi : (2^n,) complex array, normalised pure state.
    n   : number of qubits.

    Returns
    -------
    gamma : float in [0, 1].
    """
    psi_tensor = psi.reshape([2] * n)
    purity_sum = 0.0

    for k in range(n):
        # Reshape ψ as M_k of shape (2, 2^(n-1)) with qubit k as row index.
        # Move axis k to position 0 (leading axis).
        axes = [k] + [i for i in range(n) if i != k]
        M_k = np.transpose(psi_tensor, axes).reshape(2, 2 ** (n - 1))

        # rho_k = M_k M_k†, then Tr(rho_k²) = Tr((M_k M_k†)(M_k M_k†))
        rho_k = M_k @ M_k.conj().T              # (2, 2)
        purity_k = float(np.real(np.trace(rho_k @ rho_k)))
        purity_sum += purity_k

    gamma = 2.0 * (1.0 - purity_sum / n)
    return float(max(0.0, gamma))
sigma_x = np.array([[0, 1],  [1, 0]],   dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0],  [0, -1]],  dtype=complex)
I2 = np.eye(2, dtype=complex)

Sx_single = sigma_x / 2
Sy_single = sigma_y / 2
Sz_single = sigma_z / 2


def safe_tag(x: Any) -> str:
    s = str(x)
    for a, b in [(" ", ""), ("/", "-"), ("\\", "-"), (".", "p"), ("+", "")]:
        s = s.replace(a, b)
    return s


def operator_on_site(op_local, site, N):
    ops = [op_local if i == site else I2 for i in range(N)]
    full = ops[0]
    for op in ops[1:]:
        full = np.kron(full, op)
    return full


def build_hamiltonian(N, J=1.0, b=0.1):
    """Eq. (44) spin-ring Hamiltonian with PBC."""
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)

    Sx_ops = [operator_on_site(Sx_single, i, N) for i in range(N)]
    Sy_ops = [operator_on_site(Sy_single, i, N) for i in range(N)]
    Sz_ops = [operator_on_site(Sz_single, i, N) for i in range(N)]

    for i in range(N):
        j = (i + 1) % N
        H += -J * (Sx_ops[i] @ Sx_ops[j]
                   + Sy_ops[i] @ Sy_ops[j]
                   + Sz_ops[i] @ Sz_ops[j])

    for i in range(N):
        alpha_i = 2 * np.pi * i / N
        H += b * (np.cos(alpha_i) * Sx_ops[i]
                  + np.sin(alpha_i) * Sy_ops[i])

    return H


def thermal_state(H, T, degeneracy_tol=1e-10):
    """
    Build thermal density matrix rho(T) = exp(-beta H) / Z.

    Uses energy shift E - E0 for numerical stability.
    For T <= 0, returns the ground-state projector. If the ground state
    is degenerate within degeneracy_tol, returns the equal mixture over
    the degenerate ground-state subspace.
    """
    energies, vecs = np.linalg.eigh(H)

    if T <= 0:
        E0 = energies[0]
        ground_idx = np.where(np.abs(energies - E0) < degeneracy_tol)[0]

        rho0 = np.zeros((H.shape[0], H.shape[0]), dtype=complex)
        for idx in ground_idx:
            psi = vecs[:, idx]
            rho0 += np.outer(psi, psi.conj())

        rho0 /= len(ground_idx)
        return rho0

    beta = 1.0 / T
    shifted = energies - energies[0]

    weights = np.exp(-beta * shifted)
    probs = weights / np.sum(weights)

    rho = vecs @ np.diag(probs) @ vecs.conj().T

    # Optional cleanup for numerical precision
    rho = 0.5 * (rho + rho.conj().T)
    rho /= np.trace(rho)

    return rho


def parse_optimizer_list(text: str) -> List[str]:
    if text.lower().strip() == "all":
        return STIEFEL_OPTIMIZERS[:]
    opts = [x.strip() for x in text.split(",") if x.strip()]
    bad = [x for x in opts if x not in STIEFEL_OPTIMIZERS]
    if bad:
        raise ValueError(f"Unknown optimizer(s): {bad}. Valid: {STIEFEL_OPTIMIZERS}")
    return opts


def build_method_specs(methods: str, stiefel_optimizers: List[str]) -> List[Dict[str, str | None]]:
    methods = methods.strip().lower().replace("_", "-")
    specs: List[Dict[str, str | None]] = []

    def add_gcg():
        specs.append({"solver_key": "gcg", "optimizer": None, "tag": "gcg", "label": "GCG (Algorithm A)"})

    def add_stiefel(opts):
        for opt in opts:
            specs.append({
                "solver_key": "stiefel",
                "optimizer": opt,
                "tag": f"stiefel-{safe_tag(opt)}",
                "label": f"Stiefel / Method-II ({opt})",
            })

    def add_stiefel_gcg():
        specs.append({
            "solver_key": "stiefel_gcg",
            "optimizer": None,
            "tag": "stiefel-gcg",
            "label": "Stiefel-GCG hybrid",
        })

    if methods == "gcg":
        add_gcg()
    elif methods in ["stiefel", "method-ii"]:
        add_stiefel(stiefel_optimizers)
    elif methods in ["stiefel-gcg", "method-ii-gcg"]:
        add_stiefel_gcg()
    elif methods == "all3":
        add_gcg()
        add_stiefel(stiefel_optimizers[:1])
        add_stiefel_gcg()
    elif methods in ["stiefel-all-optimizers", "method-ii-all-optimizers"]:
        add_stiefel(STIEFEL_OPTIMIZERS)
    elif methods == "all":
        add_gcg()
        add_stiefel(STIEFEL_OPTIMIZERS if stiefel_optimizers == STIEFEL_OPTIMIZERS else stiefel_optimizers)
        add_stiefel_gcg()
    else:
        raise ValueError(f"Unknown --methods={methods}")

    seen = set()
    out = []
    for s in specs:
        key = (s["solver_key"], s["optimizer"])
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def call_solver(rho, N, spec, n_restarts, k_max_extra, max_iter):
    solver_key = spec["solver_key"]
    optimizer = spec.get("optimizer")

    kwargs_common = dict(
        rho=rho,
        n=N,
        n_restarts=n_restarts,
        k_max_extra=k_max_extra,
        verbose=False,
        measure="meyerwallach",
    )

    if solver_key == "stiefel":
        kwargs = dict(kwargs_common)
        kwargs["optimizer"] = optimizer
        if max_iter is not None:
            kwargs["max_iter"] = max_iter
        res = convex_roof_stiefel(**kwargs)
        return float(res["C_stiefel"]), res

    if solver_key == "gcg":
        kwargs = dict(kwargs_common)
        if max_iter is not None:
            kwargs["max_iter"] = max_iter
        res = convex_roof_gcg(**kwargs)
        return float(res["C_gcg"]), res

    if solver_key == "stiefel_gcg":
        kwargs = dict(kwargs_common)
        if max_iter is not None:
            kwargs["max_iter"] = max_iter
        res = convex_roof_stiefel_gcg(**kwargs)
        return float(res["C_stiefel_gcg"]), res

    raise ValueError(f"Unknown solver_key={solver_key}")


def run_one_point(args):
    """
    args = (N, b, T, spec, n_restarts, k_max_extra, max_iter)
    """
    N, b, T, spec, n_restarts, k_max_extra, max_iter = args
    t0 = time.time()

    H = build_hamiltonian(N, J=1.0, b=b)
    rho = thermal_state(H, T)

    eigvals = np.linalg.eigvalsh(rho)
    rank = int(np.sum(eigvals > 1e-10))

    if rank == 1:
        energies, vecs = np.linalg.eigh(rho)
        psi = vecs[:, -1]
        gamma = float(_meyerwallach_pure_vector(psi, N))
        used_method = "pure"
        raw = None
    else:
        gamma, raw = call_solver(rho, N, spec, n_restarts, k_max_extra, max_iter)
        used_method = spec["tag"]

    return {
        "N": int(N),
        "b": float(b),
        "T": float(T),
        "gamma": float(gamma),
        "rank": int(rank),
        "elapsed": time.time() - t0,
        "method_tag": str(spec["tag"]),
        "method_label": str(spec["label"]),
        "used_method": used_method,
    }


def compute_all(N_list, T_list, b_values, method_specs, n_restarts, k_max_extra, max_iter, workers):
    jobs = []
    for spec in method_specs:
        for N in N_list:
            for T in T_list:
                for b in b_values:
                    jobs.append((N, float(b), float(T), spec, n_restarts, k_max_extra, max_iter))

    print(f"  Total jobs: {len(jobs)}")
    print(f"  Methods: {[s['tag'] for s in method_specs]}")

    t_pool_start = time.time()
    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        future_to_job = {ex.submit(run_one_point, j): j for j in jobs}
        for fut in as_completed(future_to_job):
            j = future_to_job[fut]
            try:
                r = fut.result()
                results.append(r)
                completed += 1
                if completed % max(1, len(jobs) // 20) == 0 or completed == len(jobs):
                    pct = 100.0 * completed / len(jobs)
                    elapsed = time.time() - t_pool_start
                    eta = elapsed / completed * (len(jobs) - completed)
                    print(f"    [{completed:>5}/{len(jobs)} {pct:5.1f}%] "
                          f"{r['method_tag']:<20} N={r['N']} b={r['b']:.3f} "
                          f"T={r['T']:.1e} gamma={r['gamma']:.4f} rank={r['rank']} "
                          f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")
            except Exception as exc:
                print(f"    [FAIL] {j}: {exc!r}")

    print(f"\n  All jobs done in {time.time() - t_pool_start:.1f} s.\n")

    results_by_method: Dict[str, Dict[int, Dict[float, np.ndarray]]] = {}
    labels_by_method: Dict[str, str] = {}
    raw_results_by_method: Dict[str, list] = {}

    for spec in method_specs:
        tag = spec["tag"]
        labels_by_method[tag] = spec["label"]
        raw_results_by_method[tag] = [r for r in results if r["method_tag"] == tag]
        results_by_method[tag] = {}
        for N in N_list:
            results_by_method[tag][N] = {}
            for T in T_list:
                pts = [(r["b"], r["gamma"]) for r in raw_results_by_method[tag]
                       if r["N"] == N and abs(r["T"] - T) < 1e-15]
                pts.sort(key=lambda x: x[0])
                results_by_method[tag][N][float(T)] = np.array([p[1] for p in pts], dtype=float)

    expected = len(b_values)
    for tag in results_by_method:
        for N in N_list:
            for T in T_list:
                got = len(results_by_method[tag][N][float(T)])
                if got != expected:
                    print(f"  WARNING: method={tag} N={N} T={T} has {got} points, expected {expected}")

    return results_by_method, labels_by_method, raw_results_by_method


def save_data(data_path, results_by_method, labels_by_method, b_values, T_list, N_list, method_specs, raw_results_by_method):
    save_dict: Dict[str, Any] = {
        "b_values": np.asarray(b_values, dtype=float),
        "T_values": np.asarray(T_list, dtype=float),
        "N_values": np.asarray(N_list, dtype=int),
        "method_tags": np.asarray(list(results_by_method.keys()), dtype=object),
        "method_labels": np.asarray([labels_by_method[k] for k in results_by_method.keys()], dtype=object),
        "method_specs": np.asarray(method_specs, dtype=object),
        "raw_results_by_method": np.asarray([raw_results_by_method], dtype=object),
    }

    for tag, by_N in results_by_method.items():
        tag_clean = safe_tag(tag)
        for N in N_list:
            for T in T_list:
                key = f"gamma__{tag_clean}__N{N}_T{T:.0e}"
                save_dict[key] = by_N[N][float(T)]

    np.savez(data_path, **save_dict)
    print(f"  Data saved → {data_path}")


def load_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Saved data file not found: {data_path}")

    data = np.load(data_path, allow_pickle=True)
    b_values = data["b_values"]
    T_list = list(map(float, data["T_values"]))
    N_list = list(map(int, data["N_values"])) if "N_values" in data.files else [2, 3, 4, 5]
    method_tags = list(map(str, data["method_tags"]))
    method_labels_arr = list(map(str, data["method_labels"])) if "method_labels" in data.files else method_tags
    labels_by_method = dict(zip(method_tags, method_labels_arr))

    results_by_method: Dict[str, Dict[int, Dict[float, np.ndarray]]] = {}
    for tag in method_tags:
        tag_clean = safe_tag(tag)
        results_by_method[tag] = {}
        for N in N_list:
            results_by_method[tag][N] = {}
            for T in T_list:
                key = f"gamma__{tag_clean}__N{N}_T{T:.0e}"
                if key in data.files:
                    results_by_method[tag][N][float(T)] = data[key]
                else:
                    print(f"  WARNING: missing key {key}")
                    results_by_method[tag][N][float(T)] = np.array([])

    print(f"  Loaded methods: {method_tags}")
    print(f"  b/J grid: {b_values[0]:.3f} → {b_values[-1]:.3f} ({len(b_values)} points)")
    print(f"  T values: {T_list}")
    return results_by_method, labels_by_method, b_values, T_list, N_list


def plot_fig4_for_method(results_by_N_T, b_values, T_values, save_path, title_suffix=""):
    n_values = sorted(results_by_N_T.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    linestyles = ["--", "-.", (0, (3, 1, 1, 1, 1, 1)), ":"]
    markers = ["o", "s", "^", "D"]

    for idx, N in enumerate(n_values):
        ax = axes[idx]
        for j, T in enumerate(T_values):
            if T not in results_by_N_T[N]:
                continue
            gammas = np.asarray(results_by_N_T[N][T], dtype=float)
            if gammas.size == 0:
                continue
            ax.plot(b_values[:len(gammas)], gammas,
                    linestyle=linestyles[j % len(linestyles)],
                    marker=markers[j % len(markers)],
                    markersize=2.5,
                    linewidth=1.4,
                    label=f"T = {T:g} J/$k_B$")
        ax.set_xlabel(r"$b / J$")
        ax.set_ylabel(r"$\gamma$  (Meyer-Wallach)")
        ax.set_title(f"N = {N}")
        ax.set_xlim(0, max(b_values))
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=8)

    title = "Fig. 4 reproduction: Meyer-Wallach γ vs b/J"
    if title_suffix:
        title += f"  ({title_suffix})"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


def nearest_temperature(T_values, target):
    arr = np.asarray(T_values, dtype=float)
    idx = int(np.argmin(np.abs(arr - float(target))))
    return float(arr[idx])


def plot_comparison(results_by_method, labels_by_method, b_values, T_values, N_list, save_path, comparison_T):
    T_sel = nearest_temperature(T_values, comparison_T)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, N in enumerate(sorted(N_list)):
        ax = axes[idx]
        for tag, by_N in results_by_method.items():
            gammas = np.asarray(by_N[N].get(T_sel, []), dtype=float)
            if gammas.size == 0:
                continue
            ax.plot(b_values[:len(gammas)], gammas, linewidth=1.5, label=labels_by_method.get(tag, tag))
        ax.set_xlabel(r"$b / J$")
        ax.set_ylabel(r"$\gamma$")
        ax.set_title(f"N = {N}, T = {T_sel:g} J/$k_B$")
        ax.set_xlim(0, max(b_values))
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=7)

    fig.suptitle("Method comparison for Fig. 4 at one temperature", fontsize=12, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison plot saved → {save_path}")


def make_suffix(methods, optimizers, coarse):
    parts = [methods]
    opt_tag = "allopts" if optimizers == "all" else safe_tag(optimizers)
    parts.append(opt_tag)
    if coarse:
        parts.append("coarse")
    return "_" + "_".join(parts)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", type=str, default="all3",
                        choices=["gcg", "stiefel", "method-II", "stiefel-gcg",
                                 "stiefel_gcg", "method-II-gcg", "all3",
                                 "stiefel-all-optimizers", "all"],
                        help="Which solver set to run.")
    parser.add_argument("--stiefel-optimizers", type=str, default="L-BFGS-B",
                        help="Comma list or 'all'. Valid: L-BFGS-B,Powell,SLSQP,TNC,trust-constr.")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--input-npz", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default="fig4_all_methods")
    parser.add_argument("--n-restarts", type=int, default=10)
    parser.add_argument("--k-max-extra", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--coarse", action="store_true", help="15 b-points instead of 50.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf"])
    parser.add_argument("--comparison-T", type=float, default=1e-2)
    args = parser.parse_args()

    N_list = [2, 3, 4, 5]
    T_list = [1e-4, 1e-3, 1e-2, 5e-2]

    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = make_suffix(args.methods, args.stiefel_optimizers, args.coarse and not args.reload)
    data_path = Path(args.input_npz) if args.input_npz else out_dir / f"{args.output_prefix}{suffix}.npz"

    if args.reload:
        print("\nRELOAD MODE")
        print(f"Input file: {data_path}\n")
        results_by_method, labels_by_method, b_values, T_list_loaded, N_list_loaded = load_data(data_path)
    else:
        n_points = 15 if args.coarse else 50
        b_values = np.linspace(0.001, 1.0, n_points)
        stiefel_opts = parse_optimizer_list(args.stiefel_optimizers)
        method_specs = build_method_specs(args.methods, stiefel_opts)

        print("\nCOMPUTE MODE")
        print("Reproducing Fig. 4 of Rothlisberger, Lehmann, Loss (2009)")
        print(f"Methods: {[s['tag'] for s in method_specs]}")
        print(f"N values: {N_list}")
        print(f"T values: {T_list}")
        print(f"b/J grid: {b_values[0]:.3f} → {b_values[-1]:.3f} ({len(b_values)} points)")
        print(f"workers={args.workers}, restarts={args.n_restarts}, k_max_extra={args.k_max_extra}, max_iter={args.max_iter}")
        print(f"mode={'COARSE' if args.coarse else 'FULL'}\n")

        results_by_method, labels_by_method, raw_results_by_method = compute_all(
            N_list, T_list, b_values,
            method_specs=method_specs,
            n_restarts=args.n_restarts,
            k_max_extra=args.k_max_extra,
            max_iter=args.max_iter,
            workers=args.workers,
        )

        save_data(
            data_path,
            results_by_method,
            labels_by_method,
            b_values,
            T_list,
            N_list,
            method_specs,
            raw_results_by_method,
        )
        T_list_loaded = T_list
        N_list_loaded = N_list

    for tag, results_by_N_T in results_by_method.items():
        plot_path = out_dir / f"{args.output_prefix}_{safe_tag(tag)}{suffix}.{args.format}"
        plot_fig4_for_method(
            results_by_N_T,
            b_values,
            T_list_loaded,
            plot_path,
            title_suffix=labels_by_method.get(tag, tag),
        )

    if len(results_by_method) > 1:
        comp_path = out_dir / f"{args.output_prefix}_comparison_T{safe_tag(args.comparison_T)}{suffix}.{args.format}"
        plot_comparison(
            results_by_method,
            labels_by_method,
            b_values,
            T_list_loaded,
            N_list_loaded,
            comp_path,
            args.comparison_T,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()


def safe_tag(x: Any) -> str:
    s = str(x)
    for a, b in [(" ", ""), ("/", "-"), ("\\", "-"), (".", "p"), ("+", "")]:
        s = s.replace(a, b)
    return s


def operator_on_site(op_local, site, N):
    ops = [op_local if i == site else I2 for i in range(N)]
    full = ops[0]
    for op in ops[1:]:
        full = np.kron(full, op)
    return full


def build_hamiltonian(N, J=1.0, b=0.1):
    """Eq. (44) spin-ring Hamiltonian with PBC."""
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=complex)

    Sx_ops = [operator_on_site(Sx_single, i, N) for i in range(N)]
    Sy_ops = [operator_on_site(Sy_single, i, N) for i in range(N)]
    Sz_ops = [operator_on_site(Sz_single, i, N) for i in range(N)]

    for i in range(N):
        j = (i + 1) % N
        H += -J * (Sx_ops[i] @ Sx_ops[j]
                   + Sy_ops[i] @ Sy_ops[j]
                   + Sz_ops[i] @ Sz_ops[j])

    for i in range(N):
        alpha_i = 2 * np.pi * i / N
        H += b * (np.cos(alpha_i) * Sx_ops[i]
                  + np.sin(alpha_i) * Sy_ops[i])

    return H


def thermal_state(H, T, degeneracy_tol=1e-10):
    """
    Build thermal density matrix rho(T) = exp(-beta H) / Z.

    Uses energy shift E - E0 for numerical stability.
    For T <= 0, returns the ground-state projector. If the ground state
    is degenerate within degeneracy_tol, returns the equal mixture over
    the degenerate ground-state subspace.
    """
    energies, vecs = np.linalg.eigh(H)

    if T <= 0:
        E0 = energies[0]
        ground_idx = np.where(np.abs(energies - E0) < degeneracy_tol)[0]

        rho0 = np.zeros((H.shape[0], H.shape[0]), dtype=complex)
        for idx in ground_idx:
            psi = vecs[:, idx]
            rho0 += np.outer(psi, psi.conj())

        rho0 /= len(ground_idx)
        return rho0

    beta = 1.0 / T
    shifted = energies - energies[0]

    weights = np.exp(-beta * shifted)
    probs = weights / np.sum(weights)

    rho = vecs @ np.diag(probs) @ vecs.conj().T

    # Optional cleanup for numerical precision
    rho = 0.5 * (rho + rho.conj().T)
    rho /= np.trace(rho)

    return rho


def parse_optimizer_list(text: str) -> List[str]:
    if text.lower().strip() == "all":
        return STIEFEL_OPTIMIZERS[:]
    opts = [x.strip() for x in text.split(",") if x.strip()]
    bad = [x for x in opts if x not in STIEFEL_OPTIMIZERS]
    if bad:
        raise ValueError(f"Unknown optimizer(s): {bad}. Valid: {STIEFEL_OPTIMIZERS}")
    return opts


def build_method_specs(methods: str, stiefel_optimizers: List[str]) -> List[Dict[str, str | None]]:
    methods = methods.strip().lower().replace("_", "-")
    specs: List[Dict[str, str | None]] = []

    def add_gcg():
        specs.append({"solver_key": "gcg", "optimizer": None, "tag": "gcg", "label": "GCG (Algorithm A)"})

    def add_stiefel(opts):
        for opt in opts:
            specs.append({
                "solver_key": "stiefel",
                "optimizer": opt,
                "tag": f"stiefel-{safe_tag(opt)}",
                "label": f"Stiefel / Method-II ({opt})",
            })

    def add_stiefel_gcg():
        specs.append({
            "solver_key": "stiefel_gcg",
            "optimizer": None,
            "tag": "stiefel-gcg",
            "label": "Stiefel-GCG hybrid",
        })

    if methods == "gcg":
        add_gcg()
    elif methods in ["stiefel", "method-ii"]:
        add_stiefel(stiefel_optimizers)
    elif methods in ["stiefel-gcg", "method-ii-gcg"]:
        add_stiefel_gcg()
    elif methods == "all3":
        add_gcg()
        add_stiefel(stiefel_optimizers[:1])
        add_stiefel_gcg()
    elif methods in ["stiefel-all-optimizers", "method-ii-all-optimizers"]:
        add_stiefel(STIEFEL_OPTIMIZERS)
    elif methods == "all":
        add_gcg()
        add_stiefel(STIEFEL_OPTIMIZERS if stiefel_optimizers == STIEFEL_OPTIMIZERS else stiefel_optimizers)
        add_stiefel_gcg()
    else:
        raise ValueError(f"Unknown --methods={methods}")

    seen = set()
    out = []
    for s in specs:
        key = (s["solver_key"], s["optimizer"])
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def call_solver(rho, N, spec, n_restarts, k_max_extra, max_iter):
    solver_key = spec["solver_key"]
    optimizer = spec.get("optimizer")

    kwargs_common = dict(
        rho=rho,
        n=N,
        n_restarts=n_restarts,
        k_max_extra=k_max_extra,
        verbose=False,
        measure="meyerwallach",
    )

    if solver_key == "stiefel":
        kwargs = dict(kwargs_common)
        kwargs["optimizer"] = optimizer
        if max_iter is not None:
            kwargs["max_iter"] = max_iter
        res = convex_roof_stiefel(**kwargs)
        return float(res["C_stiefel"]), res

    if solver_key == "gcg":
        kwargs = dict(kwargs_common)
        if max_iter is not None:
            kwargs["max_iter"] = max_iter
        res = convex_roof_gcg(**kwargs)
        return float(res["C_gcg"]), res

    if solver_key == "stiefel_gcg":
        kwargs = dict(kwargs_common)
        if max_iter is not None:
            kwargs["max_iter"] = max_iter
        res = convex_roof_stiefel_gcg(**kwargs)
        return float(res["C_stiefel_gcg"]), res

    raise ValueError(f"Unknown solver_key={solver_key}")


def run_one_point(args):
    """
    args = (N, b, T, spec, n_restarts, k_max_extra, max_iter)
    """
    N, b, T, spec, n_restarts, k_max_extra, max_iter = args
    t0 = time.time()

    H = build_hamiltonian(N, J=1.0, b=b)
    rho = thermal_state(H, T)

    eigvals = np.linalg.eigvalsh(rho)
    rank = int(np.sum(eigvals > 1e-10))

    if rank == 1:
        energies, vecs = np.linalg.eigh(rho)
        psi = vecs[:, -1]
        gamma = float(_meyerwallach_pure_vector(psi, N))
        used_method = "pure"
        raw = None
    else:
        gamma, raw = call_solver(rho, N, spec, n_restarts, k_max_extra, max_iter)
        used_method = spec["tag"]

    return {
        "N": int(N),
        "b": float(b),
        "T": float(T),
        "gamma": float(gamma),
        "rank": int(rank),
        "elapsed": time.time() - t0,
        "method_tag": str(spec["tag"]),
        "method_label": str(spec["label"]),
        "used_method": used_method,
    }


def compute_all(N_list, T_list, b_values, method_specs, n_restarts, k_max_extra, max_iter, workers):
    jobs = []
    for spec in method_specs:
        for N in N_list:
            for T in T_list:
                for b in b_values:
                    jobs.append((N, float(b), float(T), spec, n_restarts, k_max_extra, max_iter))

    print(f"  Total jobs: {len(jobs)}")
    print(f"  Methods: {[s['tag'] for s in method_specs]}")

    t_pool_start = time.time()
    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        future_to_job = {ex.submit(run_one_point, j): j for j in jobs}
        for fut in as_completed(future_to_job):
            j = future_to_job[fut]
            try:
                r = fut.result()
                results.append(r)
                completed += 1
                if completed % max(1, len(jobs) // 20) == 0 or completed == len(jobs):
                    pct = 100.0 * completed / len(jobs)
                    elapsed = time.time() - t_pool_start
                    eta = elapsed / completed * (len(jobs) - completed)
                    print(f"    [{completed:>5}/{len(jobs)} {pct:5.1f}%] "
                          f"{r['method_tag']:<20} N={r['N']} b={r['b']:.3f} "
                          f"T={r['T']:.1e} gamma={r['gamma']:.4f} rank={r['rank']} "
                          f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")
            except Exception as exc:
                print(f"    [FAIL] {j}: {exc!r}")

    print(f"\n  All jobs done in {time.time() - t_pool_start:.1f} s.\n")

    results_by_method: Dict[str, Dict[int, Dict[float, np.ndarray]]] = {}
    labels_by_method: Dict[str, str] = {}
    raw_results_by_method: Dict[str, list] = {}

    for spec in method_specs:
        tag = spec["tag"]
        labels_by_method[tag] = spec["label"]
        raw_results_by_method[tag] = [r for r in results if r["method_tag"] == tag]
        results_by_method[tag] = {}
        for N in N_list:
            results_by_method[tag][N] = {}
            for T in T_list:
                pts = [(r["b"], r["gamma"]) for r in raw_results_by_method[tag]
                       if r["N"] == N and abs(r["T"] - T) < 1e-15]
                pts.sort(key=lambda x: x[0])
                results_by_method[tag][N][float(T)] = np.array([p[1] for p in pts], dtype=float)

    expected = len(b_values)
    for tag in results_by_method:
        for N in N_list:
            for T in T_list:
                got = len(results_by_method[tag][N][float(T)])
                if got != expected:
                    print(f"  WARNING: method={tag} N={N} T={T} has {got} points, expected {expected}")

    return results_by_method, labels_by_method, raw_results_by_method


def save_data(data_path, results_by_method, labels_by_method, b_values, T_list, N_list, method_specs, raw_results_by_method):
    save_dict: Dict[str, Any] = {
        "b_values": np.asarray(b_values, dtype=float),
        "T_values": np.asarray(T_list, dtype=float),
        "N_values": np.asarray(N_list, dtype=int),
        "method_tags": np.asarray(list(results_by_method.keys()), dtype=object),
        "method_labels": np.asarray([labels_by_method[k] for k in results_by_method.keys()], dtype=object),
        "method_specs": np.asarray(method_specs, dtype=object),
        "raw_results_by_method": np.asarray([raw_results_by_method], dtype=object),
    }

    for tag, by_N in results_by_method.items():
        tag_clean = safe_tag(tag)
        for N in N_list:
            for T in T_list:
                key = f"gamma__{tag_clean}__N{N}_T{T:.0e}"
                save_dict[key] = by_N[N][float(T)]

    np.savez(data_path, **save_dict)
    print(f"  Data saved → {data_path}")


def load_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Saved data file not found: {data_path}")

    data = np.load(data_path, allow_pickle=True)
    b_values = data["b_values"]
    T_list = list(map(float, data["T_values"]))
    N_list = list(map(int, data["N_values"])) if "N_values" in data.files else [2, 3, 4, 5]
    method_tags = list(map(str, data["method_tags"]))
    method_labels_arr = list(map(str, data["method_labels"])) if "method_labels" in data.files else method_tags
    labels_by_method = dict(zip(method_tags, method_labels_arr))

    results_by_method: Dict[str, Dict[int, Dict[float, np.ndarray]]] = {}
    for tag in method_tags:
        tag_clean = safe_tag(tag)
        results_by_method[tag] = {}
        for N in N_list:
            results_by_method[tag][N] = {}
            for T in T_list:
                key = f"gamma__{tag_clean}__N{N}_T{T:.0e}"
                if key in data.files:
                    results_by_method[tag][N][float(T)] = data[key]
                else:
                    print(f"  WARNING: missing key {key}")
                    results_by_method[tag][N][float(T)] = np.array([])

    print(f"  Loaded methods: {method_tags}")
    print(f"  b/J grid: {b_values[0]:.3f} → {b_values[-1]:.3f} ({len(b_values)} points)")
    print(f"  T values: {T_list}")
    return results_by_method, labels_by_method, b_values, T_list, N_list


def plot_fig4_for_method(results_by_N_T, b_values, T_values, save_path, title_suffix=""):
    n_values = sorted(results_by_N_T.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    linestyles = ["--", "-.", (0, (3, 1, 1, 1, 1, 1)), ":"]
    markers = ["o", "s", "^", "D"]

    for idx, N in enumerate(n_values):
        ax = axes[idx]
        for j, T in enumerate(T_values):
            if T not in results_by_N_T[N]:
                continue
            gammas = np.asarray(results_by_N_T[N][T], dtype=float)
            if gammas.size == 0:
                continue
            ax.plot(b_values[:len(gammas)], gammas,
                    linestyle=linestyles[j % len(linestyles)],
                    marker=markers[j % len(markers)],
                    markersize=2.5,
                    linewidth=1.4,
                    label=f"T = {T:g} J/$k_B$")
        ax.set_xlabel(r"$b / J$")
        ax.set_ylabel(r"$\gamma$  (Meyer-Wallach)")
        ax.set_title(f"N = {N}")
        ax.set_xlim(0, max(b_values))
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=8)

    title = "Fig. 4 reproduction: Meyer-Wallach γ vs b/J"
    if title_suffix:
        title += f"  ({title_suffix})"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


def nearest_temperature(T_values, target):
    arr = np.asarray(T_values, dtype=float)
    idx = int(np.argmin(np.abs(arr - float(target))))
    return float(arr[idx])


def plot_comparison(results_by_method, labels_by_method, b_values, T_values, N_list, save_path, comparison_T):
    T_sel = nearest_temperature(T_values, comparison_T)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, N in enumerate(sorted(N_list)):
        ax = axes[idx]
        for tag, by_N in results_by_method.items():
            gammas = np.asarray(by_N[N].get(T_sel, []), dtype=float)
            if gammas.size == 0:
                continue
            ax.plot(b_values[:len(gammas)], gammas, linewidth=1.5, label=labels_by_method.get(tag, tag))
        ax.set_xlabel(r"$b / J$")
        ax.set_ylabel(r"$\gamma$")
        ax.set_title(f"N = {N}, T = {T_sel:g} J/$k_B$")
        ax.set_xlim(0, max(b_values))
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=7)

    fig.suptitle("Method comparison for Fig. 4 at one temperature", fontsize=12, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison plot saved → {save_path}")


def make_suffix(methods, optimizers, coarse):
    parts = [methods]
    opt_tag = "allopts" if optimizers == "all" else safe_tag(optimizers)
    parts.append(opt_tag)
    if coarse:
        parts.append("coarse")
    return "_" + "_".join(parts)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", type=str, default="all3",
                        choices=["gcg", "stiefel", "method-II", "stiefel-gcg",
                                 "stiefel_gcg", "method-II-gcg", "all3",
                                 "stiefel-all-optimizers", "all"],
                        help="Which solver set to run.")
    parser.add_argument("--stiefel-optimizers", type=str, default="L-BFGS-B",
                        help="Comma list or 'all'. Valid: L-BFGS-B,Powell,SLSQP,TNC,trust-constr.")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--input-npz", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default="fig4_all_methods")
    parser.add_argument("--n-restarts", type=int, default=10)
    parser.add_argument("--k-max-extra", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--coarse", action="store_true", help="15 b-points instead of 50.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf"])
    parser.add_argument("--comparison-T", type=float, default=1e-2)
    args = parser.parse_args()

    N_list = [2, 3, 4, 5]
    T_list = [1e-4, 1e-3, 1e-2, 5e-2]

    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = make_suffix(args.methods, args.stiefel_optimizers, args.coarse and not args.reload)
    data_path = Path(args.input_npz) if args.input_npz else out_dir / f"{args.output_prefix}{suffix}.npz"

    if args.reload:
        print("\nRELOAD MODE")
        print(f"Input file: {data_path}\n")
        results_by_method, labels_by_method, b_values, T_list_loaded, N_list_loaded = load_data(data_path)
    else:
        n_points = 15 if args.coarse else 50
        b_values = np.linspace(0.001, 1.0, n_points)
        stiefel_opts = parse_optimizer_list(args.stiefel_optimizers)
        method_specs = build_method_specs(args.methods, stiefel_opts)

        print("\nCOMPUTE MODE")
        print("Reproducing Fig. 4 of Rothlisberger, Lehmann, Loss (2009)")
        print(f"Methods: {[s['tag'] for s in method_specs]}")
        print(f"N values: {N_list}")
        print(f"T values: {T_list}")
        print(f"b/J grid: {b_values[0]:.3f} → {b_values[-1]:.3f} ({len(b_values)} points)")
        print(f"workers={args.workers}, restarts={args.n_restarts}, k_max_extra={args.k_max_extra}, max_iter={args.max_iter}")
        print(f"mode={'COARSE' if args.coarse else 'FULL'}\n")

        results_by_method, labels_by_method, raw_results_by_method = compute_all(
            N_list, T_list, b_values,
            method_specs=method_specs,
            n_restarts=args.n_restarts,
            k_max_extra=args.k_max_extra,
            max_iter=args.max_iter,
            workers=args.workers,
        )

        save_data(
            data_path,
            results_by_method,
            labels_by_method,
            b_values,
            T_list,
            N_list,
            method_specs,
            raw_results_by_method,
        )
        T_list_loaded = T_list
        N_list_loaded = N_list

    for tag, results_by_N_T in results_by_method.items():
        plot_path = out_dir / f"{args.output_prefix}_{safe_tag(tag)}{suffix}.{args.format}"
        plot_fig4_for_method(
            results_by_N_T,
            b_values,
            T_list_loaded,
            plot_path,
            title_suffix=labels_by_method.get(tag, tag),
        )

    if len(results_by_method) > 1:
        comp_path = out_dir / f"{args.output_prefix}_comparison_T{safe_tag(args.comparison_T)}{suffix}.{args.format}"
        plot_comparison(
            results_by_method,
            labels_by_method,
            b_values,
            T_list_loaded,
            N_list_loaded,
            comp_path,
            args.comparison_T,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
