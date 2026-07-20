"""
fig5.py
-------
Reproduce Fig. 5 of:
  Rothlisberger, Lehmann, Loss
  "Numerical evaluation of convex-roof entanglement measures"
  PRA 80, 042301 (2009)

Plots 1 - γ_max(T) vs T on log-log axes for N = 2, 3, 4, 5
(and optionally 7, 10).

γ_max(T) = max over b of γ( ρ(T, b) )

Two methods are supported:
  • method = "stiefel" : Algorithm B (Euler-Hurwitz / Stiefel manifold)
                        with selectable optimizer.
  • method = "gcg"     : Algorithm A (Generalised Conjugate Gradient).

Two modes:
  • COMPUTE: run sweeps and save data to .npz.
  • RELOAD : reload data from .npz and regenerate the plot.

Run:
    # Method II (Stiefel), L-BFGS-B
    python fig5.py

    # Method II with a different optimizer
    python fig5.py --optimizer Powell

    # Method I (GCG)
    python fig5.py --method gcg

    # Reload existing Method II data, save as PDF
    python fig5.py --reload --input-npz outputs/fig5_data.npz --format pdf

    # Skip N=7, N=10 (faster)
    python fig5.py --skip-large
"""

from __future__ import annotations

import os, sys, time, argparse
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hubbard_gme.convex_roof import (
    _meyerwallach_pure_vector,
    convex_roof_stiefel,
    convex_roof_gcg,
)

from fig4_full import build_hamiltonian, thermal_state


# ──────────────────────────────────────────────────────────────────────────
# Temperature grid — extended to 10⁻⁶ to match the paper's x-axis range
# ──────────────────────────────────────────────────────────────────────────

T_GRID = np.array([
    1e-6, 3e-6,
    1e-5, 3e-5,
    1e-4, 3e-4,
    1e-3, 3e-3,
    1e-2, 3e-2,
    1e-1,
])

B_GRID_DEFAULT = np.linspace(0.005, 0.8, 30)


# ──────────────────────────────────────────────────────────────────────────
# Worker — method-aware
# ──────────────────────────────────────────────────────────────────────────

def run_one_point(args):
    """
    Compute γ(ρ(T, b)) for one (N, b, T) point.

    args : (N, b, T, n_restarts, k_max_extra, method, optimizer)
    """
    N, b, T, n_restarts, k_max_extra, method, optimizer = args

    t0 = time.time()
    H = build_hamiltonian(N, J=1.0, b=b)
    rho = thermal_state(H, T)

    eigvals = np.linalg.eigvalsh(rho)
    # Relaxed threshold (was 1e-10) so the pure-state shortcut triggers
    # at very low T, where ρ is numerically rank-1 but the second
    # eigenvalue may be ~1e-8 due to floating-point noise.
    rank = int(np.sum(eigvals > 1e-8))

    if rank == 1:
        _, vecs = np.linalg.eigh(rho)
        psi = vecs[:, -1]
        gamma = _meyerwallach_pure_vector(psi, N)
        used_method = "pure"
    elif method == "stiefel":
        res = convex_roof_stiefel(
            rho, N,
            n_restarts=n_restarts,
            k_max_extra=k_max_extra,
            optimizer=optimizer,
            verbose=False,
            measure="meyerwallach",
        )
        gamma = res["C_stiefel"]
        used_method = f"stiefel/{optimizer}"
    elif method == "gcg":
        res = convex_roof_gcg(
            rho, N,
            n_restarts=n_restarts,
            k_max_extra=k_max_extra,
            verbose=False,
            measure="meyerwallach",
        )
        gamma = res["C_gcg"]
        used_method = "gcg"
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "N": N, "b": b, "T": T,
        "gamma": gamma, "rank": rank,
        "elapsed": time.time() - t0,
        "method": used_method,
    }


# ──────────────────────────────────────────────────────────────────────────
# γ_max(T) at one (N, T) via parallel b-sweep
# ──────────────────────────────────────────────────────────────────────────

def gamma_max_for_NT_parallel(N, T, b_grid, n_restarts, k_max_extra,
                              method, optimizer, workers):
    jobs = [(N, b, T, n_restarts, k_max_extra, method, optimizer)
            for b in b_grid]

    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_one_point, j) for j in jobs]
        for f in as_completed(futures):
            results.append(f.result())

    results.sort(key=lambda r: r["b"])
    gammas = np.array([r["gamma"] for r in results])
    b_arr  = np.array([r["b"]     for r in results])
    i_max  = int(np.argmax(gammas))

    return {
        "gamma_max": float(gammas[i_max]),
        "b_best":    float(b_arr[i_max]),
        "b_grid":    b_arr,
        "gammas":    gammas,
    }


# ──────────────────────────────────────────────────────────────────────────
# Reuse Fig. 4 data when method matches
# ──────────────────────────────────────────────────────────────────────────

def extract_gamma_max_from_fig4(npz_path, method, optimizer):
    """
    Return None if file doesn't exist OR if its method tag doesn't match.
    Returns {N: {T: gamma_max}} otherwise.
    """
    if not os.path.exists(npz_path):
        return None, "file not found"

    data = np.load(npz_path, allow_pickle=True)

    # Verify method tag if present
    file_method = data["method"][0] if "method" in data.files else None
    file_opt    = data["optimizer"][0] if "optimizer" in data.files else None

    if file_method is not None:
        if file_method != method:
            return None, f"file is method={file_method}, want method={method}"
        if method == "stiefel" and file_opt is not None and file_opt != optimizer:
            return None, (f"file is optimizer={file_opt}, "
                          f"want optimizer={optimizer}")

    T_values = data["T_values"]
    result = {}
    for N in [2, 3, 4, 5]:
        result[N] = {}
        for T in T_values:
            key = f"gamma_N{N}_T{T:.0e}"
            if key in data.files:
                result[N][float(T)] = float(np.max(data[key]))
    return result, "ok"


# ──────────────────────────────────────────────────────────────────────────
# b_FWHM at T=0  (depends only on N, method-independent)
# ──────────────────────────────────────────────────────────────────────────

def compute_b_fwhm_T0(N, b_grid=None):
    if b_grid is None:
        b_grid = np.linspace(0.001, 1.5, 200)
    gammas = np.zeros(len(b_grid))
    for i, b in enumerate(b_grid):
        H = build_hamiltonian(N, J=1.0, b=b)
        _, vecs = np.linalg.eigh(H)
        gammas[i] = _meyerwallach_pure_vector(vecs[:, 0], N)
    below = np.where(gammas < 0.5)[0]
    if len(below) == 0:
        return float(b_grid[-1])
    i = below[0]
    if i == 0:
        return float(b_grid[0])
    b1, b2 = b_grid[i-1], b_grid[i]
    g1, g2 = gammas[i-1], gammas[i]
    if g1 == g2:
        return float(b1)
    return float(b1 + (b2 - b1) * (g1 - 0.5) / (g1 - g2))


# ──────────────────────────────────────────────────────────────────────────
# Save / Load
# ──────────────────────────────────────────────────────────────────────────

def save_data(data_path, results_by_N, b_fwhm_by_N, N_all,
              method=None, optimizer=None):
    save_dict = {}
    for N in N_all:
        Ts = np.array(sorted(results_by_N[N].keys()))
        gmax = np.array([results_by_N[N][T] for T in Ts])
        save_dict[f"T_N{N}"]         = Ts
        save_dict[f"gamma_max_N{N}"] = gmax
    save_dict["b_fwhm_N"]    = np.array([b_fwhm_by_N[N] for N in N_all])
    save_dict["N_for_fwhm"]  = np.array(N_all)
    if method is not None:
        save_dict["method"] = np.array([method], dtype=object)
    if optimizer is not None:
        save_dict["optimizer"] = np.array([optimizer], dtype=object)
    np.savez(data_path, **save_dict)
    print(f"  Data saved → {data_path}")


def load_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Saved data file not found: {data_path}")

    data = np.load(data_path, allow_pickle=True)

    method_str = ""
    if "method" in data.files:
        method_str = str(data["method"][0])
        if "optimizer" in data.files:
            method_str += f"/{data['optimizer'][0]}"

    N_for_fwhm = list(data["N_for_fwhm"])
    b_fwhm_arr = data["b_fwhm_N"]
    b_fwhm_by_N = {int(N): float(b) for N, b in zip(N_for_fwhm, b_fwhm_arr)}

    results_by_N = {}
    for N in N_for_fwhm:
        N = int(N)
        Ts   = data[f"T_N{N}"]
        gmax = data[f"gamma_max_N{N}"]
        results_by_N[N] = {float(T): float(g) for T, g in zip(Ts, gmax)}

    print(f"  Loaded data for N = {sorted(results_by_N.keys())}")
    if method_str:
        print(f"  Method tag in file: {method_str}")
    return results_by_N, b_fwhm_by_N, method_str


# ──────────────────────────────────────────────────────────────────────────
# Plot — paper style
# ──────────────────────────────────────────────────────────────────────────

def plot_fig5(results_by_N, b_fwhm_by_N, save_path, title_suffix=""):
    """
    Paper Fig. 5 style:
      - Lines only, no markers.
      - Dashed / dash-dot / dash-dot-dot / dotted dash patterns.
      - x-axis: 10⁻⁶ to 10⁻¹  (paper range).
      - y-axis: 10⁻⁶ to 10⁰  (paper range).
      - Inset: b_FWHM vs N, N = 2..18, b = 0..1 (paper range).
      - No grid, ticks inward on all four sides.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    N_list = sorted(results_by_N.keys())

    # Line styles meant to mimic the paper:
    # small N at the bottom (blue), large N at the top (red).
    styles = [
        {"linestyle": (0, (5, 3)),                "color": "#1f3a8a"},  # dashed         - blue
        {"linestyle": (0, (5, 2, 1, 2)),          "color": "#1f3a8a"},  # dash-dot       - blue
        {"linestyle": (0, (5, 2, 1, 2, 1, 2)),    "color": "#6b21a8"},  # dash-dot-dot   - purple
        {"linestyle": (0, (1, 2)),                "color": "#7c2d12"},  # dotted         - dark red
        {"linestyle": (0, (5, 3)),                "color": "#b91c1c"},  # dashed         - red
        {"linestyle": (0, (5, 2, 1, 2)),          "color": "#b91c1c"},  # dash-dot       - red
    ]

    for i, N in enumerate(N_list):
        d = results_by_N[N]
        Ts = sorted(d.keys())
        one_minus_gamma = np.array([1.0 - d[T] for T in Ts])
        Ts = np.array(Ts)

        s = styles[i % len(styles)]
        ax.loglog(Ts, one_minus_gamma,
                  linestyle=s["linestyle"],
                  color=s["color"],
                  linewidth=1.5,
                  label=f"N = {N}")

    ax.set_xlabel(r"$T$", fontsize=13)
    ax.set_ylabel(r"$1 - \gamma_{\max}$", fontsize=13)

    # Paper axis ranges
    ax.set_xlim(1e-6, 1e-1)
    ax.set_ylim(1e-6, 1e0)

    ax.grid(False)
    ax.tick_params(direction="in", which="both", top=True, right=True,
                   labelsize=11)

    if title_suffix:
        ax.set_title(title_suffix, fontsize=11)

    ax.legend(loc="upper left", fontsize=10, frameon=False)

    # Inset: b_FWHM vs N — paper-matching range
    if b_fwhm_by_N and len(b_fwhm_by_N) >= 2:
        axin = ax.inset_axes([0.55, 0.12, 0.4, 0.35])
        Ns_inset = sorted(b_fwhm_by_N.keys())
        bs_inset = [b_fwhm_by_N[N] for N in Ns_inset]
        axin.plot(Ns_inset, bs_inset, '-', color='#1f3a8a', linewidth=1.5)
        axin.set_xlabel("N", fontsize=10)
        axin.set_ylabel(r"$b_{\mathrm{FWHM}}$", fontsize=10)
        axin.set_xlim(2, 18)
        axin.set_ylim(0, 1.05)
        axin.tick_params(direction="in", which="both", labelsize=9,
                         top=True, right=True)
        axin.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────
# Compute
# ──────────────────────────────────────────────────────────────────────────

def compute_all(N_all, b_grid, n_restarts, k_max_extra, workers,
                method, optimizer, fig4_data=None):
    results_by_N = {}

    for N in N_all:
        print(f"  ─── N = {N} ───")
        results_by_N[N] = {}

        T_to_compute = list(T_GRID)
        if fig4_data is not None and N in fig4_data:
            for T_reuse, gmax_reuse in fig4_data[N].items():
                close = [T for T in T_to_compute
                         if abs(T - T_reuse) / T_reuse < 0.01]
                for T_close in close:
                    results_by_N[N][T_close] = gmax_reuse
                    T_to_compute.remove(T_close)
                    print(f"    T = {T_reuse:.0e}  γ_max = {gmax_reuse:.5f}  "
                          f"(reused from fig4)")

        for T in T_to_compute:
            t0 = time.time()
            res = gamma_max_for_NT_parallel(
                N, T, b_grid,
                n_restarts=n_restarts,
                k_max_extra=k_max_extra,
                method=method,
                optimizer=optimizer,
                workers=workers,
            )
            results_by_N[N][T] = res["gamma_max"]
            print(f"    T = {T:.0e}  γ_max = {res['gamma_max']:.5f}  "
                  f"at b/J = {res['b_best']:.3f}  ({time.time()-t0:.1f}s)")
        print()

    print(f"  Computing b_FWHM at T=0 for inset …")
    # Extend inset N-range to match the paper (N = 2..18) when possible
    inset_N_range = sorted(set(N_all + [2, 3, 4, 5, 6, 7, 8, 9, 10,
                                       12, 14, 16, 18]))
    b_fwhm_by_N = {}
    for N in inset_N_range:
        if N > 12:
            # Skip very large N to keep this fast; only fill if requested
            continue
        b_fwhm_by_N[N] = compute_b_fwhm_T0(N)
        print(f"    N = {N}:  b_FWHM = {b_fwhm_by_N[N]:.4f}")
    print()

    return results_by_N, b_fwhm_by_N


# ──────────────────────────────────────────────────────────────────────────
# Filename helpers
# ──────────────────────────────────────────────────────────────────────────

def make_suffix(method, optimizer):
    """Per-method filename suffix so different runs don't overwrite."""
    parts = [method]
    if method == "stiefel":
        opt_clean = optimizer.lower().replace("-", "").replace("_", "")
        parts.append(opt_clean)
    return "_" + "_".join(parts)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Reproduce Fig. 5 of Rothlisberger, Lehmann, Loss (2009).",
    )
    parser.add_argument("--method", type=str, default="stiefel",
                        choices=["stiefel", "gcg"],
                        help="Convex-roof solver (default: stiefel).")
    parser.add_argument("--optimizer", type=str, default="L-BFGS-B",
                        choices=["L-BFGS-B", "Powell", "SLSQP", "TNC",
                                 "trust-constr"],
                        help="Optimizer for Stiefel method (default: L-BFGS-B). "
                             "Ignored for --method gcg.")
    parser.add_argument("--reload", action="store_true",
                        help="Reload existing data, no compute.")
    parser.add_argument("--rerun-all", action="store_true",
                        help="Re-run all N from scratch (ignore saved fig4 data).")
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip N=7 and N=10.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--n-restarts", type=int, default=10)
    parser.add_argument("--k-max-extra", type=int, default=2)
    parser.add_argument("--n-b-points", type=int, default=30,
                        help="b-grid points per (N,T) sweep.")
    parser.add_argument("--format", type=str, default="png",
                        choices=["png", "pdf"],
                        help="Output plot format.")
    parser.add_argument("--input-npz", type=str, default=None,
                        help="Path to saved fig5 .npz (for --reload).")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output plot path.")
    parser.add_argument("--fig4-npz", type=str, default=None,
                        help="Path to fig4 .npz to reuse "
                             "(default: outputs/fig4_full_<method>_<opt>.npz).")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    suffix = make_suffix(args.method, args.optimizer)

    data_path = args.input_npz or os.path.join(out_dir, f"fig5_data{suffix}.npz")
    plot_path = args.output or os.path.join(
        out_dir, f"fig5_reproduction{suffix}.{args.format}")

    if args.method == "stiefel":
        title_suffix = f"Method II (Stiefel, {args.optimizer})"
    else:
        title_suffix = "Method I (GCG)"

    # ── RELOAD ──
    if args.reload:
        print(f"\n  RELOAD MODE")
        print(f"  Input file: {data_path}\n")
        results_by_N, b_fwhm_by_N, file_method = load_data(data_path)
        if file_method:
            title_suffix = f"loaded: {file_method}"
        plot_fig5(results_by_N, b_fwhm_by_N, plot_path,
                  title_suffix=title_suffix)
        print()
        return

    # ── COMPUTE ──
    N_list_main  = [2, 3, 4, 5]
    N_list_large = [] if args.skip_large else [7, 10]
    N_all = N_list_main + N_list_large

    b_grid = np.linspace(0.005, 0.8, args.n_b_points)

    # Pick the matching fig4 file by default
    if args.fig4_npz is not None:
        fig4_npz = args.fig4_npz
    else:
        fig4_suffix = make_suffix(args.method, args.optimizer)
        fig4_npz = os.path.join(out_dir, f"fig4_full{fig4_suffix}.npz")

    print(f"\n  COMPUTE MODE")
    print(f"  Method   : {args.method}"
          f"{f'  (optimizer: {args.optimizer})' if args.method == 'stiefel' else ''}")
    print(f"  T grid   : {T_GRID}")
    print(f"  N values : {N_all}")
    print(f"  b grid   : {b_grid[0]:.3f} → {b_grid[-1]:.3f}  "
          f"({len(b_grid)} points)")
    print(f"  Workers  : {args.workers}")
    print(f"  Output   : {args.format}\n")

    fig4_data = None
    if not args.rerun_all:
        fig4_data, status = extract_gamma_max_from_fig4(
            fig4_npz, args.method, args.optimizer)
        if fig4_data is not None:
            print(f"  Reusing γ_max from {fig4_npz} ({status})\n")
        else:
            print(f"  No matching fig4 data ({status}); will compute fresh\n")

    results_by_N, b_fwhm_by_N = compute_all(
        N_all, b_grid,
        n_restarts=args.n_restarts,
        k_max_extra=args.k_max_extra,
        workers=args.workers,
        method=args.method,
        optimizer=args.optimizer,
        fig4_data=fig4_data,
    )

    save_data(data_path, results_by_N, b_fwhm_by_N, N_all,
              method=args.method,
              optimizer=args.optimizer if args.method == "stiefel" else None)
    plot_fig5(results_by_N, b_fwhm_by_N, plot_path,
              title_suffix=title_suffix)
    print()


if __name__ == "__main__":
    main()