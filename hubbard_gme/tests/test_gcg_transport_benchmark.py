"""
Reproducible before/after benchmarks for the GCG parallel-transport fix
(audit issue #5).  Companion to ``tests/test_gcg_transport.py``, which holds
the fast geometric assertions.

Two benchmarks are provided:

* ``bench_synthetic``  — adversarial states (separable mixtures with true
  C_GME = 0, plus an entangled GHZ+noise control).
* ``bench_real``       — the project's OWN cached L=6 / Ne=6 ground-state
  RDMs from ``results/rdm_pkl_results/rdm_cache``, i.e. production data.

Both run the solver twice: once from the installed package and once from a
temporary copy in which ONLY the transport line is patched, so the comparison
isolates that single change.

These are slow (minutes to hours) and are therefore NOT collected as ordinary
pytest tests.  Run them explicitly:

    python tests/test_gcg_transport_benchmark.py --which real
    python tests/test_gcg_transport_benchmark.py --which synthetic
    python tests/test_gcg_transport_benchmark.py --which both --restarts 15

Recorded outcome (2026-08, n_restarts=10, k_max_extra sweep 0/2/4):

    REAL DATA (L=6 Ne=6 U=4 t=1 ground state)
      n=3, k=8 : current 2.249928e-02  fixed 2.249928e-02  delta -1.2e-14 (tie)
      -> on production RDMs the corrected transport reproduces the existing
         values to ~1e-14; no published number changes.

    SYNTHETIC (n_restarts=15)
      classical mix   (separable, true 0) 1.78e-08 -> 1.83e-09
      product mixed   (separable, true 0) 1.11e-03 -> 1.05e-03
      random full-rank mixed  (k=12)      1.31e-03 -> 9.06e-05
      GHZ + white noise (entangled)       0.650000 -> 0.650000 (identical)
      -> gains appear only in near-zero / near-separable cases; entangled,
         well-conditioned states are unchanged.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import os
import pickle
import shutil
import sys
import tempfile
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
CACHE = os.path.join(REPO, "results", "rdm_pkl_results", "rdm_cache")

CURRENT_LINE = "        T_transp = half_step @ G @ half_step.conj().T"
FIXED_LINE = "        T_transp = half_step.conj().T @ G @ half_step"


# ── package copies ───────────────────────────────────────────────────────────

def make_patched_package(tmpdir: str) -> str:
    """Copy the package and apply ONLY the transport fix. Returns its root."""
    root = os.path.join(tmpdir, "pkg_fixed")
    os.makedirs(root, exist_ok=True)
    shutil.copytree(os.path.join(SRC, "hubbard_gme"),
                    os.path.join(root, "hubbard_gme"), dirs_exist_ok=True)
    path = os.path.join(root, "hubbard_gme", "convex_roof.py")
    text = open(path).read()
    if text.count(CURRENT_LINE) == 1:
        text = text.replace(CURRENT_LINE, FIXED_LINE)
    elif text.count(FIXED_LINE) == 1:
        pass                      # already fixed upstream; copy is the same
    else:
        raise RuntimeError(
            "could not locate the transport line to patch — update "
            "CURRENT_LINE/FIXED_LINE in this file")
    open(path, "w").write(text)
    return root


def load_solver(pkg_root: str):
    """(Re)import hubbard_gme.convex_roof from a specific package root."""
    for mod in list(sys.modules):
        if mod.startswith("hubbard_gme"):
            del sys.modules[mod]
    sys.path.insert(0, pkg_root)
    import hubbard_gme.convex_roof as cr
    importlib.reload(cr)
    sys.path.pop(0)
    return cr


# ── case builders ────────────────────────────────────────────────────────────

def synthetic_cases():
    n, dim = 3, 8
    cases = []

    rng = np.random.default_rng(0)
    w = rng.dirichlet(np.ones(3))
    rho = np.zeros((dim, dim), complex)
    for wi, idx in zip(w, [0, 3, 6]):          # |000>, |011>, |110>
        rho[idx, idx] = wi
    cases.append(("classical mix (sep, C=0)", rho, n))

    sig = np.array([[0.7, 0.1 + 0.05j], [0.1 - 0.05j, 0.3]], complex)
    cases.append(("product mixed (sep, C=0)", np.kron(np.kron(sig, sig), sig), n))

    ghz = np.zeros(dim, complex)
    ghz[0] = ghz[-1] = 1 / np.sqrt(2)
    cases.append(("GHZ + noise (entangled)",
                  0.8 * np.outer(ghz, ghz.conj()) + 0.2 * np.eye(dim) / dim, n))

    r = np.random.default_rng(5)
    A = r.normal(size=(dim, dim)) + 1j * r.normal(size=(dim, dim))
    m = A @ A.conj().T
    cases.append(("random full-rank mixed", m / np.trace(m).real, n))
    return cases


def _extract_rho(payload):
    for key in ("rho", "rho_gs", "rdm"):
        if isinstance(payload, dict) and key in payload:
            return np.asarray(payload[key])
    if isinstance(payload, dict):
        for value in payload.values():
            if (isinstance(value, np.ndarray) and value.ndim == 2
                    and value.shape[0] == value.shape[1]):
                return value
    return None


def real_cases(n_values=(3, 4, 5, 6)):
    """Production L=6 / Ne=6 ground-state RDMs from the project's rdm_cache."""
    cases = []
    for n in n_values:
        pattern = os.path.join(
            CACHE, f"rdm_v1_L6_Ne6_U4_t1_mode-uniform_state-gs_n{n}_*.pkl")
        hits = sorted(glob.glob(pattern))
        if not hits:
            continue
        rho = _extract_rho(pickle.load(open(hits[0], "rb")))
        if rho is None:
            continue
        trace = np.trace(rho).real
        if abs(trace) > 1e-14:
            rho = rho / trace
        cases.append((f"L6 Ne6 GS n={n}", rho, n))
    return cases


# ── benchmark driver ─────────────────────────────────────────────────────────

def run(cases, fixed_root, restarts=10, kmx_values=(0, 2, 4)):
    print(f"\n{'case':<26} {'kmx':>4} {'CURRENT C_gcg':>16} {'k':>3} {'t(s)':>6} | "
          f"{'FIXED C_gcg':>14} {'k':>3} {'t(s)':>6} | {'delta':>12} {'verdict':>8}")
    print("-" * 108)
    for label, rho, n in cases:
        purity = float(np.real(np.trace(rho @ rho)))
        for kmx in kmx_values:
            row = {}
            for tag, root in (("cur", SRC), ("fix", fixed_root)):
                cr = load_solver(root)
                t0 = time.time()
                res = cr.convex_roof_gcg(rho, n=n, n_restarts=restarts,
                                         k_max_extra=kmx, verbose=False)
                row[tag] = (float(res["C_gcg"]), int(res.get("k_converged", -1)),
                            time.time() - t0)
            (c0, k0, e0), (c1, k1, e1) = row["cur"], row["fix"]
            delta = c1 - c0
            verdict = ("FIXED" if delta < -1e-9
                       else "current" if delta > 1e-9 else "tie")
            print(f"{label:<26} {kmx:>4} {c0:>16.6e} {k0:>3} {e0:>6.1f} | "
                  f"{c1:>14.6e} {k1:>3} {e1:>6.1f} | {delta:>+12.3e} {verdict:>8}",
                  flush=True)
            if purity > 1 - 1e-9:
                print(f"{'':<26} {'':>4}   (pure state: solver uses the exact "
                      f"formula, transport is not exercised)")
                break                      # k sweep is meaningless for a pure rho
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=["real", "synthetic", "both"],
                    default="real")
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--kmx", type=int, nargs="+", default=[0, 2, 4])
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        fixed_root = make_patched_package(tmp)
        print("=" * 108)
        print("GCG TRANSPORT BENCHMARK — current (ambient form) vs fixed (body form)")
        print(f"  n_restarts={args.restarts}   k_max_extra sweep={args.kmx}")
        print("  lower C_gcg is better (the convex roof is a minimisation)")
        print("=" * 108)

        if args.which in ("real", "both"):
            cases = real_cases()
            if not cases:
                print(f"\n[skip] no cached RDMs found under {CACHE}")
            else:
                print("\nREAL PRODUCTION DATA")
                for label, rho, n in cases:
                    ev = np.linalg.eigvalsh(rho)
                    print(f"  {label:<16} dim={rho.shape[0]:>3} "
                          f"rank={int(np.sum(ev > 1e-12)):>3} "
                          f"purity={float(np.real(np.trace(rho @ rho))):.6f}")
                run(cases, fixed_root, args.restarts, tuple(args.kmx))

        if args.which in ("synthetic", "both"):
            print("\nSYNTHETIC / ADVERSARIAL CASES")
            run(synthetic_cases(), fixed_root, args.restarts, tuple(args.kmx))


if __name__ == "__main__":
    main()
