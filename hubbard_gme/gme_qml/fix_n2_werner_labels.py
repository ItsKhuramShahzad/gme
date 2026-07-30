#!/usr/bin/env python3
"""
fix_n2_werner_labels.py — repair n=2 records that were labelled with the
(incorrect) Werner formula instead of the convex roof.

Background
----------
For two qubits the Wootters concurrence IS the convex roof exactly, so the
correct C_target for n=2 is the concurrence. An earlier ingestion wrote the
Werner form sqrt(1 - sqrt(1 - C^2)) as C_target, which is NOT a valid GME.

This script finds every record whose solver == "werner_analytic", recomputes
the Wootters concurrence from the STORED rho (analytic, no solver run), and
rewrites C_target / C_gcg / C_wootters / solver / source. Everything else in
the record (rho, config, features) is preserved byte-for-byte.

Run (repo root):
    python gme_qml/fix_n2_werner_labels.py --dry-run     # report only
    python gme_qml/fix_n2_werner_labels.py               # apply the fix
"""
import argparse, glob, os, sys
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))       # <repo>/gme_qml
_REPO_ROOT = os.path.dirname(_THIS_DIR)                      # <repo>
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
from hubbard_gme.thermal_pipeline import _wootters_concurrence

# Anchored to the repo, NOT the CWD. With CWD-relative defaults this script
# would silently scan nothing when launched from elsewhere and report
# "0 records found", which reads like success but repairs nothing.
DIRS = [os.path.join(_REPO_ROOT, "gme_qml/data_for_ML_v2/ingested/records"),
        os.path.join(_REPO_ROOT, "gme_qml/data_for_ML/ingested/records")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dirs", nargs="+", default=DIRS)
    args = ap.parse_args()

    fixed = skipped = 0
    deltas = []
    scanned_any = False
    for d in args.dirs:
        if not os.path.isdir(d):
            print(f"  [WARN] not a directory, skipping: {d}")
            continue
        n_here = len(glob.glob(os.path.join(d, "*.npz")))
        print(f"  scanning {d}  ({n_here} npz)")
        scanned_any = scanned_any or n_here > 0
        for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
            try:
                z = np.load(p, allow_pickle=True)
            except Exception:
                continue
            if "solver" not in z or "werner" not in str(z["solver"]).lower():
                skipped += 1
                continue

            data = {k: z[k] for k in z.files}
            rho = data["rho_real"].astype(np.complex128) + 1j * data["rho_imag"]
            tr = float(np.real(np.trace(rho)))
            if abs(tr) > 1e-12:
                rho = rho / tr
            c_correct = float(_wootters_concurrence(rho))
            c_old = float(data.get("C_target", np.nan))
            deltas.append(abs(c_correct - c_old))

            data["C_target"] = np.float64(c_correct)
            data["C_gcg"] = np.float64(c_correct)
            data["C_wootters"] = np.float64(c_correct)
            data["solver"] = np.array("wootters_analytic")
            data["source"] = np.array("wootters_analytic")

            if not args.dry_run:
                tmp = p + ".tmp.npz"
                np.savez_compressed(tmp, **data)
                os.replace(tmp, p)
            fixed += 1

    if not scanned_any:
        print("\n  [ERROR] no .npz records were found in any given directory.\n"
              "          Nothing was checked — this is NOT a clean result.")
        raise SystemExit(2)

    d = np.array(deltas) if deltas else np.array([0.0])
    print(f"records with WERNER label found : {fixed}")
    print(f"records left untouched          : {skipped}")
    print(f"label error being corrected     : "
          f"median {np.median(d):.4f}  max {d.max():.4f}")
    print("DRY RUN — nothing written" if args.dry_run else "FIXED (records rewritten)")


if __name__ == "__main__":
    main()
