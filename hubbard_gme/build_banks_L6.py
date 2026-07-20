#!/usr/bin/env python3
"""
build_banks_L6.py — build RDM banks ONLY (no surfaces, no GCG, no plots).

Each bank Γ^(n)[ψ_m] is one diagonalization + the per-eigenstate RDMs, cached
in results/rdm_pkl_results/rdm_bank/. This is the cheap first stage; the
GCG labelling (with Werner n=2 + convergence history) is done afterwards by
    python gme_qml/ml_ingest_pkls.py --source banks ...

Grid: uniform ring, L in {2,4,6}, U in {1,2,3,4,6,8,10,12}, both S_z sectors.
Fully resumable — an existing bank is loaded, not rebuilt.

Run:   nohup setsid python build_banks_L6.py > build_banks.log 2>&1 &
Watch: tail -f build_banks.log
"""
import sys, time
sys.path.insert(0, "src")
from hubbard_gme.thermal_pipeline import build_or_load_rdm_bank

L_VALUES = [2, 4, 6]
U_VALUES = [1, 2, 3, 4, 6, 8, 10, 12]
T_HOP = 1.0
N_WORKERS = 8

t0 = time.time()
done = 0
for L in L_VALUES:
    for U in U_VALUES:
        for sp in (True, False):          # True = S_z=0 sector, False = full
            tag = "sz0" if sp else "full"
            print(f"[{time.strftime('%H:%M:%S')}] L={L} U={U} t={T_HOP} {tag}",
                  flush=True)
            build_or_load_rdm_bank(
                L=L, t=T_HOP, U=float(U), mode="uniform",
                n_min=2, average_site_clusters=True,
                n_workers=N_WORKERS, spin_preserving=sp,
                use_pkl_cache=True, save_pkl=True, verbose=False,
            )
            done += 1
            print(f"    done ({done} banks, {(time.time()-t0)/60:.1f} min "
                  f"elapsed)", flush=True)

print(f"\nALL BANKS BUILT: {done} banks in {(time.time()-t0)/60:.1f} min")
print("Next: python gme_qml/ml_ingest_pkls.py --source banks "
      "--results-root results/rdm_pkl_results --outdir gme_qml/data_for_ML_v2 "
      "--bank-T 0 0.1 0.25 0.5 1 2 --n-restarts 8 --max-iter 150 --k-max-extra 2")
