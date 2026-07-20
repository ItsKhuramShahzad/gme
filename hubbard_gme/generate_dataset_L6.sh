#!/bin/bash
# =============================================================================
# ML dataset factory — Hubbard rings up to L=6 (local machine, GCG labels)
#
#   For every (L, U, sector, topology): run the tT-surface task, which
#   GCG-labels the full (t, T) grid in parallel and caches everything.
#   Fully resumable: surfaces + banks are config-hashed; reruns skip work.
#
#   Grid: uniform ring only; 8 U x 2 sectors = 16 configs per L.
#   Yield per full sweep: ~70,000 records
#     L=6: 16 configs x (n=2..6 x 25t x 20T = 2,500)  ~40k   (the slow part)
#     L=4: 16 configs x (n=2..4 x 25t x 20T = 1,500)  ~24k
#     L=2: 16 configs x (n=2   x 25t x 20T =   500)   ~8k
#   n=2 uses the analytic Werner GME (no convex roof); n>=3 uses GCG.
#
#   Run it:        nohup setsid bash generate_dataset_L6.sh > factory.log 2>&1 &
#   Watch it:      tail -f factory.log
#   Package it:    (repeat any time — dedup by config hash)
#     python gme_qml/ml_ingest_pkls.py --source surface \
#         --results-root results/rdm_pkl_results --outdir gme_qml/data_for_ML_v2
# =============================================================================
set -u
PY=/home/anjum/miniconda3/envs/qchem/bin/python
WORKERS=${WORKERS:-10}

for L in 2 4 6; do
  for U in 1 2 3 4 6 8 10 12; do
    for SECTOR in "" "--full-hamiltonian"; do
      echo "=== $(date +%F\ %T)  L=$L U=$U mode=uniform sector=${SECTOR:-sz0} ==="
      $PY main.py --L "$L" --U "$U" --t 1.0 --mode uniform $SECTOR \
          --tT-surface \
          --t-scan-min 0.05 --t-scan-max 2.0 --t-scan-steps 25 \
          --T-min 0 --T-max 3 --tT-surface-T-steps 20 \
          --workers "$WORKERS"
    done
  done
done
echo "=== FACTORY SWEEP COMPLETE $(date) ==="
echo "Now run the ingestion command (see header) to package the records."
