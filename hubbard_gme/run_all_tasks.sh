#!/usr/bin/env bash
# Generate PDFs for the four tasks (thermal / 3D surface / t-scan / u-scan)
# from cached PKL files.
# Run once after computation is done — loads PKL, regenerates PDF instantly.
#
# NOTE: excited-singlet is no longer part of the standard workflow. The code
#       and the --excited-singlet flag still exist if you need them.

set -e
cd "$(dirname "$0")"

COMMON="--L 6 --Ne 6 --t 1.0 --U 10.0 --min-n 2
        --thermal-solver gcg --n-restarts 10 --max-iter 200 --k-max-extra 4
        --workers 6 --use-pkl-cache"

echo "=========================================="
echo "  Task 1: Thermal GME vs T"
echo "=========================================="
python main.py --thermal \
  --L 6 --Ne 6 --t 1.0 --U 10.0 \
  --T-min 0 --T-max 5 --T-steps 40 \
  --min-n 2 \
  --thermal-solver gcg --n-restarts 10 --max-iter 200 --k-max-extra 4 \
  --workers 6 --use-pkl-cache \
  --output-prefix L6_thermal_t1_U10

echo ""
echo "=========================================="
echo "  Task 2: 3D Surface  C_GME(t/U, T)"
echo "=========================================="
# HEAVY: t-steps × T-steps × (n=2..6) GCG solves. Reduce the grids or raise
# --workers for a faster run; drop this block if you only need Tasks 1/3/4.
python main.py --tT-surface \
  --L 6 --Ne 6 --U 10.0 \
  --t-scan-min 0.05 --t-scan-max 2.0 --t-scan-steps 20 \
  --T-min 0.0 --T-max 2.0 --tT-surface-T-steps 15 \
  --min-n 2 \
  --thermal-solver gcg --n-restarts 10 --max-iter 200 --k-max-extra 4 \
  --workers 6 --use-pkl-cache \
  --output-prefix L6_surface_U10

echo ""
echo "=========================================="
echo "  Task 3: t-scan GME vs t/U  (T = 0)"
echo "=========================================="
python main.py --t-scan \
  --L 6 --Ne 6 --U 10.0 \
  --t-scan-min 0.05 --t-scan-max 2.0 --t-scan-steps 20 \
  --min-n 2 \
  --thermal-solver gcg --n-restarts 10 --max-iter 200 --k-max-extra 4 \
  --workers 6 --use-pkl-cache \
  --output-prefix L6_tscan_U10

echo ""
echo "=========================================="
echo "  Task 4: U-scan GME vs U/t"
echo "=========================================="
python main.py --u-scan \
  --L 6 --Ne 6 --t 1.0 \
  --u-scan-min 1.0 --u-scan-max 20.0 --u-scan-steps 20 \
  --min-n 2 \
  --thermal-solver gcg --n-restarts 10 --max-iter 200 --k-max-extra 4 \
  --workers 6 --use-pkl-cache \
  --output-prefix L6_uscan_t1

echo ""
echo "All PDFs saved to outputs/"
