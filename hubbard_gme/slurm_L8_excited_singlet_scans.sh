#!/bin/bash
#SBATCH -A nano                  # Project/account
#SBATCH -p ulow                  # Partition (ice4hpc / ulow)
#SBATCH -N 1                     # One node
#SBATCH --ntasks=1               # One task per array element
#SBATCH --cpus-per-task=8        # See NOTE ON CORES below
#SBATCH --mem=128GB              # See NOTE ON MEMORY below
#SBATCH --time=14-00:00:00       # Max runtime (14 days)
#SBATCH --output=logs/es_scan_%A_%a.log
#SBATCH --error=logs/es_scan_%A_%a.err
#SBATCH -J L8_es_scan            # Job name
#SBATCH --array=1-2              # 1 = U-scan, 2 = t-scan

# ═══════════════════════════════════════════════════════════════════
#  L=8 GME scans of the FIRST EXCITED SINGLET (T=0, full Hamiltonian)
#
#    array task 1 : U-scan   t=1.0 fixed, U = 0.1 .. 5   (20 points)
#    array task 2 : t-scan   U=5.0 fixed, t = 0.05 .. 2  (20 points)
#
#  Each task covers ALL body orders n = 2..8 in a single run — unlike the
#  surface jobs, the scan loops over n internally, so there is no per-n array.
#
#  The first excited singlet is re-located at EVERY grid point by its <S^2>,
#  because its eigenstate index moves with U/t through level crossings. A
#  fixed --state-idx would silently pick a triplet at some points.
#  --singlet-k 16 was verified sufficient for L=8 (idx=4 at both ends of the
#  t range; k=24 and k=32 give identical energies).
#
#  NOTE ON CORES
#  -------------
#  run_{t,U}_dependence_gme are SERIAL — they take no n_workers argument and
#  main.py never passes one. So --workers would be ignored and pinning BLAS
#  to 1 thread would leave every extra core idle. Instead we give BLAS the
#  cores, which speeds up the dense 12870x12870 diagonalisation and the GCG
#  linear algebra. (The tT-surface jobs are the opposite case: they DO use
#  multiprocessing, so there you keep OMP_NUM_THREADS=1 and raise --workers.)
#  main.py sets these with os.environ.setdefault, so exporting them here wins.
#
#  NOTE ON MEMORY
#  --------------
#  Full Hilbert space for L=8, Ne=8 is C(16,8) = 12870. The dense eigenvector
#  matrix is 12870^2 x 16 B = 2.6 GB, and LAPACK needs a few multiples of that
#  as workspace -> roughly 10-15 GB peak. 128 GB is already generous; these
#  scans build RDMs directly from eigenvectors and never cache an RDM bank, so
#  they do NOT need the ~900 GB / ~358 GB disk that the surface jobs do.
# ═══════════════════════════════════════════════════════════════════

export TMPDIR=/unimore_home/kshahzad/tmp
mkdir -p $TMPDIR logs
module purge
module load slurm

source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv             # <-- env with numpy/scipy/openfermion/matplotlib/tqdm

# Serial scan: let BLAS use the allocated cores (see NOTE ON CORES).
NTHREADS=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$NTHREADS
export OPENBLAS_NUM_THREADS=$NTHREADS
export MKL_NUM_THREADS=$NTHREADS
export NUMEXPR_NUM_THREADS=$NTHREADS

cd /unimore_home/kshahzad/hubbard_gme   # <-- repo path on the cluster

# ═══════════════════════════════════════════════════════════════════
#  EDIT THESE ONCE
# ═══════════════════════════════════════════════════════════════════
L=8                  # sites
NE=8                 # electrons (half filling = L)

# SECTOR=""                       # "" = Sz=0 sector
SECTOR="--full-hamiltonian"       # complete Hamiltonian (all Sz)

# U-scan settings (task 1): t fixed, U swept
USCAN_T=1.0
USCAN_UMIN=0.1       # NOT 0: at U=0 the excited manifold is degenerate and
USCAN_UMAX=5.0       # the singlet search raises RuntimeError
USCAN_STEPS=20

# t-scan settings (task 2): U fixed, t swept
TSCAN_U=5.0
TSCAN_TMIN=0.05
TSCAN_TMAX=2.0
TSCAN_STEPS=20

COMMON="--L $L --Ne $NE --mode uniform $SECTOR \
        --excited-singlet --singlet-k 16 \
        --thermal-solver gcg --n-restarts 8 --k-max-extra 4 --max-iter 200 \
        --use-pkl-cache --output-dir outputs"
# ═══════════════════════════════════════════════════════════════════

echo "=========================================================="
echo " task $SLURM_ARRAY_TASK_ID on $HOSTNAME (job $SLURM_JOB_ID)"
echo " L=$L Ne=$NE  sector='$SECTOR'  BLAS threads=$NTHREADS"
echo " started: $(date)"
echo "=========================================================="

case $SLURM_ARRAY_TASK_ID in

  1)  # ---- U-scan: first excited singlet, t fixed ----
      echo "U-SCAN  t=$USCAN_T  U=$USCAN_UMIN..$USCAN_UMAX ($USCAN_STEPS points)"
      python main.py $COMMON --t $USCAN_T \
        --u-scan --u-scan-min $USCAN_UMIN --u-scan-max $USCAN_UMAX \
        --u-scan-steps $USCAN_STEPS \
        --output-prefix ES_uscan_L${L}_t${USCAN_T}
      ;;

  2)  # ---- t-scan: first excited singlet, U fixed ----
      echo "t-SCAN  U=$TSCAN_U  t=$TSCAN_TMIN..$TSCAN_TMAX ($TSCAN_STEPS points)"
      python main.py $COMMON --U $TSCAN_U \
        --t-scan --t-scan-min $TSCAN_TMIN --t-scan-max $TSCAN_TMAX \
        --t-scan-steps $TSCAN_STEPS \
        --output-prefix ES_tscan_L${L}_U${TSCAN_U}
      ;;

  *)  echo "ERROR: unknown array index $SLURM_ARRAY_TASK_ID (expected 1 or 2)"
      exit 1
      ;;
esac

status=$?
echo "=========================================================="
echo " finished: $(date)   exit=$status"
echo "=========================================================="
exit $status
