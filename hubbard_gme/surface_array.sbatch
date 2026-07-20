#!/bin/bash
#SBATCH -A nano                  # Your project/account
#SBATCH -p ulow                  # Partition name (ice4hpc / ulow)
#SBATCH -N 1                     # One node
#SBATCH --ntasks=1               # One task per array element
#SBATCH --cpus-per-task=20       # Number of cores (adjust if needed)
#SBATCH --mem=900GB              # Memory per job
#SBATCH --time=14-00:00:00       # Max runtime (14 days cpu)
#SBATCH --output=logs/output_%A_%a.log   # %A=job id, %a=array index (=n)
#SBATCH --error=logs/error_%A_%a.log
#SBATCH -J L8_surf               # Job name
#SBATCH --array=2-8              # ARRAY INDEX = body-order n (all n: 2,3,4,5,6,7,8)

# Set temporary directory to your home or scratch
export TMPDIR=/unimore_home/kshahzad/tmp
mkdir -p $TMPDIR logs
module purge
module load slurm

source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv            # <-- env with numpy/scipy/openfermion/matplotlib/tqdm

# CRITICAL: 1 BLAS thread per process so --workers doesn't oversubscribe cores
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /unimore_home/kshahzad/hubbard_gme    # <-- path to the repo on Lyra

# ═══════════════════════════════════════════════════════════════════
#  EDIT THESE ONCE — shared by every task below
# ═══════════════════════════════════════════════════════════════════
L=8                  # sites
NE=8                 # electrons (half-filling = L)
U=8.0                # interaction U (surface, t-scan, thermal)
THOP=0.1             # hopping t (fixed for thermal & u-scan)

# SECTOR=""                        # "" = Sz=0 sector (default)
SECTOR="--full-hamiltonian"        # complete Hamiltonian (all Sz)

WORKERS=${SLURM_CPUS_PER_TASK:-10} # auto-matches -c above

# THIS task's body-order n comes from the array index:
N=$SLURM_ARRAY_TASK_ID

# Common to every task (NOTE: --min-n is now set PER TASK, not here):
COMMON="--L $L --Ne $NE --thermal-solver gcg \
        --n-restarts 4 --max-iter 200 --k-max-extra 4 \
        --workers $WORKERS --use-pkl-cache $SECTOR"
# ═══════════════════════════════════════════════════════════════════

echo "Task n=$N on $HOSTNAME (job $SLURM_JOB_ID)  [L=$L Ne=$NE U=$U]"

# One SLURM job per body-order n: single-n surface (--min-n $N --tT-surface-max-n $N).
# --output-prefix ...n${N} keeps each n's PDF from overwriting the others.
python main.py --tT-surface $COMMON --U $U \
  --t-scan-min 0.05 --t-scan-max 2.0 --t-scan-steps 20 \
  --T-min 0.0 --T-max 2.0 --tT-surface-T-steps 15 \
  --min-n $N --tT-surface-max-n $N \
  --output-prefix L${L}_surface_U${U}_n${N}

# ═══════════════════════════════════════════════════════════════════
#  AFTER all array tasks finish, merge the per-n PKLs into ONE figure:
#
#    python main.py --merge-surface \
#      "results/rdm_pkl_results/thermal_runs/tTsurface_*L8_*nmax*.pkl" \
#      --output-dir outputs
#
#  (heavy n=7,8: submit separately with  #SBATCH --array=7-8 )
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
#  YOUR OTHER TASKS (thermal / t-scan / u-scan)
#  These are NOT per-n array jobs — run each as its OWN submission.
#  For a single one, comment out the surface `python` block above and
#  the `#SBATCH --array` line (or set --array=0), then uncomment ONE below.
#  NOTE: for L=8 these also go n=2..L internally and will be slow;
#  ask me to add a --max-n cap to --thermal / --t-scan if needed.
# ═══════════════════════════════════════════════════════════════════
#
# Task 1: thermal vs Temperature T
#   python main.py --thermal $COMMON --min-n 2 --t $THOP --U $U \
#     --T-min 0 --T-max 5 --T-steps 40 \
#     --output-prefix L${L}_thermal_t${THOP}_U${U}
#
# Task 3: t-scan (T=0)
#   python main.py --t-scan $COMMON --min-n 2 --U $U \
#     --t-scan-min 0.05 --t-scan-max 2.0 --t-scan-steps 20 \
#     --output-prefix L${L}_tscan_U${U}
#
# Task 4: u-scan (T=0)
#   python main.py --u-scan $COMMON --min-n 2 --t $THOP \
#     --u-scan-min 1.0 --u-scan-max 20.0 --u-scan-steps 20 \
#     --output-prefix L${L}_uscan_t${THOP}
#
#  (THOP=0.1 is the fixed hopping for thermal & u-scan.)
# ═══════════════════════════════════════════════════════════════════
