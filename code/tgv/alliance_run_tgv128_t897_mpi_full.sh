#!/bin/bash
#SBATCH --job-name=tgv128_t897_mpi_full
#SBATCH --account=def-fslien
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=90G
#SBATCH --output=tgv128_t897_mpi_full_%j.out
#SBATCH --error=tgv128_t897_mpi_full_%j.err

set -euo pipefail

# Edit modules for your Alliance cluster if needed.  The dense TGV evolution is
# true MPI.  The sparse-mode post-processing diagnostics are serial Python jobs
# because they operate on thresholded O(10^3) Fourier modes.
module purge
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 python/3.11 scipy-stack || true
module load mpi4py || true

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python - <<'PY'
from mpi4py import MPI
if MPI.COMM_WORLD.Get_rank() == 0:
    print("mpi4py OK")
    print(MPI.Get_library_version())
PY

echo "Working directory: $(pwd)"
echo "Job started: $(date)"
echo "SLURM_NTASKS=${SLURM_NTASKS:-unknown}"

# -----------------------------------------------------------------------------
# 1. True-MPI TGV evolution to DNS peak-dissipation time
# -----------------------------------------------------------------------------
srun --mpi=pmix -n ${SLURM_NTASKS} python run_tgv128_dns_peak_mpi.py \
  --N 128 \
  --dt 0.002 \
  --t-final 8.97 \
  --nu 0.000625 \
  --save-times 0,1,2,4,6,8,8.97 \
  --history-every 10 \
  --mode-rel-tol 1e-10 \
  --max-modes 2000 \
  --dns-csv dns_reference/hiocfd_ucl_512_global.csv \
  --out results_tgv128_t897_mpi

# -----------------------------------------------------------------------------
# 2. Sparse-mode diagnostics.  These are intentionally serial because the input
#    is the exported sparse Fourier snapshot, not the dense 128^3 field.
# -----------------------------------------------------------------------------
python run_shell_helicity_transfer.py \
  --modes results_tgv128_t897_mpi/tgv_peak_sparse_modes.npz \
  --out results_tgv128_t897_mpi_shell_helicity

python run_one_step_update.py \
  --modes results_tgv128_t897_mpi/tgv_peak_sparse_modes.npz \
  --dt-list 1e-4,1e-3,1e-2 \
  --out results_tgv128_t897_mpi_one_step

python run_active_triad_truncation.py \
  --modes results_tgv128_t897_mpi/tgv_peak_sparse_modes.npz \
  --max-input-modes 1200 \
  --contrib-rel-tol 1e-14 \
  --fractions 0.001,0.005,0.01,0.02,0.05,0.1,0.25,0.5,1.0 \
  --out results_tgv128_t897_mpi_active_triad_truncation

echo "Job finished: $(date)"
