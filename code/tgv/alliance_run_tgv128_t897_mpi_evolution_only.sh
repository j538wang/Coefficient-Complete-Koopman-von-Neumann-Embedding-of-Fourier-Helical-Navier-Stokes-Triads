#!/bin/bash
#SBATCH --job-name=tgv128_t897_mpi_evol
#SBATCH --account=def-fslien
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=90G
#SBATCH --output=tgv128_t897_mpi_evol_%j.out
#SBATCH --error=tgv128_t897_mpi_evol_%j.err

set -euo pipefail

# Edit modules for your Alliance cluster if needed.  The important requirement
# is that mpi4py is built against the loaded MPI stack.
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

echo "Job finished: $(date)"
