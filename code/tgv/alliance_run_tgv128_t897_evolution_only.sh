#!/bin/bash
#SBATCH --job-name=tgv128_t897_evolve
#SBATCH --account=def-fslien
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=tgv128_t897_evolve_%j.out
#SBATCH --error=tgv128_t897_evolve_%j.err

set -euo pipefail
module purge
module load python/3.11 scipy-stack || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

python run_tgv128_dns_peak.py \
  --N 128 \
  --dt 0.002 \
  --t-final 8.97 \
  --nu 0.000625 \
  --save-times 0,1,2,4,6,8,8.97 \
  --history-every 10 \
  --mode-rel-tol 1e-10 \
  --max-modes 2000 \
  --dns-csv dns_reference/hiocfd_ucl_512_global.csv \
  --out results_tgv128_t897
