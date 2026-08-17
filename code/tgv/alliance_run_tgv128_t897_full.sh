#!/bin/bash
#SBATCH --job-name=tgv128_t897_helical
#SBATCH --account=def-fslien
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=tgv128_t897_%j.out
#SBATCH --error=tgv128_t897_%j.err

set -euo pipefail

# Edit module names if your Alliance cluster uses different defaults.
module purge
module load python/3.11 scipy-stack || true

# Recommended: create and activate a venv once, then uncomment these lines.
# python -m venv $SLURM_TMPDIR/tgv_venv
# source $SLURM_TMPDIR/tgv_venv/bin/activate
# pip install --no-index numpy scipy pandas matplotlib

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "Working directory: $(pwd)"
echo "Job started: $(date)"

# -----------------------------------------------------------------------------
# 1. Evolve TGV to the DNS peak-dissipation time and compare global diagnostics
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 2. Helical RHS reconstruction, shell energy transfer, and helicity transfer
# -----------------------------------------------------------------------------
python run_shell_helicity_transfer.py \
  --modes results_tgv128_t897/tgv_peak_sparse_modes.npz \
  --out results_tgv128_t897_shell_helicity

# -----------------------------------------------------------------------------
# 3. One-step update consistency at the evolved t=8.97 snapshot
# -----------------------------------------------------------------------------
python run_one_step_update.py \
  --modes results_tgv128_t897/tgv_peak_sparse_modes.npz \
  --dt-list 1e-4,1e-3,1e-2 \
  --out results_tgv128_t897_one_step

# -----------------------------------------------------------------------------
# 4. Active helical-triad contribution and truncation diagnostic
#    If this is too slow, reduce --max-input-modes to 800 or 1000.
# -----------------------------------------------------------------------------
python run_active_triad_truncation.py \
  --modes results_tgv128_t897/tgv_peak_sparse_modes.npz \
  --max-input-modes 1200 \
  --contrib-rel-tol 1e-14 \
  --fractions 0.001,0.005,0.01,0.02,0.05,0.1,0.25,0.5,1.0 \
  --out results_tgv128_t897_active_triad_truncation

echo "Job finished: $(date)"
