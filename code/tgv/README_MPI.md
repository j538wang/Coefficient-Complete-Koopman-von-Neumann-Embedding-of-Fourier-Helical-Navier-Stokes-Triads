# True-MPI TGV 128^3 peak-dissipation benchmark

This package adds a true-MPI dense Taylor--Green vortex evolution script:

```bash
run_tgv128_dns_peak_mpi.py
```

The previous `run_tgv128_dns_peak.py` is a serial NumPy implementation.  The new
MPI script distributes the dense `128^3` pseudo-spectral evolution across MPI
ranks.  It implements a distributed complex FFT using `mpi4py.Alltoallv`:

1. physical-space x slabs on each rank;
2. local FFT in the two transverse directions `(y,z)`;
3. MPI all-to-all transpose into spectral `(ky,kz)` column pencils;
4. FFT in the remaining `x` direction.

This avoids replicating the dense `128^3` field on every rank during the
expensive time evolution.  Only the exported sparse Fourier snapshots are
collected on rank 0 for downstream helical diagnostics.

## Dependencies

Required:

```text
python
numpy
pandas
matplotlib
mpi4py
```

No `mpi4py-fft` dependency is required.

On Alliance, load a Python, MPI, and mpi4py stack that are compatible with each
other.  The supplied SLURM files use a common template, but module names can vary
by cluster and software year.

## Recommended first run

Submit the evolution-only job first:

```bash
sbatch alliance_run_tgv128_t897_mpi_evolution_only.sh
```

This runs the expensive true-MPI part and writes:

```text
results_tgv128_t897_mpi/
  tgv128_global_history.csv
  tgv128_vs_dns_global.csv
  tgv128_vs_dns_energy.png
  tgv128_vs_dns_dissipation.png
  tgv128_vs_dns_enstrophy.png
  tgv128_vs_dns_relative_errors.png
  tgv_peak_sparse_modes.npz
  tgv128_peak_summary.json
```

## Full workflow

After confirming the evolution-only job works, run:

```bash
sbatch alliance_run_tgv128_t897_mpi_full.sh
```

This performs the same true-MPI dense evolution and then runs serial sparse-mode
diagnostics:

```text
results_tgv128_t897_mpi_shell_helicity/
results_tgv128_t897_mpi_one_step/
results_tgv128_t897_mpi_active_triad_truncation/
```

The post-processing diagnostics remain serial because they operate on the
thresholded sparse Fourier snapshot rather than on the dense `128^3` grid.

## Runtime advice

Start with 1 node and 32 MPI ranks:

```text
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
```

Set all threading variables to 1.  This is an MPI decomposition, not an OpenMP
threaded job.

If the full workflow is too expensive, reduce the active-triad diagnostic:

```bash
--max-input-modes 800
```

instead of the default 1200.
