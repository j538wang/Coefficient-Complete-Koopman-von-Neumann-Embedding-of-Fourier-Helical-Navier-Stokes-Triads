# TGV128 peak-dissipation fluid benchmark package

This package runs a dealiased pseudo-spectral Taylor--Green vortex to
`t = 8.97`, compares global quantities with the public `512^3` DNS reference,
and exports a sparse Fourier snapshot for helical RHS reconstruction diagnostics.

## Main workflow on Alliance Canada

Edit the account line in the SLURM scripts:

```bash
#SBATCH --account=def-fslien
```

Then submit:

```bash
sbatch alliance_run_tgv128_t897_full.sh
```

If you only want to generate the TGV field and DNS comparison first:

```bash
sbatch alliance_run_tgv128_t897_evolution_only.sh
```

## Key parameters

The recommended default is

```bash
N = 128
dt = 0.002
t_final = 8.97
nu = 0.000625  # Re = 1600
save_times = 0,1,2,4,6,8,8.97
mode_rel_tol = 1e-10
max_modes = 2000
```

The choice `t_final=8.97` is physically motivated by the peak dissipation time
in the public `512^3` DNS reference. The DNS data are used for global context
only. The exact RHS reference for helical reconstruction is still the Cartesian
projected convolution evaluated on the same saved `128^3` Fourier snapshot.

## Outputs from `run_tgv128_dns_peak.py`

Directory: `results_tgv128_t897/`

Important files:

```text
tgv128_global_history.csv
tgv128_vs_dns_global.csv
tgv128_vs_dns_energy.png
tgv128_vs_dns_dissipation.png
tgv128_vs_dns_enstrophy.png
tgv128_vs_dns_relative_errors.png
tgv128_dense_energy_spectrum_final.csv
tgv128_dense_energy_spectrum_final.png
tgv_sparse_modes_t*.npz
tgv_peak_sparse_modes.npz
tgv128_peak_summary.json
```

## Follow-up diagnostics

The full SLURM script also runs:

```bash
python run_shell_helicity_transfer.py --modes results_tgv128_t897/tgv_peak_sparse_modes.npz --out results_tgv128_t897_shell_helicity
python run_one_step_update.py --modes results_tgv128_t897/tgv_peak_sparse_modes.npz --dt-list 1e-4,1e-3,1e-2 --out results_tgv128_t897_one_step
python run_active_triad_truncation.py --modes results_tgv128_t897/tgv_peak_sparse_modes.npz --max-input-modes 1200 --out results_tgv128_t897_active_triad_truncation
```

If the truncation diagnostic is too slow, reduce `--max-input-modes` to `800` or `1000`.

## Local smoke test

```bash
python run_tgv128_dns_peak.py --N 16 --dt 0.002 --t-final 0.02 --save-times 0,0.02 --max-modes 200 --out smoke_tgv
python run_shell_helicity_transfer.py --modes smoke_tgv/tgv_peak_sparse_modes.npz --out smoke_shell
python run_one_step_update.py --modes smoke_tgv/tgv_peak_sparse_modes.npz --out smoke_step
python run_active_triad_truncation.py --modes smoke_tgv/tgv_peak_sparse_modes.npz --max-input-modes 64 --out smoke_trunc
```
