# Coefficient-Complete Koopman--von Neumann Embedding of Fourier--Helical Navier--Stokes Triads

This public repository contains the code and processed data supporting the manuscript **Coefficient-Complete Koopman--von Neumann Embedding of Fourier--Helical Navier--Stokes Triads** by Jialin Wang, Fue-Sang Lien, and Eugene Yee.

It contains the scripts and processed outputs used for the coefficient ensemble, shared-scale graph audit, global Koopman--von Neumann (KvN) assembly, restricted finite-Fock gate calculation, spectral resource enumeration, Taylor--Green vortex (TGV) checks, and Johns Hopkins Turbulence Database (JHTDB) checks.

## Quick verification

Run the repository-level consistency check from the repository root.

```text
python verify_archive.py
```

This check verifies the SHA-256 manifest and the principal numerical values reported in the manuscript.

## Archive structure

- `code/core` contains the deterministic coefficient, graph, KvN, finite-Fock, and resource scripts.
- `code/tgv` contains the serial and MPI TGV evolution and post-processing scripts, including the batch scripts used on Alliance Canada.
- `code/jhtdb` contains the JHTDB acquisition and helical reconstruction scripts.
- `processed` contains the machine-readable CSV, JSON, TXT, and NPZ inputs and outputs used in the manuscript.
- `figures` contains the three external figure files included by the LaTeX manuscript.
- `MANIFEST.csv` records the SHA-256 digest, byte size, and path of every archived file except the manifest itself.

## Environment

Python 3.10 or later is recommended. Install the common numerical dependencies with

```text
python -m pip install -r requirements.txt
```

The distributed TGV run additionally requires an MPI implementation and `mpi4py`. JHTDB acquisition requires `givernylocal` and an access token supplied through the `JHTDB_TOKEN` environment variable or the command-line option documented by the acquisition script. No credential is stored in this archive.

The two Python packages associated with these external services can be requested with `python -m pip install -r requirements-optional.txt`; the MPI runtime and JHTDB access authorization remain system-level requirements.

## Core deterministic workflow

Run the following commands from the archive root. The first command recreates the prescribed first 800 lexicographic geometries and all eight helicity assignments. The later commands consume the resulting CSV.

```text
python code/core/phase_gauge_triad_validation.py --kmax 3 --max-geometries 800 --out derived/unified_ensemble
python code/core/make_unified_ensemble_figure.py --csv derived/unified_ensemble/phase_gauge_ensemble.csv --out derived/figures/unified_ensemble_single_g.png --summary derived/unified_ensemble/compact_summary.json
python code/core/global_scaling_compatibility_audit.py --csv derived/unified_ensemble/phase_gauge_ensemble.csv --out derived/global_scaling_audit
python code/core/kvn_global_graph_audit.py --csv derived/unified_ensemble/phase_gauge_ensemble.csv --out derived/kvn_global_graph_audit
python code/core/kvn_shared_mode_realization_audit.py --csv derived/unified_ensemble/phase_gauge_ensemble.csv --out derived/kvn_shared_mode_audit
python code/core/kvn_two_triad_gate_validation.py --csv derived/unified_ensemble/phase_gauge_ensemble.csv --out derived/kvn_two_triad_gate_validation --tau-end 0.18 --selected-cutoff 10 --selected-amplitude 2.5
python code/core/full_spectral_resource_scaling.py --max-cutoff 6 --kvn-summary derived/kvn_global_graph_audit/global_kvn_summary.json --out derived/full_spectral_resource_scaling
```

The finite-Fock command is the most demanding core calculation. Its `--plot-only` option regenerates the figure from existing scan CSV files.

## TGV workflow

The processed sparse peak state is provided at `processed/tgv/state/tgv_peak_sparse_modes.npz`. It was obtained from a dealiased periodic `128^3` pseudo-spectral calculation with `nu=1/1600`, fourth-order Runge--Kutta time stepping, `dt=0.002`, and the exported snapshot at `t=8.97`. The Alliance Canada scripts and the MPI implementation in `code/tgv` document the full evolution.

The two manuscript post-processing calculations can be repeated without rerunning the full DNS.

```text
python code/tgv/run_shell_helicity_transfer.py --modes processed/tgv/state/tgv_peak_sparse_modes.npz --out derived/tgv_shell_transfer
python code/tgv/run_active_triad_truncation.py --modes processed/tgv/state/tgv_peak_sparse_modes.npz --max-input-modes 1200 --top-csv 50000 --out derived/tgv_active_triad_truncation
```

The shell-transfer calculation uses the retained 5000-mode sparse state. The active-triad calculation selects its 1200 largest input modes and records 11,494,656 helical amplitude contributions.

## JHTDB workflow

The external checks use `isotropic1024coarse` at time indices 512, 1152, 1792, 2432, 3072, 3712, 4352, and 4992. Each query starts at server index 1 and uses stride 16 in all directions, producing a `64^3` velocity cube. Raw JHTDB fields are third-party data and are excluded from the archive. The processed per-snapshot comparisons and their eight-field average are included.

Example commands for one snapshot are

```text
python code/jhtdb/01_fetch_jhtdb_full_domain_coarse.py --dataset isotropic1024coarse --time-index 512 --stride 16 --tile-size 128 --retries 5 --outdir data_jhtdb
python code/jhtdb/04_helical_channel_reconstruction.py --input data_jhtdb/isotropic1024coarse_t000512_s16_velocity.npz --outdir derived/jhtdb_helical --mode galerkin23 --kplot-max 32
```

Repeat the commands for the other seven time indices and then pass the eight `*_total_compare.csv` files to `05_average_helical_reconstruction.py`.

## Scope

The archive reproduces the conservative coefficient and operator audits reported in the manuscript. The resource calculation is a structured primitive-count analysis. Physical Gaussian map and unmap circuits, viscous quantum evolution, fault-tolerant synthesis, and end-to-end speedup are outside the validated scope.

## Data availability

The code and processed data supporting the study are included in this repository. Raw JHTDB velocity fields are third-party data and are available through JHTDB under its access and usage conditions. The repository records the dataset, time indices, sampling stride, acquisition scripts, and processed comparison outputs needed to repeat the external-data workflow.

## Citation

Please cite the associated manuscript and this repository if you use these materials. Repository citation metadata are provided in `CITATION.cff`; the journal citation can be added after publication.
