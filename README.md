# Coefficient-Complete Koopman--von Neumann Embedding of Fourier--Helical Navier--Stokes Triads

This public repository contains the code and processed data supporting the manuscript **Coefficient-Complete Koopman--von Neumann Embedding of Fourier--Helical Navier--Stokes Triads** by Jialin Wang, Fue-Sang Lien, and Eugene Yee.

It contains the scripts and processed outputs used for the coefficient ensemble, complete $K_{\max}=3$ shared-scale graph audit, global Koopman--von Neumann (KvN) assembly, restricted Gaussian and cubic gate calculation, the complete eight-real-coordinate cutoff scan, spectral resource enumeration, Taylor--Green vortex (TGV) checks, and Johns Hopkins Turbulence Database (JHTDB) checks.

## Quick verification

Run the repository-level consistency check from the repository root.

```text
python verify_archive.py
```

This check verifies the SHA-256 manifest and the principal numerical values reported in the manuscript.
After an intentional archive update, regenerate the manifest with `python generate_manifest.py` before running the verification command.

## Archive structure

- `code/core` contains the deterministic coefficient, graph, KvN, finite-Fock, and resource scripts.
- `code/tgv` contains the serial and MPI TGV evolution and post-processing scripts, including the batch scripts used on Alliance Canada.
- `code/jhtdb` contains the JHTDB acquisition and helical reconstruction scripts.
- `processed` contains the machine-readable CSV, JSON, TXT, and NPZ inputs and outputs used in the manuscript.
- `reproducibility/eight_coordinate_cutoff` contains the self-contained revision calculation for the complete eight-real-coordinate two-triad model, including all six cutoff outputs and the independent projection-convention test.
- `figures` contains the external figure files included by the LaTeX manuscript.
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

The finite-Fock command is the most demanding diagnostic-subset calculation. Its `--plot-only` option regenerates the figure from existing scan CSV files.

## Complete Kmax=3 graph and B+ gate workflow

The complete coefficient scan contains three created-mode orientations for every closed triad. The following workflow generates the full intermediate scan, retains one representative of each reality-reduced closed support, assembles the 342-node compatibility system and the 684-coordinate KvN field, and calibrates the resource count from the complete graph.

```text
python code/core/phase_gauge_triad_validation.py --kmax 3 --max-geometries -1 --out derived/full_kmax3_ensemble
python code/core/deduplicate_reality_reduced_triads.py --csv derived/full_kmax3_ensemble/phase_gauge_ensemble.csv --out derived/full_kmax3_unique
python code/core/global_scaling_compatibility_audit.py --csv derived/full_kmax3_unique/phase_gauge_ensemble.csv --out derived/full_kmax3_global_scaling_audit
python code/core/kvn_global_graph_audit.py --csv derived/full_kmax3_unique/phase_gauge_ensemble.csv --expected-active 32880 --expected-nodes 342 --reality-reduced-full-cube --out derived/full_kmax3_kvn_graph_audit
python code/core/full_spectral_resource_scaling.py --max-cutoff 6 --kvn-summary derived/full_kmax3_kvn_graph_audit/global_kvn_summary.json --out derived/full_spectral_resource_scaling_bplus
```

The first intermediate CSV exceeds GitHub's single-file limit. The public archive therefore retains the 32,928-row deduplicated CSV used by every downstream audit, together with the deduplication summary and the script that regenerates the intermediate file. The complete downstream outputs include 60,000 scale constraints and 734,176 combined KvN terms.

The B+ gate audit instantiates every passive Gaussian map, the single-mode cubic-phase operation, and the inverse map on the four-coordinate real restriction. It uses a local tensor-product cutoff as a boundary-stress test and a passive-invariant total-number cutoff as an independent compilation check.

```text
python code/core/kvn_two_triad_gate_validation.py --csv processed/unified_ensemble/phase_gauge_ensemble.csv --out derived/kvn_two_triad_gate_validation_bplus --tau-end 0.18 --selected-cutoff 10 --selected-amplitude 2.5
python code/core/kvn_gaussian_total_number_validation.py --csv processed/unified_ensemble/phase_gauge_ensemble.csv --out derived/kvn_two_triad_gate_validation_bplus --selected-cutoff 16
python code/core/make_gaussian_bplus_figure.py --local-cutoff derived/kvn_two_triad_gate_validation_bplus/gaussian_compilation_cutoff_scan.csv --global-cutoff derived/kvn_two_triad_gate_validation_bplus/global_fock_gaussian_cutoff_scan.csv --global-strang derived/kvn_two_triad_gate_validation_bplus/global_fock_gaussian_strang_scan.csv --out derived/kvn_two_triad_gate_validation_bplus
```

The processed B+ directory records the residual after every subgate, the boundary population, the Gaussian and non-Gaussian gate counts, and the maximum and total cubic strengths.

## Complete eight-coordinate cutoff workflow

The revision calculation removes the four-coordinate real-invariant restriction and directly evolves the complete 24-term projected KvN Hamiltonian in eight real coordinates.  Every cutoff is initialized by projecting and renormalizing the same untruncated eight-mode product coherent state.  The Hamiltonian projection convention is `P_N (p_a q_b q_c) P_N`; it is not the product of three separately projected factors.  The reported covariance separately uses the truncated coordinates `Q_a^(N)=P_N q_a P_N`.  This is an internally consistent finite model, but it is not identical to projecting a complete quadratic observable after it is formed.

```text
cd reproducibility/eight_coordinate_cutoff
python run_full8_cutoff.py --cutoffs 6,8,10,12,14,16 --out .
python run_full8_cutoff.py --aggregate-only --out .
python verify_projection_convention.py --cutoff 2 --out projection_verification.json
python plot_cutoff_trend.py --results-dir . --output-stem ../../figures/eight_coordinate_cutoff_trend
```

The global total-number dimensions range from 3,003 at `N=6` to 735,471 at `N=16`.  Over the nine sampled times, the maximum last-two-shell population decreases monotonically from `2.5064e-1` to `1.0935e-5`.  Relative to the finite `N=16` reference, the complete `N=14` first-moment trajectory differs by `2.2052e-5`, and its final covariance matrix differs by `2.5170e-4`.  The additional `initial_projection_diagnostics.csv` and `.json` files quantify two finite-cutoff effects at `tau=0`: the coherent-state first-moment shrinkage and the covariance-trace difference between truncated-coordinate products and complete quadratic projection.  The first-moment bias decreases from `8.9772e-2` at `N=6` to `1.0734e-6` at `N=16`; the normalized covariance-trace convention difference decreases from `1.0054e-1` to `2.0609e-6`.  These are finite-reference results for the stated initial state, interval, sampling grid, and cutoff range; they are not an infinite-cutoff proof or an eight-mode gate compilation.

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

The archive reproduces the conservative coefficient and operator audits reported in the manuscript. The restricted four-coordinate calculation validates ideal passive Gaussian map--single-mode-cubic--inverse-map circuits under two independent truncation schemes. Direct finite-cutoff evolution now covers the complete eight-coordinate two-triad model, while gate-level compilation of that complete model remains open. The resource calculation is a structured primitive-count analysis. Physical cubic-gate imperfections, viscous quantum evolution, fault-tolerant synthesis, and end-to-end speedup remain open.

## Data availability

The code and processed data supporting the study are included in this repository. Each archived file appears in `MANIFEST.csv` with its byte size and SHA-256 digest. Raw JHTDB velocity fields are third-party data and are available through JHTDB under its access and usage conditions. The repository records the dataset, time indices, sampling stride, acquisition scripts, and processed comparison outputs needed to repeat the external-data workflow.

## Citation

Please cite the associated manuscript and this repository if you use these materials. Repository citation metadata are provided in `CITATION.cff`; the journal citation can be added after publication.
