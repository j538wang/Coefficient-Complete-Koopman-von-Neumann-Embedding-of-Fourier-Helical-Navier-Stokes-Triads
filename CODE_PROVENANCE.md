# Code provenance and evidence map

The core scripts and matching output directories were recovered from the working tree that produced the numerical values in the revised manuscript. The TGV and JHTDB components were recovered from their dedicated project directories. Selection was based on agreement between machine-readable summaries and the manuscript values. Earlier exploratory outputs with different active-case counts or different truncation parameters were excluded.

- `phase_gauge_triad_validation.py` and `make_unified_ensemble_figure.py` support the 6400-case deterministic coefficient ensemble and its compact summary.
- `global_scaling_compatibility_audit.py` supports both the 252-node diagnostic-subset graph and the complete 342-node, 60,000-constraint shared-scale audit, including the least-squares residual, parallel-edge audit, and analytic two-edge obstruction.
- `deduplicate_reality_reduced_triads.py` reduces the three created-mode orientations of each complete $K_{\max}=3$ closed support to one deterministic reality-reduced representative. The retained CSV contains 4116 geometries and 32,928 helicity-resolved rows.
- `kvn_global_graph_audit.py` supports the diagnostic-subset assembly and the complete 684-real-coordinate generator with 734,176 combined terms, full-field reconstruction residual, and energy and helicity commutator audits.
- `kvn_shared_mode_realization_audit.py` provides a smaller shared-mode realization check.
- `kvn_two_triad_gate_validation.py` supports the four-coordinate finite-Fock calculation, collective-cubic polarization residuals, explicit passive Gaussian map--single-mode-cubic--inverse-map circuits, local cutoff and amplitude scans, gate counts, cubic strengths, and the second-order Strang fit.
- `kvn_gaussian_total_number_validation.py` supplies the independent passive-invariant cutoff audit through total occupation 16. `make_gaussian_bplus_figure.py` plots the two cutoff schemes and one- through eight-step results.
- `full_spectral_resource_scaling.py` supports the exact integer-lattice enumeration through cutoff 6, the complete-$K_{\max}=3$ monomial-density calibration, and the structured pseudo-spectral primitive counts.
- `code/tgv` and `processed/tgv` support the peak-state full-field comparison and active-triad magnitude-truncation calculation.
- `code/jhtdb` and `processed/jhtdb` support acquisition, helical reconstruction, and averaging for the eight processed external fields.

The SHA-256 manifest permits byte-level verification of the distributed repository. Historical input paths have been normalized to repository-relative locations. The 121 MB pre-deduplication coefficient CSV can be regenerated deterministically and exceeds GitHub's single-file limit; the archive includes its generation and deduplication scripts, the complete retained CSV used downstream, and all reported audit outputs.
