# Code provenance and evidence map

The core scripts and matching output directories were recovered from the working tree that produced the numerical values in the revised manuscript. The TGV and JHTDB components were recovered from their dedicated project directories. Selection was based on agreement between machine-readable summaries and the manuscript values. Earlier exploratory outputs with different active-case counts or different truncation parameters were excluded.

- `phase_gauge_triad_validation.py` and `make_unified_ensemble_figure.py` support the 6400-case deterministic coefficient ensemble and its compact summary.
- `global_scaling_compatibility_audit.py` supports the 252-node shared-scale graph, least-squares residual, parallel-edge audit, and analytic two-edge obstruction.
- `kvn_global_graph_audit.py` supports the 504-real-coordinate generator, 74,112 combined terms, full-field reconstruction residual, and energy and helicity commutator audits.
- `kvn_shared_mode_realization_audit.py` provides a smaller shared-mode realization check.
- `kvn_two_triad_gate_validation.py` supports the four-coordinate finite-Fock calculation, collective-cubic polarization residuals, cutoff and amplitude scans, and second-order Strang fit.
- `full_spectral_resource_scaling.py` supports the exact integer-lattice enumeration through cutoff 6 and the structured pseudo-spectral primitive counts.
- `code/tgv` and `processed/tgv` support the peak-state full-field comparison and active-triad magnitude-truncation calculation.
- `code/jhtdb` and `processed/jhtdb` support acquisition, helical reconstruction, and averaging for the eight processed external fields.

The SHA-256 manifest permits byte-level verification of the distributed repository. Historical input paths have been normalized to repository-relative locations, and every corresponding input required for the archived audits is included under `processed`.
