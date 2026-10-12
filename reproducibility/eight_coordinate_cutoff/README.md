# Full eight-coordinate total-number cutoff scan

This directory contains a new reviewer-response calculation for the complete
four-complex-amplitude (eight-real-coordinate) two-triad network.  It does not
modify or overwrite the manuscript's existing four-coordinate benchmark.

The script expands the gauge-real complex equations into 24 real quadratic
terms, constructs the divergence-free cubic KvN generator, and projects each
complete monomial onto the global total-number space `sum(n_j) <= N`.  The
Hilbert-space dimension is `C(N+8,8)`, not a local tensor-product dimension.

The default initial magnitudes match the existing benchmark, while fixed
nonzero phases `(0.20, -0.35, 0.45, -0.25)` move the state off the all-real
invariant subspace.  These eight classical coordinates also define one common
untruncated product coherent state with real displacements
`alpha_a = q_a(0)/sqrt(2)` and zero conjugate-momentum means.  At every cutoff,
the code applies `P_N` to this same untruncated state and renormalizes the
result; it does not choose a new state that merely shares the same first
moments.  Each summary JSON records the displacements and the probability
retained by the initial projection.  The scan reports the probability in the last two total
number shells at nine sampled output times, norm conservation, classical
invariant checks, and first-moment agreement with the corresponding full
eight-coordinate classical trajectory.  The reported maximum boundary
population is therefore the maximum over those sampled times, not a certified
continuous-time supremum.

Example:

```text
python run_full8_cutoff.py --cutoffs 6,8,10,12,14,16 --out .
python verify_projection_convention.py --cutoff 2 --out projection_verification.json
python plot_cutoff_trend.py --results-dir . --output-stem eight_coordinate_cutoff_trend
```

Each cutoff writes an independent JSON summary and trajectory CSV so an
interrupted scan can be resumed one cutoff at a time.  `cutoff_summary.csv`
and `run_manifest.json` aggregate only completed cases.  The scan uses the
complete-product convention `P_N (p_a q_b q_c) P_N`.  The independent
`verify_projection_convention.py` test constructs the same operators in a
larger tensor-product space before restriction and also demonstrates that
multiplying separately projected factors gives a different boundary result.

`environment.txt` records the tested software versions.  The `N=16` run is a
finite numerical reference; the outputs do not claim convergence to the
infinite-cutoff limit.

The lower-left panel of the generated trend figure uses the complete
first-moment trajectory difference
`relative_mean_trajectory_difference_vs_reference`, matching the manuscript's
definition of `Delta_m^(16)`.  The zero differences at `N=16` are omitted from
the logarithmic lower panels and described in the figure caption.
