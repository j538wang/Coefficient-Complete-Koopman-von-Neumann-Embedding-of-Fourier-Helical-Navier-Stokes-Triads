# JHTDB helicity benchmark package v10

This package extends v9 with an explicit **helical-channel reconstruction** test.

## Existing v9 steps

1. Download JHTDB full-domain coarse velocity cubes:

```bash
python 01_fetch_jhtdb_full_domain_coarse.py --dataset isotropic1024coarse --time-index 512 --stride 16 --tile-size 128 --retries 5 --outdir data_jhtdb
```

2. Compute conservative DNS-derived reference transfer:

```bash
python 02_compute_helicity_flux_from_npz.py --input data_jhtdb/isotropic1024coarse_t000512_s16_velocity.npz --outdir results_jhtdb_v9 --mode galerkin23 --kplot-max 32
```

## New v10 step: helical-channel reconstruction

For a single snapshot:

```bash
python 04_helical_channel_reconstruction.py --input data_jhtdb/isotropic1024coarse_t000512_s16_velocity.npz --outdir results_jhtdb_helical --mode galerkin23 --kplot-max 32
```

This computes the 8 helical channels

```text
r;pq, r,p,q in {+,-}
```

and checks that their sum reproduces the pseudo-spectral reference RHS and shell transfers.

## Multi-time average

In a notebook:

```python
import glob
for f in sorted(glob.glob("data_jhtdb/*_s16_velocity.npz")):
    !python 04_helical_channel_reconstruction.py --input "$f" --outdir results_jhtdb_helical --mode galerkin23 --kplot-max 32
```

Then average the total comparisons:

```python
import glob, subprocess
csv_files = sorted(glob.glob("results_jhtdb_helical/*_helical_galerkin23_total_compare.csv"))
subprocess.run(["python", "05_average_helical_reconstruction.py", "--inputs", *csv_files, "--outdir", "results_jhtdb_helical_average", "--kplot-max", "32"], check=True)
```

## Interpretation

This is a standard-DNS helicity-transfer reconstruction benchmark. It verifies that the helicity-resolved channel construction reproduces the DNS-derived nonlinear transfer on the JHTDB forced-HIT dataset. It should not be described as a fully helical cascade validation, because `isotropic1024coarse` is not a fully helically forced dataset.
