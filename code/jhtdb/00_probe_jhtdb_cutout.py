#!/usr/bin/env python3
"""Probe JHTDB/Giverny getCutout and print the returned data structure.

This script requests a tiny velocity cutout and writes a JSON report with:
  - getCutout signature
  - xarray dimensions and coordinates
  - raw shape
  - interpreted velocity-array shape after conversion to (3, x, y, z)

Use this before downloading many tiles.
"""
from __future__ import annotations
import argparse, os, inspect, json
from pathlib import Path
import numpy as np


def _pick_velocity_dataarray(ds):
    """Return an xarray DataArray for velocity from a Dataset/DataArray-like object."""
    if hasattr(ds, "data_vars"):
        if "velocity" in ds.data_vars:
            return ds["velocity"]
        # giverny often names the returned variable velocity_XXXX
        for name in ds.data_vars:
            if "velocity" in name.lower():
                return ds[name]
        first = list(ds.data_vars)[0]
        return ds[first]
    return ds


def velocity_to_cxyz(data) -> np.ndarray:
    """Convert givernylocal velocity cutout to array shape (3, nx, ny, nz).

    givernylocal local cutouts are commonly returned as xarray Dataset with a
    DataArray ordered as (zcoor, ycoor, xcoor, values).  This function uses dim
    names when available and falls back to that ordering for plain arrays.
    """
    da = _pick_velocity_dataarray(data)

    if hasattr(da, "dims") and hasattr(da, "transpose"):
        dims = list(da.dims)
        lower = [str(d).lower() for d in dims]
        try:
            xdim = dims[next(i for i, d in enumerate(lower) if d.startswith("x") or "xcoor" in d)]
            ydim = dims[next(i for i, d in enumerate(lower) if d.startswith("y") or "ycoor" in d)]
            zdim = dims[next(i for i, d in enumerate(lower) if d.startswith("z") or "zcoor" in d)]
        except StopIteration as exc:
            raise ValueError(f"Could not identify x/y/z dimensions from dims={dims}") from exc
        # remaining dimension should be vector component/value dimension.
        comp_dims = [d for d in dims if d not in (xdim, ydim, zdim)]
        if len(comp_dims) != 1:
            raise ValueError(f"Expected one component dimension in dims={dims}, got {comp_dims}")
        cdim = comp_dims[0]
        arr = da.transpose(cdim, xdim, ydim, zdim).to_numpy()
        if arr.shape[0] != 3:
            raise ValueError(f"Expected 3 velocity components after transpose, got shape={arr.shape}")
        return np.asarray(arr, dtype=np.float64)

    arr = np.asarray(da)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D array after removing time axis, got shape={arr.shape}")
    if arr.shape[-1] == 3:
        # givernylocal fallback convention: (z, y, x, c). Convert to (c, x, y, z).
        arr = np.transpose(arr, (3, 2, 1, 0))
    elif arr.shape[0] == 3:
        # Assume already (c, x, y, z).
        pass
    else:
        raise ValueError(f"Cannot identify component axis in shape={arr.shape}")
    return np.asarray(arr, dtype=np.float64)


def summarize_xarray(data):
    report = {"object_type": str(type(data))}
    if hasattr(data, "data_vars"):
        report["attrs"] = {str(k): str(v) for k, v in data.attrs.items()}
        report["coords"] = {str(k): {"shape": list(np.asarray(v).shape), "first": float(np.asarray(v).ravel()[0]) if np.asarray(v).size else None,
                                      "last": float(np.asarray(v).ravel()[-1]) if np.asarray(v).size else None}
                              for k, v in data.coords.items()}
        report["data_vars"] = {}
        for name, var in data.data_vars.items():
            report["data_vars"][str(name)] = {
                "dims": [str(d) for d in var.dims],
                "shape": list(var.shape),
                "dtype": str(var.dtype),
            }
    elif hasattr(data, "dims"):
        report["dims"] = [str(d) for d in data.dims]
        report["shape"] = list(data.shape)
    elif hasattr(data, "shape"):
        report["shape"] = list(data.shape)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="isotropic1024coarse")
    ap.add_argument("--token", default=os.environ.get("JHTDB_TOKEN", ""))
    ap.add_argument("--time-index", type=int, default=512,
                    help="integer JHTDB timepoint index. For isotropic1024coarse, your givernylocal metadata reports valid range [1, 5024]; use 512 for debug.")
    ap.add_argument("--x0", type=int, default=1, help="JHTDB spatial start index. givernylocal cutout indices are 1-based for this dataset; use 1 by default.")
    ap.add_argument("--y0", type=int, default=1, help="JHTDB spatial start index; use 1 by default.")
    ap.add_argument("--z0", type=int, default=1, help="JHTDB spatial start index; use 1 by default.")
    ap.add_argument("--n", type=int, default=8, help="number of contiguous DNS points in each direction")
    ap.add_argument("--out", default="jhtdb_probe_structure.json")
    args = ap.parse_args()
    token = args.token.strip()
    if not token:
        raise SystemExit("Set JHTDB_TOKEN or pass --token")

    from givernylocal.turbulence_dataset import turb_dataset
    from givernylocal.turbulence_toolkit import getCutout
    signature = str(inspect.signature(getCutout))
    print("getCutout signature:", signature)
    cube = turb_dataset(args.dataset, os.getcwd(), token)

    xyzt_axes_ranges = np.array([
        [args.x0, args.x0 + args.n - 1],
        [args.y0, args.y0 + args.n - 1],
        [args.z0, args.z0 + args.n - 1],
        [args.time_index, args.time_index],
    ], dtype=np.int64)
    xyzt_strides = np.array([1, 1, 1, 1], dtype=np.int64)
    print("dataset:", args.dataset)
    print("var: velocity")
    print("xyzt_axes_ranges:", xyzt_axes_ranges)
    print("xyzt_strides:", xyzt_strides)

    data = getCutout(cube, "velocity", xyzt_axes_ranges, xyzt_strides, verbose=True)
    converted = velocity_to_cxyz(data)
    report = summarize_xarray(data)
    report["getCutout_signature"] = signature
    report["converted_velocity_shape_cxyz"] = list(converted.shape)
    report["converted_velocity_minmax"] = [float(np.nanmin(converted)), float(np.nanmax(converted))]
    report["xyzt_axes_ranges"] = xyzt_axes_ranges.tolist()
    report["xyzt_strides"] = xyzt_strides.tolist()

    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("SUCCESS: wrote", out)
    print(json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    main()
