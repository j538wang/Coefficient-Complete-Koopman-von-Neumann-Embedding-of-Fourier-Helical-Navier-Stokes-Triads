#!/usr/bin/env python3
"""Fetch full-domain coarse velocity cubes from JHTDB for spectral helicity-transfer tests.

This version uses integer timepoint indices because the local givernylocal
getCutout housekeeping currently requires integer t_range values for cutouts.
Start by probing --time-index 512 or another valid integer timepoint, then fetch a full-domain coarse cube. For isotropic1024coarse, the local metadata may report valid timepoints [1, 5024].
"""
from __future__ import annotations
import argparse, os, time as _time
from pathlib import Path
import numpy as np

DNS_GRID_SIZE = 1024
DOMAIN_LENGTH = 2.0 * np.pi
# givernylocal/JHTDB cutout ranges are 1-based and inclusive for isotropic1024coarse.
# The full periodic coarse grid uses server indices 1, 1+stride, ..., 1+(N-1)*stride.
SERVER_INDEX_ORIGIN = 1


def _pick_velocity_dataarray(ds):
    """Return an xarray DataArray for velocity from a Dataset/DataArray-like object."""
    if hasattr(ds, "data_vars"):
        if "velocity" in ds.data_vars:
            return ds["velocity"]
        for name in ds.data_vars:
            if "velocity" in name.lower():
                return ds[name]
        return ds[list(ds.data_vars)[0]]
    return ds


def _extract_velocity_array(data) -> np.ndarray:
    """Convert givernylocal velocity cutout to shape (3, nx, ny, nz).

    Important: the local cutout API often returns the DataArray as
    (zcoor, ycoor, xcoor, values).  We use xarray dimension names when
    available and convert explicitly to (component, x, y, z).  This prevents
    silent x/z axis swaps in the assembled spectral cube.
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
        raise ValueError(f"Expected 4D velocity cutout after removing singleton time axis, got shape {arr.shape}")
    if arr.shape[-1] == 3:
        # givernylocal fallback convention: (z, y, x, c). Convert to (c, x, y, z).
        arr = np.transpose(arr, (3, 2, 1, 0))
    elif arr.shape[0] == 3:
        pass
    else:
        raise ValueError(f"Cannot identify velocity component axis in shape {arr.shape}")
    return np.asarray(arr, dtype=np.float64)

def _fetch_one_velocity_tile(cube, getCutout, time_index: int, stride: int,
                             x0: int, x1: int, y0: int, y1: int, z0: int, z1: int,
                             retries: int = 3, sleep: float = 2.0, verbose: bool = False) -> np.ndarray:
    # givernylocal expects integer ranges.  Endpoints are inclusive.
    xyzt_axes_ranges = np.array([
        [int(x0), int(x1)],
        [int(y0), int(y1)],
        [int(z0), int(z1)],
        [int(time_index), int(time_index)],
    ], dtype=np.int64)
    xyzt_strides = np.array([int(stride), int(stride), int(stride), 1], dtype=np.int64)

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            data = getCutout(cube, "velocity", xyzt_axes_ranges, xyzt_strides, verbose=verbose)
            return _extract_velocity_array(data)
        except Exception as exc:
            last_exc = exc
            print(f"  attempt {attempt}/{retries} failed: {exc!r}")
            if attempt < retries:
                _time.sleep(sleep)
    raise last_exc


def _probe(cube, getCutout, time_index: int) -> None:
    print("Probing a tiny 8^3 tile first...")
    tile = _fetch_one_velocity_tile(cube, getCutout, time_index, 1, SERVER_INDEX_ORIGIN, SERVER_INDEX_ORIGIN + 7, SERVER_INDEX_ORIGIN, SERVER_INDEX_ORIGIN + 7, SERVER_INDEX_ORIGIN, SERVER_INDEX_ORIGIN + 7, retries=1, verbose=True)
    print(f"Probe success, tiny tile shape: {tile.shape}")


def fetch_velocity_cutout(dataset: str, token: str, time_index: int, stride: int,
                          tile_size: int = 128, retries: int = 3) -> np.ndarray:
    try:
        from givernylocal.turbulence_dataset import turb_dataset  # type: ignore
        from givernylocal.turbulence_toolkit import getCutout  # type: ignore
    except Exception as exc:
        raise RuntimeError("Could not import givernylocal. Install with `pip install --upgrade givernylocal`.") from exc

    token = token.strip()
    cube = turb_dataset(dataset, os.getcwd(), token)
    _probe(cube, getCutout, time_index)

    if DNS_GRID_SIZE % stride != 0:
        raise ValueError(f"DNS_GRID_SIZE={DNS_GRID_SIZE} must be divisible by stride={stride}")
    if tile_size % stride != 0:
        raise ValueError(f"tile_size={tile_size} must be divisible by stride={stride}")
    if DNS_GRID_SIZE % tile_size != 0:
        raise ValueError(f"DNS_GRID_SIZE={DNS_GRID_SIZE} must be divisible by tile_size={tile_size}")

    expected = DNS_GRID_SIZE // stride
    out = np.empty((3, expected, expected, expected), dtype=np.float64)
    ntiles = DNS_GRID_SIZE // tile_size
    tile_points = tile_size // stride
    total_tiles = ntiles ** 3
    print(f"Fetching full-domain coarse cube in {total_tiles} tiles: dataset={dataset}, "
          f"time_index={time_index}, stride={stride}, tile_size={tile_size}")
    counter = 0
    for ix in range(ntiles):
        x0 = SERVER_INDEX_ORIGIN + ix * tile_size
        x1 = x0 + tile_size - stride
        ox0 = ix * tile_points; ox1 = ox0 + tile_points
        for iy in range(ntiles):
            y0 = SERVER_INDEX_ORIGIN + iy * tile_size
            y1 = y0 + tile_size - stride
            oy0 = iy * tile_points; oy1 = oy0 + tile_points
            for iz in range(ntiles):
                z0 = SERVER_INDEX_ORIGIN + iz * tile_size
                z1 = z0 + tile_size - stride
                oz0 = iz * tile_points; oz1 = oz0 + tile_points
                counter += 1
                print(f"Tile {counter}/{total_tiles}: x={x0}:{x1}:{stride}, y={y0}:{y1}:{stride}, z={z0}:{z1}:{stride}")
                try:
                    tile = _fetch_one_velocity_tile(cube, getCutout, time_index, stride, x0, x1, y0, y1, z0, z1, retries=retries)
                except Exception as exc:
                    raise RuntimeError(
                        f"JHTDB tile query failed for time_index={time_index}, "
                        f"x={x0}:{x1}:{stride}, y={y0}:{y1}:{stride}, z={z0}:{z1}:{stride}. "
                        "If probe succeeded but a tile fails, retry later, lower --tile-size, or use SciServer. "
                        f"Original error: {exc!r}"
                    ) from exc
                if tile.shape != (3, tile_points, tile_points, tile_points):
                    raise ValueError(f"Unexpected tile shape {tile.shape}; expected {(3, tile_points, tile_points, tile_points)}")
                out[:, ox0:ox1, oy0:oy1, oz0:oz1] = tile
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch full-domain coarse JHTDB velocity cube.")
    parser.add_argument("--dataset", default="isotropic1024coarse")
    parser.add_argument("--time-index", type=int, default=512, help="Integer JHTDB timepoint index. For isotropic1024coarse, valid range is usually [1, 5024]; use 512 for debug.")
    parser.add_argument("--stride", type=int, default=16, choices=[4, 8, 16, 32])
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--outdir", default="data_jhtdb")
    parser.add_argument("--token", default=os.environ.get("JHTDB_TOKEN", ""))
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    token = args.token.strip()
    if not token:
        raise SystemExit("Please provide --token or set environment variable JHTDB_TOKEN.")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    u = fetch_velocity_cutout(args.dataset, token, args.time_index, args.stride, args.tile_size, args.retries)
    out = outdir / f"{args.dataset}_t{args.time_index:06d}_s{args.stride}_velocity.npz"
    np.savez_compressed(out, u=u, dataset=args.dataset, time_index=args.time_index, stride=args.stride,
                        domain_length=DOMAIN_LENGTH, dns_grid_size=DNS_GRID_SIZE, server_index_origin=SERVER_INDEX_ORIGIN)
    print(f"wrote {out}")
    print(f"velocity shape: {u.shape}; file contains full periodic coarse cube")


if __name__ == "__main__":
    main()
