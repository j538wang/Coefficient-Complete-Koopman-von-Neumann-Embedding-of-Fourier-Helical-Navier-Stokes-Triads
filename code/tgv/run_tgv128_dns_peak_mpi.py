#!/usr/bin/env python3
"""
MPI pseudo-spectral Taylor--Green vortex benchmark at the DNS peak time.

This is a true MPI version of run_tgv128_dns_peak.py.  The dense TGV
evolution is distributed across MPI ranks using a slab/pencil-style
complex-to-complex FFT implemented with mpi4py Alltoallv:

  physical x-slabs  -> local FFT in (y,z) -> all-to-all column transpose
  -> FFT in x -> distributed spectral columns.

The script evolves the 128^3 Taylor--Green vortex to t=8.97, compares global
energy/dissipation/enstrophy histories with the public 512^3 DNS reference,
and exports sparse Fourier snapshots compatible with the helical RHS,
shell/helicity transfer, one-step, and truncation diagnostic scripts.

Only mpi4py is required for MPI parallelism; mpi4py-fft is not required.

Recommended Alliance execution:

  srun --mpi=pmix -n 32 python run_tgv128_dns_peak_mpi.py \
    --N 128 --dt 0.002 --t-final 8.97 --nu 0.000625 \
    --save-times 0,1,2,4,6,8,8.97 \
    --history-every 10 --mode-rel-tol 1e-10 --max-modes 2000 \
    --dns-csv dns_reference/hiocfd_ucl_512_global.csv \
    --out results_tgv128_t897_mpi
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mpi4py import MPI

Vector = Tuple[int, int, int]
EPS = 1e-14


def counts_and_displs(n: int, size: int) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.full(size, n // size, dtype=np.int64)
    counts[: n % size] += 1
    displs = np.zeros(size, dtype=np.int64)
    displs[1:] = np.cumsum(counts[:-1])
    return counts, displs


def fftfreq_int(N: int) -> np.ndarray:
    return np.fft.fftfreq(N, d=1.0 / N).astype(int)


def parse_list_floats(text: str) -> List[float]:
    if text is None or str(text).strip() == "":
        return []
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def time_tag(t: float) -> str:
    return f"t{t:08.3f}".replace(".", "p")


def shell_index_from_components(kx: int, ky: int, kz: int, width: float = 1.0) -> int:
    return int(math.floor(math.sqrt(kx*kx + ky*ky + kz*kz) / width + 1e-12))


class MPIFFT3D:
    """Distributed complex 3D FFT with physical x-slab and spectral-column layout."""

    def __init__(self, N: int, comm: MPI.Comm):
        self.N = int(N)
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.ks = fftfreq_int(N)
        self.x_counts, self.x_displs = counts_and_displs(N, self.size)
        self.M = N * N  # flattened (ky,kz) columns
        self.col_counts, self.col_displs = counts_and_displs(self.M, self.size)
        self.nx = int(self.x_counts[self.rank])
        self.x0 = int(self.x_displs[self.rank])
        self.ncol = int(self.col_counts[self.rank])
        self.c0 = int(self.col_displs[self.rank])
        self.x_indices = np.arange(self.x0, self.x0 + self.nx, dtype=np.int64)
        self.col_indices = np.arange(self.c0, self.c0 + self.ncol, dtype=np.int64)
        self.ky_indices = self.col_indices // N
        self.kz_indices = self.col_indices % N
        self.kx_vals = self.ks[:, None].astype(float)
        self.ky_vals = self.ks[self.ky_indices][None, :].astype(float)
        self.kz_vals = self.ks[self.kz_indices][None, :].astype(float)
        self.k2 = self.kx_vals**2 + self.ky_vals**2 + self.kz_vals**2
        self.zero_mask = self.k2 < EPS
        self.dtype_mpi = MPI.DOUBLE_COMPLEX

        # Precompute Alltoallv counts/displacements in complex numbers.
        self.f_sendcounts = np.asarray([self.nx * int(c) for c in self.col_counts], dtype=np.int32)
        self.f_sdispls = np.zeros(self.size, dtype=np.int32)
        self.f_sdispls[1:] = np.cumsum(self.f_sendcounts[:-1])
        self.f_recvcounts = np.asarray([int(xc) * self.ncol for xc in self.x_counts], dtype=np.int32)
        self.f_rdispls = np.zeros(self.size, dtype=np.int32)
        self.f_rdispls[1:] = np.cumsum(self.f_recvcounts[:-1])

        self.i_sendcounts = np.asarray([int(xc) * self.ncol for xc in self.x_counts], dtype=np.int32)
        self.i_sdispls = np.zeros(self.size, dtype=np.int32)
        self.i_sdispls[1:] = np.cumsum(self.i_sendcounts[:-1])
        self.i_recvcounts = np.asarray([self.nx * int(cc) for cc in self.col_counts], dtype=np.int32)
        self.i_rdispls = np.zeros(self.size, dtype=np.int32)
        self.i_rdispls[1:] = np.cumsum(self.i_recvcounts[:-1])

    def forward_scalar(self, phys_x: np.ndarray) -> np.ndarray:
        """physical x-slab -> normalized spectral columns."""
        N = self.N
        if phys_x.shape != (self.nx, N, N):
            raise ValueError(f"phys_x shape {phys_x.shape} incompatible with {(self.nx,N,N)}")
        yz = np.fft.fft2(phys_x, axes=(1, 2)).reshape(self.nx, self.M)
        send_chunks = []
        for r in range(self.size):
            c0 = int(self.col_displs[r]); c1 = c0 + int(self.col_counts[r])
            send_chunks.append(np.ascontiguousarray(yz[:, c0:c1]).ravel())
        sendbuf = np.concatenate(send_chunks) if send_chunks else np.empty(0, dtype=np.complex128)
        recvbuf = np.empty(int(np.sum(self.f_recvcounts)), dtype=np.complex128)
        self.comm.Alltoallv(
            [sendbuf, self.f_sendcounts, self.f_sdispls, self.dtype_mpi],
            [recvbuf, self.f_recvcounts, self.f_rdispls, self.dtype_mpi],
        )
        pencil = np.empty((N, self.ncol), dtype=np.complex128)
        off = 0
        for r in range(self.size):
            nr = int(self.x_counts[r])
            cnt = nr * self.ncol
            block = recvbuf[off:off+cnt].reshape(nr, self.ncol)
            x0 = int(self.x_displs[r])
            pencil[x0:x0+nr, :] = block
            off += cnt
        spec = np.fft.fft(pencil, axis=0) / (N**3)
        return spec

    def backward_scalar(self, spec_col: np.ndarray) -> np.ndarray:
        """normalized spectral columns -> physical x-slab."""
        N = self.N
        if spec_col.shape != (N, self.ncol):
            raise ValueError(f"spec_col shape {spec_col.shape} incompatible with {(N,self.ncol)}")
        pencil = np.fft.ifft(spec_col * (N**3), axis=0)
        send_chunks = []
        for r in range(self.size):
            x0 = int(self.x_displs[r]); x1 = x0 + int(self.x_counts[r])
            send_chunks.append(np.ascontiguousarray(pencil[x0:x1, :]).ravel())
        sendbuf = np.concatenate(send_chunks) if send_chunks else np.empty(0, dtype=np.complex128)
        recvbuf = np.empty(int(np.sum(self.i_recvcounts)), dtype=np.complex128)
        self.comm.Alltoallv(
            [sendbuf, self.i_sendcounts, self.i_sdispls, self.dtype_mpi],
            [recvbuf, self.i_recvcounts, self.i_rdispls, self.dtype_mpi],
        )
        yz = np.empty((self.nx, self.M), dtype=np.complex128)
        off = 0
        for r in range(self.size):
            nc = int(self.col_counts[r])
            cnt = self.nx * nc
            block = recvbuf[off:off+cnt].reshape(self.nx, nc)
            c0 = int(self.col_displs[r])
            yz[:, c0:c0+nc] = block
            off += cnt
        phys = np.fft.ifft2(yz.reshape(self.nx, N, N), axes=(1, 2))
        return phys

    def dealias_mask(self, frac: float = 2.0/3.0) -> np.ndarray:
        cutoff = frac * (self.N // 2)
        return (np.abs(self.kx_vals) <= cutoff) & (np.abs(self.ky_vals) <= cutoff) & (np.abs(self.kz_vals) <= cutoff)


def project_spec(grid: MPIFFT3D, uhat: List[np.ndarray]) -> List[np.ndarray]:
    k2 = grid.k2
    dot = grid.kx_vals * uhat[0] + grid.ky_vals * uhat[1] + grid.kz_vals * uhat[2]
    out = [arr.copy() for arr in uhat]
    mask = k2 > EPS
    factor = np.zeros_like(dot)
    factor[mask] = dot[mask] / k2[mask]
    out[0] -= grid.kx_vals * factor
    out[1] -= grid.ky_vals * factor
    out[2] -= grid.kz_vals * factor
    for c in range(3):
        out[c][~mask] = 0.0
    return out


def enforce_dealias_spec(grid: MPIFFT3D, uhat: List[np.ndarray]) -> List[np.ndarray]:
    mask = grid.dealias_mask()
    out = [arr.copy() for arr in uhat]
    for c in range(3):
        out[c][~mask] = 0.0
    return out


def copy_spec(uhat: List[np.ndarray]) -> List[np.ndarray]:
    return [a.copy() for a in uhat]


def add_spec(a: List[np.ndarray], b: List[np.ndarray], scale: float = 1.0) -> List[np.ndarray]:
    return [a[i] + scale * b[i] for i in range(3)]


def lincomb_spec(terms: List[Tuple[float, List[np.ndarray]]]) -> List[np.ndarray]:
    out = [np.zeros_like(terms[0][1][i]) for i in range(3)]
    for coeff, arr in terms:
        for i in range(3):
            out[i] += coeff * arr[i]
    return out


def pseudospectral_rhs_mpi(grid: MPIFFT3D, uhat_in: List[np.ndarray], nu: float, dealias: bool = True) -> List[np.ndarray]:
    uhat = copy_spec(uhat_in)
    if dealias:
        uhat = enforce_dealias_spec(grid, uhat)
    uhat = project_spec(grid, uhat)

    # Physical velocities on local x-slab.
    u_phys = [grid.backward_scalar(uhat[c]) for c in range(3)]

    nhat = []
    k_components = [grid.kx_vals, grid.ky_vals, grid.kz_vals]
    for i in range(3):
        conv_i = np.zeros((grid.nx, grid.N, grid.N), dtype=np.complex128)
        for j in range(3):
            grad_ij = grid.backward_scalar(1j * k_components[j] * uhat[i])
            conv_i += u_phys[j] * grad_ij
        nhat_i = -grid.forward_scalar(conv_i)
        nhat.append(nhat_i)

    if dealias:
        nhat = enforce_dealias_spec(grid, nhat)
    nhat = project_spec(grid, nhat)
    if nu != 0.0:
        for c in range(3):
            nhat[c] -= nu * grid.k2 * uhat[c]
    return nhat


def rk4_step_mpi(grid: MPIFFT3D, uhat: List[np.ndarray], dt: float, nu: float, dealias: bool = True) -> List[np.ndarray]:
    k1 = pseudospectral_rhs_mpi(grid, uhat, nu, dealias)
    k2 = pseudospectral_rhs_mpi(grid, add_spec(uhat, k1, 0.5*dt), nu, dealias)
    k3 = pseudospectral_rhs_mpi(grid, add_spec(uhat, k2, 0.5*dt), nu, dealias)
    k4 = pseudospectral_rhs_mpi(grid, add_spec(uhat, k3, dt), nu, dealias)
    out = [uhat[i] + (dt/6.0)*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(3)]
    if dealias:
        out = enforce_dealias_spec(grid, out)
    return project_spec(grid, out)


def tgv_initial_spec_mpi(grid: MPIFFT3D, amplitude: float = 1.0) -> List[np.ndarray]:
    N = grid.N
    x = 2.0 * np.pi * grid.x_indices / N
    y = 2.0 * np.pi * np.arange(N) / N
    z = 2.0 * np.pi * np.arange(N) / N
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    u0 = amplitude * np.sin(X) * np.cos(Y) * np.cos(Z)
    u1 = -amplitude * np.cos(X) * np.sin(Y) * np.cos(Z)
    u2 = np.zeros_like(u0)
    return [grid.forward_scalar(u0), grid.forward_scalar(u1), grid.forward_scalar(u2)]


def kinetic_energy_mpi(grid: MPIFFT3D, uhat: List[np.ndarray]) -> float:
    local = 0.5 * sum(float(np.sum(np.abs(uhat[c])**2)) for c in range(3))
    return float(grid.comm.allreduce(local, op=MPI.SUM))


def enstrophy_mpi(grid: MPIFFT3D, uhat: List[np.ndarray]) -> float:
    modal = np.zeros_like(grid.k2, dtype=float)
    for c in range(3):
        modal += np.abs(uhat[c])**2
    local = 0.5 * float(np.sum(grid.k2 * modal))
    return float(grid.comm.allreduce(local, op=MPI.SUM))


def active_mode_count_mpi(grid: MPIFFT3D, uhat: List[np.ndarray], rel_tol: float = 1e-12) -> int:
    mag = np.sqrt(sum(np.abs(uhat[c])**2 for c in range(3)))
    local_max = float(np.max(mag)) if mag.size else 0.0
    global_max = float(grid.comm.allreduce(local_max, op=MPI.MAX))
    tol = rel_tol * global_max
    local_count = int(np.count_nonzero(mag > tol))
    # Exclude zero mode if this rank owns column (ky,kz)=(0,0).
    if grid.c0 <= 0 < grid.c0 + grid.ncol:
        loc = 0 - grid.c0
        if mag[0, loc] > tol:
            local_count -= 1
    return int(grid.comm.allreduce(local_count, op=MPI.SUM))


def dense_energy_spectrum_mpi(grid: MPIFFT3D, uhat: List[np.ndarray], shell_width: float = 1.0) -> Dict[int, float]:
    max_shell = int(math.ceil(math.sqrt(3.0) * (grid.N // 2) / shell_width)) + 2
    local = np.zeros(max_shell, dtype=float)
    modal = 0.5 * sum(np.abs(uhat[c])**2 for c in range(3))
    for ix, kx in enumerate(grid.ks):
        for jl, col in enumerate(grid.col_indices):
            iy = int(col // grid.N); iz = int(col % grid.N)
            ky = int(grid.ks[iy]); kz = int(grid.ks[iz])
            if kx == 0 and ky == 0 and kz == 0:
                continue
            sh = shell_index_from_components(int(kx), ky, kz, shell_width)
            if sh < max_shell:
                local[sh] += float(modal[ix, jl].real)
    global_arr = np.zeros_like(local)
    grid.comm.Allreduce(local, global_arr, op=MPI.SUM)
    return {int(i): float(v) for i, v in enumerate(global_arr) if v > 0.0}


def transversality_error_mpi(grid: MPIFFT3D, uhat: List[np.ndarray]) -> float:
    mag = np.sqrt(sum(np.abs(uhat[c])**2 for c in range(3)))
    dot = grid.kx_vals*uhat[0] + grid.ky_vals*uhat[1] + grid.kz_vals*uhat[2]
    local_num = float(np.sum(np.abs(dot)**2))
    local_den = float(np.sum((grid.k2 + EPS) * mag**2))
    num = grid.comm.allreduce(local_num, op=MPI.SUM)
    den = grid.comm.allreduce(local_den, op=MPI.SUM)
    return float(math.sqrt(num / (den + 1e-300)))


def save_json_root(path: Path, data: dict, rank: int) -> None:
    if rank == 0:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)


def find_dns_csv(path_text: str | None) -> Path | None:
    if path_text and path_text.lower() != "auto":
        p = Path(path_text)
        return p if p.exists() else None
    candidates = [
        Path("dns_reference/hiocfd_ucl_512_global.csv"),
        Path("public_TGV_DNS_reference_csv/hiocfd_ucl_512_global.csv"),
        Path("hiocfd_ucl_512_global.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def read_dns_reference(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    needed = {"t", "energy", "dissipation", "enstrophy"}
    if not needed.issubset(df.columns):
        raise ValueError(f"DNS CSV must contain {needed}; got {df.columns.tolist()}")
    return df[["t", "energy", "dissipation", "enstrophy"]].copy()


def interpolate_dns(dns: pd.DataFrame, times: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"time": times})
    for col in ["energy", "dissipation", "enstrophy"]:
        out[f"dns_{col}"] = np.interp(times, dns["t"].to_numpy(), dns[col].to_numpy())
    return out


def save_spectrum_plot(spec: Dict[int, float], path: Path, title: str) -> None:
    rows = sorted(spec.items())
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    if rows:
        ax.semilogy([r[0] for r in rows], [r[1] for r in rows], marker="o", linestyle="-")
    ax.set_xlabel("wavenumber shell")
    ax.set_ylabel("modal kinetic energy")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.30)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def local_sparse_entries(grid: MPIFFT3D, uhat: List[np.ndarray], rel_tol: float) -> Tuple[List[Tuple[float, Vector, List[complex]]], float]:
    mag = np.sqrt(sum(np.abs(uhat[c])**2 for c in range(3)))
    local_max = float(np.max(mag)) if mag.size else 0.0
    global_max = float(grid.comm.allreduce(local_max, op=MPI.MAX))
    tol = rel_tol * global_max
    entries: List[Tuple[float, Vector, List[complex]]] = []
    for ix, kx in enumerate(grid.ks):
        for jl, col in enumerate(grid.col_indices):
            if mag[ix, jl] <= tol:
                continue
            iy = int(col // grid.N); iz = int(col % grid.N)
            ky = int(grid.ks[iy]); kz = int(grid.ks[iz])
            if kx == 0 and ky == 0 and kz == 0:
                continue
            vals = [complex(uhat[c][ix, jl]) for c in range(3)]
            entries.append((float(mag[ix, jl]), (int(kx), ky, kz), vals))
    return entries, global_max


def enforce_sparse_conjugate(modes: Dict[Vector, np.ndarray]) -> Dict[Vector, np.ndarray]:
    out = dict(modes)
    for k, u in list(modes.items()):
        kn = (-k[0], -k[1], -k[2])
        if kn not in out:
            out[kn] = np.conjugate(u)
    return out


def save_snapshot_mpi(grid: MPIFFT3D, uhat: List[np.ndarray], t: float, args, out: Path, final_alias: bool = False) -> Path | None:
    entries, global_max = local_sparse_entries(grid, uhat, args.mode_rel_tol)
    gathered = grid.comm.gather(entries, root=0)
    if grid.rank != 0:
        return None

    flat = [item for sub in gathered for item in sub]
    flat.sort(key=lambda x: x[0], reverse=True)
    if args.max_modes and args.max_modes > 0:
        flat = flat[:args.max_modes]
    modes: Dict[Vector, np.ndarray] = {k: np.asarray(vals, dtype=np.complex128) for _, k, vals in flat}
    if args.enforce_sparse_conjugate:
        modes = enforce_sparse_conjugate(modes)

    keys = np.asarray(list(modes.keys()), dtype=int)
    vals = np.stack([modes[tuple(k)] for k in keys], axis=0) if len(keys) else np.zeros((0, 3), dtype=np.complex128)
    energy_sparse = 0.5 * float(np.sum(np.abs(vals)**2)) if len(vals) else 0.0
    metadata = {
        "case": "tgv128_peak_dns_context_mpi",
        "time": float(t),
        "N": int(args.N),
        "dt": float(args.dt),
        "t_final": float(args.t_final),
        "nu": float(args.nu),
        "Re": float(1.0 / args.nu) if args.nu != 0 else None,
        "mpi_size": int(grid.size),
        "mode_rel_tol": float(args.mode_rel_tol),
        "max_modes_requested": int(args.max_modes),
        "global_max_mode_magnitude": float(global_max),
        "retained_modes": int(len(modes)),
        "retained_sparse_energy": energy_sparse,
        "conjugate_completion_applied": bool(args.enforce_sparse_conjugate),
    }
    tag = time_tag(t)
    path = out / f"tgv_sparse_modes_{tag}.npz"
    np.savez_compressed(path, keys=keys, vals=vals, metadata=json.dumps(metadata))

    # Sparse spectrum from retained modes.
    spec: Dict[int, float] = {}
    for k, u in modes.items():
        sh = shell_index_from_components(k[0], k[1], k[2])
        spec[sh] = spec.get(sh, 0.0) + 0.5 * float(np.vdot(u, u).real)
    pd.DataFrame([{"shell": k, "energy": v} for k, v in sorted(spec.items())]).to_csv(
        out / f"tgv_sparse_energy_spectrum_{tag}.csv", index=False
    )
    save_spectrum_plot(spec, out / f"tgv_sparse_energy_spectrum_{tag}.png", f"Sparse TGV spectrum at t={t:g}")

    if final_alias:
        shutil.copyfile(path, out / "tgv_peak_sparse_modes.npz")
        shutil.copyfile(path, out / "tgv_short_time_sparse_modes.npz")
    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--dt", type=float, default=0.002)
    p.add_argument("--t-final", type=float, default=8.97)
    p.add_argument("--nu", type=float, default=0.000625)
    p.add_argument("--amplitude", type=float, default=1.0)
    p.add_argument("--save-times", type=str, default="0,1,2,4,6,8,8.97")
    p.add_argument("--history-every", type=int, default=10)
    p.add_argument("--mode-rel-tol", type=float, default=1e-10)
    p.add_argument("--max-modes", type=int, default=2000)
    p.add_argument("--dense-active-rel-tol", type=float, default=1e-12)
    p.add_argument("--dns-csv", type=str, default="auto")
    p.add_argument("--out", type=str, default="results_tgv128_t897_mpi")
    p.add_argument("--no-dealias", action="store_true")
    p.add_argument("--no-enforce-sparse-conjugate", dest="enforce_sparse_conjugate", action="store_false")
    p.set_defaults(enforce_sparse_conjugate=True)
    return p.parse_args()


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    args = parse_args()
    dealias = not args.no_dealias
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    steps_float = args.t_final / args.dt
    steps = int(round(steps_float))
    if abs(steps * args.dt - args.t_final) > 1e-10:
        raise ValueError(f"t-final={args.t_final} is not an integer multiple of dt={args.dt}; got {steps_float}")

    save_times = sorted(set([t for t in parse_list_floats(args.save_times) if -1e-12 <= t <= args.t_final + 1e-12]))
    save_steps: Dict[int, float] = {}
    for t in save_times:
        n = int(round(t / args.dt))
        if abs(n * args.dt - t) > max(1e-10, 0.25 * args.dt):
            raise ValueError(f"save time {t} is not compatible with dt={args.dt}")
        save_steps[n] = float(n * args.dt)
    save_steps[steps] = float(args.t_final)

    grid = MPIFFT3D(args.N, comm)
    if rank == 0:
        print("MPI TGV peak benchmark")
        print(f"  ranks={grid.size}, N={args.N}, dt={args.dt}, t_final={args.t_final}, steps={steps}, nu={args.nu}")
        print(f"  local x counts: {grid.x_counts.tolist()}")
        print(f"  local column counts: {grid.col_counts.tolist()}")
        print(f"  dealias={dealias}, sparse rel_tol={args.mode_rel_tol}, max_modes={args.max_modes}")
        print(f"  save steps: {save_steps}", flush=True)

    uhat = tgv_initial_spec_mpi(grid, amplitude=args.amplitude)
    rows = []
    snapshot_paths: List[str] = []
    t0 = time.time()

    def record(step: int, t: float, runtime_so_far: float) -> None:
        E = kinetic_energy_mpi(grid, uhat)
        Om = enstrophy_mpi(grid, uhat)
        eps = 2.0 * args.nu * Om
        active = active_mode_count_mpi(grid, uhat, rel_tol=args.dense_active_rel_tol)
        if rank == 0:
            rows.append({
                "step": step,
                "time": t,
                "energy": E,
                "dissipation": eps,
                "enstrophy": Om,
                "dense_active_modes": active,
                "runtime_seconds": runtime_so_far,
            })

    record(0, 0.0, 0.0)
    if 0 in save_steps:
        p = save_snapshot_mpi(grid, uhat, 0.0, args, out, final_alias=(steps == 0))
        if rank == 0 and p is not None:
            snapshot_paths.append(str(p))

    for n in range(1, steps + 1):
        uhat = rk4_step_mpi(grid, uhat, args.dt, args.nu, dealias=dealias)
        t = n * args.dt
        if n % args.history_every == 0 or n == steps or n in save_steps:
            record(n, t, time.time() - t0)
        if n in save_steps:
            if rank == 0:
                print(f"  saving sparse snapshot at t={t:.5f}, step={n}", flush=True)
            p = save_snapshot_mpi(grid, uhat, t, args, out, final_alias=(n == steps))
            if rank == 0 and p is not None:
                snapshot_paths.append(str(p))
        if n % max(1, steps // 20) == 0 or n == steps:
            E = kinetic_energy_mpi(grid, uhat)
            eps = 2.0 * args.nu * enstrophy_mpi(grid, uhat)
            if rank == 0:
                print(f"  step {n:6d}/{steps}, t={t:8.4f}, E={E:.10e}, eps={eps:.10e}", flush=True)

    runtime = time.time() - t0
    final_E = kinetic_energy_mpi(grid, uhat)
    final_Om = enstrophy_mpi(grid, uhat)
    final_eps = 2.0 * args.nu * final_Om
    final_active = active_mode_count_mpi(grid, uhat, rel_tol=args.dense_active_rel_tol)
    trans_err = transversality_error_mpi(grid, uhat)
    dense_spec = dense_energy_spectrum_mpi(grid, uhat)

    if rank == 0:
        history = pd.DataFrame(rows).drop_duplicates(subset=["step"], keep="last").sort_values("step")
        history.to_csv(out / "tgv128_global_history.csv", index=False)
        pd.DataFrame([{"shell": k, "energy": v} for k, v in sorted(dense_spec.items())]).to_csv(
            out / "tgv128_dense_energy_spectrum_final.csv", index=False
        )
        save_spectrum_plot(dense_spec, out / "tgv128_dense_energy_spectrum_final.png", f"Dense MPI TGV spectrum at t={args.t_final:g}")

        dns_summary = {}
        dns_path = find_dns_csv(args.dns_csv)
        if dns_path is not None:
            print(f"  loading DNS reference: {dns_path}", flush=True)
            dns = read_dns_reference(dns_path)
            dns_peak_idx = int(np.argmax(dns["dissipation"].to_numpy()))
            dns_peak_time = float(dns.iloc[dns_peak_idx]["t"])
            dns_peak_eps = float(dns.iloc[dns_peak_idx]["dissipation"])
            dns_interp = interpolate_dns(dns, history["time"].to_numpy())
            merged = pd.concat([history.reset_index(drop=True), dns_interp.drop(columns=["time"])], axis=1)
            for col in ["energy", "dissipation", "enstrophy"]:
                merged[f"relerr_{col}"] = np.abs(merged[col] - merged[f"dns_{col}"]) / (np.abs(merged[f"dns_{col}"]) + 1e-300)
            merged.to_csv(out / "tgv128_vs_dns_global.csv", index=False)

            for col, ylabel in [("energy", "kinetic energy"), ("dissipation", "dissipation"), ("enstrophy", "enstrophy")]:
                fig, ax = plt.subplots(figsize=(7.0, 4.5))
                ax.plot(dns["t"], dns[col], linestyle="-", linewidth=1.6, label="public $512^3$ DNS")
                ax.plot(history["time"], history[col], linestyle="--", linewidth=1.8, label=f"present MPI ${args.N}^3$ run")
                if col == "dissipation":
                    ax.axvline(dns_peak_time, color="0.3", linestyle=":", linewidth=1.2, label=f"DNS peak $t={dns_peak_time:.2f}$")
                else:
                    ax.axvline(dns_peak_time, color="0.3", linestyle=":", linewidth=1.2)
                ax.set_xlabel("time")
                ax.set_ylabel(ylabel)
                ax.set_title(f"TGV {ylabel}: MPI run vs DNS reference")
                ax.grid(True, alpha=0.30)
                ax.legend(frameon=True)
                fig.tight_layout()
                fig.savefig(out / f"tgv128_vs_dns_{col}.png", dpi=240)
                plt.close(fig)

            fig, ax = plt.subplots(figsize=(7.0, 4.5))
            for col in ["energy", "dissipation", "enstrophy"]:
                ax.semilogy(merged["time"], merged[f"relerr_{col}"], marker="o", markersize=2.0, linestyle="-", label=col)
            ax.axvline(dns_peak_time, color="0.3", linestyle=":", linewidth=1.2, label=f"DNS peak $t={dns_peak_time:.2f}$")
            ax.set_xlabel("time")
            ax.set_ylabel("relative difference from DNS")
            ax.set_title("Global TGV relative differences against DNS reference")
            ax.grid(True, which="both", alpha=0.30)
            ax.legend(frameon=True)
            fig.tight_layout()
            fig.savefig(out / "tgv128_vs_dns_relative_errors.png", dpi=240)
            plt.close(fig)

            final_row = merged.iloc[-1]
            dns_summary = {
                "dns_csv": str(dns_path),
                "dns_peak_time": dns_peak_time,
                "dns_peak_dissipation": dns_peak_eps,
                "final_relerr_energy": float(final_row["relerr_energy"]),
                "final_relerr_dissipation": float(final_row["relerr_dissipation"]),
                "final_relerr_enstrophy": float(final_row["relerr_enstrophy"]),
                "final_dns_energy": float(final_row["dns_energy"]),
                "final_dns_dissipation": float(final_row["dns_dissipation"]),
                "final_dns_enstrophy": float(final_row["dns_enstrophy"]),
            }
        else:
            print("  DNS reference not found; skipping DNS comparison.", flush=True)

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.plot(history["time"], history["energy"], label="energy")
        ax.set_xlabel("time")
        ax.set_ylabel("kinetic energy")
        ax.set_title(f"Present MPI TGV run, N={args.N}")
        ax.grid(True, alpha=0.30)
        fig.tight_layout()
        fig.savefig(out / "tgv128_energy_history.png", dpi=240)
        plt.close(fig)

        summary = {
            "case": "tgv128_peak_dns_benchmark_mpi",
            "N": int(args.N),
            "dt": float(args.dt),
            "t_final": float(args.t_final),
            "nu": float(args.nu),
            "Re": float(1.0/args.nu) if args.nu != 0 else None,
            "dealias": bool(dealias),
            "steps": int(steps),
            "mpi_size": int(grid.size),
            "runtime_seconds": float(runtime),
            "final_energy": float(final_E),
            "final_dissipation": float(final_eps),
            "final_enstrophy": float(final_Om),
            "final_dense_active_modes": int(final_active),
            "final_transversality_error": float(trans_err),
            "sparse_snapshot_paths": snapshot_paths,
            **dns_summary,
        }
        with open(out / "tgv128_peak_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"Saved results to {out}", flush=True)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
