#!/usr/bin/env python3
"""
Common utilities for extended Fourier--helical Navier--Stokes nonlinear RHS
benchmarks.

Conventions
-----------
* Periodic domain [0, 2*pi)^3.
* Fourier coefficients are normalized as u_hat = fftn(u) / N^3.
* The inverse transform is u = ifftn(u_hat * N^3).
* Wavevectors are integer indices compatible with this domain.

The utilities here are intentionally explicit rather than maximally optimized.
They are designed for validation studies where the retained active spectral
support is thresholded to O(10^2--10^3) modes, not for production DNS.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
import json
import math
import time

import numpy as np

Vector = Tuple[int, int, int]
EPS = 1e-14


def v_add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_neg(a: Vector) -> Vector:
    return (-a[0], -a[1], -a[2])


def v_to_array(v: Vector) -> np.ndarray:
    return np.asarray(v, dtype=float)


def vec_norm(v: Vector) -> float:
    return float(np.linalg.norm(v_to_array(v)))


def shell_index(v: Vector, width: float = 1.0) -> int:
    """Integer shell index based on |k|."""
    return int(math.floor(vec_norm(v) / width + 1e-12))


def project_incompressible(k: Vector, u: np.ndarray) -> np.ndarray:
    karr = v_to_array(k)
    kk = float(np.dot(karr, karr))
    if kk < EPS:
        return np.zeros(3, dtype=complex)
    return u - karr * (np.dot(karr, u) / kk)


def make_perp_unit(k: np.ndarray) -> np.ndarray:
    k_norm = np.linalg.norm(k)
    if k_norm < EPS:
        raise ValueError("zero wavevector has no transverse basis")
    khat = k / k_norm
    axes = np.eye(3)
    idx = int(np.argmin(np.abs(axes @ khat)))
    a = axes[idx]
    e1 = np.cross(khat, a)
    n = np.linalg.norm(e1)
    if n < EPS:
        a = axes[(idx + 1) % 3]
        e1 = np.cross(khat, a)
        n = np.linalg.norm(e1)
    return e1 / n


def helical_basis(k: Vector, s: int) -> np.ndarray:
    """Return h_s(k), satisfying i k x h_s = s |k| h_s and k.h_s=0."""
    karr = v_to_array(k)
    kn = np.linalg.norm(karr)
    if kn < EPS:
        raise ValueError("zero wavevector has no helical basis")
    khat = karr / kn
    e1 = make_perp_unit(karr)
    e2 = np.cross(khat, e1)
    e2 = e2 / np.linalg.norm(e2)
    return (e1 + 1j * s * e2) / np.sqrt(2.0)


def decompose_helical(k: Vector, u: np.ndarray) -> Tuple[complex, complex]:
    """Return (u_+(k), u_-(k)) for a transverse vector u(k)."""
    hp = helical_basis(k, +1)
    hm = helical_basis(k, -1)
    return np.vdot(hp, u), np.vdot(hm, u)


def reconstruct_from_helical(k: Vector, up: complex, um: complex) -> np.ndarray:
    return up * helical_basis(k, +1) + um * helical_basis(k, -1)


def fftfreq_int(N: int) -> np.ndarray:
    """Integer Fourier wavevector ordering for numpy FFT on [0,2*pi)."""
    return np.fft.fftfreq(N, d=1.0 / N).astype(int)


def make_k_arrays(N: int):
    ks = fftfreq_int(N)
    kx, ky, kz = np.meshgrid(ks, ks, ks, indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    return ks, kx, ky, kz, k2


def dealias_mask(N: int, frac: float = 2.0 / 3.0) -> np.ndarray:
    ks, kx, ky, kz, _ = make_k_arrays(N)
    kmax = N // 2
    cutoff = frac * kmax
    return (np.abs(kx) <= cutoff) & (np.abs(ky) <= cutoff) & (np.abs(kz) <= cutoff)


def tgv_initial_uhat(N: int, amplitude: float = 1.0) -> np.ndarray:
    """Taylor--Green vortex on [0,2*pi)^3, normalized Fourier coefficients."""
    x = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u = np.zeros((N, N, N, 3), dtype=float)
    u[..., 0] = amplitude * np.sin(X) * np.cos(Y) * np.cos(Z)
    u[..., 1] = -amplitude * np.cos(X) * np.sin(Y) * np.cos(Z)
    u[..., 2] = 0.0
    uhat = np.empty((N, N, N, 3), dtype=complex)
    for c in range(3):
        uhat[..., c] = np.fft.fftn(u[..., c]) / (N ** 3)
    return uhat


def kinetic_energy(uhat: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(uhat) ** 2))


def project_uhat(uhat: np.ndarray) -> np.ndarray:
    N = uhat.shape[0]
    _, kx, ky, kz, k2 = make_k_arrays(N)
    out = uhat.copy()
    kk = k2.astype(float)
    mask = kk > 0
    dot = kx * out[..., 0] + ky * out[..., 1] + kz * out[..., 2]
    for c, kc in enumerate([kx, ky, kz]):
        out[..., c][mask] = out[..., c][mask] - kc[mask] * dot[mask] / kk[mask]
    out[~mask, :] = 0.0
    return out


def enforce_dealias(uhat: np.ndarray, frac: float = 2.0 / 3.0) -> np.ndarray:
    mask = dealias_mask(uhat.shape[0], frac)
    out = uhat.copy()
    out[~mask, :] = 0.0
    return out


def physical_from_uhat(uhat: np.ndarray) -> np.ndarray:
    N = uhat.shape[0]
    u = np.empty_like(uhat, dtype=complex)
    for c in range(3):
        u[..., c] = np.fft.ifftn(uhat[..., c] * (N ** 3))
    return u


def pseudospectral_rhs(uhat: np.ndarray, nu: float = 0.0, dealias: bool = True) -> np.ndarray:
    """Return d u_hat / dt = -P[(u.grad)u] - nu k^2 u_hat."""
    N = uhat.shape[0]
    if dealias:
        uhat = enforce_dealias(uhat)
    uhat = project_uhat(uhat)
    ks, kx, ky, kz, k2 = make_k_arrays(N)
    u_phys = physical_from_uhat(uhat)
    # Derivatives in physical space: du_i/dx_j.
    grad = np.empty((N, N, N, 3, 3), dtype=complex)
    for i in range(3):
        for j, kj in enumerate([kx, ky, kz]):
            grad[..., i, j] = np.fft.ifftn((1j * kj * uhat[..., i]) * (N ** 3))
    conv = np.zeros((N, N, N, 3), dtype=complex)
    for i in range(3):
        conv[..., i] = u_phys[..., 0] * grad[..., i, 0] + u_phys[..., 1] * grad[..., i, 1] + u_phys[..., 2] * grad[..., i, 2]
    nhat = np.empty_like(uhat)
    for i in range(3):
        nhat[..., i] = -np.fft.fftn(conv[..., i]) / (N ** 3)
    if dealias:
        mask = dealias_mask(N)
        nhat[~mask, :] = 0.0
    nhat = project_uhat(nhat)
    if nu != 0.0:
        nhat = nhat - nu * k2[..., None] * uhat
    return nhat


def rk4_step(uhat: np.ndarray, dt: float, nu: float, dealias: bool = True) -> np.ndarray:
    k1 = pseudospectral_rhs(uhat, nu=nu, dealias=dealias)
    k2 = pseudospectral_rhs(uhat + 0.5 * dt * k1, nu=nu, dealias=dealias)
    k3 = pseudospectral_rhs(uhat + 0.5 * dt * k2, nu=nu, dealias=dealias)
    k4 = pseudospectral_rhs(uhat + dt * k3, nu=nu, dealias=dealias)
    out = uhat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    if dealias:
        out = enforce_dealias(out)
    out = project_uhat(out)
    return out


def uhat_to_modes(uhat: np.ndarray, rel_tol: float = 1e-10, abs_tol: float = 0.0,
                  max_modes: Optional[int] = None, enforce_conjugate: bool = False) -> Dict[Vector, np.ndarray]:
    """Convert dense uhat[N,N,N,3] to a sparse dict of active Fourier modes."""
    N = uhat.shape[0]
    ks = fftfreq_int(N)
    mag = np.linalg.norm(uhat, axis=-1)
    max_mag = float(np.max(mag))
    tol = max(abs_tol, rel_tol * max_mag)
    idx = np.argwhere(mag > tol)
    entries: List[Tuple[float, Vector, np.ndarray]] = []
    for i, j, k in idx:
        v = (int(ks[i]), int(ks[j]), int(ks[k]))
        if v == (0, 0, 0):
            continue
        entries.append((float(mag[i, j, k]), v, uhat[i, j, k, :].astype(complex)))
    entries.sort(key=lambda x: x[0], reverse=True)
    if max_modes is not None and max_modes > 0 and len(entries) > max_modes:
        entries = entries[:max_modes]
    modes = {v: project_incompressible(v, u) for _, v, u in entries}
    if enforce_conjugate:
        # Add conjugate partners if missing. This may exceed max_modes slightly.
        for v, u in list(modes.items()):
            nv = v_neg(v)
            if nv not in modes:
                modes[nv] = np.conjugate(u)
    return modes


def modes_to_arrays(modes: Dict[Vector, np.ndarray]):
    keys = list(modes.keys())
    U = np.stack([modes[k] for k in keys], axis=0) if keys else np.zeros((0,3), complex)
    return keys, U


def modes_energy(modes: Dict[Vector, np.ndarray]) -> float:
    return 0.5 * float(sum(np.vdot(u, u).real for u in modes.values()))


def transversality_error(modes: Dict[Vector, np.ndarray]) -> float:
    num = 0.0
    den = 0.0
    for k, u in modes.items():
        num += abs(np.dot(v_to_array(k), u)) ** 2
        den += np.vdot(u, u).real
    return float(np.sqrt(num) / (np.sqrt(den) + 1e-300))


def conjugate_symmetry_error(modes: Dict[Vector, np.ndarray]) -> float:
    num = 0.0
    den = 0.0
    seen = set()
    for k, u in modes.items():
        if k in seen:
            continue
        nk = v_neg(k)
        v = modes.get(nk, np.zeros(3, dtype=complex))
        diff = v - np.conjugate(u)
        num += np.vdot(diff, diff).real
        den += np.vdot(u, u).real + np.vdot(v, v).real
        seen.add(k); seen.add(nk)
    return float(np.sqrt(num) / (np.sqrt(den) + 1e-300))


def cartesian_pair_rhs(modes: Dict[Vector, np.ndarray]) -> Tuple[Dict[Vector, np.ndarray], int]:
    """Explicit projected Fourier convolution over the retained sparse modes."""
    keys = list(modes.keys())
    rhs: Dict[Vector, np.ndarray] = {}
    count = 0
    for p in keys:
        up = modes[p]
        for q in keys:
            k = v_add(p, q)
            if k == (0,0,0):
                continue
            uq = modes[q]
            scalar = 1j * np.dot(v_to_array(q), up)
            contrib = -scalar * uq
            contrib = project_incompressible(k, contrib)
            rhs[k] = rhs.get(k, np.zeros(3, dtype=complex)) + contrib
            count += 1
    return rhs, count


def helical_pair_rhs(modes: Dict[Vector, np.ndarray], store_contributions: bool = False,
                     contrib_tol: float = 0.0):
    """Helical decomposition/reconstruction of the explicit sparse convolution.

    If store_contributions=True, returns a list of records
    (k,p,q,sk,sp,sq,dotus,contrib_abs) where dotus is the scalar output
    helical amplitude RHS contribution. This can be large; use with thresholded
    active mode sets.
    """
    keys = list(modes.keys())
    # Precompute bases and amplitudes.
    basis: Dict[Tuple[Vector,int], np.ndarray] = {}
    amp: Dict[Tuple[Vector,int], complex] = {}
    for k in keys:
        for s in (+1, -1):
            h = helical_basis(k, s)
            basis[(k,s)] = h
            amp[(k,s)] = np.vdot(h, modes[k])
    rhs: Dict[Vector, np.ndarray] = {}
    records = [] if store_contributions else None
    count = 0
    for p in keys:
        qarr_cache = None
        for q in keys:
            k = v_add(p, q)
            if k == (0,0,0):
                continue
            qarr = v_to_array(q)
            for sp in (+1, -1):
                up = amp[(p,sp)]
                if abs(up) <= contrib_tol:
                    continue
                vp = up * basis[(p,sp)]
                qdotvp = np.dot(qarr, vp)
                for sq in (+1, -1):
                    uq = amp[(q,sq)]
                    if abs(up * uq) <= contrib_tol:
                        continue
                    vq = uq * basis[(q,sq)]
                    raw = -(1j * qdotvp) * vq
                    raw = project_incompressible(k, raw)
                    if np.linalg.norm(raw) <= contrib_tol:
                        continue
                    for sk in (+1, -1):
                        hk = helical_basis(k, sk)
                        dotus = np.vdot(hk, raw)
                        vec = dotus * hk
                        rhs[k] = rhs.get(k, np.zeros(3, dtype=complex)) + vec
                        count += 1
                        if store_contributions and abs(dotus) > contrib_tol:
                            records.append((k, p, q, sk, sp, sq, dotus, abs(dotus)))
    return rhs, count, records


def dict_l2_norm(modes: Dict[Vector, np.ndarray]) -> float:
    return float(np.sqrt(sum(np.vdot(u, u).real for u in modes.values())))


def dict_difference_norm(a: Dict[Vector, np.ndarray], b: Dict[Vector, np.ndarray]) -> float:
    keys = set(a.keys()) | set(b.keys())
    s = 0.0
    z = np.zeros(3, dtype=complex)
    for k in keys:
        d = a.get(k, z) - b.get(k, z)
        s += np.vdot(d, d).real
    return float(np.sqrt(s))


def relative_error(a: Dict[Vector, np.ndarray], b: Dict[Vector, np.ndarray]) -> float:
    return dict_difference_norm(a, b) / (dict_l2_norm(b) + 1e-300)


def max_modewise_relative_error(a: Dict[Vector, np.ndarray], b: Dict[Vector, np.ndarray], floor: float = 1e-14) -> float:
    vals = []
    z = np.zeros(3, dtype=complex)
    for k in set(a.keys()) | set(b.keys()):
        nb = np.linalg.norm(b.get(k, z))
        if nb > floor:
            vals.append(np.linalg.norm(a.get(k, z) - b.get(k, z)) / nb)
    return float(max(vals)) if vals else 0.0


def shell_energy_transfer(uhat_modes: Dict[Vector, np.ndarray], rhs_modes: Dict[Vector, np.ndarray], shell_width: float = 1.0) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for k, u in uhat_modes.items():
        if k in rhs_modes:
            sh = shell_index(k, shell_width)
            out[sh] = out.get(sh, 0.0) + float(np.vdot(u, rhs_modes[k]).real)
    return out


def shell_helicity_transfer(uhat_modes: Dict[Vector, np.ndarray], rhs_modes: Dict[Vector, np.ndarray], shell_width: float = 1.0) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for k, u in uhat_modes.items():
        if k not in rhs_modes:
            continue
        kn = vec_norm(k)
        if kn < EPS:
            continue
        rhs = rhs_modes[k]
        for s in (+1, -1):
            h = helical_basis(k, s)
            us = np.vdot(h, u)
            dus = np.vdot(h, rhs)
            sh = shell_index(k, shell_width)
            out[sh] = out.get(sh, 0.0) + float(s * kn * np.real(np.conjugate(us) * dus))
    return out


def save_modes_npz(path: str | Path, modes: Dict[Vector, np.ndarray], metadata: Optional[dict] = None) -> None:
    path = Path(path)
    keys = np.asarray(list(modes.keys()), dtype=int)
    vals = np.stack([modes[tuple(k)] for k in keys], axis=0) if len(keys) else np.zeros((0,3), complex)
    np.savez_compressed(path, keys=keys, vals=vals, metadata=json.dumps(metadata or {}))


def load_modes_npz(path: str | Path) -> Tuple[Dict[Vector, np.ndarray], dict]:
    data = np.load(path, allow_pickle=False)
    keys = data["keys"]
    vals = data["vals"]
    metadata = json.loads(str(data.get("metadata", "{}")))
    modes = {tuple(map(int, k)): vals[i].astype(complex) for i, k in enumerate(keys)}
    return modes, metadata


def save_json(path: str | Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def energy_spectrum_by_shell(modes: Dict[Vector, np.ndarray], shell_width: float = 1.0) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for k,u in modes.items():
        sh = shell_index(k, shell_width)
        out[sh] = out.get(sh, 0.0) + 0.5 * float(np.vdot(u,u).real)
    return out

# -----------------------------------------------------------------------------
# Dense spectral global diagnostics for Taylor--Green vortex evolution
# -----------------------------------------------------------------------------

def enstrophy_dense(uhat: np.ndarray) -> float:
    """Return enstrophy Omega = 0.5 * sum_k |omega_hat(k)|^2.

    For a projected divergence-free Fourier field this is equivalent to
    0.5 * sum_k |k|^2 |u_hat(k)|^2.  This convention matches the public TGV
    reference in which epsilon = 2 * nu * Omega.
    """
    N = uhat.shape[0]
    _, _, _, _, k2 = make_k_arrays(N)
    modal_u2 = np.sum(np.abs(uhat) ** 2, axis=-1)
    return 0.5 * float(np.sum(k2 * modal_u2))


def dissipation_dense(uhat: np.ndarray, nu: float) -> float:
    """Return kinetic-energy dissipation epsilon = 2 nu Omega."""
    return 2.0 * float(nu) * enstrophy_dense(uhat)


def dense_energy_spectrum_by_shell(uhat: np.ndarray, shell_width: float = 1.0) -> Dict[int, float]:
    """Energy spectrum by integer shell from a dense Fourier array."""
    N = uhat.shape[0]
    ks = fftfreq_int(N)
    out: Dict[int, float] = {}
    for i, kx in enumerate(ks):
        for j, ky in enumerate(ks):
            for l, kz in enumerate(ks):
                k = (int(kx), int(ky), int(kz))
                if k == (0, 0, 0):
                    continue
                e = 0.5 * float(np.vdot(uhat[i, j, l, :], uhat[i, j, l, :]).real)
                if e <= 0.0:
                    continue
                sh = shell_index(k, width=shell_width)
                out[sh] = out.get(sh, 0.0) + e
    return out


def dense_active_mode_count(uhat: np.ndarray, rel_tol: float = 1e-12, abs_tol: float = 0.0) -> int:
    """Count dense Fourier modes whose vector norm exceeds a threshold."""
    mag = np.linalg.norm(uhat, axis=-1)
    tol = max(abs_tol, rel_tol * float(np.max(mag)))
    # Exclude zero mode if present.
    count = int(np.count_nonzero(mag > tol))
    N = uhat.shape[0]
    if mag[0, 0, 0] > tol:
        count -= 1
    return count
