#!/usr/bin/env python3
"""
phase_gauge_triad_validation.py

Fourth-round validation for the Spectral Triadic Transfer Quantum Map (STTQM).

Goal
----
The third-round ensemble scan showed that:
  1. projected helical Navier--Stokes triad coefficients satisfy both the
     conservative energy constraint C_K + C_P^* + C_Q^* = 0 and the
     helicity constraint s_K |K| C_K + s_P |P| C_P^* + s_Q |Q| C_Q^* = 0
     to machine precision;
  2. the simplest single-g canonical three-wave Hamiltonian is generally too
     restrictive;
  3. a common-phase real metric embedding works only for a subset of triads,
     while many triads are classified as complex_metric under the previous test.

This fourth-round code tests whether the complex_metric classification is merely
caused by arbitrary helical-mode phase conventions.  It applies a phase-gauge
transformation to the three retained helical amplitudes,

    u_K = exp(i theta_K) y_K,
    u_P = exp(i theta_P) y_P,
    u_Q = exp(i theta_Q) y_Q,

which transforms the reduced triad coefficients as

    C_K' = C_K exp(i delta),
    C_P' = C_P exp(-i delta),
    C_Q' = C_Q exp(-i delta),

where delta = theta_P + theta_Q - theta_K.  Therefore only one effective phase
parameter is relevant for the coefficient embedding problem.

For each active triad and helicity combination, the script scans delta and finds
whether the transformed coefficient triple admits a common-phase real metric form

    C_m' = -i lambda eta_m,  eta_m real.

The key diagnostics are:
  - baseline metric class before phase gauge;
  - optimized gauge residual after phase gauge;
  - whether phase gauge converts complex_metric cases into real-metric cases;
  - whether the remaining obstruction is gauge-invariant, especially the phase
    non-collinearity between C_P and C_Q.

Usage
-----
    python phase_gauge_triad_validation.py --out phase_gauge_results

Outputs
-------
    phase_gauge_results/
      summary.txt
      phase_gauge_ensemble.csv
      phase_gauge_counts.csv
      phase_gauge_by_helicity.csv
      baseline_vs_gauge_metric_counts.csv
      cp_cq_collinearity_by_helicity.csv
      conservation_residual_statistics.csv
      energy_helicity_residual_hist.png
      energy_vs_helicity_residual_scatter.png
      baseline_vs_gauge_eta_im_rel.png
      gauge_eta_im_rel_hist.png
      cp_cq_collinearity_hist.png
      phase_gauge_counts.png
      phase_gauge_by_helicity.png
      baseline_vs_gauge_scatter.png

Notes
-----
This code deliberately avoids assuming a specific closed-form Waleffe coefficient
convention.  It extracts each coefficient directly from the projected Fourier
convolution and then projects back onto the selected helical polarization.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Vector = Tuple[int, int, int]

EPS = 1e-14
SIGNATURE_ZERO_REL_TOL = 1e-12


# -----------------------------------------------------------------------------
# Basic vector and helical-basis utilities
# -----------------------------------------------------------------------------

def v_to_array(v: Vector) -> np.ndarray:
    return np.asarray(v, dtype=float)


def v_add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_neg(a: Vector) -> Vector:
    return (-a[0], -a[1], -a[2])


def norm(v: Vector) -> float:
    return float(np.linalg.norm(v_to_array(v)))


def canonical_vec(v: Vector) -> Vector:
    """Return a deterministic representative of the pair {v, -v}."""
    if v == (0, 0, 0):
        return v
    for x in v:
        if x > 0:
            return v
        if x < 0:
            return v_neg(v)
    return v


def make_perp_unit(k: np.ndarray) -> np.ndarray:
    """Construct a real unit vector perpendicular to k."""
    k_norm = np.linalg.norm(k)
    if k_norm < EPS:
        raise ValueError("zero wavevector")
    k_hat = k / k_norm
    # Pick the coordinate axis least aligned with k to avoid near-zero cross product.
    axes = np.eye(3)
    idx = int(np.argmin(np.abs(axes @ k_hat)))
    a = axes[idx]
    e1 = np.cross(k_hat, a)
    e1_norm = np.linalg.norm(e1)
    if e1_norm < EPS:
        # Fallback should rarely be needed.
        a = axes[(idx + 1) % 3]
        e1 = np.cross(k_hat, a)
        e1_norm = np.linalg.norm(e1)
    return e1 / e1_norm


def helical_basis(k_vec: Vector, s: int) -> np.ndarray:
    """
    Return a normalized complex helical basis vector h_s(k) satisfying

        i k x h_s(k) = s |k| h_s(k),
        k . h_s(k) = 0,
        h_s(k)^* . h_s(k) = 1.

    The phase convention is deterministic but arbitrary; phase-gauge validation
    explicitly tests the effect of such conventions.
    """
    k = v_to_array(k_vec)
    k_norm = np.linalg.norm(k)
    if k_norm < EPS:
        raise ValueError("zero wavevector")
    k_hat = k / k_norm
    e1 = make_perp_unit(k)
    e2 = np.cross(k_hat, e1)
    e2 = e2 / np.linalg.norm(e2)
    # With this convention, i k x h_s = s |k| h_s.
    return (e1 + 1j * s * e2) / np.sqrt(2.0)


def basis_diagnostics(k_vec: Vector, s: int) -> Dict[str, float]:
    h = helical_basis(k_vec, s)
    k = v_to_array(k_vec)
    k_norm = np.linalg.norm(k)
    transverse = abs(np.dot(k, h))
    eigen = np.linalg.norm(1j * np.cross(k, h) - s * k_norm * h)
    nrm = abs(np.vdot(h, h) - 1.0)
    return {
        "transverse_error": float(transverse),
        "eigen_error": float(eigen),
        "norm_error": float(nrm),
    }


# -----------------------------------------------------------------------------
# Projected Fourier convolution and helical coefficient extraction
# -----------------------------------------------------------------------------

def project_incompressible(k_vec: Vector, u: np.ndarray) -> np.ndarray:
    k = v_to_array(k_vec)
    kk = float(np.dot(k, k))
    if kk < EPS:
        return np.zeros(3, dtype=complex)
    P = np.eye(3) - np.outer(k, k) / kk
    return P @ u


def add_mode(modes: Dict[Vector, np.ndarray], k: Vector, amp: complex, h: np.ndarray) -> None:
    modes[k] = modes.get(k, np.zeros(3, dtype=complex)) + amp * h


def build_real_modes(K: Vector, P: Vector, Q: Vector,
                     sK: int, sP: int, sQ: int,
                     aK: complex, aP: complex, aQ: complex) -> Dict[Vector, np.ndarray]:
    """
    Build a real-valued Fourier field from three selected helical amplitudes and
    their conjugate modes.  We use h_s(-k) through the same deterministic basis
    convention, and enforce u(-k)=u(k)^* exactly by insertion.
    """
    modes: Dict[Vector, np.ndarray] = {}
    for k, s, a in [(K, sK, aK), (P, sP, aP), (Q, sQ, aQ)]:
        h = helical_basis(k, s)
        add_mode(modes, k, a, h)
        modes[v_neg(k)] = np.conjugate(modes[k])
    return modes


def nonlinear_rhs_modes(modes: Dict[Vector, np.ndarray]) -> Dict[Vector, np.ndarray]:
    """Compute N(k) = -P_k sum_{a+b=k} i b_j u_j(a) u(b)."""
    out: Dict[Vector, np.ndarray] = {}
    keys = list(modes.keys())
    for a in keys:
        ua = modes[a]
        for b in keys:
            ub = modes[b]
            k = v_add(a, b)
            if k == (0, 0, 0):
                continue
            b_arr = v_to_array(b)
            scalar = 1j * np.dot(b_arr, ua)
            contrib = -scalar * ub
            out[k] = out.get(k, np.zeros(3, dtype=complex)) + contrib
    for k in list(out.keys()):
        out[k] = project_incompressible(k, out[k])
    return out


def amplitude_rhs(K: Vector, P: Vector, Q: Vector,
                  sK: int, sP: int, sQ: int,
                  aK: complex, aP: complex, aQ: complex) -> Tuple[complex, complex, complex]:
    modes = build_real_modes(K, P, Q, sK, sP, sQ, aK, aP, aQ)
    rhs = nonlinear_rhs_modes(modes)
    hK = helical_basis(K, sK)
    hP = helical_basis(P, sP)
    hQ = helical_basis(Q, sQ)
    NK = np.vdot(hK, rhs.get(K, np.zeros(3, dtype=complex)))
    NP = np.vdot(hP, rhs.get(P, np.zeros(3, dtype=complex)))
    NQ = np.vdot(hQ, rhs.get(Q, np.zeros(3, dtype=complex)))
    return NK, NP, NQ


def extract_coefficients(K: Vector, P: Vector, Q: Vector,
                         sK: int, sP: int, sQ: int) -> Tuple[complex, complex, complex]:
    """
    Extract coefficients for the reduced triad convention

        d u_K = C_K u_P u_Q,
        d u_P = C_P u_K u_Q^*,
        d u_Q = C_Q u_K u_P^*.

    Each coefficient is isolated by activating only the two amplitudes appearing
    in the corresponding product.
    """
    CK, _, _ = amplitude_rhs(K, P, Q, sK, sP, sQ, 0.0, 1.0, 1.0)
    _, CP, _ = amplitude_rhs(K, P, Q, sK, sP, sQ, 1.0, 0.0, 1.0)
    _, _, CQ = amplitude_rhs(K, P, Q, sK, sP, sQ, 1.0, 1.0, 0.0)
    return CK, CP, CQ


# -----------------------------------------------------------------------------
# Embedding diagnostics
# -----------------------------------------------------------------------------

def fit_single_g(CK: complex, CP: complex, CQ: complex) -> Tuple[complex, np.ndarray, float, float]:
    """
    Fit the single-g canonical three-wave model:

        C_K^H = -i g,
        C_P^H = -i g^*,
        C_Q^H = -i g^*.
    """
    A = np.array([
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [-1.0, 0.0],
    ], dtype=float)
    b = np.array([CK.real, CK.imag, CP.real, CP.imag, CQ.real, CQ.imag], dtype=float)
    x_y, *_ = np.linalg.lstsq(A, b, rcond=None)
    g = complex(x_y[0], x_y[1])
    Cpred = np.array([-1j * g, -1j * np.conjugate(g), -1j * np.conjugate(g)], dtype=complex)
    C = np.array([CK, CP, CQ], dtype=complex)
    res = float(np.linalg.norm(Cpred - C))
    rel = res / (float(np.linalg.norm(C)) + 1e-300)
    return g, Cpred, res, rel


def common_phase_metric(C: np.ndarray) -> Dict[str, object]:
    """
    Test the common-phase real metric form:

        C_m = -i lambda eta_m, eta_m real.

    We compute eta_raw = i C_m and test whether all eta_raw components can be
    rotated by a common phase into real numbers.
    """
    eta_raw = 1j * np.asarray(C, dtype=complex)
    nrm = float(np.linalg.norm(eta_raw))
    if nrm < 1e-300:
        return {
            "lambda_phase": 1.0 + 0j,
            "eta_real": np.zeros(3),
            "eta_rot": eta_raw,
            "eta_im_rel": 0.0,
            "reconstruction_res": 0.0,
            "metric_class": "singular",
            "sign_pattern": "000",
        }
    idx = int(np.argmax(np.abs(eta_raw)))
    phase = np.angle(eta_raw[idx])
    lam = np.exp(1j * phase)
    eta_rot = eta_raw * np.exp(-1j * phase)
    eta_real = np.real(eta_rot)
    eta_im_rel = float(np.max(np.abs(np.imag(eta_rot))) / (np.linalg.norm(eta_rot) + 1e-300))
    C_recon = -1j * lam * eta_real
    rec_res = float(np.linalg.norm(C_recon - C))

    # Remove the arbitrary factorization scale before testing for a zero weight.
    # This makes the singular classification invariant under
    # (lambda, eta) -> (a lambda, eta/a).
    eta_scale = float(np.linalg.norm(eta_real))
    eta_unit = eta_real / max(eta_scale, 1e-300)
    signs = []
    for val in eta_unit:
        if val > SIGNATURE_ZERO_REL_TOL:
            signs.append("+")
        elif val < -SIGNATURE_ZERO_REL_TOL:
            signs.append("-")
        else:
            signs.append("0")
    sign_pattern = "".join(signs)

    if eta_im_rel > 1e-8:
        metric_class = "complex_metric"
    elif "0" in signs:
        metric_class = "singular"
    elif all(s == "+" for s in signs):
        metric_class = "positive_definite"
    elif all(s == "-" for s in signs):
        metric_class = "negative_definite"
    else:
        metric_class = "indefinite"

    return {
        "lambda_phase": lam,
        "eta_real": eta_real,
        "eta_rot": eta_rot,
        "eta_im_rel": eta_im_rel,
        "reconstruction_res": rec_res,
        "metric_class": metric_class,
        "sign_pattern": sign_pattern,
    }


def phase_gauge_transform(C: np.ndarray, delta: float) -> np.ndarray:
    """Apply the effective triad phase gauge parameter delta."""
    CK, CP, CQ = C
    return np.array([
        CK * np.exp(1j * delta),
        CP * np.exp(-1j * delta),
        CQ * np.exp(-1j * delta),
    ], dtype=complex)


def cp_cq_collinearity(CP: complex, CQ: complex) -> float:
    """
    Gauge-invariant relative phase obstruction between C_P and C_Q.

    Under u_m = exp(i theta_m) y_m, C_P and C_Q acquire the same phase factor,
    so their relative phase is invariant.  The metric embedding can be made real
    by phase gauge only if C_P and C_Q are collinear in the complex plane.
    This diagnostic is |sin(arg C_P - arg C_Q)|.
    """
    denom = abs(CP) * abs(CQ)
    if denom < 1e-300:
        return 0.0
    return float(abs(np.imag(CP * np.conjugate(CQ))) / denom)


def principal_line_phase(values: np.ndarray) -> float:
    """
    Return the orientation angle psi of the real line through the origin that
    best fits complex values in a least-squares sense.

    The objective is to minimize sum Im(values * exp(-i psi))^2.  The solution
    is psi = 0.5 arg(sum values^2), with a possible pi/2 branch; both branches
    are checked by the caller when needed.
    """
    vals = np.asarray(values, dtype=complex)
    if np.linalg.norm(vals) < 1e-300:
        return 0.0
    return 0.5 * float(np.angle(np.sum(vals ** 2)))


def optimize_phase_gauge(C: np.ndarray, ngrid: int = 0) -> Dict[str, object]:
    """
    Apply a deterministic analytic phase-gauge convention.

    Under the triad phase gauge,

        C_K' = C_K exp(i delta),
        C_P' = C_P exp(-i delta),
        C_Q' = C_Q exp(-i delta).

    For C_K != 0 we choose delta = -arg(C_K), so the transformed C_K is
    real and nonnegative.  The invariant identities then force the two partner
    coefficients to be real in every nondegenerate compatible triad.  If C_K is
    numerically zero, the first nonzero partner coefficient fixes the phase.
    This convention removes the branch ambiguity of the earlier residual-only
    optimization and makes global-helicity-reversal pairs classify identically.

    The ngrid argument is retained only for command-line compatibility.
    """
    C = np.asarray(C, dtype=complex)
    CK, CP, CQ = C
    c_norm = float(np.linalg.norm(C))
    if c_norm < 1e-300:
        emb = common_phase_metric(C)
        return {"delta_opt": 0.0, "C_gauge": C.copy(), "metric": emb}
    zero_tol = SIGNATURE_ZERO_REL_TOL * c_norm
    if abs(CK) > zero_tol:
        delta = -float(np.angle(CK))
    elif abs(CP) > zero_tol:
        delta = float(np.angle(CP))
    else:
        delta = float(np.angle(CQ))
    delta = float(np.mod(delta, 2.0 * np.pi))
    best_C = phase_gauge_transform(C, delta)
    best_emb = common_phase_metric(best_C)
    return {
        "delta_opt": delta,
        "C_gauge": best_C,
        "metric": best_emb,
    }


def coefficient_orientations(C: np.ndarray,
                             kappas: Tuple[float, float, float]) -> List[Dict[str, object]]:
    """Return the three exact created-mode orientations of a closed triad.

    The original variables satisfy

        dot u_K = C_K u_P u_Q,
        dot u_P = C_P u_K u_Q^*,
        dot u_Q = C_Q u_K u_P^*.

    Besides the K-created orientation, the same closed system can be written with
    P created by using (u_P, u_K, u_Q^*) or with Q created by using
    (u_Q, u_K, u_P^*).  Under the deterministic reality convention
    u_s(-k) = -u_s(k)^*, the conjugated variable differs from the stored -k
    amplitude only by an admissible phase of pi.
    """
    CK, CP, CQ = np.asarray(C, dtype=complex)
    kK, kP, kQ = kappas
    return [
        {
            "created_mode": "K",
            "variables": "(u_K,u_P,u_Q)",
            "uses_conjugate_partner": False,
            "C_oriented": np.array([CK, CP, CQ], dtype=complex),
            "kappa_oriented": np.array([kK, kP, kQ], dtype=float),
        },
        {
            "created_mode": "P",
            "variables": "(u_P,u_K,u_Q^*)",
            "uses_conjugate_partner": True,
            "C_oriented": np.array([CP, CK, np.conjugate(CQ)], dtype=complex),
            "kappa_oriented": np.array([kP, kK, kQ], dtype=float),
        },
        {
            "created_mode": "Q",
            "variables": "(u_Q,u_K,u_P^*)",
            "uses_conjugate_partner": True,
            "C_oriented": np.array([CQ, CK, np.conjugate(CP)], dtype=complex),
            "kappa_oriented": np.array([kQ, kK, kP], dtype=float),
        },
    ]


def audit_positive_diagonal_rescaling(C: np.ndarray,
                                      kappas: Tuple[float, float, float],
                                      active: bool,
                                      zero_rel_tol: float = SIGNATURE_ZERO_REL_TOL,
                                      audit_tol: float = 1e-10) -> Dict[str, object]:
    """Audit exact local realization by a positively rescaled canonical block.

    For an oriented, gauge-real coefficient triple (A, -B, -D), A,B,D>0 and
    A=B+D, choose beta=||C||_2, g=i beta, and

        r_0 = sqrt(BD)/beta,
        r_1 = sqrt(AD)/beta,
        r_2 = sqrt(AB)/beta.

    With a_j=r_j u_j, the canonical Hermitian three-wave equations reconstruct
    all three directed fluid coefficients.  Physical energy uses the positive
    diagonal weights 1/r_j^2, while helicity uses the same positive metric
    multiplied by its signed-wavenumber factors.  Every residual returned here
    is scale normalized.
    """
    C = np.asarray(C, dtype=complex)
    c_norm = float(np.linalg.norm(C))
    empty = {
        "rescale_status": "inactive" if not active else "unclassified",
        "rescale_created_mode": "",
        "rescale_variables": "",
        "rescale_uses_conjugate_partner": False,
        "rescale_candidate_count": 0,
        "rescale_delta": np.nan,
        "rescale_realness_rel": np.nan,
        "rescale_A": np.nan,
        "rescale_B": np.nan,
        "rescale_D": np.nan,
        "rescale_beta": np.nan,
        "rescale_r0": np.nan,
        "rescale_r1": np.nan,
        "rescale_r2": np.nan,
        "rescale_condition": np.nan,
        "rescale_balance_rel": np.nan,
        "rescale_coefficient_rel": np.nan,
        "rescale_energy_weight_rel": np.nan,
        "rescale_helicity_weight_rel": np.nan,
        "rescale_oriented_C0_real": np.nan,
        "rescale_oriented_C0_imag": np.nan,
        "rescale_oriented_C1_real": np.nan,
        "rescale_oriented_C1_imag": np.nan,
        "rescale_oriented_C2_real": np.nan,
        "rescale_oriented_C2_imag": np.nan,
    }
    if not active or c_norm <= 1e-300:
        return empty

    # Singularity is invariant under coefficient permutation and conjugation.
    base_gauge = optimize_phase_gauge(C)
    base_real = np.real(np.asarray(base_gauge["C_gauge"], dtype=complex))
    base_unit = base_real / max(float(np.linalg.norm(base_real)), 1e-300)
    if np.any(np.abs(base_unit) <= zero_rel_tol):
        empty["rescale_status"] = "singular"
        return empty

    candidates: List[Dict[str, object]] = []
    for oriented in coefficient_orientations(C, kappas):
        D0 = np.asarray(oriented["C_oriented"], dtype=complex)
        if abs(D0[0]) <= zero_rel_tol * c_norm:
            continue
        delta = -float(np.angle(D0[0]))
        Dg = np.array([
            D0[0] * np.exp(1j * delta),
            D0[1] * np.exp(-1j * delta),
            D0[2] * np.exp(-1j * delta),
        ], dtype=complex)
        realness_rel = float(np.linalg.norm(np.imag(Dg)) / (c_norm + 1e-300))
        A, c1, c2 = np.real(Dg)
        if realness_rel > audit_tol:
            continue
        if not (A > zero_rel_tol * c_norm and
                c1 < -zero_rel_tol * c_norm and
                c2 < -zero_rel_tol * c_norm):
            continue

        B, D = -float(c1), -float(c2)
        A = float(A)
        beta = c_norm
        r0 = float(np.sqrt(B * D) / beta)
        r1 = float(np.sqrt(A * D) / beta)
        r2 = float(np.sqrt(A * B) / beta)
        r = np.array([r0, r1, r2], dtype=float)
        pred = np.array([
            beta * r1 * r2 / r0,
            -beta * r0 * r2 / r1,
            -beta * r0 * r1 / r2,
        ], dtype=complex)
        coeff_rel = float(np.linalg.norm(pred - Dg) / (c_norm + 1e-300))
        balance_rel = float(abs(A - B - D) / (A + B + D + 1e-300))

        weights = 1.0 / (r ** 2)
        energy_weight_rel = float(
            abs(weights[0] - weights[1] - weights[2]) /
            (abs(weights[0]) + abs(weights[1]) + abs(weights[2]) + 1e-300)
        )
        kap = np.asarray(oriented["kappa_oriented"], dtype=float)
        helicity_terms = kap * weights
        helicity_weight_rel = float(
            abs(helicity_terms[0] - helicity_terms[1] - helicity_terms[2]) /
            (np.sum(np.abs(helicity_terms)) + 1e-300)
        )
        condition = float(np.max(r) / np.min(r))
        passed = max(realness_rel, coeff_rel, balance_rel,
                     energy_weight_rel, helicity_weight_rel) <= audit_tol
        candidate = {
            "rescale_status": "pass" if passed else "residual_failure",
            "rescale_created_mode": str(oriented["created_mode"]),
            "rescale_variables": str(oriented["variables"]),
            "rescale_uses_conjugate_partner": bool(oriented["uses_conjugate_partner"]),
            "rescale_delta": float(np.mod(delta, 2.0 * np.pi)),
            "rescale_realness_rel": realness_rel,
            "rescale_A": A,
            "rescale_B": B,
            "rescale_D": D,
            "rescale_beta": beta,
            "rescale_r0": r0,
            "rescale_r1": r1,
            "rescale_r2": r2,
            "rescale_condition": condition,
            "rescale_balance_rel": balance_rel,
            "rescale_coefficient_rel": coeff_rel,
            "rescale_energy_weight_rel": energy_weight_rel,
            "rescale_helicity_weight_rel": helicity_weight_rel,
            "rescale_oriented_C0_real": float(Dg[0].real),
            "rescale_oriented_C0_imag": float(Dg[0].imag),
            "rescale_oriented_C1_real": float(Dg[1].real),
            "rescale_oriented_C1_imag": float(Dg[1].imag),
            "rescale_oriented_C2_real": float(Dg[2].real),
            "rescale_oriented_C2_imag": float(Dg[2].imag),
        }
        candidates.append(candidate)

    if not candidates:
        empty["rescale_status"] = "no_admissible_orientation"
        return empty

    # A nonsingular conservative real triple has exactly one coefficient whose
    # sign differs from the other two, so exactly one created-mode orientation
    # should pass.  Preserve multiplicity as an explicit audit result.
    candidates.sort(key=lambda item: (
        0 if item["rescale_status"] == "pass" else 1,
        item["rescale_coefficient_rel"],
    ))
    best = candidates[0]
    best["rescale_candidate_count"] = len(candidates)
    if len(candidates) != 1 and best["rescale_status"] == "pass":
        best["rescale_status"] = "multiple_orientations"
    return best


# -----------------------------------------------------------------------------
# Geometry generation
# -----------------------------------------------------------------------------

def integer_vectors(kmax: int) -> List[Vector]:
    vals = range(-kmax, kmax + 1)
    out: List[Vector] = []
    for v in itertools.product(vals, vals, vals):
        if v == (0, 0, 0):
            continue
        out.append(v)
    return out


def clean_geometry(K: Vector, P: Vector, Q: Vector) -> bool:
    if K == (0, 0, 0) or P == (0, 0, 0) or Q == (0, 0, 0):
        return False
    if K == P or K == Q or P == Q:
        return False
    # Exclude cases where one retained positive representative equals the negative of another.
    if K == v_neg(P) or K == v_neg(Q) or P == v_neg(Q):
        return False
    # Avoid collinear degenerate triads; they tend to have weak/zero interaction.
    p = v_to_array(P)
    q = v_to_array(Q)
    if np.linalg.norm(np.cross(p, q)) < EPS:
        return False
    return True


def generate_geometries(kmax: int, max_geometries: int) -> List[Tuple[Vector, Vector, Vector]]:
    vecs = integer_vectors(kmax)
    seen = set()
    out: List[Tuple[Vector, Vector, Vector]] = []
    for P in vecs:
        for Q in vecs:
            K = v_add(P, Q)
            if K == (0, 0, 0):
                continue
            if max(abs(x) for x in K) > kmax:
                continue
            # Canonicalize K and transform P,Q consistently if needed.
            Kc = canonical_vec(K)
            if Kc != K:
                Pc, Qc = v_neg(P), v_neg(Q)
            else:
                Pc, Qc = P, Q
            # Sort P,Q to avoid duplicate pair order.
            pair = sorted([Pc, Qc])
            P2, Q2 = pair[0], pair[1]
            K2 = v_add(P2, Q2)
            if K2 != Kc:
                # If sorting did not preserve K due to sign handling, skip.
                continue
            if not clean_geometry(Kc, P2, Q2):
                continue
            key = (Kc, P2, Q2)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
            if max_geometries > 0 and len(out) >= max_geometries:
                return out
    return out


# -----------------------------------------------------------------------------
# Ensemble scan
# -----------------------------------------------------------------------------

def invariant_residuals(
    K: Vector,
    P: Vector,
    Q: Vector,
    sK: int,
    sP: int,
    sQ: int,
    CK: complex,
    CP: complex,
    CQ: complex,
    eps: float = 1e-300,
) -> Dict[str, float | complex]:
    """Return normalized energy and helicity invariant residuals."""
    energy_res = CK + np.conjugate(CP) + np.conjugate(CQ)
    energy_den = abs(CK) + abs(CP) + abs(CQ) + eps

    hK = sK * norm(K)
    hP = sP * norm(P)
    hQ = sQ * norm(Q)
    helicity_res = hK * CK + hP * np.conjugate(CP) + hQ * np.conjugate(CQ)
    helicity_den = abs(hK * CK) + abs(hP * CP) + abs(hQ * CQ) + eps

    return {
        "energy_res": energy_res,
        "energy_res_abs": float(abs(energy_res)),
        "energy_res_rel": float(abs(energy_res) / energy_den),
        "helicity_res": helicity_res,
        "helicity_res_abs": float(abs(helicity_res)),
        "helicity_res_rel": float(abs(helicity_res) / helicity_den),
    }


def conservation_residual_statistics(active: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics for energy and helicity invariant residuals."""
    rows = []
    for name, col in [("R_E", "energy_res_rel"), ("R_H", "helicity_res_rel")]:
        vals = active[col].to_numpy(dtype=float)
        rows.append({
            "residual": name,
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "q99": float(np.quantile(vals, 0.99)),
            "q999": float(np.quantile(vals, 0.999)),
        })
    return pd.DataFrame(rows)


def scan(args: argparse.Namespace) -> pd.DataFrame:
    geometries = generate_geometries(args.kmax, args.max_geometries)
    rows = []
    for gid, (K, P, Q) in enumerate(geometries):
        for sK, sP, sQ in itertools.product([1, -1], repeat=3):
            try:
                CK, CP, CQ = extract_coefficients(K, P, Q, sK, sP, sQ)
            except Exception as exc:
                rows.append({
                    "geom_id": gid, "K": str(K), "P": str(P), "Q": str(Q),
                    "sK": sK, "sP": sP, "sQ": sQ,
                    "helicity_tuple": f"({sK:+d},{sP:+d},{sQ:+d})",
                    "status": "error", "error": str(exc),
                })
                continue

            C = np.array([CK, CP, CQ], dtype=complex)
            C_norm = float(np.linalg.norm(C))
            status = "active" if C_norm > args.active_tol else "inactive"
            inv_res = invariant_residuals(K, P, Q, sK, sP, sQ, CK, CP, CQ)
            energy_res = complex(inv_res["energy_res"])
            helicity_res = complex(inv_res["helicity_res"])
            energy_res_abs = float(inv_res["energy_res_abs"])
            energy_res_rel = float(inv_res["energy_res_rel"])
            helicity_res_abs = float(inv_res["helicity_res_abs"])
            helicity_res_rel = float(inv_res["helicity_res_rel"])
            g_fit, Cpred, single_res, single_rel = fit_single_g(CK, CP, CQ)
            base = common_phase_metric(C)
            gauge = optimize_phase_gauge(C, ngrid=args.delta_grid)
            gm = gauge["metric"]
            Cg = gauge["C_gauge"]
            col = cp_cq_collinearity(CP, CQ)
            kappas = (sK * norm(K), sP * norm(P), sQ * norm(Q))
            rescale = audit_positive_diagonal_rescaling(
                C,
                kappas,
                active=(status == "active"),
                zero_rel_tol=SIGNATURE_ZERO_REL_TOL,
                audit_tol=args.rescale_tol,
            )

            gauge_success = (status == "active") and (float(gm["eta_im_rel"]) < args.gauge_tol)
            base_real = str(base["metric_class"]) != "complex_metric"
            gauge_real = str(gm["metric_class"]) != "complex_metric"
            converted = (not base_real) and gauge_real and status == "active"
            still_complex = (not gauge_real) and status == "active"

            # Basis diagnostic maxima across the three selected modes.
            diags = [basis_diagnostics(K, sK), basis_diagnostics(P, sP), basis_diagnostics(Q, sQ)]
            basis_trans_max = max(d["transverse_error"] for d in diags)
            basis_eig_max = max(d["eigen_error"] for d in diags)
            basis_norm_max = max(d["norm_error"] for d in diags)

            rows.append({
                "geom_id": gid,
                "K": str(K), "P": str(P), "Q": str(Q),
                "K_norm": norm(K), "P_norm": norm(P), "Q_norm": norm(Q),
                "sK": sK, "sP": sP, "sQ": sQ,
                "helicity_tuple": f"({sK:+d},{sP:+d},{sQ:+d})",
                "status": status,
                "CK_real": CK.real, "CK_imag": CK.imag,
                "CP_real": CP.real, "CP_imag": CP.imag,
                "CQ_real": CQ.real, "CQ_imag": CQ.imag,
                "C_norm": C_norm,
                "energy_res_real": energy_res.real,
                "energy_res_imag": energy_res.imag,
                "energy_res_abs": energy_res_abs,
                "energy_res_rel": energy_res_rel,
                "helicity_res_real": helicity_res.real,
                "helicity_res_imag": helicity_res.imag,
                "helicity_res_abs": helicity_res_abs,
                "helicity_res_rel": helicity_res_rel,
                "single_g_real": g_fit.real,
                "single_g_imag": g_fit.imag,
                "single_g_res": single_res,
                "single_g_res_rel": single_rel,
                "baseline_eta_im_rel": float(base["eta_im_rel"]),
                "baseline_metric_class": str(base["metric_class"]),
                "baseline_sign_pattern": str(base["sign_pattern"]),
                "baseline_reconstruction_res": float(base["reconstruction_res"]),
                "delta_opt": float(gauge["delta_opt"]),
                "gauge_eta_im_rel": float(gm["eta_im_rel"]),
                "gauge_metric_class": str(gm["metric_class"]),
                "gauge_sign_pattern": str(gm["sign_pattern"]),
                "gauge_reconstruction_res": float(gm["reconstruction_res"]),
                "gauge_CK_real": Cg[0].real, "gauge_CK_imag": Cg[0].imag,
                "gauge_CP_real": Cg[1].real, "gauge_CP_imag": Cg[1].imag,
                "gauge_CQ_real": Cg[2].real, "gauge_CQ_imag": Cg[2].imag,
                "cp_cq_collinearity": col,
                "phase_gauge_success": bool(gauge_success),
                "converted_by_gauge": bool(converted),
                "still_complex_after_gauge": bool(still_complex),
                "basis_transverse_error_max": basis_trans_max,
                "basis_eigen_error_max": basis_eig_max,
                "basis_norm_error_max": basis_norm_max,
                **rescale,
            })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Output and plots
# -----------------------------------------------------------------------------

def save_pie(series: pd.Series, title: str, path: str) -> None:
    """Save a colorblind-safe pie chart with hatches and an external legend."""
    series = series[series > 0]
    total = series.sum()
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]
    hatches = ["///", "\\\\", "xx", "..", "--"]
    wedges, _texts, autotexts = ax.pie(
        series.values,
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
        startangle=90,
        counterclock=False,
        colors=colors[:len(series)],
        wedgeprops={"edgecolor":"black", "linewidth":0.9},
        textprops={"fontsize":9},
    )
    for w, h in zip(wedges, hatches):
        w.set_hatch(h)
    legend_labels = [f"{idx}: {int(val)} ({100*val/total:.1f}%)" for idx, val in series.items()]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True, fontsize=9)
    ax.set_title(title)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def _apply_hatches(ax, hatches=None):
    """Apply one hatch per stacked category, not one hatch per bar patch."""
    if hatches is None:
        hatches = ["///", "\\", "xx", "..", "--", "++"]
    containers = getattr(ax, "containers", [])
    if containers:
        for ci, container in enumerate(containers):
            hatch = hatches[ci % len(hatches)]
            for patch in container.patches:
                patch.set_edgecolor("black")
                patch.set_linewidth(0.45)
                patch.set_hatch(hatch)
    else:
        for i, patch in enumerate(ax.patches):
            patch.set_edgecolor("black")
            patch.set_linewidth(0.45)
            patch.set_hatch(hatches[i % len(hatches)])

def _hist_percent(ax, data, bins, label, color, hatch=None, alpha=0.65):
    weights = np.ones_like(np.asarray(data, dtype=float)) * 100.0 / max(len(data), 1)
    n, b, patches = ax.hist(data, bins=bins, weights=weights, label=label, color=color,
                            alpha=alpha, edgecolor="black", linewidth=0.35)
    if hatch:
        for p in patches:
            p.set_hatch(hatch)
    return n, b, patches


def plot_outputs(df: pd.DataFrame, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    active = df[df["status"] == "active"].copy()
    if active.empty:
        return

    # Energy/helicity invariant residual comparison for the same active ensemble.
    conservation_residual_statistics(active).to_csv(
        os.path.join(out, "conservation_residual_statistics.csv"), index=False
    )

    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    bins = np.linspace(-18.0, -10.0, 81)
    weights = np.ones(len(active), dtype=float) * 100.0 / len(active)
    ax.hist(np.log10(active["energy_res_rel"].clip(lower=1e-18)),
            bins=bins, weights=weights, alpha=0.62, label=r"$R_E$",
            color="#56B4E9", edgecolor="black", linewidth=0.30, hatch="//")
    ax.hist(np.log10(active["helicity_res_rel"].clip(lower=1e-18)),
            bins=bins, weights=weights, alpha=0.62, label=r"$R_H$",
            color="#E69F00", edgecolor="black", linewidth=0.30, hatch="\\")
    ax.set_xlabel(r"$\log_{10}$ normalized invariant residual")
    ax.set_ylabel("fraction of active triads (%)")
    ax.set_title("Energy and helicity conservation residuals")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "energy_helicity_residual_hist.png"), dpi=260)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    x = np.log10(active["energy_res_rel"].clip(lower=1e-18))
    y = np.log10(active["helicity_res_rel"].clip(lower=1e-18))
    ax.scatter(x, y, s=9, alpha=0.35, facecolor="#0072B2", edgecolor="black", linewidth=0.15)
    lo = min(float(x.min()), float(y.min()))
    hi = max(float(x.max()), float(y.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0)
    ax.set_xlabel(r"$\log_{10} R_E$")
    ax.set_ylabel(r"$\log_{10} R_H$")
    ax.set_title("Triad-level energy versus helicity residual")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "energy_vs_helicity_residual_scatter.png"), dpi=260)
    plt.close(fig)

    # Histograms: use percentage, not raw counts, so the y-axis is proportional.
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, max(active["baseline_eta_im_rel"].max(), active["gauge_eta_im_rel"].max()), 70)
    _hist_percent(ax, active["baseline_eta_im_rel"].to_numpy(), bins, "baseline", "#56B4E9", hatch="//", alpha=0.70)
    _hist_percent(ax, active["gauge_eta_im_rel"].to_numpy(), bins, "phase-gauge optimized", "0.35", hatch="\\\\", alpha=0.65)
    ax.set_xlabel(r"max relative imaginary part of rotated metric")
    ax.set_ylabel("fraction of active triads (%)")
    ax.set_title("Metric-realness residual before and after phase gauge")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "baseline_vs_gauge_eta_im_rel.png"), dpi=260)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    _hist_percent(ax, active["gauge_eta_im_rel"].to_numpy(), bins=80, label="gauge residual", color="#009E73", hatch="//")
    ax.set_xlabel(r"optimized gauge residual")
    ax.set_ylabel("fraction of active triads (%)")
    ax.set_title("Phase-gauge optimized metric residual")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gauge_eta_im_rel_hist.png"), dpi=260)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    _hist_percent(ax, active["cp_cq_collinearity"].to_numpy(), bins=80, label="collinearity", color="#CC79A7", hatch="xx")
    ax.set_xlabel(r"$|\sin(\arg C_P-\arg C_Q)|$")
    ax.set_ylabel("fraction of active triads (%)")
    ax.set_title(r"Gauge-invariant $C_P$--$C_Q$ phase obstruction")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "cp_cq_collinearity_hist.png"), dpi=260)
    plt.close(fig)

    plt.figure(figsize=(6, 6))
    plt.scatter(active["baseline_eta_im_rel"], active["gauge_eta_im_rel"], s=8, alpha=0.35,
                facecolor="#0072B2", edgecolor="black", linewidth=0.15)
    lim = max(float(active["baseline_eta_im_rel"].max()), float(active["gauge_eta_im_rel"].max()), 1e-12)
    plt.plot([0, lim], [0, lim], "--", color="black", linewidth=1)
    plt.xlabel("baseline metric residual")
    plt.ylabel("phase-gauge optimized residual")
    plt.title("Does phase gauge reduce the metric obstruction?")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "baseline_vs_gauge_scatter.png"), dpi=260)
    plt.close()

    counts = active["gauge_metric_class"].value_counts().sort_index()
    counts.to_csv(os.path.join(out, "phase_gauge_counts.csv"), header=["count"])
    save_pie(counts, "Metric classes after phase-gauge optimization", os.path.join(out, "phase_gauge_counts.png"))

    pivot = pd.crosstab(active["helicity_tuple"], active["gauge_metric_class"])
    pivot.to_csv(os.path.join(out, "phase_gauge_by_helicity.csv"))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=["#0072B2", "#D55E00", "#009E73", "#CC79A7"])
    _apply_hatches(ax)
    ax.set_ylabel("count")
    ax.set_title("Phase-gauge metric classes by helicity")
    ax.set_xlabel("")
    ax.legend(frameon=True, fontsize=8)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "phase_gauge_by_helicity.png"), dpi=260)
    plt.close(fig)

    cross = pd.crosstab(active["baseline_metric_class"], active["gauge_metric_class"])
    cross.to_csv(os.path.join(out, "baseline_vs_gauge_metric_counts.csv"))

    # Collinearity by helicity summary.
    helicity_col = active.groupby("helicity_tuple").agg(
        count=("status", "size"),
        energy_res_rel_max=("energy_res_rel", "max"),
        helicity_res_rel_max=("helicity_res_rel", "max"),
        helicity_res_rel_median=("helicity_res_rel", "median"),
        cp_cq_collinearity_median=("cp_cq_collinearity", "median"),
        cp_cq_collinearity_mean=("cp_cq_collinearity", "mean"),
        cp_cq_collinearity_max=("cp_cq_collinearity", "max"),
        gauge_eta_im_rel_median=("gauge_eta_im_rel", "median"),
        gauge_eta_im_rel_mean=("gauge_eta_im_rel", "mean"),
        gauge_success_count=("phase_gauge_success", "sum"),
        converted_count=("converted_by_gauge", "sum"),
    ).reset_index()
    helicity_col["gauge_success_fraction"] = helicity_col["gauge_success_count"] / helicity_col["count"]
    helicity_col["converted_fraction"] = helicity_col["converted_count"] / helicity_col["count"]
    helicity_col.to_csv(os.path.join(out, "cp_cq_collinearity_by_helicity.csv"), index=False)


def write_summary(df: pd.DataFrame, args: argparse.Namespace, out: str) -> None:
    active = df[df["status"] == "active"].copy()
    lines: List[str] = []
    lines.append("Phase-gauge triad validation summary")
    lines.append("====================================")
    lines.append(f"kmax: {args.kmax}")
    lines.append(f"max_geometries: {args.max_geometries}")
    lines.append(f"delta_grid: {args.delta_grid}")
    lines.append(f"total rows: {len(df)}")
    lines.append(f"active rows: {len(active)}")
    lines.append(f"inactive/error rows: {len(df)-len(active)}")
    lines.append("")
    if active.empty:
        lines.append("No active triads found.")
    else:
        lines.append("Energy constraint residual C_K + C_P^* + C_Q^*:")
        lines.append(f"  max abs: {active['energy_res_abs'].max():.3e}")
        lines.append(f"  mean abs: {active['energy_res_abs'].mean():.3e}")
        lines.append(f"  max rel: {active['energy_res_rel'].max():.3e}")
        lines.append(f"  mean rel: {active['energy_res_rel'].mean():.3e}")
        lines.append(f"  median rel: {active['energy_res_rel'].median():.3e}")
        lines.append(f"  99% rel quantile: {active['energy_res_rel'].quantile(0.99):.3e}")
        lines.append("")
        lines.append("Helicity constraint residual sK|K| C_K + sP|P| C_P^* + sQ|Q| C_Q^*:")
        lines.append(f"  max abs: {active['helicity_res_abs'].max():.3e}")
        lines.append(f"  mean abs: {active['helicity_res_abs'].mean():.3e}")
        lines.append(f"  max rel: {active['helicity_res_rel'].max():.3e}")
        lines.append(f"  mean rel: {active['helicity_res_rel'].mean():.3e}")
        lines.append(f"  median rel: {active['helicity_res_rel'].median():.3e}")
        lines.append(f"  99% rel quantile: {active['helicity_res_rel'].quantile(0.99):.3e}")
        lines.append("")
        lines.append("Conservation residual summary table written to conservation_residual_statistics.csv.")
        lines.append("Figures written to energy_helicity_residual_hist.png and energy_vs_helicity_residual_scatter.png.")
        lines.append("")
        lines.append("Single-g residual:")
        lines.append(f"  median rel: {active['single_g_res_rel'].median():.3e}")
        lines.append(f"  mean rel: {active['single_g_res_rel'].mean():.3e}")
        lines.append(f"  max rel: {active['single_g_res_rel'].max():.3e}")
        lines.append("")
        lines.append("Metric realness residual before/after phase gauge:")
        lines.append(f"  baseline eta_im_rel median: {active['baseline_eta_im_rel'].median():.3e}")
        lines.append(f"  gauge eta_im_rel median: {active['gauge_eta_im_rel'].median():.3e}")
        lines.append(f"  baseline eta_im_rel mean: {active['baseline_eta_im_rel'].mean():.3e}")
        lines.append(f"  gauge eta_im_rel mean: {active['gauge_eta_im_rel'].mean():.3e}")
        lines.append(f"  gauge eta_im_rel max: {active['gauge_eta_im_rel'].max():.3e}")
        lines.append("")
        lines.append("Gauge-invariant C_P/C_Q collinearity obstruction:")
        lines.append(f"  median |sin phase diff|: {active['cp_cq_collinearity'].median():.3e}")
        lines.append(f"  mean |sin phase diff|: {active['cp_cq_collinearity'].mean():.3e}")
        lines.append(f"  max |sin phase diff|: {active['cp_cq_collinearity'].max():.3e}")
        lines.append("")
        lines.append("Baseline metric class counts:")
        for cls, cnt in active['baseline_metric_class'].value_counts().sort_index().items():
            lines.append(f"  {cls}: {cnt} ({100*cnt/len(active):.2f}%)")
        lines.append("")
        lines.append("Gauge-optimized metric class counts:")
        for cls, cnt in active['gauge_metric_class'].value_counts().sort_index().items():
            lines.append(f"  {cls}: {cnt} ({100*cnt/len(active):.2f}%)")
        lines.append("")
        converted = int(active['converted_by_gauge'].sum())
        successes = int(active['phase_gauge_success'].sum())
        lines.append(f"Phase-gauge success count: {successes} ({100*successes/len(active):.2f}%)")
        lines.append(f"Converted complex->real count: {converted} ({100*converted/len(active):.2f}%)")
        lines.append("")
        lines.append("Positive diagonal rescaling audit after created-mode reorientation:")
        for cls, cnt in active['rescale_status'].value_counts().sort_index().items():
            lines.append(f"  {cls}: {cnt} ({100*cnt/len(active):.2f}%)")
        passed = active[active['rescale_status'] == 'pass']
        if not passed.empty:
            lines.append("  created-mode counts:")
            for mode, cnt in passed['rescale_created_mode'].value_counts().sort_index().items():
                lines.append(f"    {mode}: {cnt}")
            lines.append(f"  max realness residual: {passed['rescale_realness_rel'].max():.3e}")
            lines.append(f"  max coefficient reconstruction residual: {passed['rescale_coefficient_rel'].max():.3e}")
            lines.append(f"  max coefficient-balance residual: {passed['rescale_balance_rel'].max():.3e}")
            lines.append(f"  max energy-weight residual: {passed['rescale_energy_weight_rel'].max():.3e}")
            lines.append(f"  max helicity-weight residual: {passed['rescale_helicity_weight_rel'].max():.3e}")
            lines.append(f"  scaling condition median: {passed['rescale_condition'].median():.6g}")
            lines.append(f"  scaling condition 99% quantile: {passed['rescale_condition'].quantile(0.99):.6g}")
            lines.append(f"  scaling condition maximum: {passed['rescale_condition'].max():.6g}")
        lines.append("")
        lines.append("Interpretation:")
        lines.append("  Energy and helicity residuals near machine precision show that the same")
        lines.append("  active ensemble is conservative in both quadratic invariants before metric")
        lines.append("  classification and positive-rescaling construction are interpreted.")
        lines.append("  Every nonsingular gauge-real triple has exactly one created-mode orientation")
        lines.append("  with sign pattern (+,-,-), and the positive diagonal scaling reconstructs")
        lines.append("  all three directed coefficients and both weighted number invariants.")
        lines.append("  Singular cases are not covered by a finite strictly positive scaling.")
        lines.append("  This is a local closed-triad construction; consistency of one scale per")
        lines.append("  shared Fourier-helicity mode across the assembled graph is not inferred.")
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def validate_positive_rescaling_audit(df: pd.DataFrame,
                                      audit_tol: float) -> Dict[str, object]:
    """Fail closed unless every nonsingular active case passes the full audit."""
    active = df[df["status"] == "active"].copy()
    nonsingular = active[active["gauge_metric_class"] == "indefinite"].copy()
    singular = active[active["gauge_metric_class"] == "singular"].copy()
    failures: List[str] = []

    bad_status = nonsingular[nonsingular["rescale_status"] != "pass"]
    if not bad_status.empty:
        failures.append(f"{len(bad_status)} nonsingular cases did not pass")
    bad_unique = nonsingular[nonsingular["rescale_candidate_count"] != 1]
    if not bad_unique.empty:
        failures.append(f"{len(bad_unique)} nonsingular cases lack a unique orientation")
    bad_singular = singular[singular["rescale_status"] != "singular"]
    if not bad_singular.empty:
        failures.append(f"{len(bad_singular)} singular cases were misclassified")

    residual_columns = [
        "rescale_realness_rel",
        "rescale_coefficient_rel",
        "rescale_balance_rel",
        "rescale_energy_weight_rel",
        "rescale_helicity_weight_rel",
    ]
    residual_maxima = {}
    for col in residual_columns:
        maximum = float(nonsingular[col].max())
        residual_maxima[col] = maximum
        if maximum > audit_tol:
            failures.append(f"{col} maximum {maximum:.3e} exceeds {audit_tol:.3e}")

    # Global helicity reversal is an independent paired check of branch and
    # created-mode determinism.  Scaling vectors must agree pairwise.
    lookup = {
        (int(row.geom_id), int(row.sK), int(row.sP), int(row.sQ)): row
        for row in active.itertuples(index=False)
    }
    seen = set()
    reversal_pairs = 0
    reversal_status_mismatches = 0
    reversal_mode_mismatches = 0
    max_reversal_scale_rel = 0.0
    for key, row in lookup.items():
        opposite = (key[0], -key[1], -key[2], -key[3])
        pair_key = tuple(sorted((key, opposite)))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        other = lookup.get(opposite)
        if other is None:
            failures.append(f"missing global-helicity-reversal partner for {key}")
            continue
        reversal_pairs += 1
        reversal_status_mismatches += int(row.rescale_status != other.rescale_status)
        reversal_mode_mismatches += int(str(row.rescale_created_mode) != str(other.rescale_created_mode))
        if row.rescale_status == "pass":
            r1 = np.array([row.rescale_r0, row.rescale_r1, row.rescale_r2], dtype=float)
            r2 = np.array([other.rescale_r0, other.rescale_r1, other.rescale_r2], dtype=float)
            rel = float(np.linalg.norm(r1 - r2) / (np.linalg.norm(r1) + 1e-300))
            max_reversal_scale_rel = max(max_reversal_scale_rel, rel)
    if reversal_status_mismatches:
        failures.append(f"{reversal_status_mismatches} reversal status mismatches")
    if reversal_mode_mismatches:
        failures.append(f"{reversal_mode_mismatches} reversal created-mode mismatches")
    if max_reversal_scale_rel > audit_tol:
        failures.append(
            f"reversal scale mismatch {max_reversal_scale_rel:.3e} exceeds {audit_tol:.3e}"
        )

    if failures:
        raise RuntimeError("Positive diagonal rescaling audit failed: " + "; ".join(failures))

    return {
        "active_cases": int(len(active)),
        "nonsingular_cases": int(len(nonsingular)),
        "singular_cases": int(len(singular)),
        "reversal_pairs": int(reversal_pairs),
        "max_reversal_scale_rel": max_reversal_scale_rel,
        "residual_maxima": residual_maxima,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-gauge validation for helical triad metric embeddings.")
    parser.add_argument("--kmax", type=int, default=3, help="Maximum integer wavevector component.")
    parser.add_argument("--max-geometries", type=int, default=800,
                        help="Maximum geometric triads to sample before helicity expansion. Use -1 for all.")
    parser.add_argument("--delta-grid", type=int, default=1440,
                        help="Deprecated compatibility option; the analytic gauge is used.")
    parser.add_argument("--out", type=str, default="phase_gauge_results", help="Output directory.")
    parser.add_argument("--active-tol", type=float, default=1e-12, help="Coefficient norm threshold for active triads.")
    parser.add_argument("--gauge-tol", type=float, default=1e-8, help="Threshold for phase-gauge real-metric success.")
    parser.add_argument("--rescale-tol", type=float, default=1e-10,
                        help="Normalized tolerance for positive diagonal rescaling audit.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = scan(args)
    audit_report = validate_positive_rescaling_audit(df, args.rescale_tol)
    df.to_csv(os.path.join(args.out, "phase_gauge_ensemble.csv"), index=False)
    plot_outputs(df, args.out)
    write_summary(df, args, args.out)

    active = df[df["status"] == "active"]
    print(f"Saved outputs to: {args.out}")
    print(f"Rows: {len(df)}, active: {len(active)}")
    print("Validated positive-rescaling audit:", audit_report)
    if not active.empty:
        print("Gauge metric classes:", dict(active['gauge_metric_class'].value_counts().sort_index()))
        print(f"Baseline eta_im_rel median: {active['baseline_eta_im_rel'].median():.3e}")
        print(f"Gauge eta_im_rel median: {active['gauge_eta_im_rel'].median():.3e}")
        print(f"Max energy residual rel: {active['energy_res_rel'].max():.3e}")
        print(f"Max helicity residual rel: {active['helicity_res_rel'].max():.3e}")
        print(f"Phase-gauge successes: {int(active['phase_gauge_success'].sum())}/{len(active)}")
        print("Positive-rescaling statuses:", dict(active['rescale_status'].value_counts().sort_index()))
        passed = active[active['rescale_status'] == 'pass']
        if not passed.empty:
            print("Created-mode counts:", dict(passed['rescale_created_mode'].value_counts().sort_index()))
            print(f"Max rescaled coefficient residual: {passed['rescale_coefficient_rel'].max():.3e}")
            print(f"Max rescaled helicity-weight residual: {passed['rescale_helicity_weight_rel'].max():.3e}")


if __name__ == "__main__":
    main()
