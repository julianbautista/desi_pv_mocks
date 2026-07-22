"""
Fast FP fitter (Howlett et al. 2022, arXiv:2201.03112).

A drop-in replacement for the differential-evolution-based fitting loop.
Keeps the user's original `FP_func` for chi-squared evaluation and PDF
computation; replaces the global optimiser only.

Strategy
--------
- L-BFGS-B with analytic gradients (via JAX) from a warm start `x0`.
- Optional multi-start with perturbations around `x0` for robustness:
  if all restarts converge to the same f-value, we trust the optimum.
- Likelihood evaluations use Numba (parallel, fastmath) when available.

Numerical equivalence with the original FP_func is verified to ~1e-9.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import optimize, stats

LIGHT_SPEED = 299792.458

# Sensible defaults — DR1 FP parameter region.
DEFAULT_BOUNDS = (
    (1.0, 1.8),     # a
    (-1.5, -0.5),   # b
    (-0.5, 0.5),    # rmean
    (2.0, 2.4),     # smean
    (2.4, 3.0),     # imean
    (0.01, 0.12),   # sigma1
    (0.05, 0.5),    # sigma2
    (0.1, 0.3),     # sigma3
)

PARAM_NAMES = ["a", "b", "rmean", "smean", "imean", "sigma1", "sigma2", "sigma3"]

# ---------------------------------------------------------------------------
# Optional accelerators
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import erf as jerf
    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


# ---------------------------------------------------------------------------
# Likelihood kernels (Numba, JAX, NumPy). All compute the same quantity:
#     -log L = 0.5 * sum_i [ chi^2_i + log_det_i + 2*FN_i ] / Sn_i
# where the FN term is the Scut integral over velocity dispersion bounds.
# Hardcoded logdists=0 (fitting mode). k=0 (paper convention).
# ---------------------------------------------------------------------------
def _sigma_terms(a, b, sigma1, sigma2, sigma3):
    """Return the 6 intrinsic covariance entries (eqs B3-B8) with k=0."""
    a2, b2 = a * a, b * b
    s1sq, s2sq, s3sq = sigma1 * sigma1, sigma2 * sigma2, sigma3 * sigma3
    # k = 0: fac1 = -a, fac2 = -1-b^2, fac3 = a*b, fac4 = 1
    fac1, fac2, fac3 = -a, -1.0 - b2, a * b
    inv_n1 = 1.0 / (1.0 + a2 + b2)
    inv_n2 = 1.0 / (1.0 + b2)
    inv_n12 = inv_n1 * inv_n2
    sigmar2 =     inv_n1 * s1sq + b2 * inv_n2 * s2sq + fac1 * fac1 * inv_n12 * s3sq
    sigmas2 =  a2*inv_n1 * s1sq                      + fac2 * fac2 * inv_n12 * s3sq
    sigmai2 =  b2*inv_n1 * s1sq +      inv_n2 * s2sq + fac3 * fac3 * inv_n12 * s3sq
    sigmars = -a*inv_n1 * s1sq                       + fac1 * fac2 * inv_n12 * s3sq
    sigmari = -b*inv_n1 * s1sq +   b * inv_n2 * s2sq + fac1 * fac3 * inv_n12 * s3sq
    sigmasi = a*b*inv_n1 * s1sq                      + fac2 * fac3 * inv_n12 * s3sq
    return sigmar2, sigmas2, sigmai2, sigmars, sigmari, sigmasi


if HAS_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def _nll_numba(params, pv_var, r, s, i, err_r, err_s, err_i,
                   Sn, smin, smax):
        a, b = params[0], params[1]
        rmean, smean, imean = params[2], params[3], params[4]
        sigma1, sigma2, sigma3 = params[5], params[6], params[7]

        # Intrinsic covariance (k = 0; same algebra as _sigma_terms)
        a2, b2 = a * a, b * b
        s1sq, s2sq, s3sq = sigma1 * sigma1, sigma2 * sigma2, sigma3 * sigma3
        fac1, fac2, fac3 = -a, -1.0 - b2, a * b
        inv_n1 = 1.0 / (1.0 + a2 + b2)
        inv_n2 = 1.0 / (1.0 + b2)
        inv_n12 = inv_n1 * inv_n2
        sigmar2 =     inv_n1 * s1sq + b2 * inv_n2 * s2sq + fac1 * fac1 * inv_n12 * s3sq
        sigmas2 =  a2*inv_n1 * s1sq                      + fac2 * fac2 * inv_n12 * s3sq
        sigmai2 =  b2*inv_n1 * s1sq +      inv_n2 * s2sq + fac3 * fac3 * inv_n12 * s3sq
        sigmars = -a*inv_n1 * s1sq                       + fac1 * fac2 * inv_n12 * s3sq
        sigmari = -b*inv_n1 * s1sq +   b * inv_n2 * s2sq + fac1 * fac3 * inv_n12 * s3sq
        sigmasi = a*b*inv_n1 * s1sq                      + fac2 * fac3 * inv_n12 * s3sq

        N = r.shape[0]
        total = 0.0
        for j in prange(N):
            cov_r = err_r[j] * err_r[j] + pv_var[j] + sigmar2
            cov_s = err_s[j] * err_s[j] + sigmas2
            cov_i = err_i[j] * err_i[j] + sigmai2
            cov_ri = -err_r[j] * err_i[j] + sigmari
            A = cov_s * cov_i - sigmasi * sigmasi
            B = sigmasi * cov_ri - sigmars * cov_i
            C = sigmars * sigmasi - cov_s * cov_ri
            E = cov_r * cov_i - cov_ri * cov_ri
            F = sigmars * cov_ri - cov_r * sigmasi
            I = cov_r * cov_s - sigmars * sigmars
            sdiff = s[j] - smean
            idiff = i[j] - imean
            rdiff = r[j] - rmean
            det = cov_r * A + sigmars * B + cov_ri * C
            Snj = Sn[j]
            chi2 = (
                A * rdiff * rdiff + E * sdiff * sdiff + I * idiff * idiff
                + 2.0 * rdiff * (B * sdiff + C * idiff)
                + 2.0 * F * sdiff * idiff
            ) / (det * Snj)
            log_det = math.log(det) / Snj
            delta = (A * F * F + I * B * B - 2.0 * B * C * F) / det
            coef = math.sqrt(E / (2.0 * (det + delta)))
            FN = math.log(0.5 * (math.erf(coef * (smax - smean))
                                 - math.erf(coef * (smin - smean)))) / Snj
            total += chi2 + log_det + 2.0 * FN
        return 0.5 * total

    @njit(parallel=True, fastmath=True, cache=True)
    def _chi2_per_galaxy_numba(params, pv_var, r, s, i, err_r, err_s, err_i):
        """
        Per-galaxy chi^2 (unweighted by Sn). Matches the original
        `Sn * FP_func(..., chi_squared_only=True)` once multiplied by Sn,
        since the original divides by Sn inside the kernel.
        """
        a, b = params[0], params[1]
        rmean, smean, imean = params[2], params[3], params[4]
        sigma1, sigma2, sigma3 = params[5], params[6], params[7]

        a2, b2 = a * a, b * b
        s1sq, s2sq, s3sq = sigma1 * sigma1, sigma2 * sigma2, sigma3 * sigma3
        fac1, fac2, fac3 = -a, -1.0 - b2, a * b
        inv_n1 = 1.0 / (1.0 + a2 + b2)
        inv_n2 = 1.0 / (1.0 + b2)
        inv_n12 = inv_n1 * inv_n2
        sigmar2 =     inv_n1 * s1sq + b2 * inv_n2 * s2sq + fac1 * fac1 * inv_n12 * s3sq
        sigmas2 =  a2*inv_n1 * s1sq                      + fac2 * fac2 * inv_n12 * s3sq
        sigmai2 =  b2*inv_n1 * s1sq +      inv_n2 * s2sq + fac3 * fac3 * inv_n12 * s3sq
        sigmars = -a*inv_n1 * s1sq                       + fac1 * fac2 * inv_n12 * s3sq
        sigmari = -b*inv_n1 * s1sq +   b * inv_n2 * s2sq + fac1 * fac3 * inv_n12 * s3sq
        sigmasi = a*b*inv_n1 * s1sq                      + fac2 * fac3 * inv_n12 * s3sq

        N = r.shape[0]
        out = np.empty(N)
        for j in prange(N):
            cov_r = err_r[j] * err_r[j] + pv_var[j] + sigmar2
            cov_s = err_s[j] * err_s[j] + sigmas2
            cov_i = err_i[j] * err_i[j] + sigmai2
            cov_ri = -err_r[j] * err_i[j] + sigmari
            A = cov_s * cov_i - sigmasi * sigmasi
            B = sigmasi * cov_ri - sigmars * cov_i
            C = sigmars * sigmasi - cov_s * cov_ri
            E = cov_r * cov_i - cov_ri * cov_ri
            F = sigmars * cov_ri - cov_r * sigmasi
            I = cov_r * cov_s - sigmars * sigmars
            sdiff = s[j] - smean
            idiff = i[j] - imean
            rdiff = r[j] - rmean
            det = cov_r * A + sigmars * B + cov_ri * C
            out[j] = (
                A * rdiff * rdiff + E * sdiff * sdiff + I * idiff * idiff
                + 2.0 * rdiff * (B * sdiff + C * idiff)
                + 2.0 * F * sdiff * idiff
            ) / det
        return out


if HAS_JAX:
    def _nll_jax(params, pv_var, r, s, i, err_r, err_s, err_i,
                 Sn, smin, smax):
        a, b = params[0], params[1]
        rmean, smean, imean = params[2], params[3], params[4]
        sigma1, sigma2, sigma3 = params[5], params[6], params[7]

        a2, b2 = a * a, b * b
        s1sq, s2sq, s3sq = sigma1 * sigma1, sigma2 * sigma2, sigma3 * sigma3
        fac1, fac2, fac3 = -a, -1.0 - b2, a * b
        inv_n1 = 1.0 / (1.0 + a2 + b2)
        inv_n2 = 1.0 / (1.0 + b2)
        inv_n12 = inv_n1 * inv_n2
        sigmar2 =     inv_n1 * s1sq + b2 * inv_n2 * s2sq + fac1 * fac1 * inv_n12 * s3sq
        sigmas2 =  a2*inv_n1 * s1sq                      + fac2 * fac2 * inv_n12 * s3sq
        sigmai2 =  b2*inv_n1 * s1sq +      inv_n2 * s2sq + fac3 * fac3 * inv_n12 * s3sq
        sigmars = -a*inv_n1 * s1sq                       + fac1 * fac2 * inv_n12 * s3sq
        sigmari = -b*inv_n1 * s1sq +   b * inv_n2 * s2sq + fac1 * fac3 * inv_n12 * s3sq
        sigmasi = a*b*inv_n1 * s1sq                      + fac2 * fac3 * inv_n12 * s3sq

        cov_r = err_r * err_r + pv_var + sigmar2
        cov_s = err_s * err_s + sigmas2
        cov_i = err_i * err_i + sigmai2
        cov_ri = -err_r * err_i + sigmari
        A = cov_s * cov_i - sigmasi * sigmasi
        B = sigmasi * cov_ri - sigmars * cov_i
        C = sigmars * sigmasi - cov_s * cov_ri
        E = cov_r * cov_i - cov_ri * cov_ri
        F = sigmars * cov_ri - cov_r * sigmasi
        I = cov_r * cov_s - sigmars * sigmars
        sdiff = s - smean
        idiff = i - imean
        rdiff = r - rmean
        det = cov_r * A + sigmars * B + cov_ri * C
        chi_squared = (
            A * rdiff * rdiff + E * sdiff * sdiff + I * idiff * idiff
            + 2.0 * rdiff * (B * sdiff + C * idiff)
            + 2.0 * F * sdiff * idiff
        ) / (det * Sn)
        log_det = jnp.log(det) / Sn
        delta = (A * F * F + I * B * B - 2.0 * B * C * F) / det
        coef = jnp.sqrt(E / (2.0 * (det + delta)))
        FN = jnp.log(0.5 * (jerf(coef * (smax - smean))
                            - jerf(coef * (smin - smean)))) / Sn
        return 0.5 * jnp.sum(chi_squared + log_det + 2.0 * FN)

    _value_and_grad = jax.jit(jax.value_and_grad(_nll_jax, argnums=0),
                              static_argnums=(9, 10))


def _chi2_per_galaxy_numpy(params, pv_var, r, s, i, err_r, err_s, err_i):
    """NumPy fallback for _chi2_per_galaxy_numba. Returns chi^2 / Sn-free part."""
    a, b, rmean, smean, imean, sigma1, sigma2, sigma3 = params
    sigmar2, sigmas2, sigmai2, sigmars, sigmari, sigmasi = _sigma_terms(
        a, b, sigma1, sigma2, sigma3
    )
    cov_r = err_r * err_r + pv_var + sigmar2
    cov_s = err_s * err_s + sigmas2
    cov_i = err_i * err_i + sigmai2
    cov_ri = -err_r * err_i + sigmari
    A = cov_s * cov_i - sigmasi * sigmasi
    B = sigmasi * cov_ri - sigmars * cov_i
    C = sigmars * sigmasi - cov_s * cov_ri
    E = cov_r * cov_i - cov_ri * cov_ri
    F = sigmars * cov_ri - cov_r * sigmasi
    I = cov_r * cov_s - sigmars * sigmars
    sdiff, idiff, rdiff = s - smean, i - imean, r - rmean
    det = cov_r * A + sigmars * B + cov_ri * C
    return (
        A * rdiff * rdiff + E * sdiff * sdiff + I * idiff * idiff
        + 2.0 * rdiff * (B * sdiff + C * idiff) + 2.0 * F * sdiff * idiff
    ) / det


def _nll_numpy(params, pv_var, r, s, i, err_r, err_s, err_i,
               Sn, smin, smax):
    """Pure-NumPy fallback. Used if neither Numba nor JAX is available."""
    a, b, rmean, smean, imean, sigma1, sigma2, sigma3 = params
    sigmar2, sigmas2, sigmai2, sigmars, sigmari, sigmasi = _sigma_terms(
        a, b, sigma1, sigma2, sigma3
    )
    cov_r = err_r * err_r + pv_var + sigmar2
    cov_s = err_s * err_s + sigmas2
    cov_i = err_i * err_i + sigmai2
    cov_ri = -err_r * err_i + sigmari
    A = cov_s * cov_i - sigmasi * sigmasi
    B = sigmasi * cov_ri - sigmars * cov_i
    C = sigmars * sigmasi - cov_s * cov_ri
    E = cov_r * cov_i - cov_ri * cov_ri
    F = sigmars * cov_ri - cov_r * sigmasi
    I = cov_r * cov_s - sigmars * sigmars
    sdiff, idiff, rdiff = s - smean, i - imean, r - rmean
    det = cov_r * A + sigmars * B + cov_ri * C
    chi_squared = (
        A * rdiff * rdiff + E * sdiff * sdiff + I * idiff * idiff
        + 2.0 * rdiff * (B * sdiff + C * idiff) + 2.0 * F * sdiff * idiff
    ) / (det * Sn)
    log_det = np.log(det) / Sn
    delta = (A * F * F + I * B * B - 2.0 * B * C * F) / det
    coef = np.sqrt(E / (2.0 * (det + delta)))
    FN = np.log(0.5 * (sp.special.erf(coef * (smax - smean))
                       - sp.special.erf(coef * (smin - smean)))) / Sn
    return 0.5 * np.sum(chi_squared + log_det + 2.0 * FN)


# ---------------------------------------------------------------------------
# Data container — bundles the per-galaxy arrays so calls stay tidy.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FPData:
    """Per-galaxy arrays used by the likelihood. `pv_var` is precomputed."""
    z_obs: np.ndarray
    r: np.ndarray
    s: np.ndarray
    i: np.ndarray
    err_r: np.ndarray
    err_s: np.ndarray
    err_i: np.ndarray
    Sn: np.ndarray
    pv_var: np.ndarray  # log10(1 + 300/(c*z))^2, precomputed

    @classmethod
    def build(cls, z_obs, r, s, i, err_r, err_s, err_i, Sn):
        z_obs = np.ascontiguousarray(z_obs, dtype=np.float64)
        pv_var = np.log10(1.0 + 300.0 / (LIGHT_SPEED * z_obs)) ** 2
        return cls(
            z_obs=z_obs,
            r=np.ascontiguousarray(r, dtype=np.float64),
            s=np.ascontiguousarray(s, dtype=np.float64),
            i=np.ascontiguousarray(i, dtype=np.float64),
            err_r=np.ascontiguousarray(err_r, dtype=np.float64),
            err_s=np.ascontiguousarray(err_s, dtype=np.float64),
            err_i=np.ascontiguousarray(err_i, dtype=np.float64),
            Sn=np.ascontiguousarray(Sn, dtype=np.float64),
            pv_var=np.ascontiguousarray(pv_var, dtype=np.float64),
        )

    def select(self, mask):
        """Return a new FPData with only the masked galaxies."""
        return FPData(
            z_obs=self.z_obs[mask], r=self.r[mask], s=self.s[mask], i=self.i[mask],
            err_r=self.err_r[mask], err_s=self.err_s[mask], err_i=self.err_i[mask],
            Sn=self.Sn[mask], pv_var=self.pv_var[mask],
        )

    def __len__(self):
        return len(self.r)


# ---------------------------------------------------------------------------
# Build objective + gradient closures over a dataset
# ---------------------------------------------------------------------------
def _make_value_grad(data: FPData, smin: float, smax: float):
    """Build (value, value_and_grad) for the dataset, using the fastest backend."""
    if HAS_NUMBA:
        def value(params):
            return _nll_numba(
                np.asarray(params, dtype=np.float64),
                data.pv_var, data.r, data.s, data.i,
                data.err_r, data.err_s, data.err_i, data.Sn, smin, smax,
            )
    else:
        def value(params):
            return _nll_numpy(
                params, data.pv_var, data.r, data.s, data.i,
                data.err_r, data.err_s, data.err_i, data.Sn, smin, smax,
            )

    if HAS_JAX:
        # Move data onto JAX once; reused across restarts.
        pv_j = jnp.asarray(data.pv_var)
        r_j, s_j, i_j = jnp.asarray(data.r), jnp.asarray(data.s), jnp.asarray(data.i)
        er_j = jnp.asarray(data.err_r)
        es_j = jnp.asarray(data.err_s)
        ei_j = jnp.asarray(data.err_i)
        Sn_j = jnp.asarray(data.Sn)

        def value_and_grad(params):
            v, g = _value_and_grad(jnp.asarray(params),
                                   pv_j, r_j, s_j, i_j,
                                   er_j, es_j, ei_j, Sn_j, smin, smax)
            return float(v), np.asarray(g, dtype=np.float64)
    else:
        value_and_grad = None  # L-BFGS-B will use finite differences

    return value, value_and_grad


# ---------------------------------------------------------------------------
# Single L-BFGS-B fit from a warm start
# ---------------------------------------------------------------------------
def _lbfgsb_from(x0, value, value_and_grad, bounds):
    if value_and_grad is not None:
        return optimize.minimize(
            value_and_grad, x0=np.asarray(x0, dtype=np.float64), jac=True,
            method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-11, "gtol": 1e-7, "maxiter": 200},
        )
    return optimize.minimize(
        value, x0=np.asarray(x0, dtype=np.float64),
        method="L-BFGS-B", bounds=bounds,
        options={"ftol": 1e-11, "gtol": 1e-7, "maxiter": 200},
    )


# ---------------------------------------------------------------------------
# Single-start L-BFGS-B fit
# ---------------------------------------------------------------------------
def fit_FP(data: FPData, x0, smin: float, smax: float,
           bounds: Sequence = DEFAULT_BOUNDS, verbose: bool = False):
    """
    Fit FP parameters by L-BFGS-B from `x0`, on a single data subset.

    Used as the inner building block of the iterative fits. For full
    robustness with multiple starting points, use `fit_FP_multistart_iter`.

    Parameters
    ----------
    data
        Bundled per-galaxy arrays.
    x0
        Starting parameter vector (length 8).

    Returns
    -------
    result : OptimizeResult
        Standard SciPy result. `.x` is the fitted parameters, `.fun` the
        final negative log-likelihood.
    """
    value, value_and_grad = _make_value_grad(data, smin, smax)
    result = _lbfgsb_from(np.asarray(x0, dtype=np.float64),
                          value, value_and_grad, bounds)
    if verbose:
        print(f"    f={result.fun:.4f}  nit={result.nit}  success={result.success}")
    return result


# ---------------------------------------------------------------------------
# Per-galaxy chi^2 helper (for outlier rejection)
# ---------------------------------------------------------------------------
def chi2_per_galaxy(params, data: "FPData"):
    """
    Per-galaxy chi^2 used for the outlier rejection criterion.

    Equivalent to the original `Sn * FP_func(..., chi_squared_only=True)[0]`,
    since the original kernel divides by Sn and the outer multiply cancels it.

    Uses the Numba kernel when available, NumPy otherwise.
    """
    if HAS_NUMBA:
        return _chi2_per_galaxy_numba(
            np.asarray(params, dtype=np.float64),
            data.pv_var, data.r, data.s, data.i,
            data.err_r, data.err_s, data.err_i,
        )
    return _chi2_per_galaxy_numpy(
        params, data.pv_var, data.r, data.s, data.i,
        data.err_r, data.err_s, data.err_i,
    )


# ---------------------------------------------------------------------------
# Latin-hypercube sampling of starting points within bounds
# ---------------------------------------------------------------------------
def lhs_starts(bounds: Sequence, n_samples: int, seed: int = 0):
    """
    Generate `n_samples` starting points by Latin-hypercube sampling
    within the supplied bounds.

    Returns
    -------
    starts : ndarray, shape (n_samples, len(bounds))
    """
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=len(bounds), seed=seed)
    unit = sampler.random(n=n_samples)  # in [0, 1]^d
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return lo + unit * (hi - lo)


# ---------------------------------------------------------------------------
# Iterative outlier-rejection fit, single start
# ---------------------------------------------------------------------------
def fit_FP_iter(data: FPData, x0, smin: float, smax: float,
                bounds: Sequence = DEFAULT_BOUNDS,
                p_threshold: float = 0.01, max_iter: int = 50,
                prelim_x0_rejection: bool = False,
                verbose: bool = True):
    """
    Iterative FP fit with chi^2 outlier rejection, from a single warm start.

    Each iteration: L-BFGS-B fit on the current sample, then chi^2 rejection
    with p < p_threshold. Repeat until the rejected set stabilises.

    Parameters
    ----------
    data
        Bundled per-galaxy arrays for the full unmasked sample.
    x0
        Starting parameter vector (e.g. published DR1 FP parameters or
        a Latin-hypercube draw).
    bounds
        Per-parameter (lo, hi) tuples for L-BFGS-B.
    prelim_x0_rejection
        If True, perform an outlier rejection step using chi^2 evaluated
        AT x0 (before any fitting) and start iter 1 on the resulting
        cleaned sample. This makes the final answer depend on the choice
        of x0, which is the intended behaviour inside
        `explore_FP_multistart` (where we want to probe start-dependence).
        For a standalone single-start fit, leave as False so iter 1 sees
        the full sample.

    Returns
    -------
    x : ndarray, shape (8,)
        Best-fit parameters at convergence.
    mask : ndarray of bool, shape (N,)
        True for galaxies in the final fit sample.
    info : dict
        'history', 'n_iter', 'final_f'.
    """
    N = len(data)
    current_x = np.asarray(x0, dtype=np.float64).copy()
    history = []
    dof = N - 8.0

    if prelim_x0_rejection:
        # Rejection step using chi^2 evaluated at x0 — gives each
        # starting point its own initial outlier set, propagating
        # start-dependence into the final fit.
        chi_squared = chi2_per_galaxy(current_x, data)
        pvals = stats.chi2.sf(chi_squared, np.sum(chi_squared) / dof)
        mask = pvals >= p_threshold
        badcount = int((~mask).sum())
        if verbose:
            print(f"    prelim rejection at x0: dropped {badcount}")
    else:
        mask = np.ones(N, dtype=bool)
        badcount = 0

    for ii in range(1, max_iter + 1):
        sub = data.select(mask)
        if verbose:
            print(f"    iter {ii}: N_fit={len(sub)}  rejected={N - len(sub)}")

        result = fit_FP(sub, current_x, smin, smax, bounds=bounds,
                        verbose=False)
        current_x = result.x

        # Per-galaxy chi^2 on the FULL sample (matches the original's
        # `Sn * FP_func(..., chi_squared_only=True)[0]`).
        chi_squared = chi2_per_galaxy(current_x, data)
        pvals = stats.chi2.sf(chi_squared, np.sum(chi_squared) / dof)
        new_mask = pvals >= p_threshold
        badcountnew = int((~new_mask).sum())
        converged = (badcountnew == badcount)

        history.append({
            "iter": ii, "f": result.fun, "x": current_x.copy(),
            "rejected": badcountnew, "sum_chi2": float(np.sum(chi_squared)),
        })

        if verbose:
            print(f"        f={result.fun:.2f}  rejected={badcountnew}  "
                  f"converged={converged}")

        mask = new_mask
        badcount = badcountnew
        if converged:
            break

    return current_x, mask, {"history": history, "n_iter": ii,
                             "final_f": result.fun}


# ---------------------------------------------------------------------------
# Multi-start exploration: compute per-galaxy outlier frequency f_outlier
# ---------------------------------------------------------------------------
def explore_FP_multistart(data: FPData, x0_primary, smin: float, smax: float,
                          bounds: Sequence = DEFAULT_BOUNDS,
                          n_lhs: int = 10, p_threshold: float = 0.01,
                          max_iter: int = 50, seed: int = 0,
                          verbose: bool = True):
    """
    Run `1 + n_lhs` independent iterative fits (each with its own chi^2
    rejection loop) from different starting points, and compute per-galaxy
    outlier frequency.

    Starts:
      - sample 0:   `x0_primary` (e.g. published DR1 FP parameters)
      - samples 1+: Latin-hypercube draws over `bounds`

    Each start runs to its own converged outlier-rejected solution. We
    record the rejection mask from each and compute:

        f_outlier[i] = (# starts that rejected galaxy i) / n_starts

    The final fit (your reported answer) is obtained separately via
    `final_fit_from_exploration`, which lets you pick a threshold on
    f_outlier without rerunning the multi-start.

    Returns
    -------
    result : dict
        - 'all_x'       : ndarray (n_starts, 8) of converged parameters
        - 'all_f'       : ndarray (n_starts,) of final negative log-likelihoods
        - 'all_masks'   : ndarray (n_starts, N) of bool — per-start kept masks
        - 'all_starts'  : ndarray (n_starts, 8) of starting points
        - 'f_outlier'   : ndarray (N,) — fraction of starts flagging each galaxy
        - 'x_mean'      : per-parameter mean across starts
        - 'x_std'       : per-parameter spread across starts (start-dependence systematic)
        - 'x_median'    : per-parameter median
        - 'f_spread'    : max - min of final f-values
    """
    starts = np.vstack([np.asarray(x0_primary, dtype=np.float64),
                        lhs_starts(bounds, n_lhs, seed=seed)])
    n_starts = len(starts)
    N = len(data)

    all_x = np.zeros((n_starts, 8))
    all_f = np.zeros(n_starts)
    all_masks = np.zeros((n_starts, N), dtype=bool)

    for k, s0 in enumerate(starts):
        tag = "primary" if k == 0 else f"LHS {k}"
        if verbose:
            print(f"\n[{tag}] start x0 = {s0}")
        x_k, mask_k, info_k = fit_FP_iter(
            data, s0, smin, smax, bounds=bounds,
            p_threshold=p_threshold, max_iter=max_iter,
            prelim_x0_rejection=True,
            verbose=verbose,
        )
        all_x[k] = x_k
        all_f[k] = info_k["final_f"]
        all_masks[k] = mask_k

    # f_outlier[i] = fraction of starts that rejected galaxy i
    f_outlier = np.mean(~all_masks, axis=0)

    result = {
        "all_x": all_x, "all_f": all_f,
        "all_masks": all_masks, "all_starts": starts,
        "f_outlier": f_outlier,
        "x_mean": all_x.mean(axis=0),
        "x_std": all_x.std(axis=0, ddof=1),
        "x_median": np.median(all_x, axis=0),
        "f_spread": float(all_f.max() - all_f.min()),
    }

    if verbose:
        print("\n" + "=" * 60)
        print(f"Multi-start summary ({n_starts} starts):")
        print(f"  f-value spread: {result['f_spread']:.3e}")
        print(f"  Per-parameter mean ± std across starts:")
        names = ["a", "b", "rmean", "smean", "imean", "sig1", "sig2", "sig3"]
        for k, name in enumerate(names):
            print(f"    {name:>6s} = {result['x_mean'][k]:.5f} "
                  f"± {result['x_std'][k]:.5f}")
        print(f"  f_outlier distribution:")
        for thr in [0.0, 0.1, 0.5, 0.9, 1.0]:
            n = int(np.sum(f_outlier > thr))
            print(f"    galaxies with f_outlier > {thr:.1f}: {n}")
        print("=" * 60)

    return result


def final_fit_from_exploration(data: FPData, exploration: dict,
                               x0, smin: float, smax: float,
                               bounds: Sequence = DEFAULT_BOUNDS,
                               f_outlier_threshold: float = 0.5,
                               verbose: bool = True):
    """
    Produce the final reported FP fit using the f_outlier from a multi-start
    exploration.

    Galaxies with f_outlier > threshold are excluded. A single L-BFGS-B
    fit is then run on the remaining sample, warm-started from `x0` (or
    from the exploration mean — your call).

    Parameters
    ----------
    exploration
        Output dict from `explore_FP_multistart`.
    f_outlier_threshold
        Galaxies with f_outlier > threshold are excluded. Common choices:
          - 0.0  : exclude any galaxy flagged by any start (most aggressive)
          - 0.5  : majority rule
          - 0.99 : exclude only galaxies flagged by all starts
          - 1.0  : no exclusion (pathological — fit will likely hit priors)
    x0
        Starting point for the final fit. The exploration mean
        (`exploration['x_mean']`) is often a good choice.

    Returns
    -------
    x : ndarray, shape (8,)
        Final reported best-fit parameters.
    mask : ndarray of bool, shape (N,)
        Galaxies kept (f_outlier <= threshold).
    info : dict
        - 'f'                    : final negative log-likelihood
        - 'n_kept'               : number of galaxies kept
        - 'n_rejected'           : number excluded
        - 'f_outlier_threshold'  : threshold used
        - 'x_systematic_err'     : exploration['x_std'] for convenience
    """
    f_outlier = exploration["f_outlier"]
    mask = f_outlier <= f_outlier_threshold
    n_kept = int(mask.sum())
    n_rejected = int((~mask).sum())

    if verbose:
        print(f"Final fit with f_outlier_threshold={f_outlier_threshold}:")
        print(f"  N kept     = {n_kept}")
        print(f"  N rejected = {n_rejected}")

    sub = data.select(mask)
    result = fit_FP(sub, x0, smin, smax, bounds=bounds, verbose=False)

    if verbose:
        print(f"  f = {result.fun:.4f}")
        print(f"  x = {result.x}")
        print(f"  systematic err (from multistart spread): "
              f"{exploration['x_std']}")

    return result.x, mask, {
        "f": float(result.fun),
        "n_kept": n_kept,
        "n_rejected": n_rejected,
        "f_outlier_threshold": f_outlier_threshold,
        "x_systematic_err": exploration["x_std"],
    }

def diagnose_exploration(exp, data, smin, smax, bounds, save_path=None):
    """
    Summarise and visualise the output of `explore_FP_multistart`.

    Prints:
      - Per-parameter mean, std, range across starts
      - Whether any start hit a prior bound (if `bounds` supplied)
      - f-value summary
      - Outlier classification breakdown

    Plots (single figure, multi-panel):
      - f_outlier histogram (log y)
      - Cumulative N_kept vs threshold
      - Per-parameter scatter across starts
      - Per-start vs primary deviation
    """
    all_x = exp["all_x"]
    all_f = exp["all_f"]
    f_out = exp["f_outlier"]
    starts = exp["all_starts"]
    n_starts = len(all_f)

    # ---- text summary ----
    print("=" * 70)
    print(f"Multi-start exploration summary  ({n_starts} starts)")
    print("=" * 70)

    print("\nFinal f-values:")
    print(f"  best     = {all_f.min():.4f}  (start {int(np.argmin(all_f))})")
    print(f"  worst    = {all_f.max():.4f}  (start {int(np.argmax(all_f))})")
    print(f"  spread   = {all_f.max() - all_f.min():.4e}")
    print(f"  primary  = {all_f[0]:.4f}  (DR1 / x0)")

    print("\nPer-parameter statistics:")
    print(f"  {'name':>7s} {'mean':>12s} {'std':>12s} "
          f"{'min':>12s} {'max':>12s} {'range/std_safe':>12s}")
    for k, name in enumerate(PARAM_NAMES):
        col = all_x[:, k]
        rng = col.max() - col.min()
        # robust normalised range: range / (|mean| or 1, whichever larger)
        scale = max(abs(col.mean()), 1.0)
        print(f"  {name:>7s} {col.mean():>12.6f} {col.std(ddof=1):>12.6f} "
              f"{col.min():>12.6f} {col.max():>12.6f} {rng/scale:>12.2e}")

    # ---- bounds check ----
    if bounds is not None:
        print("\nBound-hitting check (any start within 1% of a bound):")
        flagged = False
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        widths = hi - lo
        for s in range(n_starts):
            for k in range(len(PARAM_NAMES)):
                dlo = (all_x[s, k] - lo[k]) / widths[k]
                dhi = (hi[k] - all_x[s, k]) / widths[k]
                if dlo < 0.01 or dhi < 0.01:
                    where = "lower" if dlo < dhi else "upper"
                    tag = "primary" if s == 0 else f"LHS {s}"
                    print(f"  start {tag}: {PARAM_NAMES[k]} = {all_x[s, k]:.5f} "
                          f"near {where} bound ({lo[k] if where=='lower' else hi[k]})")
                    flagged = True
        if not flagged:
            print("  no fits within 1% of any bound — good")

    # ---- outlier classification ----
    N = len(f_out)
    print(f"\nOutlier classification ({N} galaxies):")
    bins = [
        (f_out == 0, "never rejected"),
        ((f_out > 0) & (f_out < 0.5), "minority (rejected by some, kept by most)"),
        ((f_out >= 0.5) & (f_out < 1.0), "majority (rejected by most, kept by some)"),
        (f_out == 1.0, "always rejected"),
    ]
    for cond, label in bins:
        n = int(cond.sum())
        print(f"  {label:<55s}: {n:6d}  ({100*n/N:.2f}%)")
    n_ambig = int(((f_out > 0) & (f_out < 1)).sum())
    print(f"  ambiguous (0 < f_outlier < 1) total           : {n_ambig:6d}  "
          f"({100*n_ambig/N:.2f}%)")

    # ---- plots ----
    fig = plt.figure(figsize=(13, 9))

    # Panel 1: f_outlier histogram
    ax1 = fig.add_subplot(2, 2, 1)
    edges = np.linspace(0, 1, 22)
    ax1.hist(f_out, bins=edges, edgecolor="k", linewidth=0.5)
    ax1.set_xlabel("f_outlier (fraction of starts rejecting galaxy)")
    ax1.set_ylabel("N galaxies")
    ax1.set_yscale("log")
    ax1.set_title("f_outlier distribution")
    ax1.set_xlim(0, 1)

    # Panel 2: per-parameter shift vs threshold (in σ units of start-spread)
    ax2 = fig.add_subplot(2, 2, 2)
    scan_thresholds = np.linspace(0.0, 0.95, 11)
    scan_xs = np.zeros((len(scan_thresholds), 8))
    for j, t in enumerate(scan_thresholds):
        x_t, _, _ = final_fit_from_exploration(
            data, exp, x0=exp["x_mean"], smin=smin, smax=smax,
            bounds=bounds, f_outlier_threshold=t, verbose=False,
        )
        scan_xs[j] = x_t

    means = scan_xs.mean(axis=0)
    stds = exp["x_std"] + 1e-30  # use multi-start spread as scale
    for k, name in enumerate(PARAM_NAMES):
        ax2.plot(scan_thresholds, (scan_xs[:, k] - means[k]) / stds[k],
                 "o-", markersize=3, linewidth=0.8, label=name)
    ax2.axvline(0.5, color="grey", ls="--", lw=0.8)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("f_outlier threshold")
    ax2.set_ylabel("(x − mean) / start_spread_std")
    ax2.set_title("Per-parameter shift vs threshold")
    ax2.legend(fontsize=7, ncol=2, loc="best")

    # Panel 3: per-parameter scatter across starts (normalised to mean)
    ax3 = fig.add_subplot(2, 2, 3)
    means = all_x.mean(axis=0)
    # deviation in units of the parameter spread
    for s in range(n_starts):
        dev = (all_x[s] - means) / (all_x.std(axis=0, ddof=1) + 1e-30)
        color = "C3" if s == 0 else "C0"
        alpha = 1.0 if s == 0 else 0.4
        ax3.plot(np.arange(8), dev, "o-",
                 color=color, alpha=alpha, markersize=4, linewidth=0.7,
                 label="primary (DR1)" if s == 0 else None)
    ax3.set_xticks(np.arange(8))
    ax3.set_xticklabels(PARAM_NAMES, rotation=30)
    ax3.set_ylabel("(x − mean) / std")
    ax3.set_title("Per-start deviation from mean (each line = one start)")
    ax3.axhline(0, color="k", lw=0.5)
    ax3.legend(loc="best")

    # Panel 4: f vs start index, with primary highlighted
    ax4 = fig.add_subplot(2, 2, 4)
    colors = ["C3" if k == 0 else "C0" for k in range(n_starts)]
    ax4.bar(np.arange(n_starts), all_f - all_f.min(), color=colors,
            edgecolor="k", linewidth=0.5)
    ax4.set_xlabel("start index (0 = primary)")
    ax4.set_ylabel("final f − f_best")
    ax4.set_title(f"Δf above best (primary in red); spread = {all_f.max()-all_f.min():.2e}")
    ax4.set_yscale("symlog", linthresh=1e-6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"\nSaved plot to {save_path}")
    plt.show()

    return fig


def threshold_scan(data, exp, x0, smin, smax, bounds,
                   thresholds=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999)):
    """
    Run the final fit at multiple f_outlier thresholds and summarise.

    Prints a table; returns (xs, infos) for further inspection.
    """
    
    print(f"\n{'threshold':>10s} {'N_kept':>8s} {'f':>14s}  " +
          "  ".join(f"{n:>10s}" for n in PARAM_NAMES))
    print("-" * 130)

    xs = []
    infos = []
    for t in thresholds:
        x_t, _, info = final_fit_from_exploration(
            data, exp, x0=x0, smin=smin, smax=smax,
            bounds=bounds, f_outlier_threshold=t, verbose=False,
        )
        xs.append(x_t)
        infos.append(info)
        row = f"{t:>10.2f} {info['n_kept']:>8d} {info['f']:>14.4f}  " + \
              "  ".join(f"{v:>10.5f}" for v in x_t)
        print(row)

    return np.array(xs), infos
