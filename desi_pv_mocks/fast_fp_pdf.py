"""
Fast per-galaxy log-distance PDF evaluation.

Companion to fp_fast.py. After fitting the FP parameters, this module
computes for each galaxy:
  - mean       : posterior mean of the log-distance ratio
  - err        : posterior std
  - alpha      : skew parameter of a skew-normal fit to the PDF

Two versions of each are returned: with Malmquist-bias correction (FN
normalisation) and without (`*_nmc`).

The implementation is a single parallel Numba kernel that fuses:
  - per-(distance, galaxy) chi^2 / log_det computation,
  - per-(distance, galaxy) FN integral (Owen's T + arctan2),
  - trapezoidal normalisation over the distance grid,
  - moment accumulation (∫dP, ∫d²P, ∫d³P),
  - skew-normal alpha conversion.

It also accepts a precomputed `pv_var` (peculiar-velocity term, same as
in fp_fast.FPData) so the kernel can stay free of `np.log10` allocations.

Numerical equivalence with the original notebook code is verified to
~1e-9 by the test in `test_pdf.py`.
"""

from __future__ import annotations

import math
import ctypes
import numpy as np
from numba import njit, prange
from numba.extending import get_cython_function_address

# Bind scipy.special.owens_t for use inside @njit kernels
_owens_t_addr = get_cython_function_address(
    "scipy.special.cython_special", "owens_t"
)
_owens_t_ftype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
_owens_t = _owens_t_ftype(_owens_t_addr)


# ---------------------------------------------------------------------------
# Core kernel
# ---------------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _pdf_moments_kernel(params, dbins, d_z, pv_var,
                        r, s, i, err_r, err_s, err_i,
                        kcorr, evo_corr_z, log1pz,
                        mag_low, mag_high, smin, smax):
    """
    Per-galaxy moments of the log-distance PDF.

    Loops over galaxies in parallel. For each galaxy:
      - Evaluates loglike at every distance bin (1001 inner iterations).
      - Evaluates the FN (Malmquist) integral via Owen's T per bin.
      - Normalises the PDF (trapezoidal) and accumulates the 0th-3rd moments.

    Returns six 1D arrays of length N:
      mean, err, alpha    (with Malmquist correction)
      mean_nmc, err_nmc, alpha_nmc   (without Malmquist correction)
    """
    # ---- parameter unpacking (k=0; same conventions as fp_fast) ----
    a = params[0]; b = params[1]
    rmean = params[2]; smean = params[3]; imean = params[4]
    sigma1 = params[5]; sigma2 = params[6]; sigma3 = params[7]
    a2 = a * a; b2 = b * b
    s1sq = sigma1 * sigma1; s2sq = sigma2 * sigma2; s3sq = sigma3 * sigma3
    fac1 = -a; fac2 = -1.0 - b2; fac3 = a * b
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
    nd = dbins.shape[0]
    dd = dbins[1] - dbins[0]  # uniform grid step
    log_2pi_15 = 1.5 * math.log(2.0 * math.pi)
    inv_2pi = 1.0 / (2.0 * math.pi)

    # Output arrays
    mean = np.empty(N)
    err = np.empty(N)
    alpha = np.empty(N)
    mean_nmc = np.empty(N)
    err_nmc = np.empty(N)
    alpha_nmc = np.empty(N)

    # Per-galaxy covariance terms (independent of distance bin)
    # done inside the outer loop because they depend on per-galaxy errors

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
        det = cov_r * A + sigmars * B + cov_ri * C
        log_det = math.log(det)

        sdiff = s[j] - smean
        idiff = i[j] - imean

        # FN_func constants (per galaxy, distance-independent terms)
        inv_det = 1.0 / det
        G_inv_det = math.sqrt(E) / (2.0 * F - B) * (
            C * (2.0 * F + B) - A * F - 2.0 * B * I
        ) * inv_det   # = G_paper * inv_det
        delta_nodist = (I * B * B + A * F * F - 2.0 * B * C * F) * inv_det * inv_det
        Edet = E * inv_det
        Gdet = G_inv_det * G_inv_det * det * det  # = G_paper^2 * det
        # actually Gdet = (G * det)**2 in original FN_func conventions
        # where `det` in that scope = 1/|Cov|. So Gdet = G_paper^2 / |Cov|^2 / inv_det^2
        # ...let's re-derive cleanly using the original variable conventions:
        #   In FN_func: det_FN = 1/|Cov|
        #   G_FN = sqrt(E)/(2F-B)*(...) -> this is G_paper * det_FN^(3/2)? No.
        # Match the original *exact* formulae:

        # Use original FN_func variables: in FN_func, `det` means 1/|Cov|.
        # Rebind locally to match:
        det_FN = inv_det  # so det_FN * |Cov| = 1
        # G in FN_func = sqrt(E)/(2F-B) * (...) -> this is the paper's G/det^(3/2)
        # But the only place it's used is squared: Gdet = (G*det_FN)**2
        G = math.sqrt(E) / (2.0 * F - B) * (
            C * (2.0 * F + B) - A * F - 2.0 * B * I
        )
        delta = (I * B * B + A * F * F - 2.0 * B * C * F) * det_FN * det_FN
        Edet_v = E * det_FN
        Gdet_v = (G * det_FN) ** 2
        one_p_Gdet = 1.0 + Gdet_v
        one_p_delta = 1.0 + delta
        H = math.sqrt(1.0 + Gdet_v + delta)
        sqrt_Edet_half = math.sqrt(Edet_v / 2.0)
        sqrt_2_over_Edet = math.sqrt(2.0 / Edet_v)
        sqrt_delta = math.sqrt(delta)
        Gd_sqd = G * det_FN * sqrt_delta
        Rscale = math.sqrt(2.0 * delta / det_FN) / (2.0 * F - B)
        sqrt_2_over_1pGdet = math.sqrt(2.0 / one_p_Gdet)
        sqrt_Edet_over_1pdelta = math.sqrt(Edet_v / one_p_delta)

        # G1min / G1max depend only on per-galaxy quantities (not distance bin)
        G1min = -sqrt_Edet_over_1pdelta * (smin - smean)
        G1max = -sqrt_Edet_over_1pdelta * (smax - smean)

        # ---- pass 1: compute unnormalised PDFs and trapezoidal normalisations ----
        # We need to store P[d] and P_nmc[d] across distance bins to normalise.
        # Use a stack-allocated buffer (numba allows fixed-size local arrays).
        P = np.empty(nd)
        P_nmc = np.empty(nd)
        log_norm_min = 0.0  # for numerical stability via log-sum-exp shift
        log_norm_min_nmc = 0.0

        # First pass: compute logP and logP_nmc for each bin
        logP = np.empty(nd)
        logP_nmc = np.empty(nd)

        for k in range(nd):
            d_bin = dbins[k]

            # loglike at distance d_bin (FP_func at logdists=d_bin, sumgals=False)
            rdiff = (r[j] - d_bin) - rmean
            chi2 = (
                A * rdiff * rdiff + E * sdiff * sdiff + I * idiff * idiff
                + 2.0 * rdiff * (B * sdiff + C * idiff)
                + 2.0 * F * sdiff * idiff
            ) / det
            loglike = 0.5 * (chi2 + log_det)

            # FN integral at distance d_bin: lmin/lmax depend on d_bin via d_H = 10**(-d_bin) * d_z
            log_d_H = -d_bin + math.log10(d_z[j])
            lcomm = (4.65 + 5.0 * log1pz[j] - evo_corr_z[j] + kcorr[j]
                     + 10.0 - 2.5 * math.log10(2.0 * math.pi) + 5.0 * log_d_H)
            lmin = (lcomm - mag_high) / 5.0
            lmax = (lcomm - mag_low) / 5.0

            # FN ingredients
            Rmin = (lmin - rmean - imean / 2.0) * Rscale
            Rmax = (lmax - rmean - imean / 2.0) * Rscale
            G0min = -sqrt_2_over_1pGdet * Rmin
            G0max = -sqrt_2_over_1pGdet * Rmax

            H0minmin = Gd_sqd - sqrt_Edet_half * one_p_Gdet * (smin - smean) / Rmin
            H0minmax = Gd_sqd - sqrt_Edet_half * one_p_Gdet * (smax - smean) / Rmin
            H0maxmin = Gd_sqd - sqrt_Edet_half * one_p_Gdet * (smin - smean) / Rmax
            H0maxmax = Gd_sqd - sqrt_Edet_half * one_p_Gdet * (smax - smean) / Rmax
            H1minmin = Gd_sqd - sqrt_2_over_Edet * one_p_delta * Rmin / (smin - smean)
            H1minmax = Gd_sqd - sqrt_2_over_Edet * one_p_delta * Rmin / (smax - smean)
            H1maxmin = Gd_sqd - sqrt_2_over_Edet * one_p_delta * Rmax / (smin - smean)
            H1maxmax = Gd_sqd - sqrt_2_over_Edet * one_p_delta * Rmax / (smax - smean)

            FN = (
                _owens_t(G0min, H0minmax / H) - _owens_t(G0min, H0minmin / H)
                + _owens_t(G0max, H0maxmin / H) - _owens_t(G0max, H0maxmax / H)
                + _owens_t(G1min, H1maxmin / H) - _owens_t(G1min, H1minmin / H)
                + _owens_t(G1max, H1minmax / H) - _owens_t(G1max, H1maxmax / H)
            )
            FN += inv_2pi * (
                math.atan2(H0maxmax, H) + math.atan2(H1maxmax, H)
                - math.atan2(H0maxmin, H) - math.atan2(H1maxmin, H)
                + math.atan2(H0minmin, H) + math.atan2(H1minmin, H)
                - math.atan2(H0minmax, H) - math.atan2(H1minmax, H)
            )
            if FN < 1.0e-15:
                FN = 1.0e-15
            log_FN = math.log(FN)

            logP[k] = -log_2pi_15 - loglike - log_FN
            logP_nmc[k] = -log_2pi_15 - loglike

        # log-sum-exp stabilisation: subtract max before exponentiating
        lp_max = logP[0]
        lp_max_nmc = logP_nmc[0]
        for k in range(1, nd):
            if logP[k] > lp_max:
                lp_max = logP[k]
            if logP_nmc[k] > lp_max_nmc:
                lp_max_nmc = logP_nmc[k]

        for k in range(nd):
            P[k] = math.exp(logP[k] - lp_max)
            P_nmc[k] = math.exp(logP_nmc[k] - lp_max_nmc)

        # Trapezoidal normalisation over dbins (uniform grid)
        norm = 0.5 * (P[0] + P[nd - 1])
        norm_nmc = 0.5 * (P_nmc[0] + P_nmc[nd - 1])
        for k in range(1, nd - 1):
            norm += P[k]
            norm_nmc += P_nmc[k]
        norm *= dd
        norm_nmc *= dd

        # Moments via trapezoidal: ∫ d^p P / norm
        # Compute m1, m2, m3 with weights w[0]=w[nd-1]=0.5*dd, w[k]=dd otherwise
        m1 = 0.0; m2 = 0.0; m3 = 0.0
        m1n = 0.0; m2n = 0.0; m3n = 0.0
        # endpoints
        w0 = 0.5 * dd
        d0 = dbins[0]
        d_last = dbins[nd - 1]
        m1 += w0 * d0 * P[0]
        m2 += w0 * d0 * d0 * P[0]
        m3 += w0 * d0 * d0 * d0 * P[0]
        m1 += w0 * d_last * P[nd - 1]
        m2 += w0 * d_last * d_last * P[nd - 1]
        m3 += w0 * d_last * d_last * d_last * P[nd - 1]
        m1n += w0 * d0 * P_nmc[0]
        m2n += w0 * d0 * d0 * P_nmc[0]
        m3n += w0 * d0 * d0 * d0 * P_nmc[0]
        m1n += w0 * d_last * P_nmc[nd - 1]
        m2n += w0 * d_last * d_last * P_nmc[nd - 1]
        m3n += w0 * d_last * d_last * d_last * P_nmc[nd - 1]
        for k in range(1, nd - 1):
            d_k = dbins[k]
            P_k = P[k]
            Pn_k = P_nmc[k]
            m1 += dd * d_k * P_k
            m2 += dd * d_k * d_k * P_k
            m3 += dd * d_k * d_k * d_k * P_k
            m1n += dd * d_k * Pn_k
            m2n += dd * d_k * d_k * Pn_k
            m3n += dd * d_k * d_k * d_k * Pn_k

        m1 /= norm; m2 /= norm; m3 /= norm
        m1n /= norm_nmc; m2n /= norm_nmc; m3n /= norm_nmc

        # Std and skew (gamma1)
        var = m2 - m1 * m1
        var_n = m2n - m1n * m1n
        if var < 0.0:
            var = 0.0
        if var_n < 0.0:
            var_n = 0.0
        sd = math.sqrt(var)
        sd_n = math.sqrt(var_n)
        if sd > 0.0:
            gamma1 = (m3 - 3.0 * m1 * var - m1 * m1 * m1) / (sd * sd * sd)
        else:
            gamma1 = 0.0
        if sd_n > 0.0:
            gamma1_n = (m3n - 3.0 * m1n * var_n - m1n * m1n * m1n) / (sd_n * sd_n * sd_n)
        else:
            gamma1_n = 0.0
        # Clip
        if gamma1 > 0.99:
            gamma1 = 0.99
        elif gamma1 < -0.99:
            gamma1 = -0.99
        if gamma1_n > 0.99:
            gamma1_n = 0.99
        elif gamma1_n < -0.99:
            gamma1_n = -0.99

        # alpha from gamma1 (skew-normal)
        abs_g = abs(gamma1)
        delta_sn = math.copysign(
            math.sqrt(math.pi / 2.0 / (1.0 + ((4.0 - math.pi) / (2.0 * abs_g)) ** (2.0 / 3.0))),
            gamma1,
        ) if abs_g > 0.0 else 0.0
        abs_gn = abs(gamma1_n)
        delta_sn_n = math.copysign(
            math.sqrt(math.pi / 2.0 / (1.0 + ((4.0 - math.pi) / (2.0 * abs_gn)) ** (2.0 / 3.0))),
            gamma1_n,
        ) if abs_gn > 0.0 else 0.0

        one_minus_d2 = 1.0 - delta_sn * delta_sn
        if one_minus_d2 <= 0.0:
            alpha[j] = math.copysign(1e10, delta_sn) if delta_sn != 0.0 else 0.0
        else:
            alpha[j] = delta_sn / math.sqrt(one_minus_d2)
        one_minus_d2n = 1.0 - delta_sn_n * delta_sn_n
        if one_minus_d2n <= 0.0:
            alpha_nmc[j] = math.copysign(1e10, delta_sn_n) if delta_sn_n != 0.0 else 0.0
        else:
            alpha_nmc[j] = delta_sn_n / math.sqrt(one_minus_d2n)

        mean[j] = m1
        err[j] = sd
        mean_nmc[j] = m1n
        err_nmc[j] = sd_n

    return mean, err, alpha, mean_nmc, err_nmc, alpha_nmc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def evaluate_logdist_moments(FPparams, dbins, *,
                             z_obs, z, zcmb_group,
                             r, s, i, err_r, err_s, err_i,
                             dz_cluster, kcorr, evo_corr,
                             mag_low, mag_high, smin, smax):
    """
    Per-galaxy log-distance PDF moments.

    Parameters
    ----------
    FPparams : array, shape (8,)
        Fitted FP parameters.
    dbins : array, shape (nd,)
        Log-distance ratio grid. Must be uniform.
    z_obs : array, shape (N,)
        Observed z used in the peculiar-velocity term (was data["zcmb"]).
    z : array, shape (N,)
        Heliocentric (or whatever) z used in `5*log10(1+z)` terms.
    zcmb_group : array, shape (N,)
        Group/cluster z used in the evolutionary correction term.
    r, s, i : arrays, shape (N,)
        FP coordinates (log radius, log vdisp, surface brightness).
    err_r, err_s, err_i : arrays, shape (N,)
        Per-galaxy errors on r, s, i.
    dz_cluster : array, shape (N,)
        Comoving distance to the cluster (from dist_spline(zcmb_group)).
    kcorr : array, shape (N,)
        K-correction values (zero in the published method).
    evo_corr : float
        Evolutionary correction coefficient (1.1 in the published method).
    mag_low, mag_high : float
        Apparent magnitude cuts.
    smin, smax : float
        log10(velocity dispersion) cuts.

    Returns
    -------
    dict with keys:
        mean, err, alpha           : with Malmquist correction
        mean_nmc, err_nmc, alpha_nmc : without Malmquist correction
    """
    # Precomputed per-galaxy scalars (independent of distance bin)
    LIGHT_SPEED = 299792.458
    pv_var = np.log10(1.0 + 300.0 / (LIGHT_SPEED * z_obs)) ** 2
    log1pz = np.log10(1.0 + z)
    evo_corr_z = evo_corr * zcmb_group

    # Ensure all arrays are C-contiguous float64
    def C(x):
        return np.ascontiguousarray(x, dtype=np.float64)

    out = _pdf_moments_kernel(
        C(FPparams), C(dbins), C(dz_cluster), C(pv_var),
        C(r), C(s), C(i), C(err_r), C(err_s), C(err_i),
        C(kcorr), C(evo_corr_z), C(log1pz),
        float(mag_low), float(mag_high), float(smin), float(smax),
    )
    mean, err, alpha, mean_nmc, err_nmc, alpha_nmc = out
    return {
        "mean": mean, "err": err, "alpha": alpha,
        "mean_nmc": mean_nmc, "err_nmc": err_nmc, "alpha_nmc": alpha_nmc,
    }
