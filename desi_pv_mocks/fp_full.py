"""
fp_mock_pipeline.py
-------------------
Pipeline to generate Fundamental Plane mocks from BGS spec mocks
Computes log-distance ratios with and without correction for Malmquist bias 
 
Usage:
    python mock_fp_full_claude.py <phase> <real>
 
Exemple:
    python mock_fp_full_claude.py 0 0
"""
 
import os
import subprocess
import logging
import argparse
import time
 
import h5py
import numpy as np
import scipy as sp
import pandas as pd
from astropy.io import fits
from astropy.cosmology import Planck15, FlatLambdaCDM
 
from k_correction import GAMA_KCorrection
from scipy.spatial import KDTree
 

# ---------------------------------------------------------------------------
# Configuration 
# ---------------------------------------------------------------------------
from config import load_config
cfg = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
 
 

# ---------------------------------------------------------------------------
# FP functions
# ---------------------------------------------------------------------------
def fp_params():
    return np.array([
        cfg.fp_full.a, cfg.fp_full.b, cfg.fp_full.rmean, cfg.fp_full.smean, cfg.fp_full.imean,
        cfg.fp_full.sigma1, cfg.fp_full.sigma2, cfg.fp_full.sigma3,
    ])
 
def Mmean():
    return 4.65 - 5.0 * cfg.fp_full.rmean - 2.5 * cfg.fp_full.imean - 2.5 * np.log10(2.0 * np.pi) - 15.0
 
def c():
    return cfg.fp_full.rmean - cfg.fp_full.a * cfg.fp_full.smean - cfg.fp_full.b * cfg.fp_full.imean
 
def dbins() :
    return np.linspace(cfg.fp_full.dmin, cfg.fp_full.dmax, cfg.fp_full.nd, endpoint=True)
 
 
def _compute_covariance_terms(a, b, sigma1, sigma2, sigma3, k=0.0):
    """
    Calcule les éléments de la matrice de covariance intrinsèque du FP.
    Équations B3–B8 de Howlett et al. 2022.
 
    Retourne un dict avec sigmar2, sigmas2, sigmai2, sigmars, sigmari, sigmasi.
    """
    fac1 = k * a**2 + k * b**2 - a
    fac2 = k * a - 1.0 - b**2
    fac3 = b * (k + a)
    fac4 = 1.0 - k * a
    norm1 = 1.0 + a**2 + b**2
    norm2 = 1.0 + b**2 + k**2 * (a**2 + b**2) - 2.0 * a * k
 
    return {
        "sigmar2":  1.0 / norm1 * sigma1**2 + b**2 / norm2 * sigma2**2 + fac1**2 / (norm1 * norm2) * sigma3**2,
        "sigmas2":  a**2 / norm1 * sigma1**2 + k**2 * b**2 / norm2 * sigma2**2 + fac2**2 / (norm1 * norm2) * sigma3**2,
        "sigmai2":  b**2 / norm1 * sigma1**2 + fac4**2 / norm2 * sigma2**2 + fac3**2 / (norm1 * norm2) * sigma3**2,
        "sigmars": -a / norm1 * sigma1**2 - k * b**2 / norm2 * sigma2**2 + fac1 * fac2 / (norm1 * norm2) * sigma3**2,
        "sigmari": -b / norm1 * sigma1**2 + b * fac4 / norm2 * sigma2**2 + fac1 * fac3 / (norm1 * norm2) * sigma3**2,
        "sigmasi":  a * b / norm1 * sigma1**2 - k * b * fac4 / norm2 * sigma2**2 + fac2 * fac3 / (norm1 * norm2) * sigma3**2,
    }
 
 
def FP_func(params, logdists, z_obs, r, s, i, err_r, err_s, err_i, Sn,
            smin, smax, sumgals=True, chi_squared_only=False):
    """
    Likelihood function for the Fundamental Plane 

    Parameters
    ----------
    params      : (a, b, rmean, smean, imean, sigma1, sigma2, sigma3)
    logdists    : décalages en log-distance (scalaire ou tableau 1-D)
    z_obs       : redshifts observés
    r, s, i     : variables FP (rayon effectif, dispersion, brillance)
    err_r/s/i   : erreurs observationnelles
    Sn          : poids de sélection
    smin, smax  : bornes de la coupure en dispersion
    """
    k = 0.0
    a, b, rmean, smean, imean, sigma1, sigma2, sigma3 = params
    cov = _compute_covariance_terms(a, b, sigma1, sigma2, sigma3, k)
    sigmar2  = cov["sigmar2"]
    sigmas2  = cov["sigmas2"]
    sigmai2  = cov["sigmai2"]
    sigmars  = cov["sigmars"]
    sigmari  = cov["sigmari"]
    sigmasi  = cov["sigmasi"]
 
    # Matrice de covariance totale (signal + erreurs)
    cov_r  = err_r**2 + np.log10(1.0 + 300.0 / (299792.458 * z_obs))**2 + sigmar2
    cov_s  = err_s**2 + sigmas2
    cov_i  = err_i**2 + sigmai2
    cov_ri = -1.0 * err_r * err_i + sigmari
 
    # Cofacteurs (|Cov| × Cov^{-1})
    A = cov_s * cov_i - sigmasi**2
    B = sigmasi * cov_ri - sigmars * cov_i
    C = sigmars * sigmasi - cov_s * cov_ri
    E = cov_r * cov_i - cov_ri**2
    F = sigmars * cov_ri - cov_r * sigmasi
    I = cov_r * cov_s - sigmars**2
 
    # Résidus
    sdiff = s - smean
    idiff = i - imean
    rnew  = r - np.tile(logdists, (len(r), 1)).T
    rdiff = rnew - rmean
 
    det = cov_r * A + sigmars * B + cov_ri * C
    log_det     = np.log(det) / Sn
    chi_squared = (
        A * rdiff**2 + E * sdiff**2 + I * idiff**2
        + 2.0 * rdiff * (B * sdiff + C * idiff)
        + 2.0 * F * sdiff * idiff
    ) / (det * Sn)
 
    # Terme de normalisation FN (coupure en dispersion uniquement)
    delta = (A * F**2 + I * B**2 - 2.0 * B * C * F) / det
    FN = np.log(
        0.5 * (
            sp.special.erf(np.sqrt(E / (2.0 * (det + delta))) * (smax - smean))
            - sp.special.erf(np.sqrt(E / (2.0 * (det + delta))) * (smin - smean))
        )
    ) / Sn
 
    if chi_squared_only:
        return chi_squared
    if sumgals:
        return 0.5 * np.sum(chi_squared + log_det + 2.0 * FN)
    return 0.5 * (chi_squared + log_det)
 
 
def FN_func(FPparams, zobs, er, es, ei, lmin, lmax, smin, smax):
    """
    Calcule f_n : intégrale sur la gaussienne 3-D censurée du FP.
    Prend en compte la limite en magnitude et la coupure en dispersion.
 
    Retourne log(FN) avec plancher à 1e-15 pour éviter les problèmes numériques.
    """
    k = 0.0
    a, b, rmean, smean, imean, sigma1, sigma2, sigma3 = FPparams
    cov = _compute_covariance_terms(a, b, sigma1, sigma2, sigma3, k)
    sigmar2  = cov["sigmar2"]
    sigmas2  = cov["sigmas2"]
    sigmai2  = cov["sigmai2"]
    sigmars  = cov["sigmars"]
    sigmari  = cov["sigmari"]
    sigmasi  = cov["sigmasi"]
 
    err_r  = er**2 + np.log10(1.0 + 300.0 / (299792.458 * zobs))**2 + sigmar2
    err_s  = es**2 + sigmas2
    err_i  = ei**2 + sigmai2
    cov_ri = -1.0 * er * ei + sigmari
 
    A = err_s * err_i - sigmasi**2
    B = sigmasi * cov_ri - sigmars * err_i
    C = sigmars * sigmasi - err_s * cov_ri
    E = err_r * err_i - cov_ri**2
    F = sigmars * cov_ri - err_r * sigmasi
    I = err_r * err_s - sigmars**2
 
    det   = 1.0 / (err_r * A + sigmars * B + cov_ri * C)
    G     = np.sqrt(E) / (2 * F - B) * (C * (2 * F + B) - A * F - 2.0 * B * I)
    delta = (I * B**2 + A * F**2 - 2.0 * B * C * F) * det**2
    Edet  = E * det
    Gdet  = (G * det)**2
    Rmin  = (lmin - rmean - imean / 2.0) * np.sqrt(2.0 * delta / det) / (2.0 * F - B)
    Rmax  = (lmax - rmean - imean / 2.0) * np.sqrt(2.0 * delta / det) / (2.0 * F - B)
 
    G0min = -np.sqrt(2.0 / (1.0 + Gdet)) * Rmin
    G0max = -np.sqrt(2.0 / (1.0 + Gdet)) * Rmax
    G1min = -np.sqrt(Edet / (1.0 + delta)) * (smin - smean)
    G1max = -np.sqrt(Edet / (1.0 + delta)) * (smax - smean)
 
    H         = np.sqrt(1.0 + Gdet + delta)
    H0minmin  = G * det * np.sqrt(delta) - np.sqrt(Edet / 2.0) * (1.0 + Gdet) * (smin - smean) / Rmin
    H0minmax  = G * det * np.sqrt(delta) - np.sqrt(Edet / 2.0) * (1.0 + Gdet) * (smax - smean) / Rmin
    H0maxmin  = G * det * np.sqrt(delta) - np.sqrt(Edet / 2.0) * (1.0 + Gdet) * (smin - smean) / Rmax
    H0maxmax  = G * det * np.sqrt(delta) - np.sqrt(Edet / 2.0) * (1.0 + Gdet) * (smax - smean) / Rmax
    H1minmin  = G * det * np.sqrt(delta) - np.sqrt(2.0 / Edet) * (1.0 + delta) * Rmin / (smin - smean)
    H1minmax  = G * det * np.sqrt(delta) - np.sqrt(2.0 / Edet) * (1.0 + delta) * Rmin / (smax - smean)
    H1maxmin  = G * det * np.sqrt(delta) - np.sqrt(2.0 / Edet) * (1.0 + delta) * Rmax / (smin - smean)
    H1maxmax  = G * det * np.sqrt(delta) - np.sqrt(2.0 / Edet) * (1.0 + delta) * Rmax / (smax - smean)
 
    FN  = (sp.special.owens_t(G0min, H0minmax / H) - sp.special.owens_t(G0min, H0minmin / H)
           + sp.special.owens_t(G0max, H0maxmin / H) - sp.special.owens_t(G0max, H0maxmax / H))
    FN += (sp.special.owens_t(G1min, H1maxmin / H) - sp.special.owens_t(G1min, H1minmin / H)
           + sp.special.owens_t(G1max, H1minmax / H) - sp.special.owens_t(G1max, H1maxmax / H))
    FN += 1.0 / (2.0 * np.pi) * (
        np.arctan2(H0maxmax, H) + np.arctan2(H1maxmax, H)
        - np.arctan2(H0maxmin, H) - np.arctan2(H1maxmin, H)
    )
    FN += 1.0 / (2.0 * np.pi) * (
        np.arctan2(H0minmin, H) + np.arctan2(H1minmin, H)
        - np.arctan2(H0minmax, H) - np.arctan2(H1minmax, H)
    )
 
    # Plancher numérique : évite log(0) pour des distances très improbables
    FN = np.where(FN < 1.0e-15, 1.0e-15, FN)
    return np.log(FN)
 
 
# ---------------------------------------------------------------------------
# Calcul de la distribution de log-distance
# ---------------------------------------------------------------------------
def _pdf_moments(logP_dist, dbins):
    """
    Calcule moyenne, écart-type et asymétrie d'une PDF de log-distance.
 
    Retourne (mean, err, alpha) où alpha est le paramètre d'asymétrie skew-normal.
    """
    dx = dbins[1] - dbins[0]
    # Intégration trapézoïdale vectorisée
    mean = 0.5 * np.sum(
        dbins[:-1, None] * np.exp(logP_dist[:-1]) + dbins[1:, None] * np.exp(logP_dist[1:]),
        axis=0,
    ) * dx
    err = np.sqrt(
        0.5 * np.sum(
            dbins[:-1, None]**2 * np.exp(logP_dist[:-1]) + dbins[1:, None]**2 * np.exp(logP_dist[1:]),
            axis=0,
        ) * dx - mean**2
    )
    gamma1 = (
        0.5 * np.sum(
            dbins[:-1, None]**3 * np.exp(logP_dist[:-1]) + dbins[1:, None]**3 * np.exp(logP_dist[1:]),
            axis=0,
        ) * dx
        - 3.0 * mean * err**2 - mean**3
    ) / err**3
 
    gamma1 = np.clip(gamma1, -0.99, 0.99)
    delta  = np.sign(gamma1) * np.sqrt(
        np.pi / 2.0 / (1.0 + ((4.0 - np.pi) / (2.0 * np.abs(gamma1)))**(2.0 / 3.0))
    )
    alpha  = delta / np.sqrt(1.0 - delta**2)
    return mean, err, alpha
 
 
def compute_logdist(FPparams, fpmock):
    """
    Computes log-distance ratios with and without Malquist correction.
 
    Returns a dictionary with 
    logdist, logdist_err, logdist_alpha,
    logdist_corr, logdist_corr_err, logdist_corr_alpha.
    """
    dbins_ = dbins()
    d_H   = np.outer(10.0**(-dbins_), fpmock["dz_cluster"].to_numpy())
    lmin  = (
        4.65 + 5.0 * np.log10(1.0 + fpmock["zobs"].to_numpy())
        - cfg.fp_full.evo_corr * fpmock["zcos"].to_numpy()
        + fpmock["kcorr_r"].to_numpy() + 10.0
        - 2.5 * np.log10(2.0 * np.pi)
        + 5.0 * np.log10(d_H) - cfg.fp_full.mag_high
    ) / 5.0
    lmax  = (
        4.65 + 5.0 * np.log10(1.0 + fpmock["zobs"].to_numpy())
        - cfg.fp_full.evo_corr * fpmock["zcos"].to_numpy()
        + fpmock["kcorr_r"].to_numpy() + 10.0
        - 2.5 * np.log10(2.0 * np.pi)
        + 5.0 * np.log10(d_H) - cfg.fp_full.mag_low
    ) / 5.0
 
    loglike = FP_func(
        FPparams, dbins_,
        fpmock["zobs"].to_numpy(), fpmock["r"].to_numpy(),
        fpmock["s"].to_numpy(), fpmock["i"].to_numpy(),
        fpmock["er"].to_numpy(), fpmock["es"].to_numpy(), fpmock["ei"].to_numpy(),
        np.ones(len(fpmock)), cfg.fp_full.smin, cfg.fp_full.smax,
        sumgals=False, chi_squared_only=False,
    )
 
    t0 = time.time()
    FNvals = FN_func(
        FPparams,
        fpmock["zobs"].to_numpy(), fpmock["er"].to_numpy(),
        fpmock["es"].to_numpy(), fpmock["ei"].to_numpy(),
        lmin, lmax, cfg.fp_full.smin, cfg.fp_full.smax,
    )
    logger.info(f"FN_func computed in {time.time() - t0}")
 
    ddiff = np.log10(d_H[:-1]) - np.log10(d_H[1:])
 
    # --- Sans correction Malmquist ---
    logP = -1.5 * np.log(2.0 * np.pi) - loglike
    norm = 0.5 * np.sum((np.exp(logP[1:]) + np.exp(logP[:-1])) * ddiff, axis=0)
    logP -= np.log(norm[None, :])
    mean, err, alpha = _pdf_moments(logP, dbins_)
 
    # --- Avec correction Malmquist ---
    logP_corr = -1.5 * np.log(2.0 * np.pi) - loglike - FNvals
    norm_corr  = 0.5 * np.sum((np.exp(logP_corr[1:]) + np.exp(logP_corr[:-1])) * ddiff, axis=0)
    logP_corr -= np.log(norm_corr[None, :])
    mean_c, err_c, alpha_c = _pdf_moments(logP_corr, dbins_)
 
    return {
        "logdist": mean,        "logdist_err": err,        "logdist_alpha": alpha,
        "logdist_corr": mean_c, "logdist_corr_err": err_c, "logdist_corr_alpha": alpha_c,
    }
 
 
# ---------------------------------------------------------------------------
# Chargement & filtrage des données
# ---------------------------------------------------------------------------
def load_spec_data(path, usecols=None, delta_chi2_cut = 30.0):
    """
    Loads spec catalog and applies quality cut in redshift 
    """
    #spec_keys = [
    #    "targetid", "survey", "program", "healpix", "morphtype",
    #    "z", "zerr", "mag_r", 
    #    "mag_r_err",
        #"mag_err_r", 
    #    "mag_g", "mag_z",
    #    "sersic", 
    #    "zwarn",
        #"deltachi2", 
    #    "circ_radius", "circ_radius_err", "BA_ratio",
    #]
    df = pd.read_csv(path, usecols=usecols)
    before = len(df)
    #df = df[df["deltachi2"] >= delta_chi2_cut].reset_index(drop=True)
    df = df[df["zwarn"] == 0].reset_index(drop=True)
    logger.info(f"spec : {before} → {len(df)} entries after cut on zwarn == 0")
    return df
 
 
def load_fp_catalog(path):
    """
    Loads FP catalog and keeps only calibrators.
    """
    keys = ["targetid", "ra", "dec", "zcmb", "zcmb_group", "ppxf_vdisp", "ppxf_vdisp_err",
            "r", "er", "s", "es", "i", "ei", "Sn", "FPcalibrator"]
    df = pd.read_csv(path, usecols=keys)
    logger.info(f"FP catalogue : {len(df)} galaxies total")
    df = df[df["FPcalibrator"] == 1].reset_index(drop=True)
    logger.info(f"FP catalogue : {len(df)} calibrators kept")

    return df
 
 
def load_mock_hdf5(infile: str, spec: pd.DataFrame) -> pd.DataFrame:
    """Charge un mock HDF5 et fusionne avec le catalogue spec."""
    fpmock: dict = {}
    with h5py.File(infile, "r") as f:
        for key in f.keys():
            if key == "vel":
                fpmock["vx"] = f["vel"][:, 0]
                fpmock["vy"] = f["vel"][:, 1]
                fpmock["vz"] = f["vel"][:, 2]
            else:
                fpmock[key] = f[key][()]
            if key in ("survey", "program"):
                fpmock[key] = fpmock[key].astype("U")
 
    df = pd.DataFrame.from_dict(fpmock)
    before = len(df)
    df = df.merge(spec, how="inner", on=["targetid", "survey", "program", "healpix"])
    logger.info("Mock merged with spec properties : %d → %d lignes", before, len(df))
    return df
 
 
def filter_mock(fpmock: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cuts based on the following parameters:
    redshift, morphology, ellipticity, color, magnitude and completeness
    """
    steps = [
        ("initial", len(fpmock)),
    ]
 
    fpmock = fpmock[(fpmock["zobs"] >= cfg.fp_full.zmin) & (fpmock["zobs"] <= cfg.fp_full.zmax)]
    steps.append(("redshift", len(fpmock)))
 
    fpmock = fpmock[
        (fpmock["morphtype"] == "DEV") |
        ((fpmock["morphtype"] == "SER") & (fpmock["sersic"] > 2.5))
    ]
    steps.append(("morphology", len(fpmock)))
 
    fpmock = fpmock[fpmock["BA_ratio"] > 0.3]
    steps.append(("ellipticity", len(fpmock)))
 
    #- v1 cuts
    if cfg.data_fp_full_version == 'v1':
        fpmock = fpmock[
            (fpmock["col_obs"] > 0.68) &
            (fpmock["col_obs"] > 1.3 * (fpmock["app_mag"] - fpmock["mag_z"]) - 0.12) &
            (fpmock["col_obs"] < 2.0 * (fpmock["app_mag"] - fpmock["mag_z"]) - 0.15)
        ]
    # v2 cuts
    elif cfg.data_fp_full_version == 'v2':
        fpmock = fpmock[
            (fpmock["col_obs"] > 0.68) &
            (fpmock["col_obs"] > 0.85 * (fpmock["app_mag"] - fpmock["mag_z"]) + 0.30) &
            (fpmock["col_obs"] < 2.0 * (fpmock["app_mag"] - fpmock["mag_z"]) - 0.15)
        ]
    else:
        logger.error(f" No color cuts defined for version {cfg.data_fp_full_version} !")
    steps.append(("color", len(fpmock)))
 
    fpmock = fpmock[(fpmock["app_mag"] > cfg.fp_full.mag_low) & (fpmock["app_mag"] < cfg.fp_full.mag_high)]
    steps.append(("magnitude", len(fpmock)))
 
    mask = np.random.rand(len(fpmock)) < fpmock[cfg.comp_field].values
    fpmock = fpmock[mask]
    steps.append(("completeness", len(fpmock)))
 
    for name, n in steps:
        logger.info(f"  {name} : {n} galaxies")
 
    return fpmock
 
 
# ---------------------------------------------------------------------------
# Génération des propriétés FP synthétiques
# ---------------------------------------------------------------------------
def generate_fp_properties(fpmock: pd.DataFrame) -> pd.DataFrame:
    """
    Draw FP properties (s, i, r) from absolute magnitude  
    using the conditional distribution of the FP. 
    """
    k = 0.0
    a, b, sigma1, sigma2, sigma3 = cfg.fp_full.a, cfg.fp_full.b, cfg.fp_full.sigma1, cfg.fp_full.sigma2, cfg.fp_full.sigma3
    cov = _compute_covariance_terms(a, b, sigma1, sigma2, sigma3, k)
 
    sigmar2, sigmas2, sigmai2 = cov["sigmar2"], cov["sigmas2"], cov["sigmai2"]
    sigmars, sigmari, sigmasi  = cov["sigmars"], cov["sigmari"], cov["sigmasi"]
 
    sigmaM2  = 25.0 * sigmar2 + 25.0 * sigmari + 6.25 * sigmai2
    sigmaMs  = -5.0 * sigmars - 2.5 * sigmasi
    sigmaMi  = -5.0 * sigmari - 2.5 * sigmai2
 
    hats = cfg.fp_full.smean + sigmaMs / sigmaM2 * (fpmock["abs_mag"].to_numpy() - Mmean())
    hati = cfg.fp_full.imean + sigmaMi / sigmaM2 * (fpmock["abs_mag"].to_numpy() - Mmean())
    sigma_cond = np.array([
        [sigmas2 - sigmaMs**2 / sigmaM2,       sigmasi - sigmaMs * sigmaMi / sigmaM2],
        [sigmasi - sigmaMs * sigmaMi / sigmaM2, sigmai2 - sigmaMi**2 / sigmaM2],
    ])
 
    rng = np.random.default_rng()
    means = np.column_stack([hats, hati])
    draw  = rng.multivariate_normal(np.zeros(2), sigma_cond, size=len(fpmock)) + means
 
    fpmock = fpmock.copy()
    fpmock["s"] = draw[:, 0]
    fpmock["i"] = draw[:, 1]
    fpmock["r"] = (4.65 - fpmock["abs_mag"].to_numpy() - 2.5 * fpmock["i"].to_numpy()
                   - 2.5 * np.log10(2.0 * np.pi) - 15.0) / 5.0
    return fpmock
 
 
def assign_fp_errors(fpmock: pd.DataFrame, fp_data: pd.DataFrame) -> pd.DataFrame:
    """
    Assign FP errors to each mock galaxy by finding the closest real galaxy
    in the normalized (r, s, i) space. Use the 2nd closest neighbour to avoid a
    trivial self-assignment.
    """
    def _norm(arr):
        lo, hi = np.amin(arr), np.amax(arr)
        return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
 
    mock_r = fpmock["r"].to_numpy()
    mock_s = fpmock["s"].to_numpy()
    mock_i = fpmock["i"].to_numpy()
 
    tree = KDTree(np.column_stack([
        _norm(fp_data["r"].to_numpy()),
        _norm(fp_data["s"].to_numpy()),
        _norm(fp_data["i"].to_numpy()),
    ]))
    query_pts = np.column_stack([_norm(mock_r), _norm(mock_s), _norm(mock_i)])
    _, neighbour = tree.query(query_pts, k=2)
 
    idx = neighbour[:, 1]
    fpmock = fpmock.copy()
    fpmock["er"] = fp_data["er"].to_numpy()[idx]
    fpmock["es"] = fp_data["es"].to_numpy()[idx]
    fpmock["ei"] = fp_data["ei"].to_numpy()[idx]
    return fpmock
 
 
def perturb_fp_observations(fpmock: pd.DataFrame) -> pd.DataFrame:
    """
    Perturb the FP variables by the observational errors.
 
    The covariance matrix of the errors is :
        [[err_r² + sigma_pec², 0,            -err_r * err_i],
         [0,                   err_s²,        0            ],
         [-err_r * err_i,      0,             err_i²       ]]
 
    Vectorized across the entire catalogue to avoid the Python loop.
    """
    rnew = fpmock["r"].to_numpy() + fpmock["logdist_true"].to_numpy()
    er   = fpmock["er"].to_numpy()
    es   = fpmock["es"].to_numpy()
    ei   = fpmock["ei"].to_numpy()
    zobs = fpmock["zobs"].to_numpy()
 
    sigma_pec = np.log10(1.0 + 300.0 / (299792.458 * zobs))
    var_r = er**2 + sigma_pec**2
 
    rng = np.random.default_rng()
    n   = len(fpmock)
 
    # Tirage indépendant pour s
    noise_s = rng.normal(0.0, es, size=n)
 
    # Tirage corrélé pour (r, i) via décomposition de Cholesky 2×2
    # Cov = [[var_r, -er*ei], [-er*ei, ei²]]
    L11 = np.sqrt(var_r)
    L21 = -er * ei / L11
    L22 = np.sqrt(np.maximum(ei**2 - L21**2, 0.0))
 
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    noise_r = L11 * z1
    noise_i = L21 * z1 + L22 * z2
 
    fpmock = fpmock.copy()
    fpmock["r"] = rnew + noise_r
    fpmock["s"] = fpmock["s"].to_numpy() + noise_s
    fpmock["i"] = fpmock["i"].to_numpy() + noise_i
    return fpmock
 
 
# ---------------------------------------------------------------------------
# Ajustement du Plan Fondamental
# ---------------------------------------------------------------------------
def fit_fundamental_plane(fpmock: pd.DataFrame) -> tuple:
    """
    Fits the FP with differential evolution with outlier rejection

    Returns  (FPparams_array, data_fit_DataFrame, badcount_int).
    """
    bounds = [
        (1.0, 1.8),    # a
        (-1.5, -0.5),  # b
        (-0.5, 0.5),   # rmean
        (2.0, 2.4),    # smean
        (2.4, 3.0),    # imean
        (0.01, 0.12),  # sigma1
        (0.05, 0.5),   # sigma2
        (0.1, 0.3),    # sigma3
    ]
 
    # Initialisation avec les paramètres des données réelles
    data_bestfit = fp_params()
    chi_sq = fpmock["Sn"].to_numpy() * FP_func(
        data_bestfit, 0.0,
        fpmock["zobs"].to_numpy(), fpmock["r"].to_numpy(),
        fpmock["s"].to_numpy(), fpmock["i"].to_numpy(),
        fpmock["er"].to_numpy(), fpmock["es"].to_numpy(), fpmock["ei"].to_numpy(),
        fpmock["Sn"].to_numpy(), cfg.fp_full.smin, cfg.fp_full.smax,
        sumgals=False, chi_squared_only=True,
    )[0]
    dof = np.sum(chi_sq) / (len(fpmock) - 8.0)
    pvals     = sp.stats.chi2.sf(chi_sq, dof)
    data_fit  = fpmock[pvals >= 0.01].reset_index(drop=True)
    badcount  = int(np.sum(pvals < 0.01))
    
    logger.info(f"Init FP : chi2={np.sum(chi_sq):.1f}  n_fit={len(data_fit)} n_out={badcount}" )
 
    converged = False
    t0 = time.time()
 
    while not converged:
        result = sp.optimize.differential_evolution(
            FP_func,
            bounds=bounds,
            args=(
                0.0,
                data_fit["zobs"].to_numpy(), data_fit["r"].to_numpy(),
                data_fit["s"].to_numpy(), data_fit["i"].to_numpy(),
                data_fit["er"].to_numpy(), data_fit["es"].to_numpy(),
                data_fit["ei"].to_numpy(), data_fit["Sn"].to_numpy(),
                cfg.fp_full.smin, cfg.fp_full.smax,
            ),
            maxiter=10000,
            tol=1.0e-6,
            disp=False,
        )
 
        chi_sq = fpmock["Sn"].to_numpy() * FP_func(
            result.x, 0.0,
            fpmock["zobs"].to_numpy(), fpmock["r"].to_numpy(),
            fpmock["s"].to_numpy(), fpmock["i"].to_numpy(),
            fpmock["er"].to_numpy(), fpmock["es"].to_numpy(), fpmock["ei"].to_numpy(),
            fpmock["Sn"].to_numpy(), cfg.fp_full.smin, cfg.fp_full.smax,
            sumgals=False, chi_squared_only=True,
        )[0]
        dof       = np.sum(chi_sq) / (len(fpmock) - 8.0)
        pvals     = sp.stats.chi2.sf(chi_sq, dof)
        data_fit  = fpmock[pvals >= 0.01].reset_index(drop=True)
        badcount_new = int(np.sum(pvals < 0.01))
        converged    = (badcount == badcount_new)
 
        logger.info(
            "FP iter : chi2=%.1f  n_fit=%d  n_out=%d→%d  converged=%s",
            np.sum(chi_sq), len(data_fit), badcount, badcount_new, converged,
        )
        badcount = badcount_new
 
    logger.info(f"FP fit done in {time.time() - t0} sec")

    return result.x, data_fit, badcount
 
 
# ---------------------------------------------------------------------------
# Calcul de Sn (poids de sélection par volume)
# ---------------------------------------------------------------------------
def compute_selection_weights(fpmock, cosmo, lumred_spline):
    """
    Computes the weight Sn = fraction of survey volume in which each galaxy 
    would be observed given its magniture 
    """
    Vmin = (1.0 + cfg.fp_full.zmin)**3 * cosmo.comoving_distance(cfg.fp_full.zmin).value**3
    Vmax = (1.0 + cfg.fp_full.zmax)**3 * cosmo.comoving_distance(cfg.fp_full.zmax).value**3
 
    Dlim = 10.0**(
        (cfg.fp_full.mag_high - fpmock["app_mag"].to_numpy()
         + 5.0 * np.log10(fpmock["dz"].to_numpy())
         + 5.0 * np.log10(1.0 + fpmock["zobs"].to_numpy())) / 5.0
    )
    zlim = lumred_spline(Dlim)
 
    Sn = np.where(
        zlim >= cfg.fp_full.zmax, 1.0,
        np.where(zlim < cfg.fp_full.zmin, 0.0, (Dlim**3 - Vmin) / (Vmax - Vmin)),
    )
    fpmock = fpmock.copy()
    fpmock["Sn"] = Sn
    return fpmock
 
 
# ---------------------------------------------------------------------------
# Écriture du catalogue de sortie
# ---------------------------------------------------------------------------
def write_output_catalog(outfile: str, fpmock: pd.DataFrame, FPparams: np.ndarray,
                          chi2: float, badcount: int) -> None:
    """Écrit le catalogue mock final au format FITS."""
    a, b, rmean, smean, imean, s1, s2, s3 = FPparams
    hdr = fits.Header({
        "a": a, "b": b, "c": rmean - a * smean - b * imean,
        "rmean": rmean, "smean": smean, "imean": imean,
        "sigma1": s1, "sigma2": s2, "sigma3": s3,
        "chi2": chi2, "nFP": len(fpmock), "nout": badcount,
    })
 
    col_defs = [
        ("RA",                "D", "ra"),
        ("DEC",               "D", "dec"),
        ("ZOBS",              "D", "zobs"),
        ("ZCOS",              "D", "zcos"),
        ("vx",                "D", "vx"),
        ("vy",                "D", "vy"),
        ("vz",                "D", "vz"),
        ("r",                 "D", "r"),
        ("er",                "D", "er"),
        ("s",                 "D", "s"),
        ("es",                "D", "es"),
        ("i",                 "D", "i"),
        ("ei",                "D", "ei"),
        ("Sn",                "D", "Sn"),
        ("LOGDIST_TRUE",      "D", "logdist_true"),
        ("LOGDIST",           "D", "logdist"),
        ("LOGDIST_ERR",       "D", "logdist_err"),
        ("LOGDIST_ALPHA",     "D", "logdist_alpha"),
        ("LOGDIST_CORR",      "D", "logdist_corr"),
        ("LOGDIST_CORR_ERR",  "D", "logdist_corr_err"),
        ("LOGDIST_CORR_ALPHA","D", "logdist_corr_alpha"),
    ]
    columns = [fits.Column(name=n, format=f, array=fpmock[k].to_numpy())
               for n, f, k in col_defs]
 
    hdu = fits.BinTableHDU.from_columns(columns, header=hdr)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    hdu.writeto(outfile, overwrite=True)
    logger.info(f"Catalog with {len(fpmock)} galaxies written to : {outfile}")
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline FP mock — calcul des log-distances pour BGS AbacusSummit."
    )
    parser.add_argument("config_file", type=str, help="Configuration file path (yaml format)")
    parser.add_argument("phase",        type=int, help="Phase (0–24)")
    parser.add_argument("real", type=int, help="Realisation (0-26)")
    args = parser.parse_args()
    if not (0 <= args.phase <= 24):
        parser.error("phase should be between 0 and 24")
    if not (0 <= args.real <= 26):
        parser.error("real should be between 0 and 26")
    return args
 
 
# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    phase, real = args.phase, args.real
    logger.info("=== Pipeline FP  phase=%03d  real=%03d ===", phase, real)
    
    global cfg
    cfg = load_config(args.config_file)
    
    outfile = cfg.mock_fp_full_data.format(phase=phase, real=real)
    if os.path.exists(outfile) and not cfg.fp_full.overwrite:
        logger.info("Already exists: %s — skipped", outfile)
        return

    # --- Cosmologies ---
    cosmo = FlatLambdaCDM(H0=100, Om0=0.3151)   #  DESI fiducial cosmology 
    zvals = np.logspace(-5.0, 3.0, 10000)
    lumred_spline = sp.interpolate.interp1d((1.0 + zvals) * cosmo.comoving_distance(zvals), zvals)
 
    # Corrections K (cosmologie Planck15 as in the original mocks)
    k_r = GAMA_KCorrection(Planck15, cfg.kcorr_r_path)
    k_g = GAMA_KCorrection(Planck15, cfg.kcorr_g_path)
 
    # --- Reading reference data catalogs ---
    spec   = load_spec_data(cfg.spec_csv, usecols=cfg.spec_keys)
    fp_data = load_fp_catalog(cfg.data_fp_full)
 
    # --- Reading mock ---
    infile = cfg.mock_bgs_spec_data.format(phase=phase, real=real)
    logger.info(f"Reading : {infile}")
    fpmock = load_mock_hdf5(infile, spec)
 
    fpmock["kcorr_r"] = k_r.k(fpmock["zobs"], fpmock["col_obs"])
    fpmock["kcorr_g"] = k_g.k(fpmock["zobs"], fpmock["col_obs"])
 
    logger.info("--- Applying cuts on mock ---")
    fpmock = filter_mock(fpmock)
 
    # --- Cosmological Distances ---
    fpmock["dz"]          = cosmo.comoving_distance(fpmock["zobs"].to_numpy()).value
    fpmock["dz_cluster"]  = cosmo.comoving_distance(fpmock["zcos"].to_numpy()).value
    fpmock["logdist_true"]= np.log10(fpmock["dz"].to_numpy() / fpmock["dz_cluster"].to_numpy())
 
    # --- Generating synthetic FP properties ---
    fpmock = generate_fp_properties(fpmock)
    fpmock = assign_fp_errors(fpmock, fp_data)
    fpmock = perturb_fp_observations(fpmock)
 
    # --- Final cut on dispersion --- 
    fpmock = fpmock[
        (fpmock["s"] >= cfg.fp_full.smin) & (fpmock["s"] <= cfg.fp_full.smax)
    ].reset_index(drop=True)
    logger.info(f"After dispersion cut : {len(fpmock)} galaxies")
 
    # --- Selection weights ---
    fpmock = compute_selection_weights(fpmock, cosmo, lumred_spline)
 
    # --- Fitting for the Fundamenta Plane ---
    FPparams, data_fit, badcount = fit_fundamental_plane(fpmock)
    fpmock = data_fit
    logger.info(f"Mock after final outlier rejection : {len(fpmock)} galaxies")
 
    # --- Log-distances ---
    logdist_results = compute_logdist(FPparams, fpmock)
    for col, vals in logdist_results.items():
        fpmock[col] = vals
 
    # --- Writing output mock ---
    chi2_final = float(np.sum(
        fpmock["Sn"].to_numpy() * FP_func(
            FPparams, 0.0,
            fpmock["zobs"].to_numpy(), fpmock["r"].to_numpy(),
            fpmock["s"].to_numpy(), fpmock["i"].to_numpy(),
            fpmock["er"].to_numpy(), fpmock["es"].to_numpy(), fpmock["ei"].to_numpy(),
            fpmock["Sn"].to_numpy(), cfg.fp_full.smin, cfg.fp_full.smax,
            sumgals=False, chi_squared_only=True,
        )[0]
    ))
    write_output_catalog(outfile, fpmock, FPparams, chi2_final, badcount)
 
    logger.info("=== Updating permissions ===")
    result = subprocess.run(
        ["chgrp", "-R", "desi", cfg.mock_fp_full_dir],
        check=True,
    )
    logger.info("=== Pipeline done ===")



if __name__ == "__main__":
    main()