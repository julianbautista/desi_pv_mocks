#!/usr/bin/env python
# coding: utf-8
"""
TFR Mock Dataset Generator
==========================
Generates a simulated Tully-Fisher dataset by:
  1. Merging spec photometric + spectroscopic catalogs with SGA-2020.
  2. Applying all photometric corrections used in the TF analysis.
  3. Cross-matching with a BGS mock.
  4. Applying photometric and morphological cuts for late-type galaxies.
  5. Generating mock rotational velocities via an inverted TFR best fit.
  6. Generating mock TFR distance moduli.
"""

from dataclasses import dataclass
import os
import shutil
import h5py
import pickle
import healpy as hp
import pandas as pd
import numpy as np
import scipy as sp
import logging

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from glob import glob
from itertools import groupby

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.table import Table
from csaps import csaps
from scipy.odr import Model, ODR, RealData
from scipy.spatial import KDTree
from scipy.stats import binned_statistic
from tqdm import tqdm

#sys.path.append(TF_MOCK_PATH)
import TF_photoCorrect as tfpc  # noqa: E402  (project-local, must come after sys.path)

#from hyperfit.linfit import LinFit          # noqa: E402
from hyperfit_v2 import MultiLinFit         # noqa: E402
#from line_fits import hyperfit_line_multi   # noqa: E402



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

@dataclass
class Config:
    
    # ── Matching ──────────────────────────────────────────────────────────────
    # SGA center match threshold: (angular sep) / (D26/2) < center_match_frac
    center_match_frac: float = 0.1

    # ── Velocity limits for CDF re-sampling [log10(km/s)] ────────────────────
    logv_min: float = 1.0
    logv_max: float = 3.0

    # ── Binning ───────────────────────────────────────────────────────────────
    # Minimum number of galaxies per magnitude bin before merging
    min_bin_count: int = 50
    # Magnitude bin width for building the logv CDF
    mr_bin_width: float = 0.05

    # ── TFR fitting ───────────────────────────────────────────────────────────
    # Calibration sample size per TFR fit realisation
    calib_sample_size: int = 4_200
    # Number of Monte-Carlo realisations used to average TFR fit parameters
    n_realisations: int = 25

    # ── Velocity error floor [km/s] ───────────────────────────────────────────
    # Previously hard-coded in some branches; currently not applied
    logv_err_floor: float = 7.0

    # ── Paths ─────────────────────────────────────────────────────────────────
    spec_name = 'loa'
    mock_version = 'v2.0'
    tf_data_version = 'v2'
    tf_mock_version = mock_version + '.0'
    comp_field = 'Y3_COMP'
    overwrite = False

    pv_path: str = "/global/cfs/cdirs/desi/science/td/pv"
    sga_file: str = "/global/cfs/cdirs/cosmo/data/sga/2020/SGA-2020.fits"
    dust_map: str = (
        "/global/cfs/cdirs/desi/public/papers/mws/desi_dust/y2/v1/maps/"
        "desi_dust_gr_512.fits"
    )
    spec_cat_name: str = pv_path + "/redshift_data/Y1/iron_fullsweep_catalogue_z012.csv"
        
    specsp: str = (#pv_path + "/redshift_data/Y1/specprod_iron_healpix_z015.csv"
                "/global/cfs/cdirs/desi/science/td/pv/fpgalaxies/Y3/s_syst/"
                "loa_healpix_data__pre_FP_analysis.csv"
    )

    

    # ── Paths ────────────────────────────────────
    tf_data_file: str = (
        pv_path + f"/tfgalaxies/Y3/DESI-DR2_TF_pv_cat_{tf_data_version}.fits"
    )

    mock_path = os.path.join(pv_path, "mocks/DR2")
    
    mock_infile = (
        f"{mock_path}/BGS_{spec_name}/{mock_version}/data/"+
        "BGS_PV_AbacusSummit_spec_c000_ph{phase:03d}_r{real:03d}.dat.hdf5"
    )

    tfr_pickle: str = os.path.join(pv_path, "tfgalaxies/Y3",
                "cov_ab_iron_jointTFR_varyV0-dwarfsAlex_z0p1_zbins0p005_weightsVmax-1_dVsys_KAD-20250810.pickle")
    
    mock_tf_full_dir: str = (mock_path + f"/TF_mocks/full_mocks/{tf_mock_version}")

    mock_tf_full_outfile = (mock_tf_full_dir +
        "/TF_AbacusSummit_c000_ph{phase:03d}_r{real:03d}.fits"
    )

# Default instance — import and use directly, or override fields as needed
cfg = Config()



# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def profile_histogram(x, y, xbins, *, yerr=None, weights=None,
                      median=False, weighted=False):
    """Compute a profile histogram from scattered data.

    Parameters
    ----------
    x, y : array-like
        Independent and dependent variables.
    xbins : array-like
        Bin edges for *x*.
    yerr : array-like, optional
        Per-point uncertainties on *y* (used when ``weighted=True``).
    weights : array-like, optional
        Explicit weights (overrides *yerr* when ``weighted=True``).
    median : bool
        Use median instead of (weighted) mean as the central value.
    weighted : bool
        Weight the summary statistics by 1/yerr² (or *weights*).

    Returns
    -------
    N : ndarray  – unweighted counts per bin.
    h : ndarray  – central value per bin.
    e : ndarray  – uncertainty on the central value per bin.
    """
    N = binned_statistic(x, y, bins=xbins, statistic="count").statistic

    if weighted:
        if yerr is None and weights is None:
            raise ValueError("Provide either yerr or weights when weighted=True.")
        w = weights if weights is not None else 1.0 / yerr ** 2
        W, H, _ = binned_statistic(x, [w, w * y, w * y ** 2],
                                   bins=xbins, statistic="sum").statistic
        h = H / W
        e = 1.0 / np.sqrt(W)
    else:
        mean, mean2 = binned_statistic(x, [y, y ** 2],
                                       bins=xbins, statistic="mean").statistic
        h = mean
        e = np.sqrt((mean2 - mean ** 2) / (N - 1))

    if median:
        h = binned_statistic(x, y, bins=xbins, statistic="median").statistic

    return N, h, e


def downsample(catalog: pd.DataFrame, size: int) -> pd.DataFrame:
    """Return *size* rows drawn without replacement from *catalog*."""
    if size >= len(catalog):
        return catalog.copy()
    idx = np.random.choice(len(catalog), size, replace=False)
    return catalog.iloc[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Quality cuts
# ─────────────────────────────────────────────────────────────────────────────

def alex_cuts_velocity(catalog: pd.DataFrame, *,
                       logv_name: str = "logv_rot",
                       distmod_name: str = "MU_ZCMB",
                       vmin: float = 70.0,
                       vmax: float = 300.0,
                       h: float = 1.0) -> pd.Series:
    """Boolean mask: galaxies passing Alex's velocity cuts (Aug. 2025).

    Keeps galaxies with
      * vmin < V_rot < vmax  (flat bounds)
      * V_rot < min(vmax, 10^{0.3*(mu - 34 + 5log h) + 2})  (distance-dependent)
    """
    logV_min = np.log10(vmin)
    logV_max = np.log10(vmax)
    mu_obs   = catalog[distmod_name]
    logV_M_max = np.minimum(logV_max, 0.3 * (mu_obs - (34.0 + 5.0 * np.log10(h))) + 2.0)
    return (
        (catalog[logv_name] > logV_min)
        & (catalog[logv_name] < logV_max)
        & (catalog[logv_name] < logV_M_max)
    )


def alex_cuts_dwarf(catalog: pd.DataFrame, *,
                    rmag_name: str = "R_ABSMAG_SB26",
                    distmod_name: str = "MU_ZCMB",
                    h: float = 1.0) -> pd.Series:
    """Boolean mask: galaxies that are *not* classified as dwarfs (Aug. 2025).

    Keeps galaxies where m_r <= min(17.75, mu_CMB - 17 + 5 log h).
    """
    M_lim   = -17.0 + 5.0 * np.log10(h)
    R_lim   = np.minimum(17.75, catalog[distmod_name] + M_lim)
    return catalog[rmag_name] <= R_lim


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_spec_catalog() -> pd.DataFrame:
    """Load, merge, and filter the spec fullsweep + specprod catalogs."""
    sw_keys = [
        "targetid", "survey", "program", "healpix",
        "target_ra", "target_dec",
        "z", "zerr", "zwarn", "inbasiccuts", "has_corrupt_phot",
        "mag_g", "mag_r", "mag_z",
        "morphtype", "sersic", "BA_ratio",
        "circ_radius", "circ_radius_err", "uncor_radius", "SGA_id", "radius_SB25",
    ]
    sp_keys = [
        "targetid", "survey", "program", "healpix",
        "mag_err_g", "mag_err_r", "mag_err_z", "deltachi2",
    ]
    merge_keys = ["targetid", "survey", "program", "healpix"]

    spec = pd.read_csv(cfg.spec_cat_name, usecols=sw_keys)
    specsp = pd.read_csv(cfg.specsp, usecols=sp_keys)
    spec = pd.merge(spec, specsp, on=merge_keys, how="inner")

    # Spectroscopic pipeline selection
    select = (spec["SGA_id"] > 0) & (spec["deltachi2"] >= 25) & (spec["zwarn"] == 0)
    spec = spec.loc[select].copy()

    # Merge SGA-2020 data
    sga_cols = [
        "SGA_id", "SGA_ra", "SGA_dec",
        "D26",
        "G_MAG_SB26", "G_MAG_SB26_ERR",
        "R_MAG_SB26", "R_MAG_SB26_ERR",
        "Z_MAG_SB26", "Z_MAG_SB26_ERR",
    ]
    sgacat = Table.read(cfg.sga_file, "ELLIPSE")
    sgacat.rename_column("SGA_ID", "SGA_id")
    sgacat.rename_column("RA",     "SGA_ra")
    sgacat.rename_column("DEC",    "SGA_dec")
    sgacat = sgacat[sga_cols].to_pandas()
    sgacat = sgacat.loc[sgacat["R_MAG_SB26"] >= 0]

    spec = pd.merge(spec, sgacat, on="SGA_id", how="inner").dropna()

    # Keep only galaxy-center spectra
    coords_sga  = SkyCoord(ra=spec["SGA_ra"].values,  dec=spec["SGA_dec"].values,  unit="deg")
    coords_spec = SkyCoord(ra=spec["target_ra"].values, dec=spec["target_dec"].values, unit="deg")
    sep2d = coords_spec.separation(coords_sga)
    center_ok = (2.0 * sep2d.to_value("arcmin") / spec["D26"].values) < cfg.center_match_frac
    spec = spec.loc[center_ok].copy()

    log.info("spec catalog loaded: %d galaxies", len(spec))
    return spec


def load_mock(mockfile: str, spec: pd.DataFrame) -> pd.DataFrame:
    """Load a BGS mock HDF5 file and cross-match it with *spec*."""
    merge_keys = ["targetid", "survey", "program", "healpix"]
    mock_dict: dict = {}

    with h5py.File(mockfile, "r") as f:
        for key in f.keys():
            if key == "vel":
                mock_dict["vx"] = f["vel"][:, 0]
                mock_dict["vy"] = f["vel"][:, 1]
                mock_dict["vz"] = f["vel"][:, 2]
            else:
                mock_dict[key] = f[key][()]
            if key in ("survey", "program"):
                mock_dict[key] = mock_dict[key].astype("U")

    mock = pd.DataFrame.from_dict(mock_dict)
    mock = mock.merge(spec, on=merge_keys, how="inner")
    log.info("Mock loaded: %d galaxies after cross-match", len(mock))
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# Photometric corrections
# ─────────────────────────────────────────────────────────────────────────────

def apply_photo_corrections(spec: pd.DataFrame) -> pd.DataFrame:
    """Apply imaging systematics, k-correction, MW dust, and internal dust.

    Returns *spec* with new columns ``{G,R,Z}_MAG_SB26_CORR`` and their
    ``_ERR_CORR`` counterparts (r-band only gets the internal-dust correction).
    """
    spec = spec.copy()

    # 1. Imaging photometric systematics (N/S)
    c = SkyCoord(spec["target_ra"], spec["target_dec"], unit="deg")
    is_north = (c.galactic.b > 0) & (spec["target_dec"] > 32.375)
    spec["photsys"] = np.where(is_north, "N", "S")
    A_sys, A_sys_err = tfpc.BASS_corr(spec["photsys"])

    # 2. K-correction to z = 0.1
    valid_z = spec["z"] > 0
    kc_grz = tfpc.k_corr(
        spec["z"][valid_z],
        [spec[f"{b}_MAG_SB26"][valid_z] for b in ("G", "R", "Z")],
        [spec[f"{b}_MAG_SB26_ERR"][valid_z] for b in ("G", "R", "Z")],
        z_corr=0.1,
    )
    A_k = np.zeros((len(spec), 3))
    A_k[valid_z] = kc_grz

    # 3. MW dust correction
    ebv = Table.read(cfg.dust_map)
    A_dust, A_dust_err = tfpc.MW_dust(spec["target_ra"].values,
                                       spec["target_dec"].values, ebv)
    for band_idx, band in enumerate("grz"):
        nan_mask = np.isnan(A_dust[band_idx])
        if nan_mask.any():
            log.warning("NaN MW dust correction for band %s – zeroing %d entries",
                        band, nan_mask.sum())
            A_dust[band_idx][nan_mask]     = 0.0
            A_dust_err[band_idx][nan_mask] = 0.0

    # 4. Apply MW + k + sys to all three bands
    for i, band in enumerate("GRZ"):
        spec[f"{band}_MAG_SB26_tmp"] = (
            spec[f"{band}_MAG_SB26"] - A_dust[i] + A_sys + A_k[:, i]
        )
        spec[f"{band}_MAG_SB26_ERR_tmp"] = np.sqrt(
            spec[f"{band}_MAG_SB26_ERR"] ** 2 + A_dust_err[i] ** 2 + A_sys_err ** 2
        )

    # 5. Internal dust correction (r-band only)
    ba_bins   = np.arange(0.1, 1.0, 0.1)
    ba_centre = 0.5 * (ba_bins[1:] + ba_bins[:-1])
    ba_err    = 0.5 * np.diff(ba_bins)

    m_r_median = np.median(spec["R_MAG_SB26_tmp"])
    m_r, _, _  = binned_statistic(spec["BA_ratio"], spec["R_MAG_SB26_tmp"],
                                   statistic="median", bins=ba_bins)
    n_bin, _, _ = binned_statistic(spec["BA_ratio"], spec["R_MAG_SB26_tmp"],
                                   statistic="count", bins=ba_bins)
    m_r_err, _, _ = binned_statistic(spec["BA_ratio"], spec["R_MAG_SB26_tmp"],
                                     statistic="std", bins=ba_bins)
    m_r_err /= np.sqrt(n_bin)

    linear_fit = lambda coeff, x: coeff[0] * x + coeff[1]
    odr_result = ODR(
        RealData(ba_centre, m_r - m_r_median, sx=ba_err, sy=m_r_err),
        Model(linear_fit),
        beta0=[1.0, 1.0],
    ).run()
    log.info("Internal dust fit:   %s ± %s", odr_result.beta, odr_result.sd_beta)

    A_int, A_int_err = tfpc.internal_dust(spec["BA_ratio"].values,
                                           odr_result.beta, odr_result.sd_beta)
    spec["R_MAG_SB26_CORR"]     = spec["R_MAG_SB26_tmp"] - A_int
    spec["R_MAG_SB26_ERR_CORR"] = np.sqrt(spec["R_MAG_SB26_ERR_tmp"] ** 2
                                           + A_int_err ** 2)
    return spec


# ─────────────────────────────────────────────────────────────────────────────
# TFR selection cuts
# ─────────────────────────────────────────────────────────────────────────────

def apply_tf_selection(mock: pd.DataFrame) -> pd.DataFrame:
    """Apply late-type galaxy cuts (Saulder+ 2023) to *mock*.

    Returns the filtered catalog and logs row counts at each step.
    """
    n0 = len(mock)
    log.info("Cross-matched spec+mock catalog: %d", n0)

    # Photometric quality
    bad_phot = (mock["inbasiccuts"] == 0) | (mock["has_corrupt_phot"] == 1)
    mock = mock.loc[~bad_phot].copy()
    log.info("After photometric cuts:          %d", len(mock))

    # Inclination / axial ratio
    mock = mock.loc[mock["BA_ratio"] < np.cos(np.radians(25))].copy()
    log.info("After b/a < cos(25°):            %d", len(mock))

    # Morphology
    is_exp = mock["morphtype"] == "EXP"
    is_ser = (mock["morphtype"] == "SER") & (mock["sersic"] <= 2)
    mock = mock.loc[is_exp | is_ser].copy()
    log.info("After morphology cuts:           %d", len(mock))

    mock = mock.dropna()
    log.info("After dropping NaN:              %d", len(mock))
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# Rotational velocity generation
# ─────────────────────────────────────────────────────────────────────────────

def _build_cdf(logvrot_slice, logvrot_err_slice, bins=np.arange(1.0, 3.01, 0.01)):
    """Build a weighted CDF from logvrot values within a magnitude bin."""
    pdf, edges = np.histogram(logvrot_slice, bins=bins,
                               weights=np.ones_like(logvrot_err_slice))
    cdf     = np.cumsum(pdf) / pdf.sum()
    centres = 0.5 * (edges[1:] + edges[:-1])

    # Keep only unique CDF values (csaps requires strictly increasing)
    _, unique_idx = np.unique(cdf, return_index=True)
    return cdf[unique_idx], centres[unique_idx]


def generate_logvrot(mock: pd.DataFrame, tfrcat: pd.DataFrame,
                     cosmology) -> pd.DataFrame:
    """Assign mock log₁₀(V_rot) by resampling the Y1 TFR CDF per magnitude bin.

    Adds columns ``LOGVROT_MOCK`` and ``LOGVROT_ERR_MOCK`` to *mock*.
    """
    mock = mock.copy()

    # "True" cosmological absolute magnitude
    Mr_cos = mock["R_MAG_SB26_CORR"] - cosmology.distmod(mock["zcos"]).to_value("mag")
    mu_obs  = cosmology.distmod(mock["zobs"]).to_value("mag")
    mock["MU_OBS_MOCK"]          = mu_obs
    Mr_obs  = (mock["R_MAG_SB26_CORR"] - mu_obs).to_numpy()
    mock["R_ABSMAG_SB26_TRUE"]   = Mr_cos.to_numpy()
    mock["R_ABSMAG_SB26_MOCK"]   = Mr_obs
    mock["R_ABSMAG_SB26_ERR_MOCK"] = mock["R_MAG_SB26_ERR_CORR"].to_numpy()

    # Build adaptive magnitude bins with ≥ MIN_BIN_COUNT galaxies each
    raw_edges = np.arange(-26.0, -12.0 + cfg.mr_bin_width, cfg.mr_bin_width)
    M_r_bins  = [raw_edges[0]]
    for edge in raw_edges[1:]:
        n_in = ((tfrcat["R_ABSMAG_SB26"] > M_r_bins[-1])
                & (tfrcat["R_ABSMAG_SB26"] <= edge)).sum()
        if n_in >= cfg.min_bin_count:
            M_r_bins.append(edge)
    M_r_bins.append(raw_edges[-1])

    logvrot_mock = np.zeros(len(mock))
    logv_bins    = np.arange(cfg.logv_min, cfg.logv_max + 0.01, 0.01)

    for k in tqdm(range(len(M_r_bins) - 1), desc="CDF resampling"):
        lo, hi = M_r_bins[k], M_r_bins[k + 1]

        # Data slice for this magnitude bin
        data_mask = (tfrcat["R_ABSMAG_SB26"] > lo) & (tfrcat["R_ABSMAG_SB26"] <= hi)
        logv_data  = tfrcat["logv_rot"][data_mask].to_numpy()
        logv_err   = tfrcat["logv_rot_err"][data_mask].to_numpy()
        cdf_x, cdf_y = _build_cdf(logv_data, logv_err, bins=logv_bins)

        # Mock galaxies in this magnitude bin
        mock_mask = (Mr_cos > lo) & (Mr_cos <= hi)
        n_mock    = mock_mask.sum()
        if n_mock == 0:
            continue

        def _sample(n):
            return csaps(cdf_x, np.sort(cdf_y), np.random.uniform(size=n)).values

        samples = _sample(n_mock)

        # Re-draw any out-of-range values
        out_of_range = (samples < cfg.logv_min) | (samples > cfg.logv_max)
        while out_of_range.any():
            samples[out_of_range] = _sample(out_of_range.sum())
            out_of_range = (samples < cfg.logv_min) | (samples > cfg.logv_max)

        logvrot_mock[mock_mask.to_numpy()] = samples

    # Assign uncertainties from nearest neighbour in (logv, Mr) space
    tree = KDTree(np.c_[tfrcat["logv_rot"], tfrcat["R_ABSMAG_SB26"]])
    _, nn_idx = tree.query(np.c_[logvrot_mock, mock["R_ABSMAG_SB26_MOCK"]])

    mock["LOGVROT_MOCK"]     = logvrot_mock
    mock["LOGVROT_ERR_MOCK"] = tfrcat["logv_rot_err"].iloc[nn_idx].to_numpy()
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# TFR fitting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pack_datasets(mock_ds: pd.DataFrame, zbin_idx: np.ndarray,
                   zbin_ids: np.ndarray, logV0: float):
    """Pack the mock catalog into ``(datasets, covs)`` for ``MultiLinFit``."""
    datasets, covs = [], []
    for zid in zbin_ids:
        sel  = zbin_idx == zid
        logv = mock_ds["LOGVROT_MOCK"].to_numpy()[sel] - logV0
        dlogv = mock_ds["LOGVROT_ERR_MOCK"].to_numpy()[sel]
        mr    = mock_ds["R_MAG_SB26_CORR"].to_numpy()[sel]
        dmr   = mock_ds["R_MAG_SB26_ERR_CORR"].to_numpy()[sel]

        N   = len(logv)
        cov = np.zeros((2, 2, N))
        cov[0, 0, :] = dlogv ** 2
        cov[1, 1, :] = dmr  ** 2

        data = np.empty((2, N))
        data[0] = logv
        data[1] = mr

        datasets.append(data)
        covs.append(cov)
    return datasets, covs


def fit_tfr_mock(mock: pd.DataFrame, tfrcat: pd.DataFrame,
                 zbins: np.ndarray, n_realisations: int = 25) -> tuple:
    """Estimate TFR parameters by averaging over many random calibration draws.

    Returns
    -------
    a_avg : float       – average slope
    b_avg : ndarray     – average intercepts (one per z-bin)
    sigma_avg : float   – average intrinsic scatter
    zbin_ids : ndarray  – unique z-bin indices present in the calibration sample
    logV0 : float       – pivot log-velocity used for the fit
    """
    # Good-quality mock galaxies for calibration
    good_v   = alex_cuts_velocity(mock, logv_name="LOGVROT_MOCK",
                                  distmod_name="MU_OBS_MOCK")
    not_dwarf = alex_cuts_dwarf(mock, rmag_name="R_MAG_SB26_CORR",
                                distmod_name="MU_OBS_MOCK")
    good_mask = good_v & not_dwarf

    zbin_idx_all = np.digitize(mock["zobs"], zbins, right=True)
    inside = (zbin_idx_all > 0) & (zbin_idx_all < len(zbins))
    pool = mock.loc[good_mask & inside].copy()
    pool_zidx = zbin_idx_all[good_mask & inside]

    zbin_ids = np.sort(np.unique(pool_zidx))
    n_zbins  = len(zbin_ids)
    bounds   = [[-20.0, 0.0]] + n_zbins * [(-20.0, 20.0)] + [(0.0, 5.0)]

    a_list, b_list, sigma_list = [], [], []

    for _ in tqdm(range(n_realisations), desc="TFR fit realisations"):
        sample   = downsample(pool, cfg.calib_sample_size)
        s_zidx   = np.digitize(sample["zobs"], zbins, right=True)
        logV0    = float(np.median(sample["LOGVROT_MOCK"]))
        datasets, covs = _pack_datasets(sample, s_zidx, zbin_ids, logV0)

        hf = MultiLinFit(datasets, covs, scatter=1)
        pars, parscatter, _ = hf.optimize(bounds)

        a_list.append(pars[0])
        b_list.append(pars[1:])
        sigma_list.append(parscatter[0])

    a_avg     = float(np.mean(a_list))
    b_avg     = np.mean(b_list, axis=0)
    sigma_avg = float(np.mean(sigma_list))
    logV0     = float(np.median(mock["LOGVROT_MOCK"]))

    log.info("TFR fit:  a = %.3f,  σ = %.3f", a_avg, sigma_avg)
    return a_avg, b_avg, sigma_avg, zbin_ids, logV0


# ─────────────────────────────────────────────────────────────────────────────
# Distance modulus computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_distances(mock: pd.DataFrame,
                      a: float, b: np.ndarray, sigma: float,
                      zbins: np.ndarray, zbin_ids: np.ndarray,
                      logV0: float, cosmology) -> pd.DataFrame:
    """Compute TFR distance moduli and log-distance ratios.

    Adds ``LOGDIST_TRUE``, ``LOGDIST``, and ``LOGDIST_ERR`` to *mock*.
    """
    mock = mock.copy()

    dz  = zbins[1] - zbins[0]          # assumes uniform z-bins
    zc  = 0.5 * dz + zbins[:-1]
    mu_zc = cosmology.distmod(zc).to_value("mag")

    # Convert apparent-mag zero points to absolute-mag zero points
    B = b - mu_zc

    # Map every galaxy to its z-bin; clamp to valid range
    zbin_idx = np.digitize(mock["zobs"].to_numpy(), zbins, right=True)
    B_idx    = np.clip(zbin_idx - 1, 0, len(zbins) - 2)

    # Vectorised computation of Mr_TF and its uncertainty
    logv = mock["LOGVROT_MOCK"].to_numpy()
    dlogv = mock["LOGVROT_ERR_MOCK"].to_numpy()
    Mr_TF = a * (logv - logV0) + B[B_idx]

    # Monte-Carlo uncertainty (vectorised over galaxies)
    n_mc    = 1_000
    rng     = np.random.default_rng()
    logv_mc = rng.normal(logv[:, None], 0.434 * dlogv[:, None], size=(len(mock), n_mc))
    Mr_stat = a * (logv_mc - logV0) + B[B_idx, None]
    Mr_TF_err = np.sqrt(np.nanstd(Mr_stat, axis=1) ** 2 + sigma ** 2)

    mu_TF   = mock["R_MAG_SB26_CORR"].to_numpy() - Mr_TF
    mu_TF_err = np.sqrt(mock["R_ABSMAG_SB26_ERR_MOCK"].to_numpy() ** 2
                        + Mr_TF_err ** 2)

    mu_zcmb = cosmology.distmod(mock["zobs"].to_numpy()).to_value("mag")
    mu_zcos = cosmology.distmod(mock["zcos"].to_numpy()).to_value("mag")

    mock["LOGDIST_TRUE"] = 0.2 * (mu_zcmb - mu_zcos)
    mock["LOGDIST"]      = 0.2 * (mu_zcmb - mu_TF)
    mock["LOGDIST_ERR"]  = 0.2 * mu_TF_err
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def write_output(mock: pd.DataFrame,
                 a: float, b: np.ndarray, sigma: float,
                 outfile: str) -> str:
    """Write the final mock catalog to a FITS file.

    Returns the path to the written file.
    """
    hdr = fits.Header(
        {"NTF": len(mock), "a": a, "sigma": sigma}
        | {f"b{k+1}": b[k] for k in range(len(b))}
    )

    columns = [
        fits.Column("RA",                  "D", array=mock["ra"].to_numpy()),
        fits.Column("DEC",                 "D", array=mock["dec"].to_numpy()),
        fits.Column("ZOBS",                "D", array=mock["zobs"].to_numpy()),
        fits.Column("ZCOS",                "D", array=mock["zcos"].to_numpy()),
        fits.Column("vx",                  "D", array=mock["vx"].to_numpy()),
        fits.Column("vy",                  "D", array=mock["vy"].to_numpy()),
        fits.Column("vz",                  "D", array=mock["vz"].to_numpy()),
        fits.Column("DWARF",               "L", array=mock["DWARF"].to_numpy()),
        fits.Column("MAIN",                "L", array=mock["MAIN"].to_numpy()),
        fits.Column("LOGVROT",             "D", array=mock["LOGVROT_MOCK"].to_numpy()),
        fits.Column("LOGVROT_ERR",         "D", array=mock["LOGVROT_ERR_MOCK"].to_numpy()),
        fits.Column("R_ABSMAG_SB26",       "D", array=mock["R_ABSMAG_SB26_MOCK"].to_numpy()),
        fits.Column("R_ABSMAG_SB26_ERR",   "D", array=mock["R_ABSMAG_SB26_ERR_MOCK"].to_numpy()),
        fits.Column("R_ABSMAG_SB26_TRUE",  "D", array=mock["R_ABSMAG_SB26_TRUE"].to_numpy()),
        fits.Column("LOGDIST_TRUE",        "D", array=mock["LOGDIST_TRUE"].to_numpy()),
        fits.Column("LOGDIST",             "D", array=mock["LOGDIST"].to_numpy()),
        fits.Column("LOGDIST_ERR",         "D", array=mock["LOGDIST_ERR"].to_numpy()),
        fits.Column("Y1_COMP",             "D", array=mock["Y1_COMP"].to_numpy()),
        fits.Column("Y3_COMP",             "D", array=mock["Y3_COMP"].to_numpy()),
    ]

    hdulist = fits.BinTableHDU.from_columns(columns, header=hdr)
    hdulist.writeto(outfile, overwrite=True)
    shutil.chown(outfile, group="desi")

    log.info("Output written to %s", outfile)
    return outfile


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = ArgumentParser(
        description="TFR mock generation",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--phase", dest="phase", type=int,
                        choices=range(0, 25), required=True,
                        help="Phase number (0-24)")
    parser.add_argument("-r", "--real", dest="real", type=int,
                        choices=range(0, 27), required=True,
                        help="Realization number (0-26)")
    parser.add_argument("--seed", dest="seed", type=int, default=None,
                        help="NumPy random seed for reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    phase, real = args.phase, args.real

    log.info("==  TFR mock generation for phase %d, realization %d ===", phase, real)

    if args.seed is not None:
        np.random.seed(args.seed)
        log.info("Random seed set to %d", args.seed)

    # ── Cosmology ─────────────────────────────────────────────────────────────
    cosmology = FlatLambdaCDM(H0=100.0, Om0=0.3151)

    # ── spec catalog ──────────────────────────────────────────────────────────
    spec = load_spec_catalog()
    spec = apply_photo_corrections(spec)

    # ── TFR best-fit parameters ───────────────────────────────────────────────
    with open(cfg.tfr_pickle, "rb") as fh:
        cov_ab, tfr_samples, logV0, zmin, zmax, dz, zbins = pickle.load(fh)

    tf_par    = np.median(tfr_samples, axis=1)
    a_ref     = tf_par[0]
    b_ref     = tf_par[1:-1]
    sigma_ref = tf_par[-1]
    log.info("Reference TFR: a=%.3f, σ=%.3f", a_ref, sigma_ref)

    # ── TFR Y1 catalog ────────────────────────────────────────────────────────
    log.info("Loading data TF catalog from %s", cfg.tf_data_file)
    tfrcat = Table.read(cfg.tf_data_file)
    tfrcat["logv_rot"]     = np.log10(tfrcat["V_0p4R26"])
    tfrcat["logv_rot_err"] = 0.434 * tfrcat["V_0p4R26_ERR"] / tfrcat["V_0p4R26"]
    keep_cols = [
        "Z_DESI", "D26", "R_MAG_SB26_CORR", "R_MAG_SB26_ERR_CORR",
        "R_ABSMAG_SB26", "R_ABSMAG_SB26_ERR",
        "GOOD_MORPH", "MU_ZCMB", "MU_ZCMB_ERR",
        "V_0p4R26", "V_0p4R26_ERR", "logv_rot", "logv_rot_err",
    ]
    tfrcat = tfrcat[keep_cols].to_pandas()

    # ── Mock catalog ──────────────────────────────────────────────────────────
    mock_infile = cfg.mock_infile.format(phase=phase, real=real)
    log.info("Mock file: %s", mock_infile)

    mock = load_mock(mock_infile, spec)
    mock = apply_tf_selection(mock)

    # ── Generate rotational velocities ────────────────────────────────────────
    mock = generate_logvrot(mock, tfrcat, cosmology)

    # ── Fit TFR and compute quality flags ─────────────────────────────────────
    a_fit, b_fit, sigma_fit, zbin_ids, logV0_fit = fit_tfr_mock(
        mock, tfrcat, zbins
    )

    # Annotate quality flags on the *full* mock sample
    good_v    = alex_cuts_velocity(mock, logv_name="LOGVROT_MOCK",
                                   distmod_name="MU_OBS_MOCK")
    not_dwarf = alex_cuts_dwarf(mock, rmag_name="R_MAG_SB26_CORR",
                                distmod_name="MU_OBS_MOCK")
    mock["DWARF"] = ~not_dwarf
    mock["MAIN"]  = good_v & not_dwarf

    # ── Compute TFR distances ─────────────────────────────────────────────────
    mock = compute_distances(
        mock, a_fit, b_fit, sigma_fit, zbins, zbin_ids, logV0_fit, cosmology
    )

    # ── Maximum volume fraction ───────────────────────────────────────────────
    dist      = cosmology.luminosity_distance(np.abs(mock["zobs"]))
    d26_kpc   = 2.0 * dist.to("kpc") * np.tan(0.5 * mock["D26"].values * u.arcmin)
    dist_max_galaxy = 0.5 * d26_kpc / np.tan(0.1 * u.arcmin)
    dist_max_survey = cosmology.luminosity_distance(z=0.1)
    mock["MAX_VOL_FRAC"] = (dist_max_galaxy.to("Mpc") / dist_max_survey.to("Mpc")) ** 3

    # ── Write output ──────────────────────────────────────────────────────────
    outfile = cfg.mock_tf_out.format(phase=phase, real=real)
    write_output(mock, a_fit, b_fit, sigma_fit, outfile)
    log.info("Done")

if __name__ == "__main__":
    main()