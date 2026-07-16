"""
DESI TF Mocks Processing Pipeline
===================================
Reads in TF mocks created with make_DESI_tf_mocks.py, summarises them,
downsamples each mock to match the data n(z), and converts to clustering mocks.
Also produces a random catalogue by downsampling Chris Blake's BGS randoms.
"""

import argparse
import subprocess
import os
import h5py
import logging
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from sklearn.neighbors import KDTree

import utils

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
log = logging.getLogger(__name__)

cosmo = FlatLambdaCDM(H0=100, Om0=0.3151)

# ---------------------------------------------------------------------------
# Step 1 — Load observed data & mocks, accumulate statistics
# ---------------------------------------------------------------------------

def load_observed_data():
    """Load BGS and TF clustering data + randoms."""
    log.info("Loading observed data …")
    #bgs_data = Table.read(cfg.data_bgs_clus_data).to_pandas()
    #bgs_rand = Table.read(cfg.data_bgs_clus_rand).to_pandas()
    tf_data  = Table.read(cfg.data_tf_clus_data).to_pandas()
    #tf_rand  = Table.read(cfg.data_tf_clus_rand).to_pandas()

    tf_data["LOGDIST"]           = (
        np.log10(cosmo.luminosity_distance(tf_data["Z"]).value)
        + 5.0
        - tf_data["MU"] / 5.0
    )
    tf_data["LOGDIST_ERR"]       = tf_data["MU_ERR"] / 5.0
    tf_data["LOGDIST_GAUSS_ERR"] = tf_data["LOGDIST_ERR"].copy()

    #tf_data["LOGDIST_GAUSS_ERR"] = utils.reweight(tf_data["LOGDIST"], tf_data["LOGDIST_ERR"])



    #log.info("  BGS data: %d galaxies | BGS rand: %d", len(bgs_data), len(bgs_rand))
    #log.info("  TF  data: %d galaxies | TF  rand: %d", len(tf_data), len(tf_rand))
    #return bgs_data, bgs_rand, tf_data, tf_rand
    return tf_data

def inflate_errors(mock: pd.DataFrame, sigma: float) -> pd.DataFrame:
    """
    Inflate logdist errors to better match the data, and propagate the
    additional scatter into the measured values to conserve the pull distribution.

    sigma_new = 0.75 * sqrt(sigma_old² + (sigma_TFR/5)²)
    η_new = η_true + 0.75 * (η_obs - η_true)
    """
    mock["LOGDIST_ERR"] = 0.5 * np.sqrt(
        mock["LOGDIST_ERR"] ** 2 + (sigma / 5.0) ** 2
    )
    mock["LOGDIST"] = (
        mock["LOGDIST_TRUE"] + 0.5 * (mock["LOGDIST"] - mock["LOGDIST_TRUE"])
    )
    return mock

def read_mocks():
    mocks = []
    headers = []

    #for phase in range(cfg.n_phases):
    for phase in [cfg.tf_clus.phase]:
        log.info("  Phase %d …", phase)
        for real in range(cfg.n_reals):
            tf_file = cfg.mock_tf_full_data.format(phase=phase, real=real)
            with fits.open(tf_file) as hdu:
                header = hdu[1].header.copy()
                utils.clean_header(header)
                sigma = hdu[1].header["SIGMA"]
                headers.append(header)
            #tab = Table.read(tf_file)
            #print(tab.meta) 

            mock = Table.read(tf_file).to_pandas()
            mock['sigma'] = sigma 

            #-- Inflate errors
            if cfg.tf_clus.inflate_errors: 
                inflate_errors(mock, sigma)

            #-- Redshift cut
            mock['real'] = real
            mock = mock[ (mock["ZOBS"] >= cfg.tf_clus.zmin) 
                        &(mock["ZOBS"] <= cfg.tf_clus.zmax)]
            
            #-- Completeness cut and sub-sampling
            mock = mock[ (mock[cfg.comp_field] >= cfg.tf_clus.comp_min)] 
            mock = mock[ (np.random.uniform(size=len(mock)) < mock[cfg.comp_field])]

            mock["LOGDIST_CORR"] = mock["LOGDIST"].copy()
            mock["LOGDIST_CORR_ERR"] = mock["LOGDIST_ERR"].copy() 

            #-- Gaussianise errors
            mock["LOGDIST_GAUSS_ERR"] = utils.reweight(mock["LOGDIST_CORR"], 
                                                 mock["LOGDIST_CORR_ERR"])
            #mock["LOGDIST_GAUSS_ERR"] = mock["LOGDIST_ERR"].copy()

            #-- Zero-point calibration
            offset, _, _ = utils.weighted_avg_and_std(
                mock["LOGDIST_CORR"] - mock["LOGDIST_TRUE"],
                1.0 / mock["LOGDIST_GAUSS_ERR"] ** 2,
                )
            mock["LOGDIST_CORR"] -= offset
            
            mocks.append(mock)

    mocks = pd.concat(mocks, ignore_index=True)
    mocks['DIST'] = cosmo.comoving_distance(mocks["ZOBS"]).value
    
    # PV estimation
    mocks["PV"]      = utils.pv_from_logdist(mocks["LOGDIST_CORR"], mocks['ZOBS'], cosmo)
    mocks["PV_ERR"]  = utils.pv_from_logdist(mocks["LOGDIST_GAUSS_ERR"], mocks['ZOBS'], cosmo)
    mocks["PV_TRUE"] = utils.pv_from_logdist(mocks["LOGDIST_TRUE"], mocks['ZOBS'], cosmo)

    log.info("  Total mocks: %d galaxies | %d realisations", len(mocks), mocks['real'].nunique())

    return mocks, headers

def accumulate_mock_statistics(mocks) -> dict:
    """
    Loop over all phases/realisations and accumulate n(z) and logdist statistics.
    Returns a dict of arrays for downstream use.
    """
    log.info("Accumulating mock statistics …")
    zlims = [cfg.tf_clus.zmin, cfg.tf_clus.zmax]

    # Accumulators
    #nz_bgs_mock      = np.zeros(cfg.tf_clus.nzbin)
    #nz_bgs_mock_err2  = np.zeros(cfg.tf_clus.nzbin)
    nz_tf_mock    = np.zeros(cfg.tf_clus.nzbin)
    nz_tf_mock_err2 = np.zeros(cfg.tf_clus.nzbin)
    logdistmock  = np.zeros(cfg.tf_clus.nzbin)
    logdisterr   = np.zeros(cfg.tf_clus.nzbin)
    logdisterr_g = np.zeros(cfg.tf_clus.nzbin)
    pullmock     = np.zeros(cfg.tf_clus.nzbin)

    tfmock_count = mocks['real'].nunique()
    ngals = len(mocks)

    # Histograms
    nz = np.histogram(mocks["ZOBS"], bins=cfg.tf_clus.nzbin, range=zlims)[0]
    nz_tf_mock     = nz
    nz_tf_mock_err2 = nz ** 2
    logdistmock  = np.histogram(mocks["LOGDIST_CORR"], 
                                    bins=cfg.tf_clus.nzbin, 
                                    range=[-0.3, 0.3])[0]
    logdisterr   = np.histogram(mocks["LOGDIST_CORR_ERR"], 
                                    bins=cfg.tf_clus.nzbin, 
                                    range=[0.08, 0.30])[0]
    logdisterr_g = np.histogram(mocks["LOGDIST_GAUSS_ERR"], 
                                    bins=cfg.tf_clus.nzbin, 
                                    range=[0.08, 0.30])[0]

    pulls = (
        (mocks["LOGDIST_CORR"] - mocks["LOGDIST_TRUE"])
        / mocks["LOGDIST_GAUSS_ERR"]
    )
    pullmock  = np.histogram(pulls, bins=cfg.tf_clus.nzbin, range=[-4.0, 4.0])[0]
    mean_pull = pulls.sum()
    std_pull  = (pulls ** 2).sum()

    # Normalise
    nz_tf_mock = nz_tf_mock/tfmock_count
    nz_tf_mock_err = np.sqrt(nz_tf_mock_err2 / tfmock_count - nz_tf_mock ** 2)
    logdistmock = logdistmock/tfmock_count
    logdisterr  = logdisterr/tfmock_count
    logdisterr_g = logdisterr_g/tfmock_count
    pullmock    = pullmock/tfmock_count
    mean_pull   = mean_pull/ngals
    std_pull     = np.sqrt(std_pull / ngals - mean_pull ** 2)

    log.info(
        "Mocks processed: %d tf | Nz: mean=%.4f, std=%.4f | pull: mean=%.4f, std=%.4f",
        tfmock_count, np.mean(nz_tf_mock), np.mean(nz_tf_mock_err), mean_pull, std_pull,
    )

    stats = dict(
        nz_tf_mock=nz_tf_mock, nz_tf_mock_err=nz_tf_mock_err,
        logdistmock=logdistmock, logdisterr=logdisterr, logdisterr_g=logdisterr_g,
        pullmock=pullmock, mean_pull=mean_pull, std_pull=std_pull,
    )

    return stats


# ---------------------------------------------------------------------------
# Step 2 — Compute sub-sampling fraction and logdist bias correction
# ---------------------------------------------------------------------------

def compute_subsampling_fraction(nz_data: np.ndarray, nz_mock: np.ndarray) -> np.ndarray:
    """
    Compute per-z-bin sub-sampling fraction so mock n(z) matches data n(z).
    Values are clipped to [0, 1] and normalised to max=1.
    """
    subsampling_fraction = np.where(nz_mock > 0, nz_data / nz_mock, 1.0)
    subsampling_fraction = np.where(subsampling_fraction > 1.0, 1.0, subsampling_fraction)
    #subsampling_fraction /= subsampling_fraction.max()
    
    # Smooth in bins where the mock is already sparse (subsampling_fraction == 1)
    subsampling_fraction = np.where(subsampling_fraction == 1.0, 
                                    1.0, 
                                    savgol_filter(subsampling_fraction, 15, 1))
    for i in range(len(subsampling_fraction)):
        log.info(f"Sub-sampling fraction: bin {i} = {nz_data[i]:.1f} / {nz_mock[i]:.1f} = {subsampling_fraction[i]:.4f}") 
    
    return subsampling_fraction

def subsample_mocks(mocks: pd.DataFrame, subsampling_fraction: np.ndarray) -> pd.DataFrame:
    """
    Sub-sample each mock realisation to match the data n(z).
    Returns the modified mocks DataFrame.
    """
    zbins = np.linspace(cfg.tf_clus.zmin, cfg.tf_clus.zmax, cfg.tf_clus.nzbin+1)
    izs = utils.safe_digitize(mocks["ZOBS"], zbins)
    keep = subsampling_fraction[izs] > np.random.uniform(size=len(mocks))
    log.info(f"  Total mocks after n(z) sub-sampling: {keep.sum()} of {keep.size}")
    return mocks[keep]


def compute_logdist_bias_correction(mocks) -> CubicSpline:
    """
    Fit a cubic spline to the weighted-mean logdist residual vs. redshift.
    Returns a callable correction f(z).
    """
    zbins    = np.linspace(cfg.tf_clus.zmin, cfg.tf_clus.zmax, cfg.tf_clus.nzbin+1)
    zcen = 0.5 * (zbins[:-1] + zbins[1:])
    residual = np.zeros(cfg.tf_clus.nzbin)

    for k in range(cfg.tf_clus.nzbin):
        zlo, zhi = zbins[k], zbins[k + 1]
        idx = (mocks["ZOBS"] > zlo) & (mocks["ZOBS"] <= zhi)
    
        bias_corr = mocks['LOGDIST_CORR'][idx] - mocks['LOGDIST_TRUE'][idx]
        weight_corr = 1.0 / mocks['LOGDIST_GAUSS_ERR'][idx] ** 2
        residual[k] = utils.weighted_avg_and_std(bias_corr, weight_corr)[0]

    for i in range(len(residual)):
        log.info(f"Logdist bias correction at bin {i} zcen {zcen[i]:.2f}: {residual[i]:.5f}")
    
    logdist_bias_corr = CubicSpline(zcen, residual)
    return logdist_bias_corr

def build_pv_meshes(mocks, box):
    mock_pos = utils.radec_to_xyz(mocks["RA"], mocks["DEC"], mocks["DIST"])
    logdist_error = mocks['LOGDIST_GAUSS_ERR']
    logdist_error_mesh = utils.build_mesh(mock_pos, logdist_error, box['ngrid'], box['side'])
    #pv_error_mesh = utils.pv_from_logdist(logdist_error_mesh, mocks['ZOBS'], cosmo)
    w = logdist_error_mesh>0
    for p in np.linspace(0.1, 0.9, 9):
        log.info("LOGDIST_ERR mesh: %.1f percentile = %.4e", p*100, np.percentile(logdist_error_mesh[w], p*100))

    norm = 1/mocks['real'].nunique()
    npv_mesh = norm * utils.build_density_mesh(mock_pos, box=box, normalize=False)
    w = npv_mesh > 0 
    for p in np.linspace(0.1, 0.9, 9):
        log.info("NPV mesh: %.1f percentile = %.4e", p*100, np.percentile(npv_mesh[w], p*100))

    return npv_mesh, logdist_error_mesh

# ---------------------------------------------------------------------------
# Step 3 — Build random catalogue
# ---------------------------------------------------------------------------

def build_random_catalogue(subsampling_fraction: np.ndarray, 
                           nz_tf_mock: np.ndarray) -> pd.DataFrame:
    """
    Read Abacus base randoms, apply completeness cut and sub-sampling,
    then return a DataFrame with columns RA, DEC, Z, WEIGHT.
    """
    log.info("Reading Abacus random catalogues …")
    ra_all, dec_all, z_all, comp_all = [], [], [], []

    #for phase in range(cfg.n_real_rand):
    for phase in [cfg.tf_clus.phase]: 
        for ireal in range(cfg.n_reals):
            log.info(f"  Randoms phase {phase:03d} real {ireal:03d}")
            path = cfg.mock_bgs_base_rand.format(phase=phase, real=ireal)
            #rand = pd.read_hdf(path, columns=['ra', 'dec', 'zobs', cfg.comp_field])
            with h5py.File(path, "r") as f:
                ra   = f["ra"][...]
                dec  = f["dec"][...]
                z    = f["zobs"][...]
                comp = f[cfg.comp_field][...]
            
            #- Selection cuts
            cut  = (
                (z >= cfg.tf_clus.zmin)
                & (z <= cfg.tf_clus.zmax)
                & (comp > cfg.tf_clus.comp_min)
                #-- now we are no longer sub-sampling randoms -> apply weight! 
                #& (np.random.uniform(size=nran) < comp)
            )
            
            ra_all.append(ra[cut])
            dec_all.append(dec[cut])
            z_all.append(z[cut])
            comp_all.append(comp[cut])

    ra_cat  = np.concatenate(ra_all)
    dec_cat = np.concatenate(dec_all)
    z_cat   = np.concatenate(z_all)
    w_cat   = np.concatenate(comp_all)

    # Shuffle
    #idx = np.random.permutation(len(ra_cat))
    #ra_cat, dec_cat, z_cat, w_cat = ra_cat[idx], dec_cat[idx], z_cat[idx], w_cat[idx]
    log.info("  Total randoms before sub-sampling: %d", len(z_cat))

    # Sub-sample to match the (already subsampled) mock n(z)
    zbins = np.linspace(cfg.tf_clus.zmin, cfg.tf_clus.zmax, cfg.tf_clus.nzbin+1)
    nz_base_rand, _ = np.histogram(z_cat, bins=zbins)
    subfrac_ran  = np.where(nz_base_rand > 0, 
                            nz_tf_mock * subsampling_fraction * cfg.n_reals / nz_base_rand, 
                            0.0)
    #subfrac_ran /= subfrac_ran.max()

    izs = utils.safe_digitize(z_cat, zbins)
    cut = subfrac_ran[izs] > np.random.uniform(size=len(z_cat))
    ra_cat, dec_cat, z_cat, w_cat = ra_cat[cut], dec_cat[cut], z_cat[cut], w_cat[cut]
    log.info("  Total randoms after  sub-sampling: %d", len(z_cat))

    return pd.DataFrame({"RA": ra_cat, "DEC": dec_cat, "Z": z_cat, "WEIGHT": w_cat})


def write_random_catalogue(
    tf_rand: pd.DataFrame,
    npv_mesh: np.ndarray,
    logdist_error_mesh: np.ndarray,
    box: dict,
) -> None:
    """
    Assign logdist/PV errors to randoms via nearest-neighbour matching,
    then write the FITS random catalogue.
    """
    log.info("Building random catalogue with NN error assignment …")

    # Truncate random catalogue to rfact × expected galaxy count
    #n_target = cfg.tf_clus.rfact * int((nz_tf_mock * subsampling_fraction).sum())
    #if len(tf_rand) > n_target:
    #    idx     = np.random.choice(len(tf_rand), n_target, replace=False)
    #    tf_rand = tf_rand.iloc[idx].reset_index(drop=True)
    #log.info("  Randoms after truncation: %d", len(tf_rand))

    ran_dist = cosmo.comoving_distance(tf_rand["Z"]).value
    ran_xyz  = utils.radec_to_xyz(tf_rand["RA"], 
                                  tf_rand["DEC"], 
                                  ran_dist)

    # Nearest-neighbour logdist error
    #tree = KDTree(np.c_[gal_x, gal_y, gal_z])
    #nn   = tree.query(np.c_[ran_xyz[0], ran_xyz[1], ran_xyz[2]], return_distance=False, dualtree=True)
    #ran_lde = gal_lde[nn[:, 0]]
    ran_lde = utils.get_mesh_value(logdist_error_mesh, ran_xyz, box['lims'])
    ran_pve = utils.pv_from_logdist(ran_lde, tf_rand["Z"], cosmo)

    # Density grid lookups
    ran_npv   = utils.get_mesh_value(npv_mesh, ran_xyz, box['lims'])

    # Remove randoms outside mesh
    w = ran_npv > 0 
    tf_rand = tf_rand[w]
    ran_lde = ran_lde[w]
    ran_pve = ran_pve[w]
    ran_npv = ran_npv[w]

    log.info("  Data npv  : mean=%.4e  std=%.4e", ran_npv.mean(),   ran_npv.std())

    columns = [
        ("RA",          tf_rand["RA"].to_numpy()),
        ("DEC",         tf_rand["DEC"].to_numpy()),
        ("Z",           tf_rand["Z"].to_numpy()),
        ("WEIGHT",      tf_rand["WEIGHT"].to_numpy()),
        ("NPV",         ran_npv),
        #("NDENS",       ran_ndens),
        ("LOGDIST_ERR", ran_lde),
        ("PV_ERR",      ran_pve),
    ]
    hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name=n, format="D", array=a) for n, a in columns]
    )
    mock_tf_clus_rand = cfg.mock_tf_clus_rand.format(phase=cfg.tf_clus.phase)
    os.makedirs(os.path.dirname(mock_tf_clus_rand), exist_ok=True)
    log.info("Writing random catalogue → %s", mock_tf_clus_rand)
    hdu.writeto(mock_tf_clus_rand, overwrite=True)



# ---------------------------------------------------------------------------
# Step 5 — Process each mock into a clustering mock
# ---------------------------------------------------------------------------






def run_clustering_mock_loop(mocks, headers):
    """
    Process all phase/realisation pairs into clustering mocks.
    """
    log.info("Generating clustering mocks …")

    #for phase in range(cfg.n_phases):
    for phase in [cfg.tf_clus.phase]:
        for real in range(cfg.n_reals):
            out_file = cfg.mock_tf_clus_data.format(phase=phase, real=real)
            mock = mocks[mocks['real']==real]
            header = headers[real]
            log.info(f"  Mock phase={phase:03d} real={real:03d}:  {len(mock)} galaxies")
            log.info(f"  Writing mock to {out_file}")
            write_clustering_mock(mock, header, out_file)
            
def write_clustering_mock(mock, header, outfile) -> None:
    """Write a processed mock to a FITS binary table."""

    columns_to_keep = ['RA', 'DEC', 'ZOBS', 'NPV', 
                       'LOGDIST_CORR', 'LOGDIST_GAUSS_ERR', 'LOGDIST_TRUE', 
                       'PV', 'PV_ERR', 'PV_TRUE'] 
    mock = mock[columns_to_keep]
    mock.rename(columns={"ZOBS": "Z",
                         "LOGDIST_CORR" : "LOGDIST",
                         "LOGDIST_GAUSS_ERR": "LOGDIST_ERR"}, 
                inplace=True)

    table = Table.from_pandas(mock)
    table.meta = header
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    table.write(outfile, format='fits', overwrite=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline tf clustering "
    )
    parser.add_argument("config_file", type=str, help="Configuration file path (yaml format)")
    parser.add_argument("phase", type=int, help="Phase (0–24)")
    parser.add_argument("--seed", dest="seed", type=int, default=None,
                        help="NumPy random seed for reproducibility")
    args = parser.parse_args()
    if not (0 <= args.phase <= 24):
        parser.error("phase should be between 0 and 24")

    return args

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    #-- Set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        log.info("Random seed set to %d", args.seed)

    #-- Set config
    global cfg
    cfg = load_config(args.config_file)  
    cfg.tf_clus.phase = args.phase
    
    log.info(f"=== DESI TF clustering mocks pipeline for phase {cfg.tf_clus.phase:03d} ===")

    distmax = cosmo.comoving_distance(cfg.tf_clus.zmax).value
    box = utils.build_grid_box(distmax, cfg.tf_clus.ngrid)

    # 1. Load observed data and compute n(z) 
    tf_data = load_observed_data()
    zbins = np.linspace(cfg.tf_clus.zmin, cfg.tf_clus.zmax, cfg.tf_clus.nzbin+1)
    nz_tf_data, _ = np.histogram(tf_data["Z"], bins=zbins, weights=tf_data["WEIGHT"])

    #-- Read mocks 
    mocks, headers = read_mocks() 

    #--  Accumulate mock statistics
    stats = accumulate_mock_statistics(mocks)

    #-- Sub-sampling fraction 
    subsampling_fraction = compute_subsampling_fraction(nz_tf_data, stats["nz_tf_mock"])
    mocks = subsample_mocks(mocks, subsampling_fraction)

    #-- Compute and apply logdist bias correction
    logdist_bias_corr = compute_logdist_bias_correction(mocks)
    mocks["LOGDIST_CORR"] -= logdist_bias_corr(mocks['ZOBS'])

    #-- Create mesh with average PV error in each cell, for later use in random catalogue
    npv_mesh, logdist_error_mesh = build_pv_meshes(mocks, box)
    mock_pos = utils.radec_to_xyz(mocks['RA'], mocks['DEC'], mocks['DIST'])
    mocks['NPV'] = utils.get_mesh_value(npv_mesh, mock_pos, box['lims'])

    #-- Build random catalogue
    tf_rand_cat = build_random_catalogue(subsampling_fraction, stats["nz_tf_mock"])
    write_random_catalogue(tf_rand_cat, npv_mesh, logdist_error_mesh, box)

    #- Write clustering mocks
    run_clustering_mock_loop(mocks, headers)
        
    log.info("=== Updating permissions ===")
    result = subprocess.run(
        ["chgrp", "-R", "desi", cfg.mock_tf_clus_dir],
        check=True,
    )

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()