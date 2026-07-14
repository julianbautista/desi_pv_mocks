"""
DESI FP Mocks Processing Pipeline
===================================
Reads in FP mocks created with make_DESI_FP_mocks.py, summarises them,
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
#from sklearn.neighbors import KDTree

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
    """Load BGS and FP clustering data + randoms."""
    log.info("Loading observed data …")
    #bgs_data = Table.read(cfg.data_bgs_clus_data).to_pandas()
    #bgs_rand = Table.read(cfg.data_bgs_clus_rand).to_pandas()
    fp_data  = Table.read(cfg.data_fp_clus_data).to_pandas()
    #fp_rand  = Table.read(cfg.data_fp_clus_rand).to_pandas()

    fp_data["LOGDIST_GAUSS_ERR"] = utils.reweight(fp_data["LOGDIST"], fp_data["LOGDIST_ERR"])

    #log.info("  BGS data: %d galaxies | BGS rand: %d", len(bgs_data), len(bgs_rand))
    #log.info("  FP  data: %d galaxies | FP  rand: %d", len(fp_data), len(fp_rand))
    #return bgs_data, bgs_rand, fp_data, fp_rand
    return fp_data

def read_mocks():
    mocks = []

    #for phase in range(cfg.n_phases):
    for phase in [cfg.fp_clus.phase]:
        log.info("  Phase %d …", phase)
        for real in range(cfg.n_reals):
            fp_file = cfg.mock_fp_full_data.format(phase=phase, real=real)
            try:
                mock = Table.read(fp_file).to_pandas()

                #-- Redshift cut
                mock['real'] = real
                mock = mock[ (mock["ZOBS"] >= cfg.fp_clus.zmin) 
                            &(mock["ZOBS"] <= cfg.fp_clus.zmax)]
                
                #-- Gaussianise errors
                mock["LOGDIST_GAUSS_ERR"] = utils.reweight(mock["LOGDIST_CORR"], 
                                                     mock["LOGDIST_CORR_ERR"])

                #-- Zero-point calibration
                offset, _, _ = utils.weighted_avg_and_std(
                    mock["LOGDIST_CORR"] - mock["LOGDIST_TRUE"],
                    1.0 / mock["LOGDIST_GAUSS_ERR"] ** 2,
                    )
                mock["LOGDIST_CORR"] -= offset
                
                mocks.append(mock)
            except Exception as exc:
                log.warning("Skipping FP mock %s: %s", fp_file, exc)

    mocks = pd.concat(mocks, ignore_index=True)
    mocks['DIST'] = cosmo.comoving_distance(mocks["ZOBS"]).value
    
    # PV estimation
    mocks["PV"]      = utils.pv_from_logdist(mocks["LOGDIST_CORR"], mocks['ZOBS'], cosmo)
    mocks["PV_ERR"]  = utils.pv_from_logdist(mocks["LOGDIST_GAUSS_ERR"], mocks['ZOBS'], cosmo)
    mocks["PV_TRUE"] = utils.pv_from_logdist(mocks["LOGDIST_TRUE"], mocks['ZOBS'], cosmo)

    log.info("  Total mocks: %d galaxies | %d realisations", len(mocks), mocks['real'].nunique())

    return mocks 

def accumulate_mock_statistics(mocks) -> dict:
    """
    Loop over all phases/realisations and accumulate n(z) and logdist statistics.
    Returns a dict of arrays for downstream use.
    """
    log.info("Accumulating mock statistics …")
    zlims = [cfg.fp_clus.zmin, cfg.fp_clus.zmax]

    # Accumulators
    #nz_bgs_mock      = np.zeros(cfg.fp_clus.nzbin)
    #nz_bgs_mock_err2  = np.zeros(cfg.fp_clus.nzbin)
    nz_fp_mock    = np.zeros(cfg.fp_clus.nzbin)
    nz_fp_mock_err2 = np.zeros(cfg.fp_clus.nzbin)
    logdistmock  = np.zeros(cfg.fp_clus.nzbin)
    logdisterr   = np.zeros(cfg.fp_clus.nzbin)
    logdisterr_g = np.zeros(cfg.fp_clus.nzbin)
    pullmock     = np.zeros(cfg.fp_clus.nzbin)

    fpmock_count = mocks['real'].nunique()
    ngals = len(mocks)

    # Histograms
    nz = np.histogram(mocks["ZOBS"], bins=cfg.fp_clus.nzbin, range=zlims)[0]
    nz_fp_mock     = nz
    nz_fp_mock_err2 = nz ** 2
    logdistmock  = np.histogram(mocks["LOGDIST_CORR"], 
                                    bins=cfg.fp_clus.nzbin, 
                                    range=[-0.3, 0.3])[0]
    logdisterr   = np.histogram(mocks["LOGDIST_CORR_ERR"], 
                                    bins=cfg.fp_clus.nzbin, 
                                    range=[0.08, 0.30])[0]
    logdisterr_g = np.histogram(mocks["LOGDIST_GAUSS_ERR"], 
                                    bins=cfg.fp_clus.nzbin, 
                                    range=[0.08, 0.30])[0]

    pulls = (
        (mocks["LOGDIST_CORR"] - mocks["LOGDIST_TRUE"])
        / mocks["LOGDIST_GAUSS_ERR"]
    )
    pullmock  = np.histogram(pulls, bins=cfg.fp_clus.nzbin, range=[-4.0, 4.0])[0]
    mean_pull = pulls.sum()
    std_pull  = (pulls ** 2).sum()

    # Normalise
    nz_fp_mock = nz_fp_mock/fpmock_count
    nz_fp_mock_err = np.sqrt(nz_fp_mock_err2 / fpmock_count - nz_fp_mock ** 2)
    logdistmock = logdistmock/fpmock_count
    logdisterr  = logdisterr/fpmock_count
    logdisterr_g = logdisterr_g/fpmock_count
    pullmock    = pullmock/fpmock_count
    mean_pull   = mean_pull/ngals
    std_pull     = np.sqrt(std_pull / ngals - mean_pull ** 2)

    log.info(
        "Mocks processed: %d FP | Nz: mean=%.4f, std=%.4f | pull: mean=%.4f, std=%.4f",
        fpmock_count, np.mean(nz_fp_mock), np.mean(nz_fp_mock_err), mean_pull, std_pull,
    )

    stats = dict(
        nz_fp_mock=nz_fp_mock, nz_fp_mock_err=nz_fp_mock_err,
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
    zbins = np.linspace(cfg.fp_clus.zmin, cfg.fp_clus.zmax, cfg.fp_clus.nzbin+1)
    izs = utils.safe_digitize(mocks["ZOBS"].to_numpy(), zbins)
    keep = subsampling_fraction[izs] > np.random.uniform(size=len(mocks))
    log.info(f"  Total mocks after n(z) sub-sampling: {keep.sum()} of {keep.size}")
    return mocks[keep]


def compute_logdist_bias_correction(mocks) -> CubicSpline:
    """
    Fit a cubic spline to the weighted-mean logdist residual vs. redshift.
    Returns a callable correction f(z).
    """
    zbins    = np.linspace(cfg.fp_clus.zmin, cfg.fp_clus.zmax, cfg.fp_clus.nzbin+1)
    zcen = 0.5 * (zbins[:-1] + zbins[1:])
    residual = np.zeros(cfg.fp_clus.nzbin)

    for k in range(cfg.fp_clus.nzbin):
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
                           nz_fp_mock: np.ndarray) -> pd.DataFrame:
    """
    Read Abacus base randoms, apply completeness cut and sub-sampling,
    then return a DataFrame with columns RA, DEC, Z, WEIGHT.
    """
    log.info("Reading Abacus random catalogues …")
    ra_all, dec_all, z_all, comp_all = [], [], [], []

    #for phase in range(cfg.n_real_rand):
    for phase in [cfg.fp_clus.phase]: 
        for ireal in range(cfg.n_reals):
            log.info(f"  Randoms phase {phase:03d} real {ireal:03d}")
            path = cfg.mock_bgs_base_rand.format(phase=phase, real=ireal)
            try:
                with h5py.File(path, "r") as f:
                    ra   = f["ra"][...]
                    dec  = f["dec"][...]
                    z    = f["zobs"][...]
                    comp = f[cfg.comp_field][...]
                
                #- Selection cuts
                cut  = (
                    (z >= cfg.fp_clus.zmin)
                    & (z <= cfg.fp_clus.zmax)
                    & (comp > cfg.fp_clus.comp_min)
                    #-- now we are no longer sub-sampling randoms -> apply weight! 
                    #& (np.random.uniform(size=nran) < comp)
                )
                
                n_rand = np.sum(cut)
                n_data = (nz_fp_mock).sum()
                #print(' n_rand / n_data = ', n_rand/n_data)
                #cut &= (np.random.uniform(size=ra.size) <= n_data/n_rand)

                ra_all.append(ra[cut])
                dec_all.append(dec[cut])
                z_all.append(z[cut])
                comp_all.append(comp[cut])
            except Exception as exc:
                log.warning("Skipping random %s: %s", path, exc)

    ra_cat  = np.concatenate(ra_all)
    dec_cat = np.concatenate(dec_all)
    z_cat   = np.concatenate(z_all)
    w_cat   = np.concatenate(comp_all)

    # Shuffle
    #idx = np.random.permutation(len(ra_cat))
    #ra_cat, dec_cat, z_cat, w_cat = ra_cat[idx], dec_cat[idx], z_cat[idx], w_cat[idx]
    log.info("  Total randoms before sub-sampling: %d", len(z_cat))

    # Sub-sample to match the (already subsampled) mock n(z)
    zbins = np.linspace(cfg.fp_clus.zmin, cfg.fp_clus.zmax, cfg.fp_clus.nzbin+1)
    nz_base_rand, _ = np.histogram(z_cat, bins=zbins)
    subfrac_ran  = np.where(nz_base_rand > 0, 
                            nz_fp_mock * subsampling_fraction * cfg.n_reals / nz_base_rand, 
                            0.0)
    #subfrac_ran /= subfrac_ran.max()

    izs = utils.safe_digitize(z_cat, zbins)
    cut = subfrac_ran[izs] > np.random.uniform(size=len(z_cat))
    ra_cat, dec_cat, z_cat, w_cat = ra_cat[cut], dec_cat[cut], z_cat[cut], w_cat[cut]
    log.info("  Total randoms after  sub-sampling: %d", len(z_cat))

    return pd.DataFrame({"RA": ra_cat, "DEC": dec_cat, "Z": z_cat, "WEIGHT": w_cat})


def write_random_catalogue(
    fp_rand: pd.DataFrame,
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
    #n_target = cfg.fp_clus.rfact * int((nz_fp_mock * subsampling_fraction).sum())
    #if len(fp_rand) > n_target:
    #    idx     = np.random.choice(len(fp_rand), n_target, replace=False)
    #    fp_rand = fp_rand.iloc[idx].reset_index(drop=True)
    #log.info("  Randoms after truncation: %d", len(fp_rand))

    ran_dist = cosmo.comoving_distance(fp_rand["Z"]).value
    ran_xyz  = utils.radec_to_xyz(fp_rand["RA"], 
                                  fp_rand["DEC"], 
                                  ran_dist)

    # Nearest-neighbour logdist error
    #tree = KDTree(np.c_[gal_x, gal_y, gal_z])
    #nn   = tree.query(np.c_[ran_xyz[0], ran_xyz[1], ran_xyz[2]], return_distance=False, dualtree=True)
    #ran_lde = gal_lde[nn[:, 0]]
    ran_lde = utils.get_mesh_value(logdist_error_mesh, ran_xyz, box['lims'])
    ran_pve = utils.pv_from_logdist(ran_lde, fp_rand["Z"], cosmo)

    # Density grid lookups
    ran_npv   = utils.get_mesh_value(npv_mesh, ran_xyz, box['lims'])

    # Remove randoms outside mesh
    w = ran_npv > 0 
    fp_rand = fp_rand[w]
    ran_lde = ran_lde[w]
    ran_pve = ran_pve[w]
    ran_npv = ran_npv[w]

    log.info("  Data npv  : mean=%.4e  std=%.4e", ran_npv.mean(),   ran_npv.std())

    columns = [
        ("RA",          fp_rand["RA"].to_numpy()),
        ("DEC",         fp_rand["DEC"].to_numpy()),
        ("Z",           fp_rand["Z"].to_numpy()),
        ("WEIGHT",      fp_rand["WEIGHT"].to_numpy()),
        ("NPV",         ran_npv),
        #("NDENS",       ran_ndens),
        ("LOGDIST_ERR", ran_lde),
        ("PV_ERR",      ran_pve),
    ]
    hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name=n, format="D", array=a) for n, a in columns]
    )
    mock_fp_clus_rand = cfg.mock_fp_clus_rand.format(phase=cfg.fp_clus.phase)
    os.makedirs(os.path.dirname(mock_fp_clus_rand), exist_ok=True)
    log.info("Writing random catalogue → %s", mock_fp_clus_rand)
    hdu.writeto(mock_fp_clus_rand, overwrite=True)



# ---------------------------------------------------------------------------
# Step 5 — Process each mock into a clustering mock
# ---------------------------------------------------------------------------






def run_clustering_mock_loop(
    mocks: pd.DataFrame
):
    """
    Process all phase/realisation pairs into clustering mocks.
    """
    log.info("Generating clustering mocks …")

    #for phase in range(cfg.n_phases):
    for phase in [cfg.fp_clus.phase]:
        for real in range(cfg.n_reals):
            out_file = cfg.mock_fp_clus_data.format(phase=phase, real=real)
            mock = mocks[mocks['real']==real]
            log.info(f"  Mock phase={phase:03d} real={real:03d}:  {len(mock)} galaxies")
            log.info(f"  Writing mock to {out_file}")
            write_clustering_mock(mock, out_file)
            
def write_clustering_mock(mock: pd.DataFrame, outfile: str) -> None:
    """Write a processed mock to a FITS binary table."""

    columns_to_keep = ['RA', 'DEC', 'ZOBS', 'NPV', 
                       'LOGDIST', 'LOGDIST_GAUSS_ERR', 'LOGDIST_TRUE', 
                       'PV', 'PV_ERR', 'PV_TRUE'] 
    mock = mock[columns_to_keep]
    mock.rename(columns={"ZOBS": "Z", 
                         "LOGDIST_GAUSS_ERR": "LOGDIST_ERR"}, 
                inplace=True)

    #    ("RA",           mock["RA"].to_numpy()),
    #    ("DEC",          mock["DEC"].to_numpy()),
    #    ("Z",            mock["ZOBS"].to_numpy()),
    #    ("WEIGHT",       np.ones(len(mock))),
    #    ("NPV",          mock["NPV"].to_numpy()),
    #    #("NDENS",        mock["NDENS"].to_numpy()),
    #    ("LOGDIST",      mock["LOGDIST_CORR"].to_numpy()),
    #    ("LOGDIST_ERR",  mock["LOGDIST_GAUSS_ERR"].to_numpy()),
    #    ("LOGDIST_TRUE", mock["LOGDIST_TRUE"].to_numpy()),
    #    ("PV",           mock["PV"].to_numpy()),
    #    ("PV_ERR",       mock["PV_ERR"].to_numpy()),
    #    ("PV_TRUE",      mock["PV_TRUE"].to_numpy()),
    #]
    #hdu = fits.BinTableHDU.from_columns(
    #    [fits.Column(name=n, format="D", array=a) for n, a in columns]
    #)
    table = Table.from_pandas(mock)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    table.write(outfile, format='fits', overwrite=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline FP clustering "
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
    cfg.fp_clus.phase = args.phase
    
    
    log.info(f"=== DESI FP clustering mocks pipeline for phase {cfg.fp_clus.phase:03d} ===")

    distmax = cosmo.comoving_distance(cfg.fp_clus.zmax).value
    box = utils.build_grid_box(distmax, cfg.fp_clus.ngrid)


    # 1. Load observed data and compute n(z) 
    fp_data = load_observed_data()
    zbins = np.linspace(cfg.fp_clus.zmin, cfg.fp_clus.zmax, cfg.fp_clus.nzbin+1)
    nz_fp_data, _ = np.histogram(fp_data["Z"], bins=zbins, weights=fp_data["WEIGHT"])

    #-- Read mocks 
    mocks = read_mocks() 

    #--  Accumulate mock statistics
    stats = accumulate_mock_statistics(mocks)

    #-- Sub-sampling fraction 
    subsampling_fraction = compute_subsampling_fraction(nz_fp_data, stats["nz_fp_mock"])
    mocks = subsample_mocks(mocks, subsampling_fraction)

    #-- Compute and apply logdist bias correction
    logdist_bias_corr = compute_logdist_bias_correction(mocks)
    mocks["LOGDIST_CORR"] += logdist_bias_corr(mocks['ZOBS'])

    #-- Create mesh with average PV error in each cell, for later use in random catalogue
    npv_mesh, logdist_error_mesh = build_pv_meshes(mocks, box)
    mock_pos = utils.radec_to_xyz(mocks['RA'], mocks['DEC'], mocks['DIST'])
    mocks['NPV'] = utils.get_mesh_value(npv_mesh, mock_pos, box['lims'])

    #-- Build random catalogue
    fp_rand_cat = build_random_catalogue(subsampling_fraction, stats["nz_fp_mock"])
    write_random_catalogue(fp_rand_cat, npv_mesh, logdist_error_mesh, box)

    #--  Grid geometry and build number density mesh
    #fp_dist = cosmo.comoving_distance(fp_rand_cat["Z"].to_numpy()).value
    #fp_pos = utils.radec_to_xyz(fp_rand_cat["RA"].to_numpy(), 
    #                            fp_rand_cat["DEC"].to_numpy(),
    #                            fp_dist)
    #norm = (stats['nz_fp_mock'] * subsampling_fraction).sum()
    #npv_mesh = norm * utils.build_density_mesh(
    #    fp_pos,
    #    weights=fp_rand_cat["WEIGHT"].to_numpy(),
    #    box=box, 
    #    normalize=True
    #)

    #- Write clustering mocks
    run_clustering_mock_loop(mocks)
        


 
    log.info("=== Updating permissions ===")
    result = subprocess.run(
        ["chgrp", "-R", "desi", cfg.mock_fp_clus_dir],
        check=True,
    )

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()