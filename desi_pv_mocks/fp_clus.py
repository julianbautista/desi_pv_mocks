"""
DESI FP Mocks Processing Pipeline
===================================
Reads in FP mocks created with make_DESI_FP_mocks.py, summarises them,
downsamples each mock to match the data n(z), and converts to clustering mocks.
Also produces a random catalogue by downsampling Chris Blake's BGS randoms.
"""

import argparse
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from config import load_config
CONFIG = None 
FP_CLUS = None 

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


LIGHT_SPEED = 299_792.458  # km/s



cosmo = FlatLambdaCDM(H0=100, Om0=0.3151)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def weighted_avg_and_std(values, weights, axis=None):
    """Return (weighted mean, propagated error on mean, weighted std)."""
    avg = np.average(values, weights=weights, axis=axis)
    avg_err = np.std(values) * np.sqrt(np.sum((weights / np.sum(weights)) ** 2))
    variance = np.average((values - avg) ** 2, weights=weights, axis=axis)
    return avg, avg_err, np.sqrt(variance)


def reweight(x: np.ndarray, err: np.ndarray) -> np.ndarray:
    """Gaussianise errors via a linear tilt that leaves the mean error unchanged."""
    weight = 1.0 / err ** 2
    mean_x = x.mean()
    lam = (
        (np.sum(x * weight) - mean_x * weight.sum())
        / (np.sum(x ** 2) - len(x) * mean_x ** 2)
    )
    new_weight = weight - lam * (x - mean_x)
    #- JB : rarely the new weight can be sligthly negative
    w = new_weight <= 0 
    new_weight[w] = 1e-3
    new_err = np.sqrt(1.0 / new_weight) 
    new_err = new_err - new_err.mean() + err.mean()
    return new_err 


def radec_to_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Convert (RA, Dec, comoving distance) → Cartesian (x, y, z). Returns (3, N)."""
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    x = dist * np.cos(dec) * np.cos(ra)
    y = dist * np.cos(dec) * np.sin(ra)
    z = dist * np.sin(dec)
    return np.stack([x, y, z])


def pv_from_logdist(logdist: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Carreres et al. (2023) v1 estimator:  pv = c ln(10) η / (c(1+z)/χH(z) − 1)
    """
    denom = (
        LIGHT_SPEED * (1.0 + z)
        / (cosmo.comoving_distance(z).value * cosmo.H(z).value)
        - 1.0
    )
    return LIGHT_SPEED * np.log(10.0) * logdist / denom


def build_density_grid(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    z: np.ndarray,
    weights: np.ndarray,
    norm: float,
    grid_edges: tuple,
    ngrid: int,
    box_vol: float,
) -> np.ndarray:
    """Histogram galaxies/randoms into a 3-D number-density grid."""
    dist = cosmo.comoving_distance(z).value
    xyz  = radec_to_xyz(ra_deg, dec_deg, dist)
    lx, ly, lz, x0, y0, z0 = grid_edges
    pos  = np.vstack([xyz[0] + x0, xyz[1] + y0, xyz[2] + z0]).T
    grid, _ = np.histogramdd(
        pos,
        bins=(ngrid, ngrid, ngrid),
        range=((0, lx), (0, ly), (0, lz)),
        weights=weights,
    )
    return (norm / box_vol) * (grid / grid.sum())


def lookup_grid(
    xyz: np.ndarray,
    grid: np.ndarray,
    xlims: np.ndarray,
    ylims: np.ndarray,
    zlims: np.ndarray,
) -> np.ndarray:
    """Sample a pre-computed 3-D density grid at arbitrary positions."""
    ix = np.clip(np.digitize(xyz[0], xlims) - 1, 0, grid.shape[0] - 1)
    iy = np.clip(np.digitize(xyz[1], ylims) - 1, 0, grid.shape[1] - 1)
    iz = np.clip(np.digitize(xyz[2], zlims) - 1, 0, grid.shape[2] - 1)
    return grid[ix, iy, iz]


# ---------------------------------------------------------------------------
# Step 1 — Load observed data & mocks, accumulate statistics
# ---------------------------------------------------------------------------

def load_observed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load BGS and FP clustering data + randoms."""
    log.info("Loading observed data …")
    #bgs_data = Table.read(CONFIG.data_bgs_clus_data).to_pandas()
    bgs_rand = Table.read(CONFIG.data_bgs_clus_rand).to_pandas()
    fp_data  = Table.read(CONFIG.data_fp_clus_data).to_pandas()
    #fp_rand  = Table.read(CONFIG.data_fp_clus_rand).to_pandas()

    fp_data["LOGDIST_GAUSS_ERR"] = reweight(fp_data["LOGDIST"], fp_data["LOGDIST_ERR"])

    #log.info("  BGS data: %d galaxies | BGS rand: %d", len(bgs_data), len(bgs_rand))
    #log.info("  FP  data: %d galaxies | FP  rand: %d", len(fp_data), len(fp_rand))
    #return bgs_data, bgs_rand, fp_data, fp_rand
    return bgs_rand, fp_data


def accumulate_mock_statistics() -> dict:
    """
    Loop over all phases/realisations and accumulate n(z) and logdist statistics.
    Returns a dict of arrays for downstream use.
    """
    log.info("Accumulating mock statistics …")
    zlims = [FP_CLUS.zmin, FP_CLUS.zmax]

    # Accumulators
    nz_bgs_mock      = np.zeros(FP_CLUS.nzbin)
    nz_bgs_mock_err2  = np.zeros(FP_CLUS.nzbin)
    nz_fp_mock    = np.zeros(FP_CLUS.nzbin)
    nz_fp_mock_err2 = np.zeros(FP_CLUS.nzbin)
    logdistmock  = np.zeros(FP_CLUS.nzbin)
    logdisterr   = np.zeros(FP_CLUS.nzbin)
    logdisterr_g = np.zeros(FP_CLUS.nzbin)
    pullmock     = np.zeros(FP_CLUS.nzbin)

    mock_count = fpmock_count = 0
    ngals = 0
    mean_pull = std_pull = 0.0

    # Per-galaxy arrays (collected across all mocks, used later)
    all_z         = []
    all_ld_true   = []
    all_ld_obs    = []
    all_ld_err    = []
    all_ld_corr   = []
    all_ld_cerr   = []
    all_ld_gerr   = []

    #for phase in range(CONFIG.n_phases):
    for phase in [FP_CLUS.phase]:
        log.info("  Phase %d …", phase)
        for real in range(CONFIG.n_reals):

            # ---- BGS clustering mock ----
            bgs_file = CONFIG.mock_bgs_clus_data.format(phase=phase, real=real)
            try:
                mock = Table.read(bgs_file).to_pandas()
                nz   = np.histogram(mock["Z"], bins=FP_CLUS.nzbin, range=zlims, weights=mock["WEIGHT"])[0]
                nz_bgs_mock     += nz
                nz_bgs_mock_err2 += nz ** 2
                mock_count += 1
            except Exception as exc:
                log.warning("Skipping BGS mock %s: %s", bgs_file, exc)

            # ---- FP full mock ----
            fp_file = CONFIG.mock_fp_full_data.format(phase=phase, real=real)
            try:
                mock = Table.read(fp_file).to_pandas()
                mock = mock[(mock["ZOBS"] >= FP_CLUS.zmin) & (mock["ZOBS"] <= FP_CLUS.zmax)].copy()
                ngals += len(mock)

                mock["LOGDIST_GAUSS_ERR"] = reweight(
                    mock["LOGDIST_CORR"].to_numpy(), mock["LOGDIST_CORR_ERR"].to_numpy()
                )

                # Zero-point calibration
                offset = weighted_avg_and_std(
                    mock["LOGDIST_CORR"].to_numpy() - mock["LOGDIST_TRUE"].to_numpy(),
                    1.0 / mock["LOGDIST_GAUSS_ERR"].to_numpy() ** 2,
                )[0]
                
                mock["LOGDIST_CORR"] -= offset

                # Collect per-galaxy arrays
                all_z.append(mock["ZOBS"].to_numpy())
                all_ld_true.append(mock["LOGDIST_TRUE"].to_numpy())
                all_ld_obs.append(mock["LOGDIST"].to_numpy())
                all_ld_err.append(mock["LOGDIST_ERR"].to_numpy())
                all_ld_corr.append(mock["LOGDIST_CORR"].to_numpy())
                all_ld_cerr.append(mock["LOGDIST_CORR_ERR"].to_numpy())
                all_ld_gerr.append(mock["LOGDIST_GAUSS_ERR"].to_numpy())

                # Histograms
                nz = np.histogram(mock["ZOBS"], bins=FP_CLUS.nzbin, range=zlims)[0]
                nz_fp_mock     += nz
                nz_fp_mock_err2 += nz ** 2
                logdistmock  += np.histogram(mock["LOGDIST_CORR"], bins=FP_CLUS.nzbin, range=[-0.3, 0.3])[0]
                logdisterr   += np.histogram(mock["LOGDIST_CORR_ERR"], bins=FP_CLUS.nzbin, range=[0.08, 0.30])[0]
                logdisterr_g += np.histogram(mock["LOGDIST_GAUSS_ERR"], bins=FP_CLUS.nzbin, range=[0.08, 0.30])[0]

                pulls = (
                    (mock["LOGDIST_CORR"] - mock["LOGDIST_TRUE"])
                    / mock["LOGDIST_GAUSS_ERR"]
                )
                pullmock  += np.histogram(pulls, bins=FP_CLUS.nzbin, range=[-4.0, 4.0])[0]
                mean_pull += pulls.sum()
                std_pull  += (pulls ** 2).sum()

                fpmock_count += 1
            except Exception as exc:
                log.warning("Skipping FP mock %s: %s", fp_file, exc)

    # Normalise
    nz_bgs_mock     /= mock_count
    nz_bgs_mock_err   = np.sqrt(np.maximum(nz_bgs_mock_err2 / mock_count - nz_bgs_mock ** 2, 0))
    nz_fp_mock   /= fpmock_count
    nz_fp_mock_err = np.sqrt(np.maximum(nz_fp_mock_err2 / fpmock_count - nz_fp_mock ** 2, 0))
    logdistmock /= fpmock_count
    logdisterr  /= fpmock_count
    logdisterr_g /= fpmock_count
    pullmock    /= fpmock_count
    mean_pull   /= ngals
    std_pull     = np.sqrt(std_pull / ngals - mean_pull ** 2)

    log.info(
        "Mocks processed: %d BGS / %d FP | pull: mean=%.4f, std=%.4f",
        mock_count, fpmock_count, mean_pull, std_pull,
    )

    return dict(
        nz_bgs_mock=nz_bgs_mock, nz_bgs_mock_err=nz_bgs_mock_err,
        nz_fp_mock=nz_fp_mock, nz_fp_mock_err=nz_fp_mock_err,
        logdistmock=logdistmock, logdisterr=logdisterr, logdisterr_g=logdisterr_g,
        pullmock=pullmock, mean_pull=mean_pull, std_pull=std_pull,
        zvals=np.concatenate(all_z),
        logdists_true=np.concatenate(all_ld_true),
        logdists_obs=np.concatenate(all_ld_obs),
        logdists_err=np.concatenate(all_ld_err),
        logdists_corr=np.concatenate(all_ld_corr),
        logdists_corr_err=np.concatenate(all_ld_cerr),
        logdists_gauss_err=np.concatenate(all_ld_gerr),
    )


# ---------------------------------------------------------------------------
# Step 2 — Compute sub-sampling fraction and logdist bias correction
# ---------------------------------------------------------------------------

def compute_subsampling_fraction(nz_fp_data: np.ndarray, nz_fp_mock: np.ndarray) -> np.ndarray:
    """
    Compute per-z-bin sub-sampling fraction so mock n(z) matches data n(z).
    Values are clipped to [0, 1] and normalised to max=1.
    """
    subfrac = np.where(nz_fp_mock > 0, nz_fp_data / nz_fp_mock, 1.0)
    subfrac = np.where(subfrac > 1.0, 1.0, subfrac)
    subfrac /= subfrac.max()
    # Smooth in bins where the mock is already sparse (subfrac == 1)
    subfrac = np.where(subfrac == 1.0, 1.0, savgol_filter(subfrac, 15, 1))
    log.info("Sub-sampling fraction: min=%.3f, max=%.3f", subfrac.min(), subfrac.max())
    return subfrac


def compute_logdist_bias_correction(stats: dict) -> CubicSpline:
    """
    Fit a cubic spline to the weighted-mean logdist residual vs. redshift.
    Returns a callable correction f(z).
    """
    bins    = np.linspace(FP_CLUS.zmin, FP_CLUS.zmax, FP_CLUS.nzbin)
    midvals = 0.5 * (bins[:-1] + bins[1:])
    ld_mean = np.zeros(FP_CLUS.nzbin - 1)

    zv  = stats["zvals"]
    ldc = stats["logdists_corr"]
    ldt = stats["logdists_true"]
    lge = stats["logdists_gauss_err"]

    for k, (zlo, zhi) in enumerate(zip(bins[:-1], bins[1:])):
        idx = np.where((zv > zlo) & (zv <= zhi))[0]
        if len(idx) > 2:
            ld_mean[k] = weighted_avg_and_std(ldc[idx] - ldt[idx], 1.0 / lge[idx] ** 2)[0]

    return CubicSpline(midvals, ld_mean)


# ---------------------------------------------------------------------------
# Step 3 — Build random catalogue
# ---------------------------------------------------------------------------

def build_random_catalogue(subfrac: np.ndarray, nz_fp_mock: np.ndarray) -> pd.DataFrame:
    """
    Read Abacus base randoms, apply completeness cut and sub-sampling,
    then return a DataFrame with columns RA, DEC, Z, WEIGHT.
    """
    log.info("Reading Abacus random catalogues …")
    ra_all, dec_all, z_all = [], [], []

    #for phase in range(CONFIG.n_real_rand):
    for phase in [FP_CLUS.phase]: 
        for ireal in range(CONFIG.n_reals):
            log.info("  Randoms phase %d real %d", phase, ireal)
            path = CONFIG.mock_bgs_base_rand.format(phase=phase, real=ireal)
            try:
                with h5py.File(path, "r") as f:
                    ra   = f["ra"][...]
                    dec  = f["dec"][...]
                    z    = f["zobs"][...]
                    comp = f[CONFIG.comp_field][...]
                nran = len(ra)
                cut  = (
                    (z >= FP_CLUS.zmin)
                    & (z <= FP_CLUS.zmax)
                    & (np.random.uniform(size=nran) < comp)
                )
                ra_all.append(ra[cut])
                dec_all.append(dec[cut])
                z_all.append(z[cut])
            except Exception as exc:
                log.warning("Skipping random %s: %s", path, exc)

    ra_cat  = np.concatenate(ra_all)
    dec_cat = np.concatenate(dec_all)
    z_cat   = np.concatenate(z_all)
    w_cat   = np.ones(len(ra_cat))

    # Shuffle
    idx = np.random.permutation(len(ra_cat))
    ra_cat, dec_cat, z_cat, w_cat = ra_cat[idx], dec_cat[idx], z_cat[idx], w_cat[idx]
    log.info("  Total randoms before sub-sampling: %d", len(z_cat))

    # Sub-sample to match the (already subsampled) mock n(z)
    nzold      = np.histogram(z_cat, bins=FP_CLUS.nzbin, range=[FP_CLUS.zmin, FP_CLUS.zmax], weights=w_cat)[0]
    sfrac_ran  = np.where(nzold > 0, nz_fp_mock * subfrac / nzold, 0.0)
    sfrac_ran /= sfrac_ran.max()
    izs = np.digitize(z_cat, np.linspace(FP_CLUS.zmin, FP_CLUS.zmax, FP_CLUS.nzbin + 1)) - 1
    izs = np.clip(izs, 0, FP_CLUS.nzbin - 1)
    cut = sfrac_ran[izs] > np.random.uniform(size=len(z_cat))
    ra_cat, dec_cat, z_cat, w_cat = ra_cat[cut], dec_cat[cut], z_cat[cut], w_cat[cut]
    log.info("  Total randoms after  sub-sampling: %d", len(z_cat))

    return pd.DataFrame({"RA": ra_cat, "DEC": dec_cat, "Z": z_cat, "WEIGHT": w_cat})


# ---------------------------------------------------------------------------
# Step 4 — Build 3-D density grids
# ---------------------------------------------------------------------------

def build_grid_geometry() -> dict:
    """Return the grid geometry (box dimensions, bin edges, voxel volume)."""
    distmax = cosmo.comoving_distance(FP_CLUS.zmax).value
    lx = ly = lz = 2.0 * distmax
    dx = dy = dz = lx / FP_CLUS.ngrid
    x0 = y0 = z0 = distmax
    dvol = dx * dy * dz

    xlims = np.linspace(0.0, lx, FP_CLUS.ngrid + 1) - x0
    ylims = np.linspace(0.0, ly, FP_CLUS.ngrid + 1) - y0
    zlims = np.linspace(0.0, lz, FP_CLUS.ngrid + 1) - z0

    return dict(lx=lx, ly=ly, lz=lz, x0=x0, y0=y0, z0=z0, dvol=dvol,
                xlims=xlims, ylims=ylims, zlims=zlims)


def build_density_grids(
    bgs_rand: pd.DataFrame,
    fp_rand: pd.DataFrame,
    nz_bgs_mock: np.ndarray,
    nz_fp_mock: np.ndarray,
    subfrac: np.ndarray,
    geom: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Build BGS density grid and FP PV density grid."""
    log.info("Building 3-D density grids …")

    #bgs_dist = cosmo.comoving_distance(bgs_rand["Z"].to_numpy()).value
    ndensweigrid = build_density_grid(
        bgs_rand["RA"].to_numpy(), bgs_rand["DEC"].to_numpy(),
        bgs_rand["Z"].to_numpy(), bgs_rand["WEIGHT"].to_numpy(),
        norm=nz_bgs_mock.sum() / geom["dvol"],
        grid_edges=(geom["lx"], geom["ly"], geom["lz"],
                    geom["x0"], geom["y0"], geom["z0"]),
        ngrid=FP_CLUS.ngrid, box_vol=geom["dvol"],
    )

    npvweigrid = build_density_grid(
        fp_rand["RA"].to_numpy(), fp_rand["DEC"].to_numpy(),
        fp_rand["Z"].to_numpy(), fp_rand["WEIGHT"].to_numpy(),
        norm=(nz_fp_mock * subfrac).sum() / geom["dvol"],
        grid_edges=(geom["lx"], geom["ly"], geom["lz"],
                    geom["x0"], geom["y0"], geom["z0"]),
        ngrid=FP_CLUS.ngrid, box_vol=geom["dvol"],
    )

    log.info("  n(z) total: BGS mock=%.0f | FP mock=%.0f | FP subsampled=%.0f",
             nz_bgs_mock.sum(), nz_fp_mock.sum(), (nz_fp_mock * subfrac).sum())
    return ndensweigrid, npvweigrid


# ---------------------------------------------------------------------------
# Step 5 — Process each mock into a clustering mock
# ---------------------------------------------------------------------------

def process_mock(
    mock: pd.DataFrame,
    subfrac: np.ndarray,
    logdist_fix: CubicSpline,
    ndensweigrid: np.ndarray,
    npvweigrid: np.ndarray,
    geom: dict,
) -> pd.DataFrame:
    """
    Apply sub-sampling, Gaussianisation, zero-pointing, PV estimation,
    and density field sampling to a single mock realisation.
    Returns the processed DataFrame with all output columns.
    """
    mock = mock[(mock["ZOBS"] >= FP_CLUS.zmin) & (mock["ZOBS"] <= FP_CLUS.zmax)].copy()

    # Sub-sample to match data n(z)
    izs = np.clip(
        np.digitize(mock["ZOBS"].to_numpy(), np.linspace(FP_CLUS.zmin, FP_CLUS.zmax, FP_CLUS.nzbin + 1)) - 1,
        0, FP_CLUS.nzbin - 1,
    )
    keep = subfrac[izs] > np.random.uniform(size=len(mock))
    mock = mock.iloc[keep].copy()

    # Gaussianise errors
    mock["LOGDIST_GAUSS_ERR"] = reweight(
        mock["LOGDIST_CORR"].to_numpy(), mock["LOGDIST_CORR_ERR"].to_numpy()
    )

    # Zero-point + bias correction
    offset = weighted_avg_and_std(
        mock["LOGDIST_CORR"].to_numpy() - mock["LOGDIST_TRUE"].to_numpy(),
        1.0 / mock["LOGDIST_GAUSS_ERR"].to_numpy() ** 2,
    )[0]
    mock["LOGDIST_CORR"] -= offset + logdist_fix(mock["ZOBS"].to_numpy())

    # PV estimation
    mock["PV"]      = pv_from_logdist(mock["LOGDIST_CORR"].to_numpy(), mock["ZOBS"].to_numpy())
    mock["PV_ERR"]  = pv_from_logdist(mock["LOGDIST_GAUSS_ERR"].to_numpy(), mock["ZOBS"].to_numpy())
    mock["PV_TRUE"] = LIGHT_SPEED * (
        (1.0 + mock["ZOBS"].to_numpy()) / (1.0 + mock["ZCOS"].to_numpy()) - 1.0
    )

    # 3-D positions
    dist = cosmo.comoving_distance(mock["ZOBS"].to_numpy()).value
    xyz  = radec_to_xyz(mock["RA"].to_numpy(), mock["DEC"].to_numpy(), dist)
    xyz_shifted = xyz + np.array([[geom["x0"]], [geom["y0"]], [geom["z0"]]])
    mock["_x"] = xyz[0]
    mock["_y"] = xyz[1]
    mock["_z"] = xyz[2]

    mock["NDENS"] = lookup_grid(xyz_shifted, ndensweigrid, geom["xlims"], geom["ylims"], geom["zlims"])
    mock["NPV"]   = lookup_grid(xyz_shifted, npvweigrid,   geom["xlims"], geom["ylims"], geom["zlims"])

    return mock


def write_clustering_mock(mock: pd.DataFrame, outfile: str) -> None:
    """Write a processed mock to a FITS binary table."""
    columns = [
        ("RA",           mock["RA"].to_numpy()),
        ("DEC",          mock["DEC"].to_numpy()),
        ("Z",            mock["ZOBS"].to_numpy()),
        ("WEIGHT",       np.ones(len(mock))),
        ("NPV",          mock["NPV"].to_numpy()),
        ("NDENS",        mock["NDENS"].to_numpy()),
        ("LOGDIST",      mock["LOGDIST_CORR"].to_numpy()),
        ("LOGDIST_ERR",  mock["LOGDIST_GAUSS_ERR"].to_numpy()),
        ("LOGDIST_TRUE", mock["LOGDIST_TRUE"].to_numpy()),
        ("PV",           mock["PV"].to_numpy()),
        ("PV_ERR",       mock["PV_ERR"].to_numpy()),
        ("PV_TRUE",      mock["PV_TRUE"].to_numpy()),
    ]
    hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name=n, format="D", array=a) for n, a in columns]
    )
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    hdu.writeto(outfile, overwrite=True)


def run_clustering_mock_loop(
    subfrac: np.ndarray,
    logdist_fix: CubicSpline,
    ndensweigrid: np.ndarray,
    npvweigrid: np.ndarray,
    geom: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process all phase/realisation pairs into clustering mocks.
    Returns arrays of (x, y, z, logdist_err) accumulated across all mocks,
    used later for nearest-neighbour error assignment to randoms.
    """
    log.info("Generating clustering mocks …")
    all_x, all_y, all_z_gal, all_lde = [], [], [], []

    #for phase in range(CONFIG.n_phases):
    for phase in [FP_CLUS.phase]:
        for real in range(CONFIG.n_reals):
            fp_file = CONFIG.mock_fp_full_data.format(phase=phase, real=real)
            out_file = CONFIG.mock_fp_clus_data.format(phase=phase, real=real)
            if os.path.exists(out_file) and not FP_CLUS.overwrite:
                log.info(f"Already exists: {out_file} — skipped")
                continue

            try:
                raw  = Table.read(fp_file).to_pandas()
                mock = process_mock(
                    raw, subfrac, logdist_fix, ndensweigrid, npvweigrid, geom
                )
                log.info(f"  Writing mock {phase:d}-{real:d} to {out_file}")
                write_clustering_mock(mock, out_file)

                all_x.append(mock["_x"].to_numpy())
                all_y.append(mock["_y"].to_numpy())
                all_z_gal.append(mock["_z"].to_numpy())
                all_lde.append(mock["LOGDIST_GAUSS_ERR"].to_numpy())
            except Exception as exc:
                log.warning("Skipping mock phase=%d real=%d: %s", phase, real, exc)

    return (
        np.concatenate(all_x),
        np.concatenate(all_y),
        np.concatenate(all_z_gal),
        np.concatenate(all_lde),
    )


# ---------------------------------------------------------------------------
# Step 6 — Write random catalogue with nearest-neighbour error assignment
# ---------------------------------------------------------------------------

def write_random_catalogue(
    fp_rand: pd.DataFrame,
    gal_x: np.ndarray,
    gal_y: np.ndarray,
    gal_z: np.ndarray,
    gal_lde: np.ndarray,
    ndensweigrid: np.ndarray,
    npvweigrid: np.ndarray,
    geom: dict,
    nz_fp_mock: np.ndarray,
    subfrac: np.ndarray,
) -> None:
    """
    Assign logdist/PV errors to randoms via nearest-neighbour matching,
    then write the FITS random catalogue.
    """
    log.info("Building random catalogue with NN error assignment …")

    # Truncate random catalogue to rfact × expected galaxy count
    n_target = FP_CLUS.rfact * int((nz_fp_mock * subfrac).sum())
    if len(fp_rand) > n_target:
        idx     = np.random.choice(len(fp_rand), n_target, replace=False)
        fp_rand = fp_rand.iloc[idx].reset_index(drop=True)
    log.info("  Randoms after truncation: %d", len(fp_rand))

    ran_dist = cosmo.comoving_distance(fp_rand["Z"].to_numpy()).value
    ran_xyz  = radec_to_xyz(fp_rand["RA"].to_numpy(), fp_rand["DEC"].to_numpy(), ran_dist)
    ran_xyz_shifted = ran_xyz + np.array([[geom["x0"]], [geom["y0"]], [geom["z0"]]])

    # Nearest-neighbour logdist error
    tree = KDTree(np.c_[gal_x, gal_y, gal_z])
    nn   = tree.query(np.c_[ran_xyz[0], ran_xyz[1], ran_xyz[2]], return_distance=False, dualtree=True)
    ran_lde = gal_lde[nn[:, 0]]
    ran_pve = pv_from_logdist(ran_lde, fp_rand["Z"].to_numpy())

    # Density grid lookups
    ran_ndens = lookup_grid(ran_xyz_shifted, ndensweigrid, geom["xlims"], geom["ylims"], geom["zlims"])
    ran_npv   = lookup_grid(ran_xyz_shifted, npvweigrid,   geom["xlims"], geom["ylims"], geom["zlims"])

    log.info("  Data dens : mean=%.4e  std=%.4e", ran_ndens.mean(), ran_ndens.std())
    log.info("  Data npv  : mean=%.4e  std=%.4e", ran_npv.mean(),   ran_npv.std())

    columns = [
        ("RA",          fp_rand["RA"].to_numpy()),
        ("DEC",         fp_rand["DEC"].to_numpy()),
        ("Z",           fp_rand["Z"].to_numpy()),
        ("WEIGHT",      fp_rand["WEIGHT"].to_numpy()),
        ("NPV",         ran_npv),
        ("NDENS",       ran_ndens),
        ("LOGDIST_ERR", ran_lde),
        ("PV_ERR",      ran_pve),
    ]
    hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name=n, format="D", array=a) for n, a in columns]
    )
    mock_fp_clus_rand = CONFIG.mock_fp_clus_rand.format(phase=FP_CLUS.phase)
    os.makedirs(os.path.dirname(mock_fp_clus_rand), exist_ok=True)
    log.info("Writing random catalogue → %s", mock_fp_clus_rand)
    hdu.writeto(mock_fp_clus_rand, overwrite=True)

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
    cfg = load_config(args.config_file)
    global CONFIG, FP_CLUS
    CONFIG, FP_CLUS = cfg.CONFIG, cfg.FP_CLUS    
    FP_CLUS.phase = args.phase


    log.info(f"=== DESI FP clustering mocks pipeline for phase {FP_CLUS.phase:03d} ===")

    # 1. Load observed data and compute n(z) 
    bgs_rand, fp_data = load_observed_data()

    nz_fp_data = np.histogram(
        fp_data["Z"], bins=FP_CLUS.nzbin, range=[FP_CLUS.zmin, FP_CLUS.zmax],
        weights=fp_data["WEIGHT"],
    )[0]

    # 2. Accumulate mock statistics
    stats = accumulate_mock_statistics()

    #for k in stats.keys():
    #    log.info(f"  {k}: {stats[k]} {stats[k].size} {np.isnan(stats[k]).sum()}")

    # 3. Sub-sampling fraction & logdist bias correction
    subfrac     = compute_subsampling_fraction(nz_fp_data, stats["nz_fp_mock"])
    log.info("WARNING: not subsampling")
    subfrac = subfrac*0 + 1.0
    logdist_fix = compute_logdist_bias_correction(stats)

    # 4. Build random catalogue
    fp_rand_cat = build_random_catalogue(subfrac, stats["nz_fp_mock"])

    # 5. Grid geometry and density grids
    geom = build_grid_geometry()
    ndensweigrid, npvweigrid = build_density_grids(
        bgs_rand, fp_rand_cat,
        stats["nz_bgs_mock"], stats["nz_fp_mock"], subfrac, geom,
    )

    # 6. Generate clustering mocks
    gal_x, gal_y, gal_z, gal_lde = run_clustering_mock_loop(
        subfrac, logdist_fix, ndensweigrid, npvweigrid, geom
    )

    # 7. Write random catalogue
    write_random_catalogue(
        fp_rand_cat, gal_x, gal_y, gal_z, gal_lde,
        ndensweigrid, npvweigrid, geom, 
        stats["nz_fp_mock"], subfrac,
    )

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()