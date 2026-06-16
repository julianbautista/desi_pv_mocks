"""
DESI TF Mocks Processing Pipeline
===================================
Reads in TF mocks, summarises them, downsamples each mock to match the data
n(z), and converts to clustering mocks. Also produces a random catalogue by
downsampling Chris Blake's BGS randoms.

Architecture mirrors the FP pipeline (make_DESI_FP_clustering_mocks.py).
Shared utilities (weighted_avg_and_std, reweight, radec_to_xyz,
pv_from_logdist, build_density_grid, lookup_grid) are identical.
"""

import os
import h5py
import logging
import argparse
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
CONFIG, TF_CLUS = None, None

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
# Utility functions  (identical signatures to the FP pipeline)
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
    return np.stack([
        dist * np.cos(dec) * np.cos(ra),
        dist * np.cos(dec) * np.sin(ra),
        dist * np.sin(dec),
    ])

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
# TF-specific helpers
# ---------------------------------------------------------------------------

def inflate_errors(mock: pd.DataFrame, sigma: float) -> pd.DataFrame:
    """
    Inflate logdist errors to better match the data, and propagate the
    additional scatter into the measured values to conserve the pull distribution.

    σ_new = 0.75 * sqrt(σ_old² + (σ_TFR/5)²)
    η_new = η_true + 0.75 * (η_obs − η_true)
    """
    mock = mock.copy()
    mock["LOGDIST_ERR"] = 0.75 * np.sqrt(
        mock["LOGDIST_ERR"] ** 2 + (sigma / 5.0) ** 2
    )
    mock["LOGDIST"] = (
        mock["LOGDIST_TRUE"] + 0.75 * (mock["LOGDIST"] - mock["LOGDIST_TRUE"])
    )
    return mock


def read_tf_mock(path: str) -> tuple[pd.DataFrame, float]:
    """Read a TF full mock FITS file. Returns (DataFrame, sigma from header)."""
    with fits.open(path) as hdul:
        sigma = hdul[1].header["SIGMA"]
        df = pd.DataFrame(hdul[1].data)
    return df, sigma


# ---------------------------------------------------------------------------
# Step 1 — Load observed data
# ---------------------------------------------------------------------------

def load_observed_data() -> pd.DataFrame:
    """Load BGS and TF clustering data + randoms."""
    log.info("Loading observed data …")
    #bgs_data = Table.read(CONFIG.bgs_clus_data).to_pandas()
    #bgs_rand = Table.read(CONFIG.bgs_clus_rand).to_pandas()
    tf_data  = Table.read(CONFIG.data_tf_clus_data).to_pandas()
    tf_rand  = Table.read(CONFIG.data_tf_clus_rand).to_pandas()

    tf_data["LOGDIST"]          = (
        np.log10(cosmo.luminosity_distance(tf_data["Z"].to_numpy()).value)
        + 5.0
        - tf_data["MU"] / 5.0
    )
    tf_data["LOGDIST_ERR"]       = tf_data["MU_ERR"] / 5.0
    tf_data["LOGDIST_GAUSS_ERR"] = tf_data["LOGDIST_ERR"].copy()

    #log.info("  BGS data: %d galaxies | BGS rand: %d", len(bgs_data), len(bgs_rand))
    log.info("  TF  data: %d galaxies | TF  rand: %d", len(tf_data),  len(tf_rand))
    #return bgs_data, bgs_rand, tf_data, tf_rand
    return tf_data, tf_rand


# ---------------------------------------------------------------------------
# Step 2 — Accumulate mock statistics
# ---------------------------------------------------------------------------

def accumulate_mock_statistics() -> dict:
    """
    Loop over all phases/realisations and accumulate n(z) and logdist statistics.
    Returns a dict of arrays for downstream use.
    """
    log.info("Accumulating mock statistics …")
    zrange = [TF_CLUS.zmin, TF_CLUS.zmax]

    nzmock        = np.zeros(TF_CLUS.nzbin)
    nzmockerr2    = np.zeros(TF_CLUS.nzbin)
    nztfmock      = np.zeros(TF_CLUS.nzbin)
    nztfmockerr2  = np.zeros(TF_CLUS.nzbin)
    logdistmock   = np.zeros(TF_CLUS.nzbin)
    logdisterr    = np.zeros(TF_CLUS.nzbin)
    logdisterr_g  = np.zeros(TF_CLUS.nzbin)
    pullmock      = np.zeros(TF_CLUS.nzbin)

    mock_count = tfmock_count = 0
    ngals      = 0
    mean_pull  = std_pull = 0.0

    all_z, all_ld_true, all_ld_obs  = [], [], []
    all_ld_err, all_ld_gerr          = [], []

    for phase in range(CONFIG.n_phases):
        log.debug("  Phase %d …", phase)
        for real in range(CONFIG.n_reals):

            # ---- BGS clustering mock ----
            bgs_file = CONFIG.mock_bgs_clus_data.format(phase=phase, real=real)
            try:
                mock = Table.read(bgs_file).to_pandas()
                nz   = np.histogram(mock["Z"], bins=TF_CLUS.nzbin,
                                    range=zrange, weights=mock["WEIGHT"])[0]
                nzmock      += nz
                nzmockerr2  += nz ** 2
                mock_count  += 1
            except Exception as exc:
                log.warning("Skipping BGS mock %s: %s", bgs_file, exc)

            # ---- TF full mock ----
            tf_file = CONFIG.mock_tf_full_data.format(phase=phase, real=real)
            try:
                mock, sigma = read_tf_mock(tf_file)
                mock = mock[(mock["ZOBS"] >= TF_CLUS.zmin) & (mock["ZOBS"] <= TF_CLUS.zmax)].copy()

                # Completeness downsampling
                keep = np.random.uniform(size=len(mock)) < mock[CONFIG.comp_field].to_numpy()
                mock = mock.iloc[keep].copy()

                mock = inflate_errors(mock, sigma)
                mock["LOGDIST_GAUSS_ERR"] = reweight(
                    mock["LOGDIST"].to_numpy(), mock["LOGDIST_ERR"].to_numpy()
                )

                # Zero-point the mock
                offset = weighted_avg_and_std(
                    mock["LOGDIST"].to_numpy() - mock["LOGDIST_TRUE"].to_numpy(),
                    1.0 / mock["LOGDIST_GAUSS_ERR"].to_numpy() ** 2,
                )[0]
                mock["LOGDIST"] -= offset

                ngals += len(mock)
                all_z.append(mock["ZOBS"].to_numpy())
                all_ld_true.append(mock["LOGDIST_TRUE"].to_numpy())
                all_ld_obs.append(mock["LOGDIST"].to_numpy())
                all_ld_err.append(mock["LOGDIST_ERR"].to_numpy())
                all_ld_gerr.append(mock["LOGDIST_GAUSS_ERR"].to_numpy())

                nz = np.histogram(mock["ZOBS"], bins=TF_CLUS.nzbin, range=zrange)[0]
                nztfmock     += nz
                nztfmockerr2 += nz ** 2
                logdistmock  += np.histogram(mock["LOGDIST"],     bins=TF_CLUS.nzbin, range=[-0.3, 0.3])[0]
                logdisterr   += np.histogram(mock["LOGDIST_ERR"], bins=TF_CLUS.nzbin, range=[0.09, 0.20])[0]
                logdisterr_g += np.histogram(mock["LOGDIST_GAUSS_ERR"], bins=TF_CLUS.nzbin, range=[0.09, 0.20])[0]

                pulls = (mock["LOGDIST"] - mock["LOGDIST_TRUE"]) / mock["LOGDIST_GAUSS_ERR"]
                pullmock  += np.histogram(pulls, bins=TF_CLUS.nzbin, range=[-4.0, 4.0])[0]
                mean_pull += pulls.sum()
                std_pull  += (pulls ** 2).sum()
                tfmock_count += 1

            except Exception as exc:
                log.warning("Skipping TF mock %s: %s", tf_file, exc)

    nzmock      /= max(mock_count, 1)
    nzmockerr    = np.sqrt(np.maximum(nzmockerr2   / max(mock_count, 1)   - nzmock ** 2,   0))
    nztfmock    /= max(tfmock_count, 1)
    nztfmockerr  = np.sqrt(np.maximum(nztfmockerr2 / max(tfmock_count, 1) - nztfmock ** 2, 0))
    logdistmock /= max(tfmock_count, 1)
    logdisterr  /= max(tfmock_count, 1)
    logdisterr_g /= max(tfmock_count, 1)
    pullmock    /= max(tfmock_count, 1)
    if ngals > 0:
        mean_pull /= ngals
        std_pull   = np.sqrt(std_pull / ngals - mean_pull ** 2)

    log.info(
        "Mocks processed: %d BGS / %d TF | pull: mean=%.4f, std=%.4f",
        mock_count, tfmock_count, mean_pull, std_pull,
    )

    return dict(
        nzmock=nzmock,     nzmockerr=nzmockerr,
        nztfmock=nztfmock, nztfmockerr=nztfmockerr,
        logdistmock=logdistmock, logdisterr=logdisterr, logdisterr_g=logdisterr_g,
        pullmock=pullmock, mean_pull=mean_pull, std_pull=std_pull,
        zvals          = np.concatenate(all_z),
        logdists_true  = np.concatenate(all_ld_true),
        logdists_obs   = np.concatenate(all_ld_obs),
        logdists_err   = np.concatenate(all_ld_err),
        logdists_gerr  = np.concatenate(all_ld_gerr),
    )


# ---------------------------------------------------------------------------
# Step 3 — Sub-sampling fraction and logdist bias correction
# ---------------------------------------------------------------------------

def compute_subsampling_fraction(
    nztfdat: np.ndarray,
    nztfmock: np.ndarray,
) -> np.ndarray:
    """
    Per-z-bin sub-sampling fraction so mock n(z) matches data n(z).
    Clipped to [0, 1]; smoothed where the mock is already at capacity.
    """
    subfrac = np.where(nztfmock > 0, nztfdat / nztfmock, 1.0)
    subfrac = np.clip(subfrac, 0.0, 1.0)
    smooth  = savgol_filter(subfrac, 10, 1)
    # Only smooth bins where subsampling < 1 (i.e. mock is denser than data)
    subfrac = np.where(subfrac == 1.0, 1.0, smooth)
    log.info("Sub-sampling fraction: min=%.3f, max=%.3f", subfrac.min(), subfrac.max())
    return subfrac


def compute_logdist_bias_correction(stats: dict) -> CubicSpline:
    """
    Fit a cubic spline to the weighted-mean logdist residual vs. redshift.
    Returns a callable correction f(z).
    """
    bins    = np.linspace(TF_CLUS.zmin, TF_CLUS.zmax, TF_CLUS.nzbin)
    midvals = 0.5 * (bins[:-1] + bins[1:])
    ld_mean = np.zeros(TF_CLUS.nzbin - 1)

    zv  = stats["zvals"]
    ldo = stats["logdists_obs"]
    ldt = stats["logdists_true"]
    lge = stats["logdists_gerr"]

    for k, (zlo, zhi) in enumerate(zip(bins[:-1], bins[1:])):
        idx = np.where((zv > zlo) & (zv <= zhi))[0]
        if len(idx) > 2:
            ld_mean[k] = weighted_avg_and_std(
                ldo[idx] - ldt[idx], 1.0 / lge[idx] ** 2
            )[0]

    return CubicSpline(midvals, ld_mean)


# ---------------------------------------------------------------------------
# Step 4 — Build random catalogue
# ---------------------------------------------------------------------------

def build_random_catalogue(
    subfrac: np.ndarray,
    nztfmock: np.ndarray,
) -> pd.DataFrame:
    """
    Read Abacus base randoms, apply completeness cut and sub-sampling,
    then return a DataFrame with columns RA, DEC, Z, WEIGHT.
    """
    log.info("Reading Abacus random catalogues …")
    ra_all, dec_all, z_all = [], [], []

    for ireal in range(CONFIG.n_phases_rand):
        for isub in range(CONFIG.n_reals):
            path = CONFIG.mock_bgs_base_rand.format(phase=ireal, real=isub)
            try:
                with h5py.File(path, "r") as f:
                    ra   = f["ra"][...]
                    dec  = f["dec"][...]
                    z    = f["zobs"][...]
                    comp = f[CONFIG.comp_field][...]
                nran = len(ra)
                cut  = (
                    (z >= TF_CLUS.zmin)
                    & (z <= TF_CLUS.zmax)
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
    nzold     = np.histogram(z_cat, bins=TF_CLUS.nzbin, range=[TF_CLUS.zmin, TF_CLUS.zmax],
                             weights=w_cat)[0]
    sfrac_ran = np.where(nzold > 0, nztfmock * subfrac / nzold, 0.0)
    sfrac_ran = sfrac_ran / sfrac_ran.max()

    izs = np.clip(
        np.digitize(z_cat, np.linspace(TF_CLUS.zmin, TF_CLUS.zmax, TF_CLUS.nzbin + 1)) - 1,
        0, TF_CLUS.nzbin - 1,
    )
    cut = sfrac_ran[izs] > np.random.uniform(size=len(z_cat))
    ra_cat, dec_cat, z_cat, w_cat = ra_cat[cut], dec_cat[cut], z_cat[cut], w_cat[cut]
    log.info("  Total randoms after  sub-sampling: %d", len(z_cat))

    return pd.DataFrame({"RA": ra_cat, "DEC": dec_cat, "Z": z_cat, "WEIGHT": w_cat})


# ---------------------------------------------------------------------------
# Step 5 — Build 3-D density grids
# ---------------------------------------------------------------------------

def build_grid_geometry() -> dict:
    """Return the grid geometry (box dimensions, bin edges, voxel volume)."""
    distmax = cosmo.comoving_distance(TF_CLUS.zmax).value
    lx = ly = lz = 2.0 * distmax
    dvol  = (lx / TF_CLUS.ngrid) ** 3
    x0 = y0 = z0 = distmax
    xlims = np.linspace(0.0, lx, TF_CLUS.ngrid + 1) - x0
    ylims = np.linspace(0.0, ly, TF_CLUS.ngrid + 1) - y0
    zlims = np.linspace(0.0, lz, TF_CLUS.ngrid + 1) - z0
    return dict(lx=lx, ly=ly, lz=lz, x0=x0, y0=y0, z0=z0,
                dvol=dvol, xlims=xlims, ylims=ylims, zlims=zlims)


def build_density_grids(
    bgs_rand: pd.DataFrame,
    tf_rand: pd.DataFrame,
    nzmock: np.ndarray,
    nztfmock: np.ndarray,
    subfrac: np.ndarray,
    geom: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Build BGS density grid (NDENS) and TF PV density grid (NPV)."""
    log.info("Building 3-D density grids …")
    edges = (geom["lx"], geom["ly"], geom["lz"],
             geom["x0"], geom["y0"], geom["z0"])

    ndensweigrid = build_density_grid(
        bgs_rand["RA"].to_numpy(), bgs_rand["DEC"].to_numpy(),
        bgs_rand["Z"].to_numpy(),  bgs_rand["WEIGHT"].to_numpy(),
        norm=nzmock.sum() / geom["dvol"],
        grid_edges=edges, ngrid=TF_CLUS.ngrid, box_vol=geom["dvol"],
    )
    npvweigrid = build_density_grid(
        tf_rand["RA"].to_numpy(),  tf_rand["DEC"].to_numpy(),
        tf_rand["Z"].to_numpy(),   tf_rand["WEIGHT"].to_numpy(),
        norm=(nztfmock * subfrac).sum() / geom["dvol"],
        grid_edges=edges, ngrid=TF_CLUS.ngrid, box_vol=geom["dvol"],
    )
    log.info("  n(z) total: BGS mock=%.0f | TF mock=%.0f | TF subsampled=%.0f",
             nzmock.sum(), nztfmock.sum(), (nztfmock * subfrac).sum())
    return ndensweigrid, npvweigrid


# ---------------------------------------------------------------------------
# Step 6 — Process each mock into a clustering mock
# ---------------------------------------------------------------------------

def process_mock(
    mock: pd.DataFrame,
    sigma: float,
    subfrac: np.ndarray,
    logdist_fix: CubicSpline,
    ndensweigrid: np.ndarray,
    npvweigrid: np.ndarray,
    geom: dict,
) -> pd.DataFrame:
    """
    Apply completeness cut, error inflation, sub-sampling, Gaussianisation,
    zero-pointing, PV estimation, and density field sampling to one mock.
    """
    mock = mock[(mock["ZOBS"] >= TF_CLUS.zmin) & (mock["ZOBS"] <= TF_CLUS.zmax)].copy()

    # Completeness
    keep = np.random.uniform(size=len(mock)) < mock[CONFIG.comp_field].to_numpy()
    mock = mock.iloc[keep].copy()

    # Error inflation (TF-specific)
    mock = inflate_errors(mock, sigma)

    # Sub-sample to match data n(z)
    izs  = np.clip(
        np.digitize(mock["ZOBS"].to_numpy(),
                    np.linspace(TF_CLUS.zmin, TF_CLUS.zmax, TF_CLUS.nzbin + 1)) - 1,
        0, TF_CLUS.nzbin - 1,
    )
    keep = subfrac[izs] > np.random.uniform(size=len(mock))
    mock = mock.iloc[keep].copy()

    # Gaussianise errors
    mock["LOGDIST_GAUSS_ERR"] = reweight(
        mock["LOGDIST"].to_numpy(), mock["LOGDIST_ERR"].to_numpy()
    )

    # Zero-point + redshift-dependent bias correction
    offset = weighted_avg_and_std(
        mock["LOGDIST"].to_numpy() - mock["LOGDIST_TRUE"].to_numpy(),
        1.0 / mock["LOGDIST_GAUSS_ERR"].to_numpy() ** 2,
    )[0]
    mock["LOGDIST"] -= offset + logdist_fix(mock["ZOBS"].to_numpy())

    # PV estimation
    mock["PV"]      = pv_from_logdist(mock["LOGDIST"].to_numpy(),      mock["ZOBS"].to_numpy())
    mock["PV_ERR"]  = pv_from_logdist(mock["LOGDIST_GAUSS_ERR"].to_numpy(), mock["ZOBS"].to_numpy())
    mock["PV_TRUE"] = LIGHT_SPEED * (
        (1.0 + mock["ZOBS"].to_numpy()) / (1.0 + mock["ZCOS"].to_numpy()) - 1.0
    )

    # 3-D Cartesian positions
    dist = cosmo.comoving_distance(mock["ZOBS"].to_numpy()).value
    xyz  = radec_to_xyz(mock["RA"].to_numpy(), mock["DEC"].to_numpy(), dist)
    mock["_x"], mock["_y"], mock["_z"] = xyz[0], xyz[1], xyz[2]

    xyz_shifted = xyz + np.array([[geom["x0"]], [geom["y0"]], [geom["z0"]]])
    mock["NDENS"] = lookup_grid(xyz_shifted, ndensweigrid,
                                geom["xlims"], geom["ylims"], geom["zlims"])
    mock["NPV"]   = lookup_grid(xyz_shifted, npvweigrid,
                                geom["xlims"], geom["ylims"], geom["zlims"])
    return mock


def write_clustering_mock(mock: pd.DataFrame, outfile: str) -> None:
    """Write a processed mock to a FITS binary table."""
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    columns = [
        ("RA",           mock["RA"].to_numpy()),
        ("DEC",          mock["DEC"].to_numpy()),
        ("Z",            mock["ZOBS"].to_numpy()),
        ("WEIGHT",       np.ones(len(mock))),
        ("NPV",          mock["NPV"].to_numpy()),
        ("NDENS",        mock["NDENS"].to_numpy()),
        ("LOGDIST",      mock["LOGDIST"].to_numpy()),
        ("LOGDIST_ERR",  mock["LOGDIST_GAUSS_ERR"].to_numpy()),
        ("LOGDIST_TRUE", mock["LOGDIST_TRUE"].to_numpy()),
        ("PV",           mock["PV"].to_numpy()),
        ("PV_ERR",       mock["PV_ERR"].to_numpy()),
        ("PV_TRUE",      mock["PV_TRUE"].to_numpy()),
    ]
    hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name=n, format="D", array=a) for n, a in columns]
    )
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
    Returns (x, y, z, logdist_gauss_err) arrays accumulated across all mocks
    for nearest-neighbour error assignment to randoms.
    """
    log.info("Generating clustering mocks …")
    all_x, all_y, all_z_gal, all_lde = [], [], [], []

    for phase in range(CONFIG.n_phases):
        for real in range(CONFIG.n_reals):
            tf_file  = CONFIG.mock_tf_full_data.format(phase=phase, real=real)
            out_file = CONFIG.mock_tf_clus_data.format(phase=phase, real=real)
            try:
                raw, sigma = read_tf_mock(tf_file)
                mock = process_mock(
                    raw, sigma, subfrac, logdist_fix,
                    ndensweigrid, npvweigrid, geom,
                )
                log.info("  Writing mock ph=%03d r=%03d → %s", phase, real, out_file)
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
# Step 7 — Write random catalogue with nearest-neighbour error assignment
# ---------------------------------------------------------------------------

def write_random_catalogue(
    tf_rand: pd.DataFrame,
    gal_x: np.ndarray,
    gal_y: np.ndarray,
    gal_z: np.ndarray,
    gal_lde: np.ndarray,
    ndensweigrid: np.ndarray,
    npvweigrid: np.ndarray,
    geom: dict,
    nztfmock: np.ndarray,
    subfrac: np.ndarray,
    rfact: int = 20,
) -> None:
    """
    Assign logdist/PV errors to randoms via nearest-neighbour matching,
    then write FITS random catalogues at rfact x 10 and rfact x 1 sizes.
    """
    log.info("Building random catalogue with NN error assignment …")

    n_target = rfact * int((nztfmock * subfrac).sum())
    if len(tf_rand) > n_target:
        idx     = np.random.choice(len(tf_rand), n_target, replace=False)
        tf_rand = tf_rand.iloc[idx].reset_index(drop=True)
    log.info("  Randoms after size truncation: %d (rfact=%d)", len(tf_rand), rfact)

    ran_dist = cosmo.comoving_distance(tf_rand["Z"].to_numpy()).value
    ran_xyz  = radec_to_xyz(tf_rand["RA"].to_numpy(), tf_rand["DEC"].to_numpy(), ran_dist)
    ran_xyz_shifted = ran_xyz + np.array([[geom["x0"]], [geom["y0"]], [geom["z0"]]])

    # Nearest-neighbour logdist error assignment
    tree    = KDTree(np.c_[gal_x, gal_y, gal_z])
    nn      = tree.query(np.c_[ran_xyz[0], ran_xyz[1], ran_xyz[2]],
                         return_distance=False, dualtree=True)
    ran_lde = gal_lde[nn[:, 0]]
    ran_pve = pv_from_logdist(ran_lde, tf_rand["Z"].to_numpy())

    ran_ndens = lookup_grid(ran_xyz_shifted, ndensweigrid,
                            geom["xlims"], geom["ylims"], geom["zlims"])
    ran_npv   = lookup_grid(ran_xyz_shifted, npvweigrid,
                            geom["xlims"], geom["ylims"], geom["zlims"])

    log.info("  Random NDENS: mean=%.4e  std=%.4e", ran_ndens.mean(), ran_ndens.std())
    log.info("  Random NPV  : mean=%.4e  std=%.4e", ran_npv.mean(),   ran_npv.std())

    columns = [
        ("RA",          tf_rand["RA"].to_numpy()),
        ("DEC",         tf_rand["DEC"].to_numpy()),
        ("Z",           tf_rand["Z"].to_numpy()),
        ("WEIGHT",      tf_rand["WEIGHT"].to_numpy()),
        ("NPV",         ran_npv),
        ("NDENS",       ran_ndens),
        ("LOGDIST_ERR", ran_lde),
        ("PV_ERR",      ran_pve),
    ]

    #def _write(outfile: str, n: int) -> None:
    #    idx = np.random.choice(len(tf_rand), n, replace=True)
    hdu = fits.BinTableHDU.from_columns([
            fits.Column(name=nm, format="D", array=arr)
            for nm, arr in columns
        ])
    log.info("  Writing random catalogue → %s", CONFIG.mock_tf_clus_rand)
    os.makedirs(os.path.dirname(CONFIG.mock_tf_clus_rand), exist_ok=True)
    hdu.writeto(CONFIG.mock_tf_clus_rand, overwrite=True)

    #n_base     = int((nztfmock * subfrac).sum())
    #rand20_out = CONFIG.mock_tf_clus_rand.replace("random20", "random200")
    #_write(rand20_out,            200 * n_base)
    #_write(CONFIG.mock_tf_clus_rand,  20 * n_base)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline TF clustering "
    )
    parser.add_argument("config_file", type=str, help="Configuration file path (yaml format)")
    args = parser.parse_args()
    return args

def main() -> None:
    log.info("=== DESI TF Mocks Pipeline ===")
    args = parse_args()
    cfg = load_config(args.config_file)
    global CONFIG, TF_CLUS
    CONFIG, TF_CLUS = cfg.CONFIG, cfg.TF_CLUS  
    
    # 1. Load observed data
    tf_data, tf_rand = load_observed_data()

    nztfdat = np.histogram(
        tf_data["Z"], bins=TF_CLUS.nzbin, range=[TF_CLUS.zmin, TF_CLUS.zmax],
        weights=tf_data["WEIGHT"],
    )[0]

    # 2. Accumulate mock statistics
    stats = accumulate_mock_statistics()

    # 3. Sub-sampling fraction & logdist bias correction
    subfrac     = compute_subsampling_fraction(nztfdat, stats["nztfmock"])
    logdist_fix = compute_logdist_bias_correction(stats)

    # 4. Build random catalogue (using BGS base randoms, sub-sampled to TF n(z))
    tf_rand_cat = build_random_catalogue(subfrac, stats["nztfmock"])

    # 5. Grid geometry and density grids
    geom = build_grid_geometry()

    # Load the BGS clustering randoms for the density grid
    bgs_clus_rand = Table.read(CONFIG.mock_bgs_clus_rand).to_pandas()

    ndensweigrid, npvweigrid = build_density_grids(
        bgs_clus_rand, tf_rand_cat,
        stats["nzmock"], stats["nztfmock"], subfrac, geom,
    )

    # 6. Generate clustering mocks
    gal_x, gal_y, gal_z, gal_lde = run_clustering_mock_loop(
        subfrac, logdist_fix, ndensweigrid, npvweigrid, geom,
    )

    # 7. Write random catalogue
    write_random_catalogue(
        tf_rand_cat, gal_x, gal_y, gal_z, gal_lde,
        ndensweigrid, npvweigrid, geom, 
        stats["nztfmock"], subfrac,
    )

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()