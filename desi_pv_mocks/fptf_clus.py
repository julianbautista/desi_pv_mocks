"""
DESI Combined PV Mocks Pipeline (FP + TF)
==========================================
Lit les mocks de clustering FP et TF individuels, les combine,
calcule le champ de densité NPV sur grille 3D et écrit les catalogues
combinés (données + randoms × 20 et × 200).

Utilise les utilitaires partagés de desi_fp_mocks.py :
    weighted_avg_and_std, radec_to_xyz, build_density_grid, lookup_grid
"""

import os
import h5py
import logging
import numpy as np
import fitsio
import pandas as pd
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilitaires partagés (repris de desi_fp_mocks.py)
# ---------------------------------------------------------------------------

def weighted_avg_and_std(values, weights, axis=None):
    """(moyenne pondérée, erreur sur la moyenne, écart-type pondéré)"""
    avg = np.average(values, weights=weights, axis=axis)
    avg_err = np.std(values) * np.sqrt(np.sum((weights / np.sum(weights)) ** 2))
    variance = np.average((values - avg) ** 2, weights=weights, axis=axis)
    return avg, avg_err, np.sqrt(variance)


def radec_to_xyz(ra_deg, dec_deg, dist):
    """(RA, Dec, distance comobile) → tableau Cartésien (3, N)."""
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    return np.stack([
        dist * np.cos(dec) * np.cos(ra),
        dist * np.cos(dec) * np.sin(ra),
        dist * np.sin(dec),
    ])


def build_density_grid(ra_deg, dec_deg, z, weights, norm, geom, ngrid):
    """Histogramme 3D pondéré → grille de densité numérique."""
    dist = cosmo.comoving_distance(z).value
    xyz  = radec_to_xyz(ra_deg, dec_deg, dist)
    pos  = np.vstack([
        xyz[0] + geom["x0"],
        xyz[1] + geom["y0"],
        xyz[2] + geom["z0"],
    ]).T
    lx, ly, lz = geom["lx"], geom["ly"], geom["lz"]
    grid, _ = np.histogramdd(
        pos,
        bins=(ngrid, ngrid, ngrid),
        range=((0, lx), (0, ly), (0, lz)),
        weights=weights,
    )
    return (norm / geom["dvol"]) * (grid / grid.sum())


def lookup_grid(xyz, grid, geom):
    """Échantillonne la grille 3D aux positions xyz (3, N)."""
    xlims, ylims, zlims = geom["xlims"], geom["ylims"], geom["zlims"]
    ix = np.clip(np.digitize(xyz[0], xlims) - 1, 0, grid.shape[0] - 1)
    iy = np.clip(np.digitize(xyz[1], ylims) - 1, 0, grid.shape[1] - 1)
    iz = np.clip(np.digitize(xyz[2], zlims) - 1, 0, grid.shape[2] - 1)
    return grid[ix, iy, iz]


def write_fits_table(columns: dict, outfile: str) -> None:
    """Écrit un dictionnaire {nom: tableau} dans un fichier FITS BinTable."""
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    hdu = fits.BinTableHDU.from_columns([
        fits.Column(name=k, format="D", array=v) for k, v in columns.items()
    ])
    hdu.writeto(outfile, overwrite=True)
    log.info("Écrit → %s  (%d lignes)", outfile, len(next(iter(columns.values()))))


def read_fitsio(path: str) -> pd.DataFrame:
    """Lit un fichier FITS via fitsio et retourne un DataFrame."""
    return pd.DataFrame(fitsio.read(str(path)).byteswap().newbyteorder())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    zmin: float = 0.01
    zmax: float = 0.10
    ngrid: int  = 128
    nzbin: int  = 36
    nrealran: int = 1
    nsub: int     = 27
    n_phases: int = 25
    mock_version: float = 0.5
    comp_field: str = "Y1_COMP"
    pv_path: Path = Path("/global/cfs/cdirs/desi/science/td/pv")

    def __post_init__(self):
        mv  = self.mock_version
        pv  = self.pv_path
        mocks = pv / "mocks"

        self.base_stem    = mocks / f"BGS_base/v{mv}"
        self.fp_clus_root = mocks / f"FP_mocks/clusteringmocks/v{mv:.1f}.2"
        self.tf_clus_root = mocks / f"TF_mocks/clusteringmocks/v{mv:.1f}.3"
        self.out_root     = pv / "combinedpv/Y1/mocks"

        self.data_file    = pv / "combinedpv/Y1/PV_clustering_data.fits"
        self.random_file  = pv / "combinedpv/Y1/PV_clustering_random20.fits"

        self.fp_mock_tmpl = str(
            self.fp_clus_root
            / "FP_AbacusSummit_clustering_c000_ph{phase:03d}_r{real:03d}_v2.fits"
        )
        self.tf_mock_tmpl = str(
            self.tf_clus_root
            / "TF_AbacusSummit_clustering_c000_ph{phase:03d}_r{real:03d}.fits"
        )
        self.out_mock_tmpl = str(
            self.out_root
            / "Combined_AbacusSummit_clustering_c000_ph{phase:03d}_r{real:03d}.fits"
        )
        self.bgs_base_rand_tmpl = str(
            self.base_stem
            / "randoms/BGS_PV_AbacusSummit_base_c000_ph{phase:03d}_r{real:03d}_z0.11.ran.hdf5"
        )
        os.makedirs(self.out_root, exist_ok=True)


cfg   = Config()
cosmo = FlatLambdaCDM(H0=100, Om0=0.3151)

# ---------------------------------------------------------------------------
# Colonnes de sortie
# ---------------------------------------------------------------------------

MOCK_COLS = [
    "FP_FLAG", "RA", "DEC", "Z", "WEIGHT", "NPV", "NDENS",
    "LOGDIST", "LOGDIST_ERR", "LOGDIST_TRUE", "PV", "PV_ERR", "PV_TRUE",
]
RAND_COLS = ["FP_FLAG", "RA", "DEC", "Z", "WEIGHT", "NPV", "NDENS", "LOGDIST_ERR", "PV_ERR"]

# ---------------------------------------------------------------------------
# Étape 1 — Charger les données observées et accumuler les statistiques mock
# ---------------------------------------------------------------------------

def load_observed_nz(cfg: Config) -> dict:
    """Calcule les n(z) observées pour les sous-échantillons combined/TF/FP."""
    log.info("Chargement des données observées …")
    data    = read_fitsio(cfg.data_file)
    randoms = read_fitsio(cfg.random_file)

    zlims = [cfg.zmin, cfg.zmax]
    kw    = dict(bins=cfg.nzbin, range=zlims)

    tf = data[data["FP_FLAG"] == 0]
    fp = data[data["FP_FLAG"] == 1]

    result = dict(
        nzdata   = np.histogram(data["Z"],    weights=data["WEIGHT"],    **kw)[0],
        nztfdata = np.histogram(tf["Z"],      weights=tf["WEIGHT"],      **kw)[0],
        nzfpdata = np.histogram(fp["Z"],      weights=fp["WEIGHT"],      **kw)[0],
        sndata   = np.histogram(data["Z"],    weights=data["WEIGHT"] / data["PV_ERR"] ** 2,  **kw)[0],
        sntfdata = np.histogram(tf["Z"],      weights=tf["WEIGHT"]   / tf["PV_ERR"]   ** 2,  **kw)[0],
        snfpdata = np.histogram(fp["Z"],      weights=fp["WEIGHT"]   / fp["PV_ERR"]   ** 2,  **kw)[0],
    )
    log.info("  TF: %d  |  FP: %d  |  Total: %d", len(tf), len(fp), len(data))
    return result


def accumulate_mock_nz(cfg: Config) -> dict:
    """Boucle sur tous les mocks et accumule les n(z) moyens (combined/TF/FP)."""
    log.info("Accumulation des statistiques mock …")
    zlims = [cfg.zmin, cfg.zmax]
    kw    = dict(bins=cfg.nzbin, range=zlims)

    acc = {k: np.zeros(cfg.nzbin) for k in [
        "nzmock", "nzmock2", "nztf", "nztf2", "nzfp", "nzfp2",
        "snmock", "snmock2", "sntf", "sntf2", "snfp", "snfp2",
    ]}
    mock_count = 0

    for phase in range(cfg.n_phases):
        for real in range(cfg.nsub):
            try:
                fp = read_fitsio(cfg.fp_mock_tmpl.format(phase=phase, real=real))
                tf = read_fitsio(cfg.tf_mock_tmpl.format(phase=phase, real=real))
                fp["FP_FLAG"] = 1
                tf["FP_FLAG"] = 0
                combined = pd.concat([tf, fp], ignore_index=True)

                tf_sel = combined[combined["FP_FLAG"] == 0]
                fp_sel = combined[combined["FP_FLAG"] == 1]

                for key, cat, w in [
                    ("nzmock", combined, None),
                    ("nztf",   tf_sel,   None),
                    ("nzfp",   fp_sel,   None),
                ]:
                    nz = np.histogram(cat["Z"], weights=w, **kw)[0]
                    acc[key]       += nz
                    acc[key + "2"] += nz ** 2

                for key, cat in [
                    ("snmock", combined),
                    ("sntf",   tf_sel),
                    ("snfp",   fp_sel),
                ]:
                    sn = np.histogram(
                        cat["Z"], weights=1.0 / cat["PV_ERR"] ** 2, **kw
                    )[0]
                    acc[key]       += sn
                    acc[key + "2"] += sn ** 2

                mock_count += 1
            except Exception as exc:
                log.warning("Skipping phase=%d real=%d: %s", phase, real, exc)

    def _norm(key):
        mean = acc[key] / mock_count
        err  = np.sqrt(np.maximum(acc[key + "2"] / mock_count - mean ** 2, 0))
        return mean, err

    nzmock,  nzmockerr  = _norm("nzmock")
    nztf,    nztferr    = _norm("nztf")
    nzfp,    nzfperr    = _norm("nzfp")
    snmock,  snmockerr  = _norm("snmock")
    sntf,    sntferr    = _norm("sntf")
    snfp,    snfperr    = _norm("snfp")

    log.info("  %d mocks traités", mock_count)
    return dict(
        nzmock=nzmock, nzmockerr=nzmockerr,
        nztf=nztf,     nztferr=nztferr,
        nzfp=nzfp,     nzfperr=nzfperr,
        snmock=snmock, snmockerr=snmockerr,
        sntf=sntf,     sntferr=sntferr,
        snfp=snfp,     snfperr=snfperr,
    )

# ---------------------------------------------------------------------------
# Étape 2 — Construire le catalogue de randoms et la grille NPV
# ---------------------------------------------------------------------------

def build_random_catalogue(cfg: Config, nzmock: np.ndarray) -> pd.DataFrame:
    """Lit les randoms Abacus, applique les coupures et le sous-échantillonnage."""
    log.info("Lecture des catalogues randoms Abacus …")
    ra_all, dec_all, z_all = [], [], []

    for ireal in range(cfg.nrealran):
        for isub in range(cfg.nsub):
            path = cfg.bgs_base_rand_tmpl.format(phase=ireal, real=isub)
            try:
                with h5py.File(path, "r") as f:
                    ra   = f["ra"][...]
                    dec  = f["dec"][...]
                    z    = f["zobs"][...]
                    comp = f[cfg.comp_field][...]
                cut = (
                    (z >= cfg.zmin) & (z <= cfg.zmax)
                    & (np.random.uniform(size=len(z)) < comp)
                )
                ra_all.append(ra[cut])
                dec_all.append(dec[cut])
                z_all.append(z[cut])
            except Exception as exc:
                log.warning("Skipping random phase=%d real=%d: %s", ireal, isub, exc)

    ra_cat  = np.concatenate(ra_all)
    dec_cat = np.concatenate(dec_all)
    z_cat   = np.concatenate(z_all)
    w_cat   = np.ones(len(ra_cat))

    # Mélange
    idx = np.random.permutation(len(ra_cat))
    ra_cat, dec_cat, z_cat, w_cat = ra_cat[idx], dec_cat[idx], z_cat[idx], w_cat[idx]
    log.info("  %d randoms avant sous-échantillonnage", len(z_cat))

    # Sous-échantillonnage pour correspondre à nzmock
    nzold   = np.histogram(z_cat, bins=cfg.nzbin, range=[cfg.zmin, cfg.zmax], weights=w_cat)[0]
    sfrac   = np.where(nzold > 0, nzmock / nzold, 0.0)
    sfrac  /= sfrac.max()
    izs     = np.clip(
        np.digitize(z_cat, np.linspace(cfg.zmin, cfg.zmax, cfg.nzbin + 1)) - 1,
        0, cfg.nzbin - 1,
    )
    keep    = sfrac[izs] > np.random.uniform(size=len(z_cat))
    ra_cat, dec_cat, z_cat, w_cat = ra_cat[keep], dec_cat[keep], z_cat[keep], w_cat[keep]
    log.info("  %d randoms après sous-échantillonnage", len(z_cat))

    return pd.DataFrame({"RA": ra_cat, "DEC": dec_cat, "Z": z_cat, "WEIGHT": w_cat})


def build_grid_geometry(cfg: Config) -> dict:
    distmax = cosmo.comoving_distance(cfg.zmax).value
    lx = ly = lz = 2.0 * distmax
    x0 = y0 = z0 = distmax
    dvol  = (lx / cfg.ngrid) ** 3
    xlims = np.linspace(0.0, lx, cfg.ngrid + 1) - x0
    ylims = np.linspace(0.0, ly, cfg.ngrid + 1) - y0
    zlims = np.linspace(0.0, lz, cfg.ngrid + 1) - z0
    return dict(lx=lx, ly=ly, lz=lz, x0=x0, y0=y0, z0=z0,
                dvol=dvol, xlims=xlims, ylims=ylims, zlims=zlims)


def build_npv_grid(cfg: Config, randoms: pd.DataFrame, nzmock: np.ndarray, geom: dict) -> np.ndarray:
    """Construit la grille de densité NPV à partir des randoms."""
    log.info("Construction de la grille NPV …")
    return build_density_grid(
        randoms["RA"].to_numpy(), randoms["DEC"].to_numpy(),
        randoms["Z"].to_numpy(),  randoms["WEIGHT"].to_numpy(),
        norm=nzmock.sum(), geom=geom, ngrid=cfg.ngrid,
    )

# ---------------------------------------------------------------------------
# Étape 3 — Générer les mocks combinés
# ---------------------------------------------------------------------------

def add_npv_column(df: pd.DataFrame, npvgrid: np.ndarray, geom: dict) -> pd.DataFrame:
    """Ajoute la colonne NPV interpolée depuis la grille 3D."""
    dist = cosmo.comoving_distance(df["Z"].to_numpy()).value
    xyz  = radec_to_xyz(df["RA"].to_numpy(), df["DEC"].to_numpy(), dist)
    xyz_shifted = xyz + np.array([[geom["x0"]], [geom["y0"]], [geom["z0"]]])
    df = df.copy()
    df["NPV"] = lookup_grid(xyz_shifted, npvgrid, geom)
    return df


def generate_combined_mocks(cfg: Config, npvgrid: np.ndarray, geom: dict) -> None:
    """Boucle sur tous les mocks, combine FP+TF, ajoute NPV et écrit en FITS."""
    log.info("Génération des mocks combinés …")
    for phase in range(cfg.n_phases):
        log.info("  Phase %d …", phase)
        for real in range(cfg.nsub):
            try:
                fp = read_fitsio(cfg.fp_mock_tmpl.format(phase=phase, real=real))
                tf = read_fitsio(cfg.tf_mock_tmpl.format(phase=phase, real=real))
                fp["FP_FLAG"] = 1
                tf["FP_FLAG"] = 0
                combined = pd.concat([tf, fp], ignore_index=True)
                combined = add_npv_column(combined, npvgrid, geom)

                write_fits_table(
                    {col: combined[col].to_numpy() for col in MOCK_COLS},
                    cfg.out_mock_tmpl.format(phase=phase, real=real),
                )
            except Exception as exc:
                log.warning("Skipping mock phase=%d real=%d: %s", phase, real, exc)


def generate_combined_random(cfg: Config, npvgrid: np.ndarray, geom: dict, rfact: int) -> None:
    """Lit les randoms FP+TF, ajoute NPV et écrit le catalogue random combiné."""
    suffix = rfact  # 20 ou 200
    log.info("Génération du catalogue random combiné (×%d) …", suffix)

    fp_rand = read_fitsio(
        cfg.fp_clus_root / f"FP_AbacusSummit_clustering_randoms{suffix}_v2.fits"
    )
    tf_rand = read_fitsio(
        cfg.tf_clus_root / f"TF_AbacusSummit_clustering_randoms{suffix}.fits"
    )
    fp_rand["FP_FLAG"] = 1
    tf_rand["FP_FLAG"] = 0
    combined = pd.concat([tf_rand, fp_rand], ignore_index=True)
    combined = add_npv_column(combined, npvgrid, geom)

    outfile = str(
        cfg.out_root / f"Combined_AbacusSummit_clustering_randoms{suffix}.fits"
    )
    write_fits_table(
        {col: combined[col].to_numpy() for col in RAND_COLS},
        outfile,
    )

# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Pipeline mocks combinés FP+TF ===")

    # 1. Données observées et statistiques mock
    obs_nz   = load_observed_nz(cfg)
    mock_nz  = accumulate_mock_nz(cfg)

    # 2. Catalogue randoms et grille NPV
    geom     = build_grid_geometry(cfg)
    randoms  = build_random_catalogue(cfg, mock_nz["nzmock"])
    npvgrid  = build_npv_grid(cfg, randoms, mock_nz["nzmock"], geom)

    # 3. Mocks combinés
    generate_combined_mocks(cfg, npvgrid, geom)

    # 4. Catalogues randoms combinés (×20 et ×200)
    for rfact in (20, 200):
        generate_combined_random(cfg, npvgrid, geom, rfact=rfact)

    log.info("=== Pipeline terminé ===")


if __name__ == "__main__":
    main()