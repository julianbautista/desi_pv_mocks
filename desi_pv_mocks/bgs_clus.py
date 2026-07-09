"""
Build BGS PV clustering catalogues from AbacusSummit mocks.
 
Processes random and data mock catalogues: applies redshift/magnitude/
completeness cuts, builds a 3D number density grid from randoms, samples
the grid at galaxy positions, and writes FITS output catalogues.
"""

import subprocess
import argparse
import os
import logging
import h5py
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM, Planck15
from astropy.io import fits
from astropy.table import Table
from scipy.signal import savgol_filter

from . import utils

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from .config import load_config
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def apply_selection(redshift, app_mag, abs_mag,
                    zmin, zmax, appmaglim, absmaglim,
                    comp_subsample=False, comp=None, comp_min=0, 
                    rng=None):
    """Return boolean mask applying redshift, magnitude, and completeness cuts."""
    if rng is None:
        rng = np.random.default_rng()
    mask = (
        (redshift > zmin) & (redshift < zmax)
        & (app_mag < appmaglim)
        & (abs_mag < absmaglim)
    )
    if not comp is None: 
        mask &= (comp > comp_min)
        if comp_subsample:
            subsample = rng.uniform(size=len(redshift)) < comp
            mask &= subsample
    return mask 
 
def make_fits_table(**columns):
    """Build an astropy BinTableHDU from keyword-argument {name: array} pairs."""
    cols = [fits.Column(name=k, format="D", array=v) for k, v in columns.items()]
    return fits.BinTableHDU.from_columns(cols)

 
def write_ndens_grid(path, zmin, zmax, box, grid):
    """Write the number-density grid to a plain text file."""
    ngrid = box["ngrid"]
    with open(path, "w") as fh:
        fh.write(f"{zmin} {zmax}\n")
        fh.write(
            f"{ngrid} {ngrid} {ngrid} "
            f"{box['side']} {box['side']} {box['side']} "
            f"{box['origin']} {box['origin']} {box['origin']}\n"
        )
        for iz in range(ngrid):
            for iy in range(ngrid):
                for ix in range(ngrid):
                    fh.write(f"{grid[ix, iy, iz]}\n")
    log.info("Number density grid written to %s", path)
 
def compute_subsampling_fraction(nz_data: np.ndarray, nz_mock: np.ndarray) -> np.ndarray:
    """
    Compute per-z-bin sub-sampling fraction so mock n(z) matches data n(z).
    Values are clipped to [0, 1] and normalised to max=1.
    """
    subfrac = np.where(nz_mock > 0, nz_data / nz_mock, 1.0)
    subfrac = np.where(subfrac > 1.0, 1.0, subfrac)
    subfrac /= subfrac.max()
    # Smooth in bins where the mock is already sparse (subfrac == 1)
    subfrac = np.where(subfrac == 1.0, 1.0, savgol_filter(subfrac, 15, 1))
    log.info("Sub-sampling fraction: min=%.3f, max=%.3f", subfrac.min(), subfrac.max())
    return subfrac

def load_observed_data():
    """Load BGS clustering data + randoms."""
    log.info("Loading observed data …")
    bgs_data = Table.read(cfg.data_bgs_clus_data).to_pandas()
    #bgs_rand = Table.read(cfg.data_bgs_clus_rand).to_pandas()

    log.info(f"  BGS data: {len(bgs_data)} galaxies")
    #log.info("  FP  data: %d galaxies | FP  rand: %d", len(fp_data), len(fp_rand))
    return bgs_data#, bgs_rand


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline BGS clustering "
    )
    parser.add_argument("config_file", type=str, help="Configuration file path (yaml format)")
    parser.add_argument("phase",        type=int, help="Phase (0–24)")
    args = parser.parse_args()
    if not (0 <= args.phase <= 24):
        parser.error("phase should be between 0 and 24")

    args = parser.parse_args()
    return args

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
 
def main():
    args = parse_args()
    global cfg
    cfg = load_config(args.config_file)
    phase = args.phase
    cfg.bgs_clus.phase = phase

    comp_field = cfg.comp_field
    rng = np.random.default_rng()
 
    os.makedirs(cfg.mock_bgs_clus_dir, exist_ok=True)
    os.makedirs(cfg.mock_bgs_clus_dir+"/data", exist_ok=True)
    os.makedirs(cfg.mock_bgs_clus_dir+"/rand", exist_ok=True)

    cosmo = FlatLambdaCDM(H0=100, Om0=0.3151)
    distmax = cosmo.comoving_distance(cfg.bgs_clus.zmax).value
    box = utils.build_grid_box(distmax, cfg.bgs_clus.ngrid)
    zbins = np.linspace(cfg.bgs_clus.zmin, cfg.bgs_clus.zmax, cfg.bgs_clus.nzbin + 1)

    # ------------------------------------------------------------------
    # Phase 1: build the number density grid from all random catalogues
    # ------------------------------------------------------------------
 
    log.info(f"=== Building number density grid from randoms for phase {phase:03d} ===")
 
    ra_ran, dec_ran, z_ran, comp_ran = [], [], [], []
    counts_ran = []
    frac_skys = []

    #for phase in range(cfg.n_phases_rand):
    for real in range(cfg.n_reals):
        mock_infile_rand = cfg.mock_bgs_base_rand.format(phase=phase, real=real)
        log.info("Reading random catalogue: %s", mock_infile_rand)

        with h5py.File(mock_infile_rand, "r") as hf:
            ra   = hf["ra"][...]
            dec  = hf["dec"][...]
            zobs = hf["zobs"][...]
            absmag = hf["abs_mag"][...]
            appmag = hf["app_mag"][...]
            comp   = hf[comp_field][...]

        frac_sky = np.sum(comp[comp>cfg.bgs_clus.comp_min]) / len(ra)
        log.info(f" {frac_sky*100:.1f}% randoms selected from completeness cut of {cfg.bgs_clus.comp_min}" )

        #-- We apply redshift, magnitude cuts as well 
        #-- as a completeness cut, but we DO NOT subsample the randoms ! 
        mask = apply_selection(
            zobs, appmag, absmag,
            zmin=cfg.bgs_clus.zmin, 
            zmax=cfg.bgs_clus.zmax,
            appmaglim=cfg.bgs_clus.appmaglim, 
            absmaglim=cfg.bgs_clus.absmaglim,
            comp_min=cfg.bgs_clus.comp_min,
            comp=comp,
        )
        n_sel = mask.sum()
        log.info(f" {n_sel} / {len(ra)} randoms pass selection (z, mag, completeness)" )
        counts_ran.append(n_sel)
        frac_skys.append(frac_sky)
        ra_ran.append(ra[mask])
        dec_ran.append(dec[mask])
        z_ran.append(zobs[mask])
        comp_ran.append(comp[mask])

    ra_ran  = np.concatenate(ra_ran)
    dec_ran = np.concatenate(dec_ran)
    z_ran   = np.concatenate(z_ran)
    comp_ran = np.concatenate(comp_ran)
    weight_ran   = 1/comp_ran 
    weight_ran *= weight_ran.size/np.sum(weight_ran)
    dist_ran = cosmo.comoving_distance(z_ran).value
    log.info(f"Total randoms after selection: {len(ra_ran)}")

    n_avg = np.mean(counts_ran)
    frac_sky_avg = np.mean(frac_skys)
    log.info(f"Mean randoms per realisation/phase: {n_avg:.1f}")
    log.info(f"Mean sky fraction per realisation/phase: {frac_sky_avg:.3f}")

    pos_ran = utils.radec_to_xyz(ra_ran, dec_ran, dist_ran)
 
    #-- Compute number density in a 3D mesh with the stacked randoms
    ndensgrid = n_avg * utils.build_density_mesh(pos_ran, box, normalize=True)
    ndens_ran = utils.get_mesh_value(ndensgrid, pos_ran, box)
 
    # Save grid
    ndens_outfile = f"ndens_denssample_mock_{comp_field}_{phase}_{cfg.n_reals}.dat"
    #write_ndens_grid(ndens_outfile, cfg.bgs_clus.zmin, cfg.bgs_clus.zmax, box, ndensgrid)

    #- Compute number density in spherical shells 
    #- and we do not use weights since randoms are uniform over sky
    nzgrid = n_avg * utils.compute_nz(z_ran, zbins, cosmo, frac_sky_avg, normalize=True)  
    nz_ran = nzgrid[utils.safe_digitize(z_ran, zbins, cfg.bgs_clus.nzbin)]



    # ------------------------------------------------------------------
    # Phase 3: write data catalogues
    # ------------------------------------------------------------------
 
    bgs_data = load_observed_data()
    n_avg_data = np.sum(bgs_data['WEIGHT'])

    log.info("=== Processing data catalogues ===")
 

    #for phase in range(cfg.n_phases):
    for real in range(cfg.n_reals):            
        mock_infile_data = cfg.mock_bgs_base_data.format(phase=phase, real=real)
        mock_outfile_data = cfg.mock_bgs_clus_data.format(phase=phase, real=real)

        if os.path.exists(mock_outfile_data) and not cfg.bgs_clus.overwrite:
            log.info("Already exists: %s — skipped", mock_outfile_data)
            continue

        if not os.path.exists(mock_infile_data):
            log.warning("Missing input file: %s — skipped", mock_infile_data)
            continue

        log.info("Reading data catalogue: %s", mock_infile_data)

        with h5py.File(mock_infile_data, "r") as hf:
            ra   = hf["ra"][...]
            dec  = hf["dec"][...]
            zobs = hf["zobs"][...]
            absmag = hf["abs_mag"][...]
            appmag = hf["app_mag"][...]
            comp   = hf[comp_field][...]

        #-- We apply redshift, magnitude cuts as well 
        #-- as a completeness cut, and we DO subsample the data !
        mask = apply_selection(
            zobs, appmag, absmag, 
            zmin=cfg.bgs_clus.zmin, 
            zmax=cfg.bgs_clus.zmax,
            appmaglim=cfg.bgs_clus.appmaglim, 
            absmaglim=cfg.bgs_clus.absmaglim,
            comp_min=cfg.bgs_clus.comp_min,
            comp=comp,
            comp_subsample=True,
            rng=rng,
        )

        #-- Subsample to match data n(z)
        if cfg.bgs_clus.subsample_nz:
            frac = n_avg_data/np.sum(mask)
            print(frac)
            mask &= (rng.uniform(size=len(ra)) < frac)  

        ra, dec, zobs = ra[mask], dec[mask], zobs[mask]
        dist = cosmo.comoving_distance(zobs).value
        comp = comp[mask]
        weight_comp = 1/comp
        weight_comp *= weight_comp.size/np.sum(weight_comp) 
        weight_comp[comp == 0] = 0.
        log.info(f"  {len(ra)} galaxies pass selection")

        pos = utils.radec_to_xyz(ra, dec, dist)
        ndens_dat = utils.get_mesh_value(ndensgrid, pos, box)
        mask = (ndens_dat > 0)
        ra, dec, zobs, weight_comp, ndens_dat, comp = (ra[mask], dec[mask], zobs[mask],
            weight_comp[mask], ndens_dat[mask], comp[mask]
        )
        log.info(f"  {len(ra)} galaxies after removing zero-density objects")


        #- Compute number density in spherical shells 
        #- and we DO use weights since data are NOT uniform over sky
        nzgrid = utils.compute_nz(zobs, zbins, cosmo, frac_sky_avg, weights=weight_comp, normalize=False)  
        nz_dat = nzgrid[utils.safe_digitize(zobs, zbins, cfg.bgs_clus.nzbin)]


        hdu = make_fits_table(
            RA=ra, DEC=dec, Z=zobs,
            COMP=comp,
            WEIGHT=np.ones_like(ra),
            NDENS=ndens_dat,
            NZ=nz_dat
        )
        hdu.writeto(mock_outfile_data, overwrite=True)
        log.info(f"Written: {mock_outfile_data}")


    # ------------------------------------------------------------------
    # Phase 2: write random catalogue
    # ------------------------------------------------------------------
 
    log.info("=== Writing random catalogue ===")
    hdu = make_fits_table(
        RA=ra_ran, DEC=dec_ran, Z=z_ran,
        COMP=comp_ran,
        WEIGHT=comp_ran,
        NDENS=ndens_ran*n_avg_data/n_avg,
        NZ=nz_ran*n_avg_data/n_avg,
    )
    rand_file = cfg.mock_bgs_clus_rand.format(phase=phase)
    hdu.writeto(rand_file, overwrite=True)
    log.info(f"Written: {rand_file}")
    log.info("=== Done ===")



    log.info("=== Updating permissions ===")
    result = subprocess.run(
        ["chgrp", "-R", "desi", cfg.mock_bgs_clus_dir],
        check=True,
    )
 
    log.info("=== Done ===")
 
 
if __name__ == "__main__":
    main()