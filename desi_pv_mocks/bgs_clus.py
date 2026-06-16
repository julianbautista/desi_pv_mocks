"""
Build BGS PV clustering catalogues from AbacusSummit mocks.
 
Processes random and data mock catalogues: applies redshift/magnitude/
completeness cuts, builds a 3D number density grid from randoms, samples
the grid at galaxy positions, and writes FITS output catalogues.
"""

import subprocess
import os
import logging
import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM, Planck15
from astropy.io import fits
 

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from mock_config import CONFIG, BGS_CLUS

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
 
def apply_selection(redshift, app_mag, abs_mag, completeness, *,
                    zmin, zmax, appmaglim, absmaglim, rng=None):
    """Return boolean mask applying redshift, magnitude, and completeness cuts."""
    if rng is None:
        rng = np.random.default_rng()
    return (
        (redshift > zmin) & (redshift < zmax)
        & (app_mag < appmaglim)
        & (abs_mag < absmaglim)
        & (rng.uniform(size=len(redshift)) < completeness)
    )
 
 
def radec_z_to_xyz(ra_deg, dec_deg, redshift, cosmo):
    """Convert (RA, Dec, z) to comoving Cartesian coordinates (Mpc/h)."""
    dist = cosmo.comoving_distance(redshift).value
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    x = dist * np.cos(dec_rad) * np.cos(ra_rad)
    y = dist * np.cos(dec_rad) * np.sin(ra_rad)
    z = dist * np.sin(dec_rad)
    return x, y, z
 
 
def build_grid_box(distmax, ngrid):
    """Return grid parameters for a cube enclosing the survey volume."""
    side = 2.0 * distmax
    d = side / ngrid
    origin = distmax          # offset so coordinates are centred at the origin
    lims = np.linspace(0.0, side, ngrid + 1) - origin
    return dict(n=ngrid, side=side, d=d, origin=origin, dvol=d**3, lims=lims)
 
 
def make_fits_table(**columns):
    """Build an astropy BinTableHDU from keyword-argument {name: array} pairs."""
    cols = [fits.Column(name=k, format="D", array=v) for k, v in columns.items()]
    return fits.BinTableHDU.from_columns(cols)
 
 
def safe_digitize(values, edges, n):
    """Return grid indices clipped to [0, n-1] to guard against boundary objects."""
    return np.clip(np.digitize(values, edges) - 1, 0, n - 1)
 
 
def write_ndens_grid(path, zmin, zmax, box, grid):
    """Write the number-density grid to a plain text file."""
    n = box["n"]
    with open(path, "w") as fh:
        fh.write(f"{zmin} {zmax}\n")
        fh.write(
            f"{n} {n} {n} "
            f"{box['side']} {box['side']} {box['side']} "
            f"{box['origin']} {box['origin']} {box['origin']}\n"
        )
        for iz in range(n):
            for iy in range(n):
                for ix in range(n):
                    fh.write(f"{grid[ix, iy, iz]}\n")
    log.info("Number density grid written to %s", path)
 
 
# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
 
def main():

    comp_field = CONFIG.comp_field
    rng = np.random.default_rng()
 
    os.makedirs(CONFIG.mock_bgs_clus_dir, exist_ok=True)
    os.makedirs(CONFIG.mock_bgs_clus_dir+"/data", exist_ok=True)
    os.makedirs(CONFIG.mock_bgs_clus_dir+"/rand", exist_ok=True)

    cosmo = FlatLambdaCDM(H0=100, Om0=0.3151)
    distmax = cosmo.comoving_distance(BGS_CLUS.zmax).value
    box = build_grid_box(distmax, BGS_CLUS.ngrid)
    n = box["n"]
 
    # ------------------------------------------------------------------
    # Phase 1: build the number density grid from all random catalogues
    # ------------------------------------------------------------------
 
    log.info("=== Building number density grid from randoms ===")
 
    ra_ran, dec_ran, z_ran = [], [], []
    counts_ran = []

    for phase in range(CONFIG.n_phases_rand):
        for real in range(CONFIG.n_reals):
            mock_infile_rand = CONFIG.mock_bgs_base_rand.format(phase=phase, real=real)
            log.info("Reading random catalogue: %s", mock_infile_rand)
 
            with h5py.File(mock_infile_rand, "r") as hf:
                ra   = hf["ra"][...]
                dec  = hf["dec"][...]
                zobs = hf["zobs"][...]
                absmag = hf["abs_mag"][...]
                appmag = hf["app_mag"][...]
                comp   = hf[comp_field][...]
 
            mask = apply_selection(
                zobs, appmag, absmag, comp,
                zmin=BGS_CLUS.zmin, 
                zmax=BGS_CLUS.zmax,
                appmaglim=BGS_CLUS.appmaglim, 
                absmaglim=BGS_CLUS.absmaglim,
                rng=rng,
            )
            n_sel = mask.sum()
            log.info(f" {n_sel} / {len(ra)} randoms pass selection (z, mag, completeness)" )
            counts_ran.append(n_sel)
            ra_ran.append(ra[mask])
            dec_ran.append(dec[mask])
            z_ran.append(zobs[mask])
 
    ra_ran  = np.concatenate(ra_ran)
    dec_ran = np.concatenate(dec_ran)
    z_ran   = np.concatenate(z_ran)
    w_ran   = np.ones(len(ra_ran))
    log.info(f"Total randoms after selection: {len(ra_ran)}")
 
    x_ran, y_ran, z_ran_cart = radec_z_to_xyz(ra_ran, dec_ran, z_ran, cosmo)
 
    # Histogram into grid
    o = box["origin"]
    s = box["side"]
    wingrid, _ = np.histogramdd(
        np.c_[x_ran + o, y_ran + o, z_ran_cart + o],
        bins=(n, n, n),
        range=((0.0, s), (0.0, s), (0.0, s)),
    )
    n_avg = np.mean(counts_ran)
    log.info(f"Mean randoms per realisation/phase: {n_avg:.1f}")
    ndensgrid = (n_avg / box["dvol"]) * (wingrid / wingrid.sum())
 
    # Save grid
    ndens_outfile = f"ndens_denssample_mock_{comp_field}_{CONFIG.n_phases_rand*CONFIG.n_reals}.dat"
    write_ndens_grid(ndens_outfile, BGS_CLUS.zmin, BGS_CLUS.zmax, box, ndensgrid)
 
    # Sample grid at random positions (for the output random catalogue)
    ix = safe_digitize(x_ran, box["lims"], n)
    iy = safe_digitize(y_ran, box["lims"], n)
    iz = safe_digitize(z_ran_cart, box["lims"], n)
    ndens_ran = ndensgrid[ix, iy, iz]
 
    # ------------------------------------------------------------------
    # Phase 2: write random catalogue
    # ------------------------------------------------------------------
 
    log.info("=== Writing random catalogue ===")
    hdu = make_fits_table(
        RA=ra_ran, DEC=dec_ran, Z=z_ran,
        WEIGHT=w_ran,
        NDENS=ndens_ran,
    )
    hdu.writeto(CONFIG.mock_bgs_clus_rand, overwrite=True)
    log.info(f"Written: {CONFIG.mock_bgs_clus_rand}")
    log.info("=== Done ===")

    # ------------------------------------------------------------------
    # Phase 3: write data catalogues
    # ------------------------------------------------------------------
 
    log.info("=== Processing data catalogues ===")
 

    for phase in range(CONFIG.n_phases):
        for real in range(CONFIG.n_reals):            
            mock_infile_data = CONFIG.mock_bgs_base_data.format(phase=phase, real=real)
            mock_outfile_data = CONFIG.mock_bgs_clus_data.format(phase=phase, real=real)

            if os.path.exists(mock_outfile_data) and not BGS_CLUS.overwrite:
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
 
            mask = apply_selection(
                zobs, appmag, absmag, comp,
                zmin=BGS_CLUS.zmin, 
                zmax=BGS_CLUS.zmax,
                appmaglim=BGS_CLUS.appmaglim, 
                absmaglim=BGS_CLUS.absmaglim,
                rng=rng,
            )
            ra, dec, zobs = ra[mask], dec[mask], zobs[mask]
            log.info(f"  {len(ra)} galaxies pass selection")
 
            x_dat, y_dat, z_dat_cart = radec_z_to_xyz(ra, dec, zobs, cosmo)
 
            ix = safe_digitize(x_dat, box["lims"], n)
            iy = safe_digitize(y_dat, box["lims"], n)
            iz = safe_digitize(z_dat_cart, box["lims"], n)
            ndens_dat = ndensgrid[ix, iy, iz]
            
            hdu = make_fits_table(
                RA=ra, DEC=dec, Z=zobs,
                WEIGHT=np.ones(len(ra)),
                NDENS=ndens_dat,
            )
            hdu.writeto(mock_outfile_data, overwrite=True)
            log.info(f"Written: {mock_outfile_data}")

    log.info("=== Updating permissions ===")
    result = subprocess.run(
        ["chgrp", "-R", "desi", CONFIG.mock_bgs_clus_dir],
        check=True,
    )
    print(result)
 
    log.info("=== Done ===")
 
 
if __name__ == "__main__":
    main()