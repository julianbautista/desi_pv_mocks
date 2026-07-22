"""
Cross-match BGS AbacusSummit mocks with DESI spectroscopic catalogue.
 
Adds DR9 photometric and FastSpecFit properties to each mock galaxy by
finding its nearest neighbour in (z, M_r, g-r) space using a KD-tree
built from the spectroscopic sample.
"""
import logging
import argparse
import os
import h5py
import subprocess
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import KDTree
from astropy.cosmology import Planck15
from k_correction import GAMA_KCorrection
 

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

 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_clean_spec(k_r):
    """Load the spectro catalogue and apply quality/photometric cuts."""
    log.info(f"Loading spec catalogue from {cfg.spec_csv}")
    spec = pd.read_csv(cfg.spec_csv, usecols=cfg.spec_keys)
    log.info("  %d rows before cuts", len(spec))
 
    #spec = spec[spec["deltachi2"] >= cfg.bgs_spec.deltachi2_min]
    spec = spec[spec["zwarn"] == 0]
    spec = spec[spec["z"] <= cfg.bgs_spec.zmax]
    spec = spec[(spec["flux_g"] > 0) & (spec["flux_r"] > 0) & (spec["flux_z"] > 0)]
    spec["col"] = spec["mag_g"] - spec["mag_r"]
    spec = spec[(spec["col"] >= cfg.bgs_spec.col_min) & (spec["col"] <= cfg.bgs_spec.col_max)]
    spec["abs_mag_r"] = k_r.absolute_magnitude(spec["mag_r"], spec["z"], spec["col"])
    spec = spec[spec["mag_r"] <= cfg.bgs_spec.r_mag_lim]
 
    log.info("  %d rows after cuts", len(spec))
    
    return spec.reset_index(drop=True)
 
 
def build_kdtree(spec):
    """Build a normalised KD-tree in (z, M_r, g-r) space."""
    z   = spec["z"].to_numpy()
    mag = spec["abs_mag_r"].to_numpy()
    col = spec["col"].to_numpy()
 
    z_min, z_rng   = z.min(),   z.max()   - z.min()
    mag_min, mag_rng = mag.min(), mag.max() - mag.min()
    col_min, col_rng = col.min(), col.max() - col.min()
 
    norms = dict(z_min=z_min, z_rng=z_rng,
                 mag_min=mag_min, mag_rng=mag_rng,
                 col_min=col_min, col_rng=col_rng)
 
    data = np.c_[
        (z   - z_min)   / z_rng,
        (mag - mag_min) / mag_rng,
        (col - col_min) / col_rng,
    ]
    return KDTree(data), norms
 
 
def read_mock_hdf5(fpath):
    """Load an HDF5 mock file into a dict, decoding byte-string fields."""
    mock = {}
    with h5py.File(fpath, "r") as hf:
        for key in hf.keys():
            if key == "vel":
                mock["vx"] = hf["vel"][:, 0]
                mock["vy"] = hf["vel"][:, 1]
                mock["vz"] = hf["vel"][:, 2]
            else:
                mock[key] = hf[key][()]
        # Decode byte strings (survey, program …)
        for key in ("survey", "program"):
            if key in mock and mock[key].dtype.kind in ("S", "O"):
                mock[key] = mock[key].astype("U")
    return mock
 
 
def crossmatch_mock(mock_df, tree, norms, spec):
    """Query the KD-tree and return matched Iron rows for each mock galaxy."""
    z   = mock_df["zobs"].to_numpy()
    mag = mock_df["abs_mag"].to_numpy()
    col = mock_df["col_obs"].to_numpy()
 
    query = np.c_[
        (z   - norms["z_min"])   / norms["z_rng"],
        (mag - norms["mag_min"]) / norms["mag_rng"],
        (col - norms["col_min"]) / norms["col_rng"],
    ]
    _, neighbours = tree.query(query)
    return spec.iloc[neighbours].reset_index(drop=True)
 
 
# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Return a sub-range of *cmap* as a new colormap."""
    return mcolors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n)),
    )
 
 
def _contour_levels(counts, fractions=(0.997, 0.985, 0.95, 0.875, 0.70, 0.40, 0.10, 0.02)):
    t = np.linspace(0, counts.max(), 1000)
    integral = ((counts >= t[:, None, None]) * counts).sum(axis=(1, 2))
    f = sp.interpolate.interp1d(integral, t)
    return f(np.array(fractions))
 
 
def _axis_style(ax):
    ax.tick_params(width=1.3)
    ax.tick_params("both", length=10, which="major")
    ax.tick_params("both", length=5,  which="minor")
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
    for tick in (*ax.xaxis.get_ticklabels(), *ax.yaxis.get_ticklabels()):
        tick.set_fontsize(12)
 
 
def plot_color_magnitude(spec, mock_counts, magbins, colbins, outpath):
    """Colour–magnitude diagram: Iron hexbin + mock contours."""
    contours = _contour_levels(mock_counts)
    n_lev = len(contours)
    cmap_gray = plt.get_cmap("gray_r")
    level_colors = [cmap_gray(i * (0.8 / (n_lev - 1)) + 0.2) for i in range(n_lev)]
 
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hexbin(
        spec["abs_mag_r"], spec["col"],
        mincnt=50, gridsize=50,
        cmap=truncate_colormap(plt.get_cmap("viridis"), 0.0, 0.95),
        reduce_C_function=np.sum,
    )
    ax.contour(
        mock_counts.T, levels=contours, colors=level_colors,
        extent=[magbins.min(), magbins.max(), colbins.min(), colbins.max()],
        linewidths=2, alpha=0.9,
    )
    ax.set_xlabel(r"$M_{r}$", fontsize=14)
    ax.set_ylabel(r"$g-r$", fontsize=14, labelpad=0)
    ax.set_xlim(-24.0, -12.0)
    ax.set_ylim(-0.25, 1.3)
    _axis_style(ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    log.info("Saved %s", outpath)
    plt.close(fig)
 
 
def plot_mass_ssfr(spec, mock_counts, massbins, ssfrbins, outpath):
    """Stellar-mass vs. sSFR diagram: Iron hexbin + mock contours."""
    contours = _contour_levels(mock_counts)
    n_lev = len(contours)
    cmap_gray = plt.get_cmap("gray_r")
    level_colors = [cmap_gray(i * (0.8 / (n_lev - 1)) + 0.2) for i in range(n_lev)]
 
    log_ssfr_spec = np.log10(spec["sfr"]) - spec["logmstar"]
 
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.hexbin(
        spec["logmstar"], log_ssfr_spec,
        bins="log", mincnt=10, gridsize=80,
        cmap=truncate_colormap(plt.get_cmap("viridis"), 0.0, 0.95),
        reduce_C_function=np.sum,
    )
    ax.contour(
        mock_counts.T, levels=contours, colors=level_colors,
        extent=[massbins.min(), massbins.max(), ssfrbins.min(), ssfrbins.max()],
        linewidths=2, alpha=0.9,
    )
    ax.set_xlabel(r"$\log(M_{*}/M_{\odot})$", fontsize=14)
    ax.set_ylabel(r"$\log(\mathrm{sSFR}/\mathrm{yr})$", fontsize=14, labelpad=0)
    ax.set_xlim(6.0, 12.0)
    ax.set_ylim(-14.0, -8.0)
    _axis_style(ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    log.info("Saved %s", outpath)
    plt.close(fig)
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline BGS spectro properties "
    )
    parser.add_argument("config_file", type=str, help="Configuration file path (yaml format)")
    parser.add_argument("phase", type=int, help="Phase (0–24)")
    args = parser.parse_args()
    if not (0 <= args.phase <= 24):
        parser.error("phase should be between 0 and 24")
    return args

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
 
def main():
    args = parse_args()
    global cfg 
    cfg = load_config(args.config_file)
    phase = args.phase 

    k_r = GAMA_KCorrection(Planck15, cfg.kcorr_file)
    spec = load_and_clean_spec(k_r)
    tree, norms = build_kdtree(spec)
 
    # ------------------------------------------------------------------
    # Phase 1: write cross-matched mock files
    # ------------------------------------------------------------------
 
    log.info(f"=== Cross-matching mocks for phase {phase:03d} ===")
    os.makedirs(cfg.mock_bgs_spec_dir, exist_ok=True)
 
    #for phase in range(cfg.n_phases):
    for real in range(cfg.n_reals):
        mock_infile  = cfg.mock_bgs_base_data.format(phase=phase, real=real) 
        mock_outfile = cfg.mock_bgs_spec_data.format(phase=phase, real=real)

        if os.path.exists(mock_outfile) and not cfg.bgs_spec.overwrite:
            log.info("Already exists: %s — skipped", mock_outfile)
            continue

        if not os.path.exists(mock_infile):
            log.warning("Missing input file: %s — skipped", mock_infile)
            continue

        mock_dict = read_mock_hdf5(mock_infile)
        mock_df   = pd.DataFrame(mock_dict)
        matched   = crossmatch_mock(mock_df, tree, norms, spec)

        with h5py.File(mock_outfile, "w") as out:
            # Copy original mock datasets
            with h5py.File(mock_infile, "r") as src:
                for key in src.keys():
                    src.copy(key, out)
            # Append matched Iron columns
            for key in cfg.match_keys:
                arr = matched[key].to_numpy()
                out[key] = arr.astype("S") if arr.dtype.kind == "U" else arr

        log.info("Written: %s  (%d galaxies)", mock_outfile, len(mock_df))

    log.info("=== Updating permissions ===")
    result = subprocess.run(
        ["chgrp", "-R", "desi", cfg.mock_bgs_spec_dir],
        check=True,
    )


    # ------------------------------------------------------------------
    # Phase 2: accumulate 2D histograms for diagnostic plots
    # ------------------------------------------------------------------
    if cfg.bgs_spec.do_diagnostic_plots:
        os.makedirs(cfg.mock_bgs_spec_plot_dir, exist_ok=True)
 
        log.info("=== Accumulating histograms for diagnostic plots ===")
        magbins  = np.linspace(-24.0, -12.0, 51)
        colbins  = np.linspace(-0.25,   1.3, 51)
        massbins = np.linspace(6.0,    12.0, 31)
        ssfrbins = np.linspace(-14.0,  -8.0, 31)
    
        counts_cm   = np.zeros((50, 50))
        counts_msfr = np.zeros((30, 30))
    
        #for phase in range(1):          # expand range as needed
        for real in range(5):
            mock_outfile = cfg.mock_bgs_spec_data.format(phase=phase, real=real)
            if not os.path.exists(mock_outfile):
                continue

            mock_dict = read_mock_hdf5(mock_outfile)
            mock_df   = pd.DataFrame(mock_dict)
            merged    = mock_df.merge(spec, how="inner", on=["targetid", "survey", "program", "healpix"])
            log.info("  ph%03d r%03d: %d matched galaxies", phase, real, len(merged))

            counts_cm += np.histogram2d(
                merged["abs_mag"], merged["col_obs"], bins=(magbins, colbins)
            )[0]
            #log_ssfr = np.log10(merged["sfr"]) - merged["logmstar"]
            #counts_msfr += np.histogram2d(
            #    merged["logmstar"], log_ssfr, bins=(massbins, ssfrbins)
            #)[0]
    
        counts_cm   /= counts_cm.sum()
        #counts_msfr /= counts_msfr.sum()
 
        # ------------------------------------------------------------------
        # Phase 3: diagnostic plots
        # ------------------------------------------------------------------
        plot_color_magnitude(
            spec, counts_cm, magbins, colbins,
            cfg.mock_bgs_spec_plot_dir +f"/BGS_PV_AbacusSummit_M_vs_gr_ph{phase:03d}.png",
        )
        #plot_mass_ssfr(
        #    spec, counts_msfr, massbins, ssfrbins,
        #    cfg.mock_bgs_spec_plot_dir +"/BGS_PV_AbacusSummit_logM_vs_sSFR.png",
        #)
 
    log.info("=== Done ===")


if __name__ == "__main__":
    main()