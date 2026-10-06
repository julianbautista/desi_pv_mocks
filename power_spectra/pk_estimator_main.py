#!/usr/bin/env python3
"""
Measure the density, momentum and density-momentum cross power spectra of the
DESI peculiar-velocity Tully-Fisher (TF) mocks, using `pk_estimator`.

All settings are read from a YAML configuration file.

Usage:
    python measure_ps.py --config config.yaml
    python measure_ps.py -c config.yaml --mocks 0 1 2
"""

import argparse
import os
import sys
import logging

import numpy as np
import yaml  # pip install pyyaml
from astropy.io import fits

from desi_pv_mocks.pk_estimator import estimate_power_spectrum

REQUIRED_SECTIONS = ('cosmology', 'power_spectrum', 'grid', 'observer', 'paths', 'ps_types')


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

###############################################################################
######                                                                  ######
######   Sec 1. Configuration and catalogue input                       ######
######                                                                  ######
###############################################################################

def load_config(path):
    """Read the YAML configuration file at `path` and check its sections."""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    missing = [name for name in REQUIRED_SECTIONS if name not in config]
    if missing:
        sys.exit('\n Error: missing config section(s): ' + ', '.join(missing) + '\n')
    return config


def zero_pad(number):
    """Format `number` as a zero-padded, 3-character string (e.g. 7 -> '007')."""
    return '{:0>3}'.format(str(number))


def read_catalogue(file_name):
    """Return the data table stored in the first extension of a FITS file."""
    with fits.open(file_name) as hdul:
        return hdul[1].data


###############################################################################
######                                                                  ######
######   Sec 2. Power spectrum measurement                              ######
######                                                                  ######
###############################################################################

def measure_power_spectra(config):
    """Measure the requested power spectra for each mock in `mock_indices`."""
    # -- 1. Settings from the configuration ----------------------------------
    cosmo = config['cosmology']
    ps_cfg = config['power_spectrum']
    grid = config['grid']
    paths = config['paths']

    omega_m = cosmo['omega_m']
    omega_lambda = 1. - omega_m
    hubble_constant = cosmo['hubble_constant']
    sigv = cosmo['sigv']

    fkp = ps_cfg['fkp_d']
    fkp_v = ps_cfg['fkp_v']
    ps_multi = ps_cfg['multipoles']
    ps_types = config['ps_types']

    bulk_vel = np.array(config['observer']['bulk_vel'], dtype=float)
    output_dir = paths['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Grid and k-bin settings, common to every call of the estimator.
    grid_args = dict(kmin=grid['kmin'], kmax=grid['kmax'], nk=grid['nk'],
                     nx=grid['nx'], ny=grid['ny'], nz=grid['nz'],
                     xmin=grid['xmin'], xmax=grid['xmax'],
                     ymin=grid['ymin'], ymax=grid['ymax'],
                     zmin=grid['zmin'], zmax=grid['zmax'])
    # Cosmology settings, common to every call of the estimator.
    cosmo_args = dict(omega_m=omega_m, omega_lambda=omega_lambda,
                      hubble_constant=hubble_constant)

    # -- 2. Random catalogue (a single example random catalogue) -------------
    random_file = paths['random_file'].format(phase=config['phase'])
    log.info(f"Reading random catalogue from {random_file}")
    data_r = read_catalogue(random_file)
    longir, latir, rsfr = data_r['RA'], data_r['DEC'], data_r['Z']
    nbr_norm = data_r['NPV']

    # -- 3. Read mock galaxies -----------------------
    mock_file = paths['mock_file'].format(phase=config['phase'],
                                          real=config['real'])
    log.info(f"Reading mock catalogue from {mock_file}")
    data = read_catalogue(mock_file)
    longi, lati, rsf, nb = data['RA'], data['DEC'], data['Z'], data['NPV']
    vp, evp = data['PV'], data['PV_ERR']

    # density-PS
    if 'den' in ps_types:
        den_file=os.path.join(output_dir, f'TFMock_den_ph{config["phase"]:03d}_r{config["real"]:03d}.txt')
        log.info(f"Estimating density power spectrum, output to {den_file}") 
        estimate_power_spectrum(
            **grid_args, **cosmo_args,
            longi=longi, lati=lati, rsf=rsf, nb=nb, fkp=fkp,
            longir=longir, latir=latir, rsfr=rsfr, nbr_norm=nbr_norm,
            ps_type='den', ps_multi=ps_multi,
            file_dir=den_file)

    # momentum-PS
    if 'mom' in ps_types:
        mom_file = os.path.join(output_dir, f'TFMock_mom_ph{config["phase"]:03d}_r{config["real"]:03d}.txt')
        log.info(f"Estimating momentum power spectrum, output to {mom_file}") 
        estimate_power_spectrum(
            **grid_args, **cosmo_args,
            longiv=longi, lativ=lati, rsfv=rsf, vp=vp, evp=evp, nbmom=nb,
            fkp=fkp, fkp_v=fkp_v, sigv=sigv, bulk_vel=bulk_vel,
            ps_type='mom', ps_multi=ps_multi,
            file_dir=mom_file)

    # cross-PS
    if 'crs' in ps_types:
        crs_file=os.path.join(output_dir, f'TFMock_crs_ph{config["phase"]:03d}_r{config["real"]:03d}.txt')
        log.info(f"Estimating cross power spectrum, output to {crs_file}")  
        estimate_power_spectrum(
            **grid_args, **cosmo_args,
            longi=longi, lati=lati, rsf=rsf, nb=nb, fkp=fkp,
            longir=longir, latir=latir, rsfr=rsfr, nbr_norm=nbr_norm,
            longiv=longi, lativ=lati, rsfv=rsf, vp=vp, evp=evp, nbmom=nb,
            fkp_v=fkp_v, sigv=sigv, bulk_vel=bulk_vel,
            ps_type='crs', ps_multi=ps_multi,
            file_dir=crs_file)


def main():
    parser = argparse.ArgumentParser(
        description='Measure power spectra of the TF mocks using a YAML config file.')
    parser.add_argument('-c', '--config', required=True, help='Path to the YAML config file')
    parser.add_argument('--phase', type=int, required=True,
                        help='phase of mock between 0 and 24')
    parser.add_argument('--real', type=int, required=True,
                            help='phase of mock between 0 and 26')
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        config['phase'] = args.phase
        config['real'] = args.real
    except (OSError, yaml.YAMLError) as e:
        sys.exit('\n Error loading config: ' + str(e) + '\n')

    measure_power_spectra(config)
    log.info("Done !")

if __name__ == '__main__':
    main()
