#!/usr/bin/env python
# coding: utf-8

# # Generate a Mock Dataset
# 
# Generate a simulated TF dataset as follows:
# 
# * Merge the iron photometric + spectroscopic catalogs with the SGA2020 catalog.
# * Apply all photometric corrections used in the TF analysis.
# * Cross-match with one of the BGS mocks.
# * Apply photometric and morphological cuts used in the PV survey for late-type galaxies.
# * Generate mock rotational velocities using an inverted TFR best fit from the data.
# * Generate mock TFR distances. 

# In[1]:


import os
import shutil
import h5py
import fitsio
import pickle
import healpy as hp
import pandas as pd
import numpy as np
import scipy as sp

from itertools import groupby

from csaps import csaps
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from scipy.stats import binned_statistic
from scipy.odr import ODR, Model, RealData
from scipy.spatial import KDTree

#- Global file path for PV analysis.
#  Set to the NERSC folder /global/cfs/cdirs/desi/science/td/pv by default.
#  Set it to something else if working offline.
pvpath = '/global/cfs/cdirs/desi/science/td/pv'
mockpath = os.path.join(pvpath, 'mocks')
tfmockpath = os.path.join(mockpath, 'TF_mocks')

from corner import corner
from hyperfit.linfit import LinFit
from hyperfit_v2 import MultiLinFit
from line_fits import hyperfit_line_multi

#- Path to TF_mocks: code for Blanton's k-corrections.
import sys
sys.path.append(tfmockpath)
import TF_photoCorrect as tfpc

from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import Planck18, FlatLambdaCDM, units
from astropy.coordinates import SkyCoord, Distance

from tqdm import tqdm
from glob import glob

import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


# In[2]:


def profile_histogram(x, y, xbins, yerr=None, weights=None, median=False, weighted=False):
    """Compute a profile histogram from scattered data.
    
    Parameters
    ----------
    x : list or ndarray
        Ordinates (independent variable).
    y : list or ndarray
        Coordinates (dependent variable).
    xbins : list or ndarray
        Bin edges for the independent variable.
    yerr : list or ndarray
        Uncertainties on the dependent variable. Assumed independent.
    weights : list or ndarray
        If not None (and weighted=True), will use this instead of yerr to weight 
        the summary statistics.
    median : bool
        If true, compute median as central value; else, the (weighted) mean.
    weighted : bool
        Weight the summary statistics, either by the uncertainty in y or the 
        provided weights.
        
    Returns
    -------
    N : ndarray
        Unweighted counts per bin.
    h : ndarray
        Summary statistic (mean or median) of independent variable per bin.
    e : ndarray
        Uncertainty on the summary statistic per bin.
    """
    
    N = binned_statistic(x, y, bins=xbins, statistic='count').statistic

    if weighted:
        if (yerr is None) and (weights is None):
            raise ValueError('need to define either yerr or weights if using weighted fit.')

        if weights is None:
            # weight based on yerr
            w = 1/yerr**2
        else:
            w = weights
        W, H, E = binned_statistic(x, [w, w*y, w*y**2], bins=xbins, statistic='sum').statistic
        h = H/W
        e = 1/np.sqrt(W)
    else:
        mean, mean2 = binned_statistic(x, [y, y**2], bins=xbins, statistic='mean').statistic
        h = mean
        e = np.sqrt((mean2 - mean**2) / (N - 1))

    if median:
        h = binned_statistic(x, y, bins=xbins, statistic='median').statistic
    
    return N, h, e


parser = ArgumentParser(description='TFR mock generation', formatter_class=ArgumentDefaultsHelpFormatter)
#parser.add_argument('mockfile')
parser.add_argument('-m', '--mock', dest='mock', type=int, choices=range(0, 25),
                    help='Mock number')
parser.add_argument('-r', '--realization', dest='realization', type=int, choices=range(0, 27),
                    help='Realization number')
parser.add_argument('-v', '--version', dest='version', default='v0.5.4',
                    help='Mock version number')
args = parser.parse_args()


# ## Iron Data + SGA Catalog
# 
# Follow the procedure used in FP mock generation: read in iron data relevant for the TFR for cross-matching to the BGS mocks, producing a simulated set with realistic galaxy observables.
# 
# Here we merge the fullsweep and iron specprod catalogs to reproduce cuts when cross-matching with the mocks.
# 
# As a final step, get any missing SGA data directly from the SGA 2020 catalog.

# In[3]:


#- Read in the iron fullsweep and specprod catalogs.
sw_keys = ['targetid', 'survey', 'program', 'healpix', 'target_ra', 'target_dec',
           'z', 'zerr', 'zwarn', 'inbasiccuts', 'has_corrupt_phot',
           'mag_g', 'mag_r', 'mag_z',
           'morphtype', 'sersic', 'BA_ratio',
           'circ_radius', 'circ_radius_err', 'uncor_radius', 'SGA_id', 'radius_SB25']

ironsweep = os.path.join(pvpath, 'redshift_data/Y1/iron_fullsweep_catalogue_z012.csv')
iron = pd.read_csv(ironsweep, usecols=sw_keys)

#- Read in the spectroscopic production table generated by Caitlin Ross.
sp_keys = ['targetid', 'survey', 'program', 'healpix',
           'mag_err_g', 'mag_err_r', 'mag_err_z', 
           'deltachi2']

ironspec = os.path.join(pvpath, 'redshift_data/Y1/specprod_iron_healpix_z015.csv')
ironsp = pd.read_csv(ironspec, usecols=sp_keys)

#- Cross-match the catalogs.
iron = pd.merge(iron, ironsp, 
                left_on=['targetid', 'survey', 'program', 'healpix'],
                right_on=['targetid', 'survey', 'program', 'healpix'], how='inner')

#- Object selection from the spectro pipeline:
#  1. Valid SGA ID, which implicitly enforces a size selection.
#  2. Delta-chi2 > 25.
#  3. No redrock warnings.
select = (iron['SGA_id'] > 0) & \
         (iron['deltachi2'] >= 25) & \
         (iron['zwarn'] == 0)

iron = iron.drop(iron[~select].index)

#- Read the SGA catalog and match on SGA_ID.
#  This is needed to access R_26 and other quantities at the mag 26 isophote.
sgafile = '/global/cfs/cdirs/cosmo/data/sga/2020/SGA-2020.fits'
sgacat = Table.read(sgafile, 'ELLIPSE')
sgacat.rename_column('SGA_ID', 'SGA_id')
sgacat.rename_column('RA', 'SGA_ra')
sgacat.rename_column('DEC', 'SGA_dec')
sgacat = sgacat['SGA_ra', 'SGA_dec', 'SGA_id', 'D26', 'G_MAG_SB26', 'G_MAG_SB26_ERR', 'R_MAG_SB26', 'R_MAG_SB26_ERR', 'Z_MAG_SB26', 'Z_MAG_SB26_ERR'].to_pandas()
sgacat = sgacat.drop(sgacat[sgacat['R_MAG_SB26'] < 0].index)

iron = pd.merge(iron, sgacat, how='inner', on=['SGA_id'])


# In[4]:


#- Drop NaN
iron = iron.dropna()
iron


# ### Keep only Galaxy Centers
# 
# The iron catalog may include some off-axis measurements of SGA galaxies that pass the spectroscopic cuts. Remove them with a cone-angle cut, comparing the SGA centers (from Tractor) to the target RA, Dec in DESI. The cut is
# 
# $$
# \frac{\angle(\mathbf{r}_\mathrm{SGA}, \mathbf{r}_\mathrm{fiber})}{D_{26}/2} < 0.1
# $$
# 
# See details in Kelly's [SGA selection notebook for iron](https://github.com/DESI-UR/DESI_SGA/blob/master/TF/Y1/iron_rot_vel.ipynb).
# 
# Note that requiring a nonzero $m_{r,\mathrm{SB_{26}}}$ may remove all spectra not measured on galaxy centers, making this cut redundant.

# In[5]:


coords_sga = SkyCoord(ra=iron['SGA_ra'], dec=iron['SGA_dec'], unit='degree')
coords_iron = SkyCoord(ra=iron['target_ra'], dec=iron['target_dec'], unit='degree')
sep2d = coords_iron.separation(coords_sga)
select = (2*sep2d.to_value('arcmin') / iron['D26']) < 0.1

iron = iron.drop(iron[~select].index)


# In[6]:


iron


# ## Apply Dust and K-corrections
# 
# There are four photometric corrections that need to be applied.
# 
# 1. N vs. S imaging catalog photometric systematics.
# 2. $k$-corrections to $z=0.1$.
# 3. Global Milky Way dust corrections using the maps from Zhou+, 2024.
# 4. Per-galaxy internal dust corrections based on the galaxies' inclination angles.
# 
# The corrections, applied to the $r$-band magnitudes, are summed as
# 
# $$
# A_\mathrm{sys} + A_k + A_\mathrm{MW} + A_\mathrm{dust}
# $$

# ### Imaging Systematics

# In[7]:


#- Apply imaging survey systematics: compute photometric system (N or S)
#  Note that we need the RA,Dec of the *data*, not the mocks, because the corrections
#  are applied to the observed magnitudes from SGA (MAG_R_SB26, etc.).
c = SkyCoord(iron['target_ra'], iron['target_dec'], unit='degree')
isnorth = (c.galactic.b > 0) & (iron['target_dec'] > 32.375)
iron['photsys'] = 'S'
iron.loc[isnorth, 'photsys'] = 'N'

#- Adjust northern photometry to DECaLS
Asys, Asys_err = tfpc.BASS_corr(iron['photsys'])


# ### K Correction

# In[8]:


#- This is based on the kcorrect package by Blanton (https://kcorrect.readthedocs.io/)
#  Note that we need the RA,Dec of the *data*, not the mocks, because the corrections
#  are applied to the observed magnitudes from SGA (MAG_R_SB26, etc.).
select = iron['z'] > 0

kc = tfpc.k_corr(iron['z'][select], 
                [iron['G_MAG_SB26'][select],     iron['R_MAG_SB26'][select],     iron['Z_MAG_SB26'][select]], 
                [iron['G_MAG_SB26_ERR'][select], iron['R_MAG_SB26_ERR'][select], iron['Z_MAG_SB26_ERR'][select]], 
                z_corr=0.1)

Ak = np.zeros((len(iron), 3))
Ak[select] = kc


# ### MW Dust Correction

# In[9]:


#- Compute MW dust corrections
#  Note that we need the RA,Dec of the *data*, not the mocks, because the corrections
#  are applied to the observed magnitudes from SGA (MAG_R_SB26, etc.).
dustmap = '/global/cfs/cdirs/desi/public/papers/mws/desi_dust/y2/v1/maps/desi_dust_gr_512.fits'
ebv = Table.read(dustmap)
Adust, Adust_err = tfpc.MW_dust(iron['target_ra'].values, iron['target_dec'].values, ebv)

#- Mask out NaNs
for i, band in enumerate('grz'):
    isnan_gal = np.isnan(Adust[i])
    if np.any(isnan_gal):
        logging.info(f'Removing NaN for MW dust correction, band {band}')
        Adust[i][isnan_gal] = 0
        Adust_err[i][isnan_gal] = 0


# ### Apply MW, K-Correction, and Imaging Systematics Corrections Prior to Internal Dust Correction

# In[10]:


#- Apply MW dust, k-corrections, and photometric systematic corrections to the data.
for i, band in enumerate('GRZ'):
    iron[f'{band}_MAG_SB26_tmp'] = iron[f'{band}_MAG_SB26'] - Adust[i] + Asys + Ak[:,i]
    iron[f'{band}_MAG_SB26_ERR_tmp'] = np.sqrt(iron[f'{band}_MAG_SB26_ERR']**2 + Adust_err[i]**2 + Asys_err**2)


# ### Internal Dust Correction 
# 
# Correct m_r for the internal galactic dust, assuming that as we look through higher inclinations we're viewing the galaxy through its dust lanes. Details in [this notebook by Kelly](https://github.com/DESI-UR/DESI_SGA/blob/master/TF/Y1/TF_iron_internal-dustCorr.ipynb).

# In[11]:


#- Kelly applies an empirical fit to the internal dust in each galaxy. Steps:
#   1. Apply "known" corrections (k-correction, MW dust).
#   2. Fit m_r_corr (corrected) vs spiral b/a
#   3. Zero out this linear dependence.

#- Set up a binned data set and perform the fit
ba_bins = np.arange(0.1,1,0.1)
ba = 0.5*(ba_bins[1:] + ba_bins[:-1])
ba_err = 0.5*np.diff(ba_bins)
m_r_median = np.median(iron['R_MAG_SB26_tmp'])
m_r, _, _ = binned_statistic(iron['BA_ratio'], iron['R_MAG_SB26_tmp'], statistic='median', bins=ba_bins)
n_bin, _, _ = binned_statistic(iron['BA_ratio'], iron['R_MAG_SB26_tmp'], statistic='count', bins=ba_bins)
m_r_err, _, _ = binned_statistic(iron['BA_ratio'], iron['R_MAG_SB26_tmp'], statistic='std', bins=ba_bins)
m_r_err /= np.sqrt(n_bin)

linear_fit = lambda coeff, x: coeff[0]*x + coeff[1]
model = Model(linear_fit)
data = RealData(ba, m_r - m_r_median, sx=ba_err, sy=m_r_err)
odr = ODR(data, model, beta0=[1, 1])

result = odr.run()
coeff = result.beta
coeff_err = result.sd_beta
print('Best fit:    ', coeff)
print('uncertainty: ', coeff_err)

#fig, axes = plt.subplots(1,2, figsize=(10,4.5), tight_layout=True, sharex=True)#, sharey=True)
#ax = axes[0]
#ax.scatter(iron['BA_ratio'], iron['R_MAG_SB26_tmp'], alpha=0.01)
#ax.errorbar(ba, m_r, xerr=ba_err, yerr=m_r_err, fmt='o', color='tab:orange')
#ax.plot(ba_bins, coeff[0]*ba_bins + coeff[1] + m_r_median, color='tab:orange')
#ax.set(ylim=(19, 11),
#       ylabel='$m_{r}$',
#       xlabel='$b/a$');

#- Compute the internal dust correction
A_int, A_int_err = tfpc.internal_dust(iron['BA_ratio'].values, coeff, coeff_err)

#ax = axes[1]
#ax.scatter(iron['BA_ratio'], iron['R_MAG_SB26_tmp'] - A_int, alpha=0.01)
#m_r, _, _ = binned_statistic(iron['BA_ratio'], iron['R_MAG_SB26_tmp'] - A_int, statistic='median', bins=ba_bins)
#ax.errorbar(ba, m_r, xerr=ba_err, yerr=m_r_err, fmt='o', color='tab:orange')
#ax.set(ylim=(19, 11),
#       ylabel='$m_{r,\mathrm{corr}}$',
#       xlabel='$b/a$')

# fig.savefig('tfr_mock_internal_dust_correction.png', dpi=150);

#- Update the r-band magnitudes
iron['R_MAG_SB26_CORR'] = iron['R_MAG_SB26_tmp'] - A_int
iron['R_MAG_SB26_ERR_CORR'] = np.sqrt(iron['R_MAG_SB26_ERR_tmp']**2 + A_int_err**2)


# ## BGS Mock Catalog
# 
# Read in one of the mock catalog files and cross-match to iron.

# In[12]:

mockfile = f'/global/cfs/cdirs/desi/science/td/pv/mocks/BGS_base/v0.5/iron/BGS_PV_AbacusSummit_base_c000_ph{args.mock:03d}_r{args.realization:03d}_z0.11.dat.hdf5'

print(f'Picked mock file {os.path.basename(mockfile)}.')


# In[13]:


#- Fill the catalog as a dictionary and convert to a Pandas table.
#  Here we follow the I/O from the FP generator (thanks Cullan).
mock = {}
with h5py.File(mockfile, 'r') as f:
    for key in f.keys():
        if key == 'vel':
            #- Pack the galaxy velocities into labeled vx, vy, vz
            mock['vx'] = f['vel'][:,0]
            mock['vy'] = f['vel'][:,1]
            mock['vz'] = f['vel'][:,2]
        else:
            mock[key] = f[key][()]

        # convert strings to unicode
        if key == 'survey' or key == 'program':
            mock[key] = mock[key].astype('U')

    #- Convert to a pandas table
    mock = pd.DataFrame.from_dict(mock)

    #- Merge with iron on 4 keywords
    mock = mock.merge(iron, how='inner', on=['targetid', 'survey', 'program', 'healpix'])


# In[14]:


mock.keys()


# In[15]:


mock


# ### Apply TF Selection Cuts
# 
# Here apply the late-type galaxy cuts defined in *Target Selection for the DESI Peculiar Velocity Survey*, C. Saulder+, MNRAS 525:1106, 2023. Note that several cuts are the complement of the early-type cuts for the FP sample.

# In[16]:


mock_selection = {
    'basic cuts' : 0,
    # '0.03 < z < 0.105' : 0,
    'b/a < cos(25°)' : 0,
    'morphtype' : 0,
    'NaN' : 0
}

#- Apply target selection
print(f'Size of cross-matched iron+mock catalog ..{len(mock):.>20d}')

#- Drop data that doesn't pass the photometric cuts
select = (mock['inbasiccuts'] == 0) | (mock['has_corrupt_phot'] == 1)
mock = mock.drop(mock[select].index)
mock_selection['basic cuts'] = len(mock)
print(f'Size after photometric cuts .........{len(mock):.>20d}')

#- Redshift range cut: remove?
# select = (mock['zobs'] > 0.03) & (mock['zobs'] <= 0.105)
# mock = mock[select]
# mock_selection['0.03 < z < 0.105'] = len(mock)
# print(f'Redshift selection: 0.03 < z < 0.105 {len(mock):.>20d}')

#- B/A ratio cut:
select = mock['BA_ratio'] < np.cos(np.radians(25))
mock = mock[select]
mock_selection['b/a < cos(25°)'] = len(mock)
print(f'Ratio b/a < cos(25 deg) .............{len(mock):.>20d}')

#- Morphology cuts:
select = (mock['morphtype'] == 'EXP') | ((mock['morphtype'] == 'SER') & (mock['sersic'] <= 2))
mock = mock[select]
mock_selection['morphtype'] = len(mock)
print(f'Morphology cuts: ....................{len(mock):.>20d}')

#- Drop any rows with NaN
mock = mock.dropna()
mock_selection['NaN'] = len(mock)
print(f'Drop NaN ............................{len(mock):.>20d}')


# In[17]:


names = list(mock_selection.keys())
values = list(mock_selection.values())

#fig, ax = plt.subplots(1, 1, figsize=(5,6), tight_layout=True)
#bars = ax.bar(names, values)
#ax.set_xticklabels(names, rotation=45, ha='right')
## ax.bar_label(bars, fmt='%d')
#ax.set(ylabel='count');#, yscale='log', ylim=[1e3,1.2e4])
#fig.set_facecolor('none')
## fig.savefig('tfr_mock_cuts.png', dpi=150)


# ## Compute TFR Quantities
# 
# Assign a rotational velocity using the data. Then use this to infer $M_r$ using the calibrated TFR.

# ### Set up the Cosmology
# 
# Use a flat-$\Lambda$CDM fiducial cosmology with $H_0\equiv100$ km/s/Mpc and $\Omega_m=0.3151$.

# In[18]:


h = 1
cosmology = FlatLambdaCDM(H0=100*h, Om0=0.3151)


# ### Current Y1 TFR Best Fit
# 
# TF Y1 best-fit parameters and covariances, corresponding to v9 of the TF Y1 catalog. This comes from the TFR calibration using 7 galaxy clusters (2025-03-28) using Vmax weights to account for the galaxy size function. The TFR fit is
# 
# $$
# M_r = a \log_{10}{\left(\frac{V_\mathrm{rot}}{V_0}\right)} + b_{0\mathrm{pt}}
# $$
# 
# with intrinsic scatter $\sigma$ along the magnitude axis. See this [notebook](https://github.com/DESI-UR/DESI_SGA/blob/master/TF/Y1/TF_Y1_cluster_calibration_AnthonyUpdates_weightsVmax-1_KAD.ipynb) in the [DESI_SGA/TF/Y1](https://github.com/DESI-UR/DESI_SGA/tree/master/TF/Y1) GitHub repo.
# 
# The parameter vector includes the TFR slope $a$, global zero point $b_{0\text{pt}}$, calibration cluster intercepts $\{b_i\}$, and intrinsic scatter $\sigma$.

# In[19]:


with open('cov_ab_iron_jointTFR_varyV0-dwarfsAlex_z0p1_zbins0p005_weightsVmax-1_dVsys_KAD-20250810.pickle', 'rb') as tfr_file:
    cov_ab, tfr_samples, logV0, zmin, zmax, dz, zbins = pickle.load(tfr_file)

tf_par = np.median(tfr_samples, axis=1)
a, b, sigma = tf_par[0], tf_par[1:-1], tf_par[-1]


# In[20]:


# with open('cov_ab_iron_jointTFR_varyV0-perpdwarfs0_z0p1_binaryMLupdated_Anthony_weightsVmax-1_dVsys_KAD-20250523.pickle', 'rb') as tfr_file:
#     cov_ab, tfr_samples, logV0 = pickle.load(tfr_file)

# # Extract all best-fit parameters, including individual cluster intercepts.
# a, b0pt, b1, b2, b3, b4, b5, b6, b7, sigma = [np.median(tfr_samples[i]) for i in range(10)]

# # Store TF best-fit parameters.
# tf_par = np.asarray([a, b0pt, sigma])

# # Store the covariance of the intercept, slope, and intrinsic scatter.
# mask = np.zeros_like(cov_ab, dtype=bool)
# mask[:2, :2] = mask[:2, 9] = mask[9, :2] = mask[9,9] = True
# tf_cov = cov_ab[mask].reshape(3,3)


# In[21]:


# with open('cov_ab_iron_jointTFR_varyV0-dwarfsAlex_z0p1_zbins0p005_weightsVmax-1_dVsys_KAD-20250810.pickle', 'rb') as tfr_file:
#     cov_ab, tfr_samples, logV0, zmin, zmax, dz, zbins = pickle.load(tfr_file)

# # Center redshift values of each bin
# zc = 0.5*dz + zbins[:-1]

# # Distance modulus for each redshift bin center
# mu_zc = cosmo.distmod(zc)

# # Each redshift bin has its own 0pt
# # To put it in absolute-magnitude space, we'll convert it to an absolute magnitude using the middle of the redshift bin
# ZP = np.median(tfr_samples[1:-1], axis=1) - mu_zc.value
# ZP_err = np.sqrt(np.diagonal(cov_ab[1:-1,1:-1])) # Should include z-bin width to this uncertainty

# # First, match each galaxy to its redshift bin
# zbin_indices = np.digitize(SGA_TF['Z_DESI_CMB'], zbins, right=True)

# # For those galaxies that fall outside the calibration range, assign them to the closest bin
# zbin_indices[zbin_indices == 0] = 1
# zbin_indices[zbin_indices == len(zbins)] = len(zbins) - 1

# # Then, use that galaxy's redshift bin's zero-point to calculate the distance modulus
# SGA_TF['R_ABSMAG_SB26_TF'] = np.nan
# for i in range(len(SGA_TF)):
#     SGA_TF['R_ABSMAG_SB26_TF'][i] = slope*(np.log10(SGA_TF['V_0p4R26'][i]) - logV0) + ZP[zbin_indices[i] - 1]


# ### Current TFR Catalog
# 
# Read in the TFR catalog to sample uncertainties in $V_\mathrm{rot}$.

# In[22]:


tfr_version = 'v13'

tfrcatfile = os.path.join(pvpath, f'tfgalaxies/Y1/DESI-DR1_TF_pv_cat_{tfr_version}.fits')
tfrcat = Table.read(tfrcatfile)

#- Set minimum velocity uncertainty to 7 km/s.
# lowverr = tfrcat['V_0p4R26_ERR'] < 7.
# tfrcat['V_0p4R26_ERR'][lowverr] = 7.

tfrcat['logv_rot'] = np.log10(tfrcat['V_0p4R26'])
tfrcat['logv_rot_err'] = 0.434*tfrcat['V_0p4R26_ERR'] / tfrcat['V_0p4R26']

tfrcat = tfrcat['Z_DESI', 'D26', 'R_MAG_SB26_CORR', 'R_MAG_SB26_ERR_CORR', 'R_ABSMAG_SB26', 'R_ABSMAG_SB26_ERR', 'GOOD_MORPH', 'MU_ZCMB', 'MU_ZCMB_ERR', 'V_0p4R26', 'V_0p4R26_ERR', 'logv_rot', 'logv_rot_err'].to_pandas()
tfrcat


# In[23]:


#fig, axes = plt.subplots(1,2, figsize=(8,6), tight_layout=True, sharex=True)
#
#ax = axes[0]
#ax.errorbar(tfrcat['logv_rot'], tfrcat['R_MAG_SB26_CORR'],
#             xerr=tfrcat['logv_rot_err'],
#             yerr=tfrcat['R_MAG_SB26_ERR_CORR'],
#             fmt='.', 
#             alpha=0.5, 
#             ecolor='gray')
#
#ax.set(xlim=(0,3),
#       xlabel=r'$\log{v_\mathrm{rot}}$',
#       ylabel=r'$m_{r,\mathrm{corr}}$',
#       ylim=(20,8))
#
#ax = axes[1]
#ax.errorbar(tfrcat['logv_rot'], tfrcat['R_ABSMAG_SB26'],
#             xerr=tfrcat['logv_rot_err'],
#             yerr=tfrcat['R_ABSMAG_SB26_ERR'],
#             fmt='.', 
#             alpha=0.5, 
#             ecolor='gray')
#
#ax.set(xlim=(0,3),
#       xlabel=r'$\log{v_\mathrm{rot}}$',
#       ylabel=r'$M_{r,\mathrm{TF}}$',
#       ylim=(-12.25, -25))
#
#fig.suptitle(f'TFR Y1 Sample ({tfr_version})')
#fig.set_facecolor('none');
## fig.savefig('tfr_y1_sample.png', dpi=180);


# ### Generate Absolute Magnitudes for TFR Fitting
# 
# Procedure:
# 1. Calculate $M_{r,\mathrm{cos}}$ using $z_\mathrm{cos}$ from the mocks. This explicitly excludes mock peculiar velocities.
# 1. Compute distance moduli $\mu$ for each *observed* redshift in the mocks, $z_\mathrm{obs}$.
# 1. Compute $M_{r,\mathrm{obs}}$ using $m_{r,\mathrm{SB26}}$ from cross-matched from data and the generated $\mu$.
# 1. Scatter $\log{V_\mathrm{rot}}$ by binning the Y1 data in magnitude ($M_{r,\mathrm{cos}}$), computing the $\log{v_\mathrm{rot}}$ CDF in each bin, and randomly sampling new values.

# In[24]:


#- Use distmod with cosmological redshifts to compute a "true" absolute magnitude M_r(SB26).
#  Then compute a central value for rotational velocity using this "true" magnitude.
Mr_cos = mock['R_MAG_SB26_CORR'] - cosmology.distmod(mock['zcos']).to_value('mag')

#- Compute an observed magnitude based on the PVs in the mock catalog.
mu_obs_mock = cosmology.distmod(mock['zobs']).to_value('mag')
mock['MU_OBS_MOCK'] = mu_obs_mock
Mr_obs_mock = (mock['R_MAG_SB26_CORR'] - mu_obs_mock).to_numpy()
Mr_obs_err_mock = mock['R_MAG_SB26_ERR_CORR'].to_numpy()

#- Bin R_ABSMAG_SB26. Merge any bins with < 50 datapoints, working from the ends of the magnitude range.
bins = np.arange(-26, -12 + 0.05, 0.05)
M_r_bins = [bins[0]]
for k in np.arange(1, len(bins)):
    select = (tfrcat['R_ABSMAG_SB26'] > M_r_bins[-1]) & (tfrcat['R_ABSMAG_SB26'] <= bins[k])    
    if np.sum(select) >= 50:
        M_r_bins.append(bins[k])
M_r_bins.append(bins[-1])
N_bins = len(M_r_bins)

print(np.histogram(tfrcat['R_ABSMAG_SB26'], M_r_bins))

#- Loop through the magnitude bins and regenerate log(v_rot) by resampling the data.
#  Try to reduce resampling effects by using a smoothed version of the CDF of log(v_rot).
logvrot_mock = np.zeros_like(Mr_obs_mock)

use_weighted_fit = True

for k in tqdm(np.arange(0, N_bins-1)):
    # # TEST
    # if k > 5*N_bins//9:
    #     break
    
    M_r_min, M_r_max = M_r_bins[k], M_r_bins[k+1]

    #- Select TFR velocity data in this magnitude bin and compute the CDF of log(v_rot).
    i = (tfrcat['R_ABSMAG_SB26'] > M_r_min) & (tfrcat['R_ABSMAG_SB26'] <= M_r_max)
    logvrot_slice = tfrcat['logv_rot'][i].to_numpy()
    logvrot_err_slice = tfrcat['logv_rot_err'][i].to_numpy()

    if use_weighted_fit:
        #- Attempt to build a weighted CDF
        logvrot_bins = np.arange(1, 3.01, 0.01)
        logvrot_pdf_wt, logv_bins = np.histogram(logvrot_slice, bins=logvrot_bins, 
                                                 weights=np.ones_like(logvrot_err_slice)
                                                 # weights=1/logvrot_err_slice**2
                                                )
        logvrot_cdf = np.cumsum(logvrot_pdf_wt) / np.sum(logvrot_pdf_wt)
        
        logvrot_slice = 0.5*(logvrot_bins[1:] + logvrot_bins[:-1])

        #- Keep only the unique elements in the list
        idx = np.cumsum([len(list(g)) for k, g in groupby(logvrot_cdf)])[:-1]
        logvrot_cdf = logvrot_cdf[idx]
        logvrot_slice = logvrot_slice[idx]
        # print(logvrot_cdf.shape, logvrot_slice.shape)
    else:
        #- Default to the unweighted CDF
        logvrot_cdf = np.cumsum(logvrot_slice) / np.sum(logvrot_slice)

    #- Select mock data in this magnitude bin.
    j = (Mr_cos > M_r_min) & (Mr_cos <= M_r_max)
    # j = (Mr_obs_mock > M_r_min) & (Mr_obs_mock <= M_r_max)
    N_mock_slice = np.sum(j)
    un = np.random.uniform(size=N_mock_slice)
    logvrot_mock_slice = csaps(logvrot_cdf, np.sort(logvrot_slice), un).values
    logvrot_mock[j] = logvrot_mock_slice

    #- Apply a velocity cut of 10 to 1000 km/s.
    #  Regenerate any velocities that fall outside the valid range.
    #  Note that an intermediate variable is needed to manage the array slicing.
    bad_vrot = (logvrot_mock[j] < 1) | (logvrot_mock[j] > 3)
    while np.any(bad_vrot):
        N_regen = np.sum(bad_vrot)
        uni = np.random.uniform(size=N_regen)
        logvrot_mock_regen = logvrot_mock[j]
        logvrot_mock_regen[bad_vrot] = csaps(logvrot_cdf, np.sort(logvrot_slice), uni).values
        logvrot_mock[j] = logvrot_mock_regen
        bad_vrot = (logvrot_mock[j] < 1) | (logvrot_mock[j] > 3)

# - Finally, match mock values against the Y1 data to assign uncertainties on logvrot_mock and Mr_obs_mock.
#  Many ways to do this... here just copy the FP approach of grabbing the nearest neighbor in (log v, M_r)
#  from data and taking its uncertainty.
search_tree = KDTree(np.c_[tfrcat['logv_rot'], tfrcat['R_ABSMAG_SB26']])
search_tree.query([1.5, -20])
_, idx = search_tree.query([[x, y] for (x,y) in zip(logvrot_mock, Mr_obs_mock)])

logvrot_err_mock = tfrcat['logv_rot_err'][idx].to_numpy()
Mr_obs_err_mock = mock['R_MAG_SB26_ERR_CORR'].to_numpy()

mock['LOGVROT_MOCK'] = logvrot_mock
mock['LOGVROT_ERR_MOCK'] = logvrot_err_mock
mock['R_ABSMAG_SB26_TRUE'] = Mr_cos
mock['R_ABSMAG_SB26_MOCK'] = Mr_obs_mock
mock['R_ABSMAG_SB26_ERR_MOCK'] = Mr_obs_err_mock


# In[25]:


# k = 2*N_bins//3
# M_r_min, M_r_max = M_r_bins[k], M_r_bins[k+1]

# #- Select TFR velocity data in this magnitude bin and compute the CDF of log(v_rot).
# i = (tfrcat['R_ABSMAG_SB26'] > M_r_min) & (tfrcat['R_ABSMAG_SB26'] <= M_r_max)
# logvrot_slice = tfrcat['logv_rot'][i].to_numpy()
# logvrot_err_slice = tfrcat['logv_rot_err'][i].to_numpy()
# logvrot_cdf = np.cumsum(logvrot_slice) / np.sum(logvrot_slice)

# fig, axes = plt.subplots(1, 4, figsize=(16,4), tight_layout=True)

# logvrot_bins = np.arange(1, 3.05, 0.05)

# ax = axes[0]
# ax.hist(logvrot_slice, bins=logvrot_bins)
# ax.set(xlabel=r'$\log{v_\mathrm{rot}}$',
#        ylabel=r'count',
#        title='slice histogram')

# ax = axes[1]
# logvrot_pdf_wt, logv_bins, _ = ax.hist(logvrot_slice, bins=logvrot_bins, weights=1/logvrot_err_slice**2)
# newcdf_wt = np.cumsum(logvrot_pdf_wt) / np.sum(logvrot_pdf_wt)
# ax.set(xlabel=r'$\log{v_\mathrm{rot}}$',
#        title='weighted slice histogram')

# ax = axes[2]
# logvrot_gen = np.random.multivariate_normal(mean=logvrot_slice, cov=np.diag(logvrot_err_slice**2), size=1000).flatten()
# logvrot_pdf_rsmpl, logv_bins, _ = ax.hist(logvrot_gen, bins=logvrot_bins)
# newcdf_rsmpl = np.cumsum(logvrot_pdf_rsmpl) / np.sum(logvrot_pdf_rsmpl)
# ax.set(xlabel=r'$\log{v_\mathrm{rot}}$',
#        title='resampled histogram')

# ax = axes[3]
# ax.plot(np.sort(logvrot_slice), logvrot_cdf, label='unweighted CDF')
# ax.plot(logv_bins[:-1], newcdf_wt, label='weighted CDF')
# ax.plot(logv_bins[:-1], newcdf_rsmpl, label='resampled CDF')
# ax.set(xlabel=r'$\log{v_\mathrm{rot}}$',
#        title='CDF')
# l = ax.legend(loc='upper left', fontsize=10)


# #### Apply Alex's Velocity and Dwarf Cuts
# 
# Alex's cuts, defined August 2025, are:
# * $m_r < \min{(17.75, \mu_\mathrm{CMB} - 17 + 5\log{h})}$, a cut on dwarfs
# * $70~\mathrm{km/s} < V_\mathrm{rot}(0.4R_{26}) < 300~\mathrm{km/s}$, a vertical velocity cut
# * $V_\mathrm{rot}(0.4R_{26}) < \min{(300~\mathrm{km/s}, 10^{0.3(\mu_\mathrm{CMB} - 34 + 5\log{h}) + 2})}$, a distance-dependent velocity cut

# In[26]:


def downsample(mock, size=100):
    """Randomly downsample a mock catalog, without replacement, to some size.

    Parameters
    ----------
    mock: pandas.DataFrame
        Pandas table with a mock catalog.
    size: int
        Size of the final downsampled catalog.

    Returns
    -------
    newmock: pandas.DataFrame
        Downsampled Pandas table.
    """
    Nmock = len(mock)
    idx_downsample = np.random.choice(Nmock, size, replace=False)
    return mock.iloc[idx_downsample]

def alex_cuts_velocity(catalog, logv_name='logv_rot', distmod_name='MU_ZCMB', vmin=70., vmax=300., h=1.):
    """Apply Alex's velocity cuts (Aug. 2025).

    Parameters
    ----------
    catalog: pandas.DataFrame
        Pandas table with a catalog (data or mock).
    logv_name: str
        Name of the rotational velocity column in the table.
    distmod_name: str
        Name of the distance modulus column in the table.
    vmin: float
        Minimum velocity cut in km/s.
    vmax: float
        Maximum velocity cut in km/s.
    h: float
        Dimensionless Hubble constant.

    Returns
    -------
    select: list or np.array
        Indices of table elements passing the cuts.
    """
    logVmin, logVmax = np.log10(vmin), np.log10(vmax)
    a = 0.3
    b = 34 + 5*np.log10(h)
    mu_obs = catalog[distmod_name]
    logVMmax = np.minimum(logVmax, a*(mu_obs - b) + 2)
    select = (catalog[logv_name] > logVmin) & (catalog[logv_name] < logVmax) & (catalog[logv_name] < logVMmax)
    return select

def alex_cuts_dwarf(catalog, rmag_name='R_ABSMAG_SB26', distmod_name='MU_ZCMB', h=1.):
    """Apply Alex's dwarf galaxy cuts (Aug. 2025).

    Parameters
    ----------
    catalog: pandas.DataFrame
        Pandas table with a catalog (data or mock).
    rmag_name: str
        Name of the r-band magnitude used for computing the cut.
    distmod_name: str
        Name of the distance modulus column in the table.
    h: float
        Dimensionless Hubble constant.

    Returns
    -------
    select: list or np.array
        Indices of table elements that are *not* classified as dwarfs.
    """
    Rlim = 17.75
    Mlim = -17 + 5*np.log10(h)
    Rlim_eff = np.minimum(Rlim, catalog[distmod_name] + Mlim)
    select = catalog[rmag_name] <= Rlim_eff
    return select


# #### Plot $M_{R,\mathrm{SB26}}$ for Mocks and TFR Data
# 
# Make a side-by-side comparison of the (downsampled) mock catalog and Y1 data.
# 
# Apply the quality cuts equally to both.

# In[27]:


#- Plot Mr vs log(v_rot) for the various steps in the calculation.

#fig, axes = plt.subplots(1,2, figsize=(8,6), tight_layout=True, sharex=True, sharey=True)
#
Ntfr = len(tfrcat)
#mock_downsample = downsample(mock, Ntfr)
#
#ax = axes[0]
#ax.errorbar(mock_downsample['LOGVROT_MOCK'], mock_downsample['R_ABSMAG_SB26_MOCK'],
#            xerr=mock_downsample['LOGVROT_ERR_MOCK'],
#            yerr=mock_downsample['R_ABSMAG_SB26_ERR_MOCK'],
#            fmt='.', 
#            alpha=0.5, 
#            ecolor='gray')
#
#idx_mock_goodv = alex_cuts_velocity(mock_downsample, logv_name='LOGVROT_MOCK', distmod_name='MU_OBS_MOCK')
#idx_mock_notdwarf = alex_cuts_dwarf(mock_downsample, rmag_name='R_MAG_SB26_CORR', distmod_name='MU_OBS_MOCK')
#idx_mock_good = idx_mock_goodv & idx_mock_notdwarf
#
#ax.errorbar(mock_downsample['LOGVROT_MOCK'][idx_mock_good], mock_downsample['R_ABSMAG_SB26_MOCK'][idx_mock_good],
#             xerr=mock_downsample['LOGVROT_ERR_MOCK'][idx_mock_good],
#             yerr=mock_downsample['R_ABSMAG_SB26_ERR_MOCK'][idx_mock_good],
#             fmt='.', 
#             alpha=0.5, 
#             ecolor='gray')
#
#ax.set(xlim=(0,3),
#       xlabel=r'$\log{v_\mathrm{rot}}$',
#       ylabel=r'$M_{r,\mathrm{obs}}$',
#       ylim=(-12.25, -25),
#       title=r'Mock catalog, downsampled')
#
#ax = axes[1]
#ax.errorbar(tfrcat['logv_rot'], tfrcat['R_ABSMAG_SB26'],
#             xerr=tfrcat['logv_rot_err'],
#             yerr=tfrcat['R_ABSMAG_SB26_ERR'],
#             fmt='.', 
#             alpha=0.5, 
#             ecolor='gray')
#
idx_tfr_goodv = alex_cuts_velocity(tfrcat, logv_name='logv_rot', distmod_name='MU_ZCMB')
idx_tfr_notdwarf = alex_cuts_dwarf(tfrcat, rmag_name='R_MAG_SB26_CORR', distmod_name='MU_ZCMB')
idx_tfr_good = idx_tfr_goodv & idx_tfr_notdwarf
#
#ax.errorbar(tfrcat['logv_rot'][idx_tfr_good], tfrcat['R_ABSMAG_SB26'][idx_tfr_good],
#             xerr=tfrcat['logv_rot_err'][idx_tfr_good],
#             yerr=tfrcat['R_ABSMAG_SB26_ERR'][idx_tfr_good],
#             fmt='.', 
#             alpha=0.5, 
#             ecolor='gray')
#
#ax.set(xlim=(0,3),
#       xlabel=r'$\log{v_\mathrm{rot}}$',
#       ylabel=r'$M_{r,\mathrm{Y1}}$',
#       ylim=(-12.25, -25),
#       title=f'Y1 TFR Sample ({tfr_version})');
#
#fig.set_facecolor('none');
## fig.savefig('tfr_mock_mr_vs_logv.png', dpi=150);


# #### Apply Quality Cuts to the Full Mock Sample

# In[28]:


idx_mock_goodv = alex_cuts_velocity(mock, logv_name='LOGVROT_MOCK', distmod_name='MU_OBS_MOCK')
idx_mock_notdwarf = alex_cuts_dwarf(mock, rmag_name='R_MAG_SB26_CORR', distmod_name='MU_OBS_MOCK')
idx_mock_good = idx_mock_goodv & idx_mock_notdwarf

print(np.sum(idx_mock_good), len(mock))


# ### Compute the Maximum Volume for each Galaxy
# 
# Since the SGA is size-limited, with $D_{26}>0.2'$, there is a maximum volume within which the galaxy could be located to be included in the SGA. Calculate the maximum volume to be used as a weight in the TFR calibration.

# In[29]:


dist = cosmology.luminosity_distance(np.abs(mock['zobs']))
dist_max = cosmology.luminosity_distance(z=0.1)
d26_kpc = 2*dist.to('kpc') * np.tan(0.5*mock['D26'].values*u.arcmin)
mock_dist_max = 0.5*d26_kpc / np.tan(0.1*u.arcmin)
# surv_max = cosmology.luminosity_distance(z=0.2)

# mock['D26_kpc'] = 2*dist.to_value('kpc') * np.tan(0.5*mock['D26'].values*u.arcmin)
# mock['DIST_MAX'] = 0.5*mock['D26_kpc'].values / np.tan(0.1*u.arcmin)
# mock['MAX_VOL_FRAC'] = (1e-3 * mock['DIST_MAX'].values)**3 / dist_max.to_value('Mpc')**3
mock['MAX_VOL_FRAC'] = mock_dist_max.to('Mpc')**3 / dist_max.to('Mpc')**3


# In[30]:


#plt.figure(tight_layout=True)
#
#iron_dist = cosmology.luminosity_distance(np.abs(tfrcat['Z_DESI'].values))
#iron_d26kpc = 2*iron_dist.to('kpc') * np.tan(0.5*tfrcat['D26'].values*u.arcmin)
#iron_dist_max = 0.5*iron_d26kpc / np.tan(0.1*u.arcmin)
#
#plt.hist(mock['MAX_VOL_FRAC'], np.arange(0, 20, 0.2), density=True, alpha=0.5, label='mock')
#plt.hist(iron_dist_max.to('Mpc')**3 / dist_max.to('Mpc')**3, np.arange(0, 20, 0.2), density=True, alpha=0.5, label='Y1')
#
#plt.legend()
#
#plt.xlabel('$V_{max}$/$V(z = 0.1)$');


# ### Fit the TFR and Compute Mock Distance Moduli
# 
# Fit $M_{r,\mathrm{obs,mock}}$ versus $\log{V_\mathrm{rot,mock}}$ to get a mock TFR.
# 
# Then compute the TF distance modulus as
# 
# $$
# \mu_\mathrm{mock} = m_{r,\mathrm{SB_{26}}} - M_{r,\mathrm{obs,mock}},
# $$
# 
# where the apparent magnitude is the quantity `R_MAG_SB26_CORR` used to compute magnitudes from the cosmological and observed redshift.

# #### Create a Synthetic Data Set for TFR Fitting

# In[31]:


mock_downsample = downsample(mock, Ntfr)

idx_mock_downsample_goodv = alex_cuts_velocity(mock_downsample, logv_name='LOGVROT_MOCK', distmod_name='MU_OBS_MOCK')
idx_mock_downsample_notdwarf = alex_cuts_dwarf(mock_downsample, rmag_name='R_MAG_SB26_CORR', distmod_name='MU_OBS_MOCK')
idx_mock_downsample_good = idx_mock_downsample_goodv & idx_mock_downsample_notdwarf

print(f'Nmock      = {len(mock_downsample):6d}\n'
      f'Nmock_cuts = {len(mock_downsample[idx_mock_downsample_good]):6d}\n'
      f'Ntfr       = {len(tfrcat):6d}\n'
      f'Ntfr_cuts  = {len(tfrcat[idx_tfr_good]):6d}')


# In[32]:


zbin_idx = np.digitize(mock_downsample['zobs'], zbins, right=True)
for i in range(len(zbins) + 1):
    if i == 0:
        print(f'{i:2d}  z <= {zbins[i]:0.3f}  {np.sum(zbin_idx == i):3d} galaxies')
    elif i == len(zbins):
        print(f'{i:2d}  z > {zbins[i-1]:0.3f}  {np.sum(zbin_idx == i):3d} galaxies')
    else:
        print(f'{i:2d}  {zbins[i-1]:0.3f} < z <= {zbins[i]:0.3f}  {np.sum(zbin_idx == i):3d} galaxies')


# In[33]:


# Downsample the mock catalog after cuts to 4200 galaxies.

no_use = (zbin_idx == 0) | (zbin_idx == len(zbins))
mock_downsample = downsample(mock_downsample[idx_mock_downsample_good & ~no_use], size=4200)

zbin_idx = np.digitize(mock_downsample['zobs'], zbins, right=True)
for i in range(len(zbins) + 1):
    if i == 0:
        print(f'{i:2d}  z <= {zbins[i]:0.3f}  {np.sum(zbin_idx == i):3d} galaxies')
    elif i == len(zbins):
        print(f'{i:2d}  z > {zbins[i-1]:0.3f}  {np.sum(zbin_idx == i):3d} galaxies')
    else:
        print(f'{i:2d}  {zbins[i-1]:0.3f} < z <= {zbins[i]:0.3f}  {np.sum(zbin_idx == i):3d} galaxies')


# In[34]:


#fig, axes = plt.subplots(1,2, figsize=(8,6), sharex=True, tight_layout=True)
#
#logV0 = 0
#
_zbin_ids = np.sort(np.unique(zbin_idx))
n_zbins = len(_zbin_ids)
#
#markers = 'sDv^<>'
#
#colors = iter(plt.cm.viridis(np.linspace(0,1, n_zbins + 1)))
#
#for j, _zbin_id in enumerate(_zbin_ids):
#    select_zbin = np.isin(zbin_idx, _zbin_id)
#
#    logv = mock_downsample['LOGVROT_MOCK'][select_zbin] - logV0
#    logv_err = 0.434*mock_downsample['LOGVROT_ERR_MOCK'][select_zbin] / mock_downsample['LOGVROT_MOCK'][select_zbin]
#
#    mr26 = mock_downsample['R_MAG_SB26_CORR'][select_zbin]
#    mr26_err = mock_downsample['R_MAG_SB26_ERR_CORR'][select_zbin]
#
#    Mr26 = mock_downsample['R_ABSMAG_SB26_MOCK'][select_zbin]
#    Mr26_err = mock_downsample['R_ABSMAG_SB26_ERR_MOCK'][select_zbin]
#
#    c = next(colors)
#
#    ax = axes[0]
#    ax.errorbar(x=logv, y=mr26, xerr=logv_err, yerr=mr26_err,
#                fmt=markers[j % 2], markersize=6, alpha=0.3,
#                color=c)
#
#    ax = axes[1]
#    ax.errorbar(x=logv, y=Mr26, xerr=logv_err, yerr=Mr26_err,
#                fmt=markers[j % 2], markersize=6, alpha=0.3,
#                color=c)
#
#ax = axes[0]
#ax.set(xlim=(1.7, 2.6),
#       ylim=(18, 13),
#       xlabel=r'$\log{v_\mathrm{rot}}$',
#       ylabel=r'$m_{r,\mathrm{SB26}}$')
#
#ax = axes[1]
#ax.set(ylim=(-17.5, -23.5),
#       xlabel=r'$\log{v_\mathrm{rot}}$',
#       ylabel=r'$M_{r,\mathrm{SB26}}$');


# #### Pack the Data and Fit with `MultiLinFit`

# In[35]:


bounds = [[-20, 0]]              # Bounds on a (slope)
bounds += n_zbins*[(-20, 20)]    # Bounds on b (intercepts: z-bins)
bounds += [(0,5)]                # Bounds on sigma

logV0_mock = np.median(mock_downsample['LOGVROT_MOCK'])
print(logV0_mock)

datasets = [] # list of (2xN) arrays of the data
covs = []     # covariance of zero point and slope for each "cluster"

for j, _zbin_id in enumerate(_zbin_ids):
    select_zbin = np.isin(zbin_idx, _zbin_id)

    logv = mock_downsample['LOGVROT_MOCK'].to_numpy()[select_zbin] - logV0_mock
    dlogv = mock_downsample['LOGVROT_ERR_MOCK'].to_numpy()[select_zbin]
    mr = mock_downsample['R_MAG_SB26_CORR'].to_numpy()[select_zbin]
    dmr = mock_downsample['R_MAG_SB26_ERR_CORR'].to_numpy()[select_zbin]

    N = len(logv)
    cov = np.empty((2, 2, N))
    for i in range(N):
        cov[:,:,i] = np.array([[dlogv[i]**2, 0.], [0., dmr[i]**2]])
    covs.append(cov)

    data = np.empty((2, N))
    data[0] = logv
    data[1] = mr
    datasets.append(data)

hf = MultiLinFit(datasets, covs, scatter=1)
pars, parscatter, lnpost = hf.optimize(bounds)


# In[37]:


#pars, parscatter
#
#
## #### Try the Fit using MCMC
#
## In[38]:
#
#
## Determine logV0 for the test calibration
#logV0_mock = np.median(logvrot_mock[idx])
#logV0_mock = np.median(mock_downsample['LOGVROT_MOCK'])
#print(logV0_mock)
#
## Pack the calibration set into lists
#logv, dlogv = [], []
#mr, dmr = [], []
#weights = []
#
## Loop over redshift bins
#for j, _zbin_id in enumerate(_zbin_ids):
#    select_zbin = np.isin(zbin_idx, _zbin_id)
#
#    logv.append(mock_downsample['LOGVROT_MOCK'].to_numpy()[select_zbin] - logV0_mock)
#    dlogv.append(mock_downsample['LOGVROT_ERR_MOCK'].to_numpy()[select_zbin])
#    mr.append(mock_downsample['R_MAG_SB26_CORR'].to_numpy()[select_zbin])
#    dmr.append(mock_downsample['R_MAG_SB26_ERR_CORR'].to_numpy()[select_zbin])
#    weights.append(np.ones_like(mock_downsample['LOGVROT_MOCK'][select_zbin]))
#
#
## In[39]:
#
#
## Number of redshift bins
#bounds = [[-20, 0]]        # Bounds on a (slope)
#bounds += n_zbins*[(-20, 20)]    # Bounds on b (intercepts: z-bins)
#bounds += [(0,5)]          # Bounds on sigma
#
## logging.warning('Fit does not account for volume weights.')
#
#results = hyperfit_line_multi(logv, mr, dlogv, dmr, bounds, weights=weights, scatter=1)
#
#a_mcmc, b_mcmc, sigma_mcmc, cov_mcmc, mcmc_samples, hf = results
#
#
## #### Plot the HyperFit Results
#
## In[40]:
#
#
#labels  = ['$a$']
#labels += [f'$b_{{ {k+1} }}$' for k in np.arange(n_zbins)]
#labels += [r'$\sigma$']
#
#fig = corner(mcmc_samples.T, bins=25, smooth=1,
#             labels=labels,
#             label_kwargs={'fontsize':18},
#             labelpad=0.1,
#             levels=(1-np.exp(-0.5), 1-np.exp(-2)),
#             quantiles=[0.16, 0.5, 0.84],
#             color='tab:blue',
#             hist_kwargs={'histtype':'stepfilled', 'alpha':0.3},
#             plot_datapoints=False,
#             fill_contours=True,
#             truths=tf_par,
#             truth_color='tab:green',
#             show_titles=True,
#             title_kwargs={"fontsize": 18, 'loc':'left', 'pad':10});
#
#for ax in fig.get_axes():
#    ax.tick_params(axis='both', which='major', labelsize=16);


# In[41]:


#fig, axes = plt.subplots(2, 1, figsize=(5,7),
#                         gridspec_kw={'height_ratios':[4,1], 'hspace':0.04, 'wspace':0.25})
#
#_logv = np.arange(0, 3, 0.1) - logV0_mock
#
#color = iter(plt.cm.viridis(np.linspace(0,1, n_zbins)))
#
#for k in range(n_zbins):
#    ax = axes[0]
#    c = next(color)
#    
#    eb = ax.errorbar(x=logv[k] + logV0_mock, y=mr[k],
#                     xerr=dlogv[k], yerr=dmr[k],
#                     fmt='.', color=c,
#                     label=f'{zbins[k]:.3f}-{zbins[k+1]:.3f}')
#
#    ax.plot(_logv + logV0_mock, a_mcmc*_logv + b_mcmc[k], color=eb[0].get_color(), ls='--', alpha=0.5)
#
#    ax = axes[1]
#    logv_obs = logv[k]
#    m_obs = mr[k]
#    m_exp = a_mcmc*logv_obs + b_mcmc[k]
#    eb = ax.errorbar(x=logv_obs + logV0_mock, y=(m_exp - m_obs)/m_exp,
#                     xerr=dlogv[k], yerr=dmr[k],
#                     fmt='.', color=c)
#
#ax = axes[0]
#ax.set(xlim=[1.7, 2.5],
#       ylim=[17.25, 13],
#       xticklabels=[],
#       ylabel=r'$m_{r,\mathrm{SB26}}$')
#
#ax = axes[1]
#ax.grid(ls=':')
#
#ax.set(xlabel=r'$\log{v_\mathrm{rot}}$',
#       ylabel=r'$\Delta m_r/m_r$',
#       ylim=(-0.4,0.4));


# In[42]:


#fig, axes = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=True, figsize=(10,8), tight_layout=True)
#
#color = iter(plt.cm.cool(np.linspace(0,1, n_zbins)))
#for i in range(n_zbins):
#    c = next(color)
#    
#    row = int(i/5)
#    col = i%5
#    
#    eb = axes[row,col].errorbar(logv[i] + logV0_mock, mr[i], xerr=dlogv[i], yerr=dmr[i], fmt='.', color=c, alpha=0.1)
#    axes[row,col].plot(_logv + logV0_mock, a_mcmc*_logv + b_mcmc[i], color=c)
#    
#    axes[row,col].set(xlim=[1.7, 2.5], ylim=[17.25, 13], title=f'{zbins[i]:.3f} < z $\leq$ {zbins[i+1]:.3f}')
#
## Delete extra axes
## fig.delaxes(axs[-1,-1])
#
#fig.supxlabel(r'$\log{(V(0.4R_{26})~[\mathrm{km/s}]}$)')
#fig.supylabel(r'$m_r^{0.1} (26)$');


# #### Quick Check of Many Realizations of the "Calibration" Sample

# In[43]:


bounds = [[-20, 0]]              # Bounds on a (slope)
bounds += n_zbins*[(-20, 20)]    # Bounds on b (intercepts: z-bins)
bounds += [(0,5)]                # Bounds on sigma

a_real, b_real, sigma_real = [], [], []

for i in tqdm(np.arange(25)):
    mock_downsample = downsample(mock, Ntfr)
    
    idx_mock_downsample_goodv = alex_cuts_velocity(mock_downsample, logv_name='LOGVROT_MOCK', distmod_name='MU_OBS_MOCK')
    idx_mock_downsample_notdwarf = alex_cuts_dwarf(mock_downsample, rmag_name='R_MAG_SB26_CORR', distmod_name='MU_OBS_MOCK')
    idx_mock_downsample_good = idx_mock_downsample_goodv & idx_mock_downsample_notdwarf

    zbin_idx = np.digitize(mock_downsample['zobs'], zbins, right=True)
    no_use = (zbin_idx == 0) | (zbin_idx == len(zbins))
    mock_downsample = downsample(mock_downsample[idx_mock_downsample_good & ~no_use], size=4200)
    
    zbin_idx = np.digitize(mock_downsample['zobs'], zbins, right=True)
    logV0_mock = np.median(mock_downsample['LOGVROT_MOCK'])

    datasets = [] # list of (2xN) arrays of the data
    covs = []     # covariance of zero point and slope for each "cluster"
    
    for j, _zbin_id in enumerate(_zbin_ids):
        select_zbin = np.isin(zbin_idx, _zbin_id)
    
        logv = mock_downsample['LOGVROT_MOCK'].to_numpy()[select_zbin] - logV0_mock
        dlogv = mock_downsample['LOGVROT_ERR_MOCK'].to_numpy()[select_zbin]
        mr = mock_downsample['R_MAG_SB26_CORR'].to_numpy()[select_zbin]
        dmr = mock_downsample['R_MAG_SB26_ERR_CORR'].to_numpy()[select_zbin]
    
        N = len(logv)
        cov = np.empty((2, 2, N))
        for k in range(N):
            cov[:,:,k] = np.array([[dlogv[k]**2, 0.], [0., dmr[k]**2]])
        covs.append(cov)
    
        data = np.empty((2, N))
        data[0] = logv
        data[1] = mr
        datasets.append(data)
    
    hf = MultiLinFit(datasets, covs, scatter=1)
    pars, parscatter, lnpost = hf.optimize(bounds)

    a_real.append(pars[0])
    b_real.append(pars[1:])
    sigma_real.append(parscatter[0])


# In[44]:


# N = len(logvrot_mock)
# Ns = 150

# a_real, b_real, sigma_real = [], [], []

# for i in tqdm(np.arange(100)):
#     idx = np.random.choice(N-1, size=Ns, replace=False)

#     logV0_mock = np.median(logvrot_mock)

#     logv  = logvrot_mock[idx] - logV0_mock
#     dlogv = logvrot_err_mock[idx]
#     Mr  = Mr_obs_mock[idx]
#     dMr = Mr_obs_err_mock[idx]
#     weights = np.ones_like(logv)
#     # weights = 1/mock['MAX_VOL_FRAC'].to_numpy()[idx]
    
#     mock_dat = np.empty((2, Ns))
#     mock_cov = np.empty((2, 2, Ns))
    
#     logv, Ns, len(logv)
    
#     for k in range(Ns):
#         mock_dat[:, k] = np.array([logv[k], Mr[k]])
#         mock_cov[:,:,k] = np.array([[dlogv[k]**2, 0.], [0., dMr[k]**2]])
    
#     bounds = [[-20, 0]]                    # Bounds on a (slope)
#     bounds += [(-40,0)]                    # Bounds on b (intercepts: 0-pt + clusters)
#     bounds += [(0,5)]                      # Bounds on sigma
    
#     # logging.warning('Fit does not account for volume weights.')
    
#     hf = LinFit(mock_dat, mock_cov, weights=weights)
#     (a_bf, b_bf), sigma_bf, ll_bf = hf.optimize(bounds, verbose=False)

#     a_real.append(a_bf)
#     b_real.append(b_bf),
#     sigma_real.append(sigma_bf)
    
#     # print(f'a = {a_bf:.3f}')
#     # print(f'b = {b_bf:.3f}')
#     # print(f'sigma = {sigma_bf:.3f}')


# In[45]:


#fig, axes = plt.subplots(1,3, figsize=(12,4), tight_layout=True)
#
#ax = axes[0]
#ax.hist(a_real, bins=np.arange(-8.5,-6.4,0.1))
#ax.axvline(tf_par[0], color='tab:orange', ls='--')
#ax.axvline(np.mean(a_real), color='tab:green')
#ax.axvline(np.mean(a_real) - np.std(a_real), color='tab:green', ls='--')
#ax.axvline(np.mean(a_real) + np.std(a_real), color='tab:green', ls='--')
#ax.set(title=rf'$\hat{{a}}={np.mean(a_real):.2f}\pm{np.std(a_real):.2f}$',
#       xlabel=r'slope $a$')
#
#ax = axes[1]
#ax.hist(b_real[0], bins=np.arange(10,20.4,0.4))
#ax.axvline(tf_par[1], color='tab:orange', ls='--')
#ax.axvline(np.mean(b_real[0]), color='tab:green')
#ax.axvline(np.mean(b_real[0]) - np.std(b_real[0]), color='tab:green', ls='--')
#ax.axvline(np.mean(b_real[0]) + np.std(b_real[0]), color='tab:green', ls='--')
#ax.set(title=rf'$\hat{{b}}_0={np.mean(b_real[0]):.2f}\pm{np.std(b_real):.2f}$',
#       xlabel=r'zero point $b_1$')
#
#ax = axes[2]
#ax.hist(sigma_real, bins=np.arange(0.4,0.71,0.01))
#ax.axvline(tf_par[-1], color='tab:orange', ls='--')
#ax.axvline(np.mean(sigma_real), color='tab:green')
#ax.axvline(np.mean(sigma_real) - np.std(sigma_real), color='tab:green', ls='--')
#ax.axvline(np.mean(sigma_real) + np.std(sigma_real), color='tab:green', ls='--')
#ax.set(title=rf'$\hat{{\sigma}}={np.mean(sigma_real):.2f}\pm{np.std(sigma_real):.2f}$',
#       xlabel=r'magnitude scatter $\sigma$')
#
## fig.set_facecolor('none')
## fig.savefig(f'tfr_fit_param_spread_equal_weight_{os.path.basename(mockfile)[:-9]}.png', dpi=150)
## # fig.savefig('tfr_fit_param_spread_error_weighted.png', dpi=150)
## # fig.savefig('tfr_fit_param_spread_var_weighted.png', dpi=150)


# In[46]:


a_avg = np.average(a_real)
b_avg = np.average(b_real, axis=0)
sigma_avg = np.average(sigma_real)
a_avg, b_avg, sigma_avg


# In[47]:


a_mcmc, b_mcmc, sigma_mcmc = a_avg, b_avg, sigma_avg


# #### Apply Velocity and Dwarf Identification to the Full Sample

# In[48]:


idx_mock_goodv = alex_cuts_velocity(mock, logv_name='LOGVROT_MOCK', distmod_name='MU_OBS_MOCK')
idx_mock_notdwarf = alex_cuts_dwarf(mock, rmag_name='R_MAG_SB26_CORR', distmod_name='MU_OBS_MOCK')
idx_mock_good = idx_mock_goodv & idx_mock_notdwarf

mock['DWARF'] = np.zeros_like(mock['LOGVROT_MOCK'], dtype=bool)
mock.loc[~idx_mock_notdwarf, 'DWARF'] = True

mock['MAIN'] = np.zeros_like(mock['DWARF'], dtype=bool)
mock.loc[idx_mock_good, 'MAIN'] = True

# In[49]:


# Identify zbins for all mock galaxies
zbin_idx = np.digitize(mock['zobs'], zbins, right=True)


# ### Compute TFR Distance Modulus
# 
# Using the "measured" apparent magnitude and the TFR-predicted absolute magnitude from the "calibration" above, compute the distance modulus:
# 
# $$
# \mu_\mathrm{TF} =  m_{r,\mathrm{SB_{26}}} - M_{r,\mathrm{TF}}.
# $$
# 
# Also compute the log distance ratio
# 
# $$
# \eta = \log{\left(\frac{D_z}{D_\mathrm{TFR}}\right)}
# $$

# In[50]:


#- Compute TF absolute magnitude and uncertainties using the MCMC from HyperFit.
#  Downsample the MCMC significantly for this quick calculation.

# Redshift bin centers:
zc = 0.5*dz + zbins[:-1]
mu_zc = cosmology.distmod(zc)

# Convert each redshift bin zero point to an abs mag
B_mcmc = b_mcmc - mu_zc.value

# Compute indices for intercepts.
# For galaxies outside the calibration range, assign them to the closest bin.
B_idx = zbin_idx - 1
B_idx[zbin_idx == 0] = 0
B_idx[zbin_idx == len(zbins)] = len(zbins) - 2

# Use the calibrated TFR to compute abs mag.
logV0_mock = np.median(mock['LOGVROT_MOCK'])
Mr_TF = a_mcmc * (mock['LOGVROT_MOCK'] - logV0_mock) + B_mcmc[B_idx]

# Compute the uncertainty in TFR abs mag.
Mr_TF_err = np.zeros_like(Mr_TF)

for i in tqdm(range(len(mock))):
    logv_random = np.random.normal(mock['LOGVROT_MOCK'].iloc[i], 0.434*mock['LOGVROT_ERR_MOCK'].iloc[i], size=1000)
    Mr_stat = a_mcmc*(logv_random - logV0_mock) + B_mcmc[B_idx[i]]
    Mr_TF_err[i] = np.sqrt(np.nanstd(Mr_stat)**2 + sigma_mcmc**2)

# Mr_TF = a_mcmc * (logvrot_mock - logV0_mock) + b_mcmc
# a_sampled, b_sampled = mcmc_samples[0][::500], mcmc_samples[1][::500]
# Mr_TF_err = np.std(a_sampled * (logvrot_mock[:, np.newaxis] - logV0_mock) + b_sampled, axis=1)

mu_TF = mock['R_MAG_SB26_CORR'] - Mr_TF
mu_TF_err = np.sqrt(mock['R_ABSMAG_SB26_ERR_MOCK']**2 + Mr_TF_err**2)

mu_zcmb = cosmology.distmod(mock['zobs']).to_value('mag')
mu_zcos = cosmology.distmod(mock['zcos']).to_value('mag')

eta_true = 0.2 * (mu_zcmb - mu_zcos)
eta_mock = 0.2 * (mu_zcmb - mu_TF)
eta_err_mock = 0.2 * mu_TF_err


# In[51]:


#fig, ax = plt.subplots(1,1, figsize=(7,4.5), tight_layout=True)
#ax.errorbar(mock['zobs'], eta_mock,
#            yerr=eta_err_mock,
#            fmt='.',
#            alpha=0.1,
#            ecolor='gray')
#
#zbins = np.arange(0, 0.1025, 0.0025)
#dz = 0.5*np.diff(zbins)
#zc = 0.5*(zbins[1:] + zbins[:-1])
#
#_, eta_avg, eta_std = profile_histogram(mock['zobs'], eta_mock, zbins)
#ax.errorbar(zc, eta_avg, xerr=dz, yerr=eta_std, fmt='o', color='tab:orange')
#
#ax.set(xlabel=r'$z_\mathrm{obs}$',
#       xlim=(-0.005,0.1),
#       ylim=(-1,1),
#       ylabel=r'$\eta = \log{(D_z / D_\mathrm{TFR})}$');
#
#fig.set_facecolor('none')
## fig.savefig('tfr_mock_eta.png', dpi=150);


# In[52]:


logdist_true = np.log10(cosmology.comoving_distance(mock['zobs'].to_numpy()).value/cosmology.comoving_distance(mock['zcos'].to_numpy()).value)

#fig, axes = plt.subplots(1,2, figsize=(10,4.5), tight_layout=True, sharex=True, sharey=True)
#
#ax = axes[0]
#ax.scatter(mock['zobs'], logdist_true, marker='.', alpha=0.1)
#
#ax = axes[1]
#ax.scatter(mock['zobs'], eta_true, marker='.', alpha=0.1)
#
#zbins = np.arange(0, 0.1025, 0.0025)
#dz = 0.5*np.diff(zbins)
#zc = 0.5*(zbins[1:] + zbins[:-1])
#
#ax = axes[0]
#_, eta_avg, eta_std = profile_histogram(mock['zobs'], logdist_true, zbins)
#ax.errorbar(zc, eta_avg, xerr=dz, yerr=eta_std, fmt='o', color='tab:orange')
#ax.grid(ls=':')
#
#ax = axes[1]
#_, eta_avg, eta_std = profile_histogram(mock['zobs'], eta_true, zbins)
#ax.errorbar(zc, eta_avg, xerr=dz, yerr=eta_std, fmt='o', color='tab:orange')
#
#ax.set(xlabel=r'$z_\mathrm{obs}$',
#       xlim=(-0.005,0.1),
#       ylim=(-1,1),
#       ylabel=r'$\eta = \log{(D_z / D_\mathrm{TFR})}$')
#ax.grid(ls=':');


# In[53]:


#fig, axes = plt.subplots(1,2, figsize=(11,4.5), tight_layout=True, sharex=True, sharey=True)
#
#ax = axes[0]
#ax.scatter(mock['zobs'], eta_true, marker='.', alpha=0.1)
#
#ax = axes[1]
#ax.errorbar(mock['zobs'], eta_mock, yerr=eta_err_mock, fmt='.', alpha=0.1, ecolor='gray')
#
#zbins = np.arange(0, 0.1025, 0.0025)
#dz = 0.5*np.diff(zbins)
#zc = 0.5*(zbins[1:] + zbins[:-1])
#
#ax = axes[0]
#_, eta_avg, eta_std = profile_histogram(mock['zobs'], eta_true, zbins)
#ax.errorbar(zc, eta_avg, xerr=dz, yerr=eta_std, fmt='o', color='tab:orange')
#ax.set(xlabel=r'$z_\mathrm{obs}$',
#       xlim=(-0.005,0.1),
#       ylim=(-1,1),
#       ylabel=r'$\eta_\mathrm{true} = \log{(D_{z_\mathrm{obs}} / D_{z_\mathrm{cos}})}$')
#ax.grid(ls=':')
#
#ax = axes[1]
#_, eta_avg, eta_std = profile_histogram(mock['zobs'], eta_mock, zbins)
#ax.errorbar(zc, eta_avg, xerr=dz, yerr=eta_std, fmt='o', color='tab:orange')
#ax.set(xlabel=r'$z_\mathrm{obs}$',
#       xlim=(-0.005,0.1),
#       ylim=(-1,1),
#       ylabel=r'$\eta = \log{(D_z / D_\mathrm{TFR})}$')
#ax.grid(ls=':');
#
#fig.set_facecolor('none')
## fig.savefig('tfr_mock_eta_true_mock.png', dpi=150);


# In[54]:


#fig, ax = plt.subplots(1,1, figsize=(6,5), tight_layout=True)
#
#ax.scatter(eta_true, eta_mock, alpha=0.2, marker='.')
#
#eta_bins = np.arange(-0.2, 0.225, 0.025)
#eta_c = 0.5*(eta_bins[1:] + eta_bins[:-1])
#deta = 0.5*np.diff(eta_bins)
#_, eta_avg, eta_std = profile_histogram(eta_true, eta_mock, eta_bins)
#
#ax.errorbar(eta_c, eta_avg, xerr=deta, yerr=eta_std, fmt='o', color='tab:orange')
#
#ax.set(xlim=(-0.5,0.5),
#       xlabel=r'$\eta_\mathrm{true}$',
#       ylim=(-1,2),
#       ylabel=r'$\eta_\mathrm{mock}$');


# In[55]:


#fig, ax = plt.subplots(1,1, figsize=(6,5), tight_layout=True)
#
#ax.scatter(mock['zobs'], eta_true - eta_mock, alpha=0.2, marker='.')
#
#_, deta_avg, deta_std = profile_histogram(mock['zobs'], eta_true - eta_mock, zbins)
#ax.errorbar(zc, deta_avg, xerr=dz, yerr=deta_std, fmt='o', color='tab:orange')
#
#ax.set(xlabel=r'$z_\mathrm{obs}$',
#       xlim=(-0.005,0.1),
#       ylim=(-1,1),
#       ylabel=r'$\eta_\mathrm{true} - \eta_\mathrm{TF}$');


# In[54]:


# fig, ax = plt.subplots(1,1, figsize=(6,4), tight_layout=True)
# ax.hist((eta_mock - eta_true) / eta_err_mock, bins=np.arange(-10,10.2,0.2))
# ax.set(xlabel=r'$(\eta_\mathrm{mock} - \eta_\mathrm{true})/\sigma_\eta$',
#        yscale='log')


# ## Write Output to FITS

# In[55]:


{ 'a' : a_mcmc} | \
{ f'b{k+1}' : b_mcmc[k] for k in range(len(b_mcmc)) } | \
{ 'sigma' : sigma_mcmc }


# In[57]:


mockdir=os.path.join('/global/cfs/cdirs/desi/science/td/pv/mocks/TF_mocks/fullmocks', args.version)
if not os.path.exists(mockdir):
    os.makedirs(mockdir)

outfile = os.path.join(mockdir, os.path.basename(mockfile).replace('.dat.hdf5', '.fits').replace('BGS_PV', 'TF_extended'))

# hdr = fits.Header(dict(NTF=len(mock),
#                        a=a_mcmc,
#                        b=b_mcmc,
#                        sigma=sigma_mcmc,
#                        cov_aa=cov_mcmc[0][0],
#                        cov_ab=cov_mcmc[0][1],
#                        cov_as=cov_mcmc[0][2],
#                        cov_bb=cov_mcmc[1][1],
#                        cov_bs=cov_mcmc[1][2],
#                        cov_ss=cov_mcmc[2][2]))

hdr = fits.Header({ 'NTF' : len(mock) } | \
                  { 'a' : a_mcmc } | \
                  { f'b{k+1}' : b_mcmc[k] for k in range(len(b_mcmc)) } | \
                  { 'sigma' : sigma_mcmc })

col01 = fits.Column(name='RA',                 format='D', array=mock['ra'].to_numpy())
col02 = fits.Column(name='DEC',                format='D', array=mock['dec'].to_numpy())
col03 = fits.Column(name='ZOBS',               format='D', array=mock['zobs'].to_numpy())
col04 = fits.Column(name='ZCOS',               format='D', array=mock['zcos'].to_numpy())
col05 = fits.Column(name='vx',                 format='D', array=mock['vx'].to_numpy())
col06 = fits.Column(name='vy',                 format='D', array=mock['vy'].to_numpy())
col07 = fits.Column(name='vz',                 format='D', array=mock['vz'].to_numpy())
col08 = fits.Column(name='DWARF',              format='L', array=mock['DWARF'].to_numpy())
col09 = fits.Column(name='MAIN',               format='L', array=mock['MAIN'].to_numpy())
col10 = fits.Column(name='LOGVROT',            format='D', array=mock['LOGVROT_MOCK'].to_numpy())
col11 = fits.Column(name='LOGVROT_ERR',        format='D', array=mock['LOGVROT_ERR_MOCK'].to_numpy())
col12 = fits.Column(name='R_ABSMAG_SB26' ,     format='D', array=mock['R_ABSMAG_SB26_MOCK'].to_numpy())
col13 = fits.Column(name='R_ABSMAG_SB26_ERR',  format='D', array=mock['R_ABSMAG_SB26_ERR_MOCK'].to_numpy())
col14 = fits.Column(name='R_ABSMAG_SB26_TRUE', format='D', array=mock['R_ABSMAG_SB26_TRUE'].to_numpy())
col15 = fits.Column(name='LOGDIST_TRUE',       format='D', array=eta_true)
col16 = fits.Column(name='LOGDIST',            format='D', array=eta_mock.to_numpy())
col17 = fits.Column(name='LOGDIST_ERR',        format='D', array=eta_err_mock.to_numpy())
col18 = fits.Column(name='Y1_COMP',            format='D', array=mock['Y1_COMP'].to_numpy())
col19 = fits.Column(name='Y3_COMP',            format='D', array=mock['Y3_COMP'].to_numpy())

hdulist = fits.BinTableHDU.from_columns([col01, col02, col03, col04, col05,
                                         col06, col07, col08, col09, col10,
                                         col11, col12, col13, col14, col15,
                                         col16, col17, col18, col19],
                                        header=hdr)
hdulist.writeto(outfile, overwrite=True)

shutil.chown(outfile, group='desi')

