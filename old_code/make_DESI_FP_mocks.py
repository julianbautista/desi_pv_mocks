import sys
import h5py
import fitsio
import numpy as np
import scipy as sp
import healpy as hp
import pandas as pd
from calc_kcor import *
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import astropy.units as u
from astropy.io import fits
from astropy.cosmology import Planck15, FlatLambdaCDM, z_at_value
from astropy.table import Table
from k_correction import GAMA_KCorrection
from scipy.spatial import KDTree
import time

# Useful utilities
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def weighted_avg_and_std(values, weights, axis=None):
    average = np.average(values, weights=weights, axis=axis)
    average_err = np.std(values)*np.sqrt(np.sum((weights/np.sum(weights))**2))
    variance = np.average((values-average)**2, weights=weights, axis=axis)
    return (average, average_err, np.sqrt(variance))

# The likelihood function for the Fundamental Plane
def FP_func(params, logdists, z_obs, r, s, i, err_r, err_s, err_i, Sn, smin, smax, sumgals=True, chi_squared_only=False):

    k = 0.0
    a, b, rmean, smean, imean, sigma1, sigma2, sigma3 = params

    fac1, fac2, fac3, fac4 = k*a**2 + k*b**2 - a, k*a - 1.0 - b**2, b*(k+a), 1.0 - k*a
    norm1, norm2 = 1.0+a**2+b**2, 1.0+b**2+k**2*(a**2+b**2)-2.0*a*k
    dsigma31, dsigma23 = sigma3**2-sigma1**2, sigma2**2-sigma3**3
    sigmar2 =  1.0/norm1*sigma1**2 +      b**2/norm2*sigma2**2 + fac1**2/(norm1*norm2)*sigma3**2    ##eq. B3
    sigmas2 = a**2/norm1*sigma1**2 + k**2*b**2/norm2*sigma2**2 + fac2**2/(norm1*norm2)*sigma3**2    ##eq. B4 
    sigmai2 = b**2/norm1*sigma1**2 +   fac4**2/norm2*sigma2**2 + fac3**2/(norm1*norm2)*sigma3**2    ##eq. B5
    sigmars =  -a/norm1*sigma1**2 -   k*b**2/norm2*sigma2**2 + fac1*fac2/(norm1*norm2)*sigma3**2    ##eq. B6
    sigmari =  -b/norm1*sigma1**2 +   b*fac4/norm2*sigma2**2 + fac1*fac3/(norm1*norm2)*sigma3**2    ##eq. B7
    sigmasi = a*b/norm1*sigma1**2 - k*b*fac4/norm2*sigma2**2 + fac2*fac3/(norm1*norm2)*sigma3**2    ##eq. B8

    sigma_cov = np.array([[sigmar2, sigmars, sigmari], [sigmars, sigmas2, sigmasi], [sigmari, sigmasi, sigmai2]])

    # Compute the chi-squared and determinant (quickly!)
    cov_r = err_r**2 + np.log10(1.0 + 300.0/(LightSpeed*z_obs))**2 + sigmar2  ##r2 entry of full covarence matrix (eq.17)
    cov_s = err_s**2 + sigmas2  ##s2 entry of full covarence matrix (eq.17)
    cov_i = err_i**2 + sigmai2  ##i2 entry of full covarence matrix (eq.17)
    cov_ri = -1.0*err_r*err_i + sigmari  ##ri/ir entry of full covarence matrix (eq.17)  
    
    A = cov_s*cov_i - sigmasi**2  ##det of bottom right corner of cov//also |cn|*cov11^{-1}
    B = sigmasi*cov_ri - sigmars*cov_i #det of bottom edges of cov //also |cn|*cov12^{-1} 
    C = sigmars*sigmasi - cov_s*cov_ri  ##det of bottom left corner of cov //also |cn|*cov13^{-1}
    E = cov_r*cov_i - cov_ri**2  ## det corners //also |cn|*cov22^{-1}
    F = sigmars*cov_ri - cov_r*sigmasi ## det top/bottom left //also |cn|*cov23^{-1}
    I = cov_r*cov_s - sigmars**2 ##//also |cn|*cov33^{-1}

    sdiff, idiff = s - smean, i - imean ##s and i residuals
    rnew = r - np.tile(logdists, (len(r), 1)).T ## applies shift to r by logdists
    rdiff  = rnew - rmean ##residual of shifted r

    det = cov_r*A + sigmars*B + cov_ri*C ##determinant of covarient matrix
    log_det = np.log(det)/Sn ##log det and apply weighting to galaxies 
    chi_squared = (A*rdiff**2 + E*sdiff**2 + I*idiff**2 + 2.0*rdiff*(B*sdiff + C*idiff) + 2.0*F*sdiff*idiff)/(det*Sn)

    # Compute the FN term for the Scut only
    delta = (A*F**2 + I*B**2 - 2.0*B*C*F)/det  # This is delta_paper * det^2
    FN = np.log(0.5 * (sp.special.erf(np.sqrt(E/(2.0*(det+delta)))*(smax-smean)) - sp.special.erf(np.sqrt(E/(2.0*(det+delta)))*(smin-smean))))/Sn
    if chi_squared_only:
        return chi_squared
    elif sumgals:
        return 0.5 * np.sum(chi_squared + log_det + 2.0*FN)
    else:
        return 0.5 * (chi_squared + log_det)

# Calculates f_n (the integral over the censored 3D Gaussian of the Fundamental Plane) 
# for a magnitude limit and velocity dispersion cut. 
def FN_func(FPparams, zobs, er, es, ei, lmin, lmax, smin, smax):
    
    k = 0.0
    a, b, rmean, smean, imean, sigma1, sigma2, sigma3 = FPparams

    ##covariance matrix calculations (same as in FP_func())  eq.B1-B8
    fac1, fac2, fac3, fac4 = k*a**2 + k*b**2 - a, k*a - 1.0 - b**2, b*(k+a), 1.0 - k*a
    norm1, norm2 = 1.0+a**2+b**2, 1.0+b**2+k**2*(a**2+b**2)-2.0*a*k
    dsigma31, dsigma23 = sigma3**2-sigma1**2, sigma2**2-sigma3**3
    sigmar2 =  1.0/norm1*sigma1**2 +      b**2/norm2*sigma2**2 + fac1**2/(norm1*norm2)*sigma3**2
    sigmas2 = a**2/norm1*sigma1**2 + k**2*b**2/norm2*sigma2**2 + fac2**2/(norm1*norm2)*sigma3**2
    sigmai2 = b**2/norm1*sigma1**2 +   fac4**2/norm2*sigma2**2 + fac3**2/(norm1*norm2)*sigma3**2
    sigmars =  -a/norm1*sigma1**2 -   k*b**2/norm2*sigma2**2 + fac1*fac2/(norm1*norm2)*sigma3**2
    sigmari =  -b/norm1*sigma1**2 +   b*fac4/norm2*sigma2**2 + fac1*fac3/(norm1*norm2)*sigma3**2
    sigmasi = a*b/norm1*sigma1**2 - k*b*fac4/norm2*sigma2**2 + fac2*fac3/(norm1*norm2)*sigma3**2

    err_r = er**2 + np.log10(1.0 + 300.0/(LightSpeed*zobs))**2 + sigmar2
    err_s = es**2 + sigmas2  ##s2 entry of full covarence matrix (eq.17)
    err_i = ei**2 + sigmai2  ##i2 entry of full covarence matrix (eq.17)
    cov_ri = -1.0*er*ei + sigmari ##ri/ir entry of full covarence matrix (eq.17)

    ## |Cov|*Cov^{-1} = [[A,B,C],[B,E,F],[C,F,I]]
    A = err_s*err_i - sigmasi**2
    B = sigmasi*cov_ri - sigmars*err_i
    C = sigmars*sigmasi - err_s*cov_ri
    E = err_r*err_i - cov_ri**2
    F = sigmars*cov_ri - err_r*sigmasi
    I = err_r*err_s - sigmars**2	

    # Inverse of the determinant!!
    det = 1.0/(err_r*A + sigmars*B + cov_ri*C)  ## 1/ln.82 in FP_func()

    # Compute all the G, H and R terms
    G = np.sqrt(E)/(2*F-B)*(C*(2*F+B) - A*F - 2.0*B*I) # This is actually G_paper / det ** (3/2)
    delta = (I*B**2 + A*F**2 - 2.0*B*C*F)*det**2       # This is actually delta_paper / det
    Edet = E*det  ## now equal to \Psi_{ss}
    Gdet = (G*det)**2 # This is equal to G_paper^2 * |Cov|
    Rmin = (lmin - rmean - imean/2.0)*np.sqrt(2.0*delta/det)/(2.0*F-B)  # This is equal to Rmin_paper
    Rmax = (lmax - rmean - imean/2.0)*np.sqrt(2.0*delta/det)/(2.0*F-B)  # This is equal to Rmax_paper

    G0min = -np.sqrt(2.0/(1.0+Gdet))*Rmin ##eq. C19 (max)
    G0max = -np.sqrt(2.0/(1.0+Gdet))*Rmax ##eq. C19 (max)
    G1min = -np.sqrt(Edet/(1.0+delta))*(smin - smean) ##eq. C20
    G1max = -np.sqrt(Edet/(1.0+delta))*(smax - smean) ##eq. C20
    
    H = np.sqrt(1.0+Gdet+delta)
    H0minmin = G*det*np.sqrt(delta) - np.sqrt(Edet/2.0)*(1.0+Gdet)*(smin - smean)/Rmin  ##H * eq.C21 (updated, Rmin, smin)
    H0minmax = G*det*np.sqrt(delta) - np.sqrt(Edet/2.0)*(1.0+Gdet)*(smax - smean)/Rmin  ##H * eq.C21 (updated, Rmin, smax)
    H0maxmin = G*det*np.sqrt(delta) - np.sqrt(Edet/2.0)*(1.0+Gdet)*(smin - smean)/Rmax  ##H * eq.C21 (updated, Rmax, smin)
    H0maxmax = G*det*np.sqrt(delta) - np.sqrt(Edet/2.0)*(1.0+Gdet)*(smax - smean)/Rmax  ##H * eq.C21 (updated, Rmax, smax)
    H1minmin = G*det*np.sqrt(delta) - np.sqrt(2.0/Edet)*(1.0+delta)*Rmin/(smin - smean) ##H * eq.C22 (updated, Rmin, smin)
    H1minmax = G*det*np.sqrt(delta) - np.sqrt(2.0/Edet)*(1.0+delta)*Rmin/(smax - smean) ##H * eq.C22 (updated, Rmin, smax)
    H1maxmin = G*det*np.sqrt(delta) - np.sqrt(2.0/Edet)*(1.0+delta)*Rmax/(smin - smean) ##H * eq.C22 (updated, Rmax, smin)
    H1maxmax = G*det*np.sqrt(delta) - np.sqrt(2.0/Edet)*(1.0+delta)*Rmax/(smax - smean) ##H * eq.C22 (updated, Rmax, smax)

    FN =  sp.special.owens_t(G0min, H0minmax/H) - sp.special.owens_t(G0min, H0minmin/H) + sp.special.owens_t(G0max, H0maxmin/H) - sp.special.owens_t(G0max, H0maxmax/H)
    FN += sp.special.owens_t(G1min, H1maxmin/H) - sp.special.owens_t(G1min, H1minmin/H) + sp.special.owens_t(G1max, H1minmax/H) - sp.special.owens_t(G1max, H1maxmax/H)
    FN += 1.0/(2.0*np.pi)*(np.arctan2(H0maxmax,H) + np.arctan2(H1maxmax,H) - np.arctan2(H0maxmin,H) - np.arctan2(H1maxmin,H))
    FN += 1.0/(2.0*np.pi)*(np.arctan2(H0minmin,H) + np.arctan2(H1minmin,H) - np.arctan2(H0minmax,H) - np.arctan2(H1minmax,H))

    # This can go less than zero for very large distances if there are rounding errors, so set a floor
    # This shouldn't affect the measured logdistance ratios as these distances were already very low probability!
    index = np.where(FN < 1.0e-15)
    FN[index] = 1.0e-15

    return np.log(FN)
    
mock = int(sys.argv[1])
realisation = int(sys.argv[2])

# Constants used in the FP procedure
a, b, rmean, smean, imean, sigma1, sigma2, sigma3 = 1.182, -0.803, 0.256, 2.131, 2.457, 0.058, 0.456, 0.224 # data FP
c = rmean - a*smean - b*imean 
Mmean = 4.65 - 5.0*rmean - 2.5*imean - 2.5*np.log10(2.0*np.pi) - 15.0
LightSpeed = 299792.458
deg2rad = np.pi/180.0
mag_low, mag_high = 10.0, 18.0
zmin, zmax = 0.0033, 0.1  ##define min and max redshift cuts
smin, smax = np.log10(50.0), np.log10(420.0)
sigma_corr_exp, sigma_corr_exp_err = -0.06, 0.03
evo_corr, evo_corr_err = 1.1, 0.4
theta_ap = 0.75  ##fiber radius (on sky angular radius of each fiber) 

# Set up k-corrections for the mocks using the same cosmology 
# as Alex Smith used for the original BGS mocks. This cosmology
# is only used for the k-corrections.
k_r = GAMA_KCorrection(Planck15, "/global/cfs/cdirs/desi/science/td/pv/mocks/FP_mocks/k_corr_rband_z01.dat")
k_g = GAMA_KCorrection(Planck15, "/global/cfs/cdirs/desi/science/td/pv/mocks/FP_mocks/k_corr_gband_z01.dat")

# Set up the redshift-distance relations matching the DESI fiducial cosmology.
cosmo = FlatLambdaCDM(H0=100,Om0=0.3151)
zvals = np.logspace(-5.0, 3.0, 10000)
red_spline = sp.interpolate.interp1d(cosmo.comoving_distance(zvals), zvals)
lumred_spline = sp.interpolate.interp1d((1.0+zvals)*cosmo.comoving_distance(zvals), zvals)

# Read in the iron data for cross-matching
iron_keys = ['targetid', 'survey', 'program', 'healpix', 'morphtype', 'z', 'zerr', 'mag_r', 'mag_err_r', 'mag_g', 'mag_z',
             'sersic', 'deltachi2', 'circ_radius', 'circ_radius_err', 'BA_ratio']
iron = pd.read_csv("/global/cfs/cdirs/desi/science/td/pv/redshift_data/Y1/specprod_iron_healpix_z015.csv", usecols=iron_keys)
iron = iron.drop(iron[iron["deltachi2"] < 30.0].index) #drop entries with bad z fits
for key in iron.keys():
    print(key)

FP_data = pd.read_csv("/global/cfs/cdirs/desicollab/science/td/pv/fpgalaxies/Y1/v3/FP_pv_cat_v3.csv", usecols=['targetid', 'zcmb', 'zcmb_group', 'ppxf_vdisp', 'ppxf_vdisp_err', 'r', 'er', 's', 'es', 'i', 'ei', 'Sn', 'FPcalibrator'])
print(len(FP_data))
FP_data = FP_data.drop(FP_data[FP_data["FPcalibrator"] == 0].index)
print(len(FP_data), FP_data.keys())

# Now read in a mock and apply the same cuts as our FP catalogue.
fpmock = {}
infile = str("/global/cfs/cdirs/desi/science/td/pv/mocks/BGS_base/v0.5/iron/BGS_PV_AbacusSummit_base_c000_ph%03d_r%03d_z0.11.dat.hdf5" % (mock, realisation))
f = h5py.File(infile, 'r')
for key in f.keys():
    if key == 'vel':
        fpmock['vx'] = f['vel'][:,0]
        fpmock['vy'] = f['vel'][:,1]
        fpmock['vz'] = f['vel'][:,2]
    else:
        fpmock[key] = f[key][()]
    if key == 'survey' or key == 'program':
        fpmock[key] = fpmock[key].astype('U')
f.close()
fpmock = pd.DataFrame.from_dict(fpmock)
fpmock = fpmock.merge(iron, how='inner', on=['targetid', 'survey', 'program', 'healpix'])
fpmock['kcorr_r'] = k_r.k(fpmock["zobs"], fpmock["col_obs"])
fpmock['kcorr_g'] = k_g.k(fpmock["zobs"], fpmock["col_obs"])

print(len(fpmock))
fpmock = fpmock[((fpmock["zobs"] >= zmin) & (fpmock["zobs"] <= zmax))]
print(len(fpmock))
fpmock = fpmock[((fpmock["morphtype"] == 'DEV') | ((fpmock["morphtype"] == 'SER') & (fpmock["sersic"] > 2.5)))]
print(len(fpmock))
fpmock = fpmock[fpmock["BA_ratio"] > 0.3]
print(len(fpmock))
fpmock = fpmock[(fpmock["col_obs"] > 0.68)
                & (fpmock["col_obs"] > 1.3*(fpmock["app_mag"] - fpmock["mag_z"])-0.12) 
                & (fpmock["col_obs"] < 2.0*(fpmock["app_mag"] - fpmock["mag_z"])-0.15)]
print(len(fpmock))
fpmock = fpmock[(fpmock["app_mag"] > mag_low) & (fpmock["app_mag"] < mag_high)]
print(len(fpmock))
fpmock = fpmock.drop(fpmock[(np.random.rand(len(fpmock)) > fpmock["Y1_COMP"])].index)
print(len(fpmock))

# Set out the intrinsic scatter matrix of the data fundamental plane
k = 0.0
fac1, fac2, fac3, fac4 = k*a**2 + k*b**2 - a, k*a - 1.0 - b**2, b*(k+a), 1.0 - k*a ##r, s, and coefficients for v3 (not normalised by |v1||v2|) and i coefficient for v2 (not normalised by |v2|) in eq. B1 (for easier calculation of sigmas)
norm1, norm2 = 1.0+a**2+b**2, 1.0+b**2+k**2*(a**2+b**2)-2.0*a*k  ##square of eq. B2 
sigmar2 =  1.0/norm1*sigma1**2 +      b**2/norm2*sigma2**2 + fac1**2/(norm1*norm2)*sigma3**2    ##eq. B3
sigmas2 = a**2/norm1*sigma1**2 + k**2*b**2/norm2*sigma2**2 + fac2**2/(norm1*norm2)*sigma3**2    ##eq. B4 
sigmai2 = b**2/norm1*sigma1**2 +   fac4**2/norm2*sigma2**2 + fac3**2/(norm1*norm2)*sigma3**2    ##eq. B5
sigmars =  -a/norm1*sigma1**2 -   k*b**2/norm2*sigma2**2 + fac1*fac2/(norm1*norm2)*sigma3**2    ##eq. B6
sigmari =  -b/norm1*sigma1**2 +   b*fac4/norm2*sigma2**2 + fac1*fac3/(norm1*norm2)*sigma3**2    ##eq. B7
sigmasi = a*b/norm1*sigma1**2 - k*b*fac4/norm2*sigma2**2 + fac2*fac3/(norm1*norm2)*sigma3**2    ##eq. B8

# Now, the transformed covariance matrix elements
sigmaM2 = 25.0*sigmar2 + 25.0*sigmari + 6.25*sigmai2
sigmaMs = -5.0*sigmars - 2.5*sigmasi
sigmaMi = -5.0*sigmari - 2.5*sigmai2

# Now the conditional mean and covariance matrix. This relies on the "inferred" absolute magnitude, not the true one to ensure that the 
# conversion from r, i and apparent magnitude is correct. As such overwrite the abs_mag with a slightly different expression using observed redshift and with evolution correction
# Compute the FP variables from the input. For the errors, we'll find the nearest real FP galaxy
fpmock["dz"] = cosmo.comoving_distance(fpmock["zobs"].to_numpy()).value
fpmock["dz_cluster"] = cosmo.comoving_distance(fpmock["zcos"].to_numpy()).value
fpmock["logdist_true"] = np.log10(fpmock["dz"].to_numpy()/fpmock["dz_cluster"].to_numpy())
dz_dz = LightSpeed / cosmo.H(fpmock["zobs"].to_numpy()).value

#fpmock["abs_mag"] = fpmock["app_mag"] - 5.0*np.log10(fpmock["dz"]) - 5.0*np.log10(1.0 + fpmock["zobs"]) - 25.0 - fpmock["kcorr_r"] + evo_corr*fpmock["zobs"]
hats = smean + sigmaMs/sigmaM2*(fpmock["abs_mag"].to_numpy() - Mmean)
hati = imean + sigmaMi/sigmaM2*(fpmock["abs_mag"].to_numpy() - Mmean)
new_sigma_cov = np.array([[sigmas2 - sigmaMs**2/sigmaM2, sigmasi - sigmaMs*sigmaMi/sigmaM2], [sigmasi - sigmaMs*sigmaMi/sigmaM2, sigmai2 - sigmaMi**2/sigmaM2]])

# Now draw surface brightnesses and velocity dispersions for each mock galaxy
draw = np.array([np.random.multivariate_normal(np.array([hats[j], hati[j]]), new_sigma_cov) for j in range(len(fpmock))])

fpmock["s"] = draw[:,0]
fpmock["i"] = draw[:,1]
fpmock["r"] = (4.65 - fpmock["abs_mag"] - 2.5*fpmock["i"] - 2.5*np.log10(2.0*np.pi) - 15.0)/5.0

# Compute the FP variables from the input. For the errors, we'll find the nearest real FP galaxy
tree = KDTree(np.c_[(FP_data["r"].to_numpy() - np.amin(fpmock["r"]))/(np.amax(fpmock["r"]) - np.amin(fpmock["r"])), (FP_data["s"].to_numpy() - np.amin(fpmock["s"]))/(np.amax(fpmock["s"]) - np.amin(fpmock["s"])), (FP_data["i"].to_numpy() - np.amin(fpmock["i"]))/(np.amax(fpmock["i"]) - np.amin(fpmock["i"]))])
distance, neighbour = tree.query(np.c_[(fpmock["r"].to_numpy() - np.amin(fpmock["r"]))/(np.amax(fpmock["r"]) - np.amin(fpmock["r"])), (fpmock["s"].to_numpy() - np.amin(fpmock["s"]))/(np.amax(fpmock["s"]) - np.amin(fpmock["s"])), (fpmock["i"].to_numpy() - np.amin(fpmock["i"]))/(np.amax(fpmock["i"]) - np.amin(fpmock["i"]))], k=2)
fpmock["er"] = FP_data["er"].to_numpy()[neighbour[:,1]]
fpmock["es"] = FP_data["es"].to_numpy()[neighbour[:,1]]
fpmock["ei"] = FP_data["ei"].to_numpy()[neighbour[:,1]]

# Perturb the FP data by the observational errors
rnew = fpmock["r"].to_numpy() + fpmock["logdist_true"].to_numpy()    # Add on the peculiar velocity to the sizes
err_r = fpmock["er"].to_numpy()
err_s = fpmock["es"].to_numpy()
err_i = fpmock["ei"].to_numpy()
fpmeasure = np.zeros((len(fpmock),3))
for ii in range(len(fpmock)):
    err_cov = np.array([[err_r[ii]**2 + np.log10(1.0 + 300.0/(LightSpeed*fpmock["zobs"].to_numpy()[ii]))**2, 0.0, -1.0*err_r[ii]*err_i[ii]],
                        [0.0, err_s[ii]**2, 0.0],
                        [-1.0*err_r[ii]*err_i[ii], 0.0, err_i[ii]**2]])
    fpmeasure[ii,0:] = np.random.multivariate_normal([rnew[ii],fpmock["s"].to_numpy()[ii],fpmock["i"].to_numpy()[ii]],err_cov)

# Overwrite the mock FP properties and apparent magnitude with those with observational errors before we apply the relevant cuts
fpmock["r"] = fpmeasure[:,0]
fpmock["s"] = fpmeasure[:,1]
fpmock["i"] = fpmeasure[:,2]
#fpmock["app_mag"] = 4.65 - 5.0*fpmock["r"] - 2.5*fpmock["i"] + 5.0*np.log10(1.0 + fpmock["zobs"]) - evo_corr*fpmock["zcos"] + fpmock["kcorr_r"] + 10.0 - 2.5*np.log10(2.0*np.pi) + 5.0*np.log10(fpmock["dz"]) 

#fpmock = fpmock[(fpmock["app_mag"] > mag_low) & (fpmock["app_mag"] < mag_high)]
#print(len(fpmock))
fpmock = fpmock.drop(fpmock[((fpmock["s"] < smin) | (fpmock["s"] > smax))].index)
print(len(fpmock))
#fpmock = fpmock.drop(fpmock[fpmock["es"] > 0.15].index)
#print(len(fpmock))

# Weighting for the FP fits
Vmin, Vmax = (1.0+zmin)**3*cosmo.comoving_distance(zmin).value**3, (1.0+zmax)**3*cosmo.comoving_distance(zmax).value**3  ##cube of luminosity distance (Volume proxy) for min and max redshifts included in survey (Selection effects, for Sn calculation)
Dlim = 10.0**((mag_high - fpmock["app_mag"].to_numpy() + 5.0*np.log10(fpmock["dz"]) + 5.0*np.log10(1.0 + fpmock["zobs"].to_numpy()))/5.0) ##compute max luminosity distance at which each individual galaxy would be observable
zlim = lumred_spline(Dlim)  ##compute max redshift that each galaxy would be observable (based of luminosity)
fpmock["Sn"] = np.where(zlim >= zmax, 1.0, np.where(zlim < zmin, 0.0, (Dlim**3 - Vmin)/(Vmax - Vmin))) ##compute selection effect weighting to account for incompleteness in sample caused by redshift and magnitude cuts (based off fraction of survey volume where galaxy with magnitude m_r could be observed)
#fpmock = fpmock.drop(fpmock[fpmock["Sn"] < 0.025].index)
#print(len(fpmock))

# Look at the quality of the data FP params against the mock and use this as the first iteration for fitting the mock
converged = False
data_bestfit = np.array([a, b, rmean, smean, imean, sigma1, sigma2, sigma3])
chi_squared = fpmock["Sn"].to_numpy()*FP_func(data_bestfit, 0.0, fpmock["zobs"].to_numpy(), fpmock["r"].to_numpy(), fpmock["s"].to_numpy(), fpmock["i"].to_numpy(), fpmock["er"].to_numpy(), fpmock["es"].to_numpy(), fpmock["ei"].to_numpy(), fpmock["Sn"].to_numpy(), smin, smax, sumgals=False, chi_squared_only=True)[0]
pvals = sp.stats.chi2.sf(chi_squared, np.sum(chi_squared)/(len(fpmock) - 8.0))
data_fit = fpmock.drop(fpmock[pvals < 0.01].index).reset_index(drop=True)
badcount = len(np.where(pvals < 0.01)[0]) ##count of all removed outliers
print(data_bestfit, np.sum(chi_squared), len(data_fit), sp.stats.chi2.isf(0.01, np.sum(chi_squared)/(len(fpmock) - 8.0)), np.sum(chi_squared)/(len(fpmock) - 8.0), 0, badcount, converged)

# Start the FP fitting
start = time.time()

while not converged: ##FP fitting algorithm (see Howlett et. al., 2022 for details)
        
    avals, bvals = (1.0, 1.8), (-1.5, -0.5)      ##set bounds on priors for a,b (a,b are fundamental plane coefficeints)
    rvals, svals, ivals = (-0.5, 0.5), (2.0, 2.4), (2.4, 3.0)        ##set bounds on priors for r,s,i (FP parameters, means)
    s1vals, s2vals, s3vals = (0.01, 0.12), (0.05, 0.5), (0.1, 0.3)   ##set bounds on priors for  sig1, sig2, sig3 (intrinsic scatters in orthogonal coordinate system v1,v2,v3 - see appendix B & p.962)

    FPparams = sp.optimize.differential_evolution(FP_func, bounds=(avals, bvals, rvals, svals, ivals, s1vals, s2vals, s3vals), 
                                                  args=(0.0, data_fit["zobs"].to_numpy(), data_fit["r"].to_numpy(),
                                                        data_fit["s"].to_numpy(), data_fit["i"].to_numpy(),
                                                        data_fit["er"].to_numpy(), data_fit["es"].to_numpy(),
                                                        data_fit["ei"].to_numpy(), data_fit["Sn"].to_numpy(), smin, smax),
                                                  maxiter=10000, tol=1.0e-6, disp=False)
    chi_squared = fpmock["Sn"].to_numpy()*FP_func(FPparams.x, 0.0, fpmock["zobs"].to_numpy(), fpmock["r"].to_numpy(),
                                                  fpmock["s"].to_numpy(), fpmock["i"].to_numpy(), fpmock["er"].to_numpy(),
                                                  fpmock["es"].to_numpy(), fpmock["ei"].to_numpy(), fpmock["Sn"].to_numpy(), 
                                                  smin, smax, sumgals=False, chi_squared_only=True)[0]
    pvals = sp.stats.chi2.sf(chi_squared, np.sum(chi_squared)/(len(fpmock) - 8.0))

    data_fit = fpmock.drop(fpmock[pvals < 0.01].index).reset_index(drop=True)
    badcountnew = len(np.where(pvals < 0.01)[0])
    converged = True if badcount == badcountnew else False
    print(FPparams.x, np.sum(chi_squared), len(data_fit), sp.stats.chi2.isf(0.01, np.sum(chi_squared)/(len(fpmock) - 8.0)), 
          np.sum(chi_squared)/(len(fpmock) - 8.0), badcount, badcountnew, converged)
    badcount = badcountnew
    
print(time.time()-start)

# Restrict the mock to only data that passes the outlier cuts in the FP fitting
fpmock = data_fit

# Now use the bestfit FP to compute some logdistance ratios
dmin, dmax, nd = -1.4, 1.4, 1001 ##set interpolator limits for log-distance ratio measurments (logR offset limits)
dbins = np.linspace(dmin, dmax, nd, endpoint=True) ## build bins for log-distance interpolator (logR offset bins)

d_H = np.outer(10.0**(-dbins), fpmock["dz_cluster"].to_numpy())
z_H = red_spline(d_H)
lmin = (4.65 + 5.0*np.log10(1.0+fpmock["zobs"].to_numpy()) - evo_corr*fpmock["zcos"].to_numpy() + fpmock["kcorr_r"].to_numpy() + 10.0 - 2.5*np.log10(2.0*np.pi) + 5.0*np.log10(d_H) - mag_high)/5.0 
lmax = (4.65 + 5.0*np.log10(1.0+fpmock["zobs"].to_numpy()) - evo_corr*fpmock["zcos"].to_numpy() + fpmock["kcorr_r"].to_numpy() + 10.0 - 2.5*np.log10(2.0*np.pi) + 5.0*np.log10(d_H) - mag_low)/5.0
loglike = FP_func(FPparams.x, dbins, fpmock["zobs"].to_numpy(), fpmock["r"].to_numpy(), fpmock["s"].to_numpy(), 
                  fpmock["i"].to_numpy(), fpmock["er"].to_numpy(), fpmock["es"].to_numpy(), fpmock["ei"].to_numpy(), 
                  np.ones(len(fpmock)), smin, smax, sumgals=False, chi_squared_only=False)
start = time.time()
FNvals = FN_func(FPparams.x, fpmock["zobs"].to_numpy(), fpmock["er"].to_numpy(), fpmock["es"].to_numpy(),
                 fpmock["ei"].to_numpy(), lmin, lmax, smin, smax)
print(time.time()-start)

# Convert to the PDF for logdistance. No malmquist bias correction yet
logP_dist = -1.5*np.log(2.0*np.pi) - loglike

# normalise logP_dist
ddiff = np.log10(d_H[:-1])-np.log10(d_H[1:])
valdiff = np.exp(logP_dist[1:])+np.exp(logP_dist[0:-1])
norm = 0.5*np.sum(valdiff*ddiff, axis=0)
logP_dist -= np.log(norm[:,None]).T

# Calculate the mean and variance of the gaussian, then the skew
mean = np.sum(dbins[0:-1,None]*np.exp(logP_dist[0:-1])+dbins[1:,None]*np.exp(logP_dist[1:]), axis=0)*(dbins[1]-dbins[0])/2.0 
err = np.sqrt(np.sum(dbins[0:-1,None]**2*np.exp(logP_dist[0:-1])+dbins[1:,None]**2*np.exp(logP_dist[1:]), axis=0)*(dbins[1]-dbins[0])/2.0 - mean**2) 
gamma1 = (np.sum(dbins[0:-1,None]**3*np.exp(logP_dist[0:-1])+dbins[1:,None]**3*np.exp(logP_dist[1:]), axis=0)*(dbins[1]-dbins[0])/2.0 - 3.0*mean*err**2 - mean**3)/err**3
gamma1 = np.where(gamma1 > 0.99, 0.99, gamma1)
gamma1 = np.where(gamma1 < -0.99, -0.99, gamma1)
delta = np.sign(gamma1)*np.sqrt(np.pi/2.0*1.0/(1.0 + ((4.0 - np.pi)/(2.0*np.abs(gamma1)))**(2.0/3.0)))
scale = err*np.sqrt(1.0/(1.0 - 2.0*delta**2/np.pi))
loc = mean - scale*delta*np.sqrt(2.0/np.pi)
alpha = delta/(np.sqrt(1.0 - delta**2))

fpmock["logdist"] = mean ##mean of weighted likelihood gaussian 
fpmock["logdist_err"] = err ##varience of weighted likelihood gaussian
fpmock["logdist_alpha"] = alpha ##skew of weighted likelihood gaussian

# And now with malmquist bias correction
logP_dist = -1.5*np.log(2.0*np.pi) - loglike - FNvals
valdiff = np.exp(logP_dist[1:])+np.exp(logP_dist[0:-1])
norm = 0.5*np.sum(valdiff*ddiff, axis=0)
logP_dist -= np.log(norm[:,None]).T

# Calculate the mean and variance of the gaussian, then the skew
mean = np.sum(dbins[0:-1,None]*np.exp(logP_dist[0:-1])+dbins[1:,None]*np.exp(logP_dist[1:]), axis=0)*(dbins[1]-dbins[0])/2.0 
err = np.sqrt(np.sum(dbins[0:-1,None]**2*np.exp(logP_dist[0:-1])+dbins[1:,None]**2*np.exp(logP_dist[1:]), axis=0)*(dbins[1]-dbins[0])/2.0 - mean**2) 
gamma1 = (np.sum(dbins[0:-1,None]**3*np.exp(logP_dist[0:-1])+dbins[1:,None]**3*np.exp(logP_dist[1:]), axis=0)*(dbins[1]-dbins[0])/2.0 - 3.0*mean*err**2 - mean**3)/err**3
gamma1 = np.where(gamma1 > 0.99, 0.99, gamma1)
gamma1 = np.where(gamma1 < -0.99, -0.99, gamma1)
delta = np.sign(gamma1)*np.sqrt(np.pi/2.0*1.0/(1.0 + ((4.0 - np.pi)/(2.0*np.abs(gamma1)))**(2.0/3.0)))
scale = err*np.sqrt(1.0/(1.0 - 2.0*delta**2/np.pi))
loc = mean - scale*delta*np.sqrt(2.0/np.pi)
alpha = delta/(np.sqrt(1.0 - delta**2))

##save log-distance values
fpmock["logdist_corr"] = mean ##mean of weighted likelihood gaussian 
fpmock["logdist_corr_err"] = err ##varience of weighted likelihood gaussian
fpmock["logdist_corr_alpha"] = alpha ##skew of weighted likelihood gaussian
print(len(fpmock))

# Output the fpmock
outfile = str("/global/cfs/cdirs/desi/science/td/pv/mocks/FP_mocks/fullmocks/v0.5/FP_AbacusSummit_c000_ph%03d_r%03d_v2.fits" % (mock, realisation))
print('\nWriting out mock catalogue...')
print(outfile)
hdr = fits.Header({'a': FPparams.x[0], 'b': FPparams.x[1], 'c': FPparams.x[2] - FPparams.x[0]*FPparams.x[3] - FPparams.x[1]*FPparams.x[4] , 
                   'rmean': FPparams.x[2], 'smean': FPparams.x[3], 'imean': FPparams.x[4], 'sigma1': FPparams.x[5], 'sigma2': FPparams.x[6], 
                   'sigma3': FPparams.x[7], 'chi2': np.sum(chi_squared), 'nFP': len(fpmock), 'nout': badcount})
col1 = fits.Column(name='RA',format='D',array=fpmock["ra"].to_numpy())
col2 = fits.Column(name='DEC',format='D',array=fpmock["dec"].to_numpy())
col3 = fits.Column(name='ZOBS',format='D',array=fpmock["zobs"].to_numpy())
col4 = fits.Column(name='ZCOS',format='D',array=fpmock["zcos"].to_numpy())
col5 = fits.Column(name='vx',format='D',array=fpmock["vx"].to_numpy())
col6 = fits.Column(name='vy',format='D',array=fpmock["vy"].to_numpy())
col7 = fits.Column(name='vz',format='D',array=fpmock["vz"].to_numpy())
col8 = fits.Column(name='r',format='D',array=fpmock["r"].to_numpy())
col9 = fits.Column(name='er',format='D',array=fpmock["er"].to_numpy())
col10 = fits.Column(name='s',format='D',array=fpmock["s"].to_numpy())
col11 = fits.Column(name='es',format='D',array=fpmock["es"].to_numpy())
col12 = fits.Column(name='i',format='D',array=fpmock["i"].to_numpy())
col13 = fits.Column(name='ei',format='D',array=fpmock["ei"].to_numpy())
col14 = fits.Column(name='Sn',format='D',array=fpmock["Sn"].to_numpy())
col15 = fits.Column(name='LOGDIST_TRUE',format='D',array=fpmock["logdist_true"].to_numpy())
col16 = fits.Column(name='LOGDIST',format='D',array=fpmock["logdist"].to_numpy())
col17 = fits.Column(name='LOGDIST_ERR',format='D',array=fpmock["logdist_err"].to_numpy())
col18 = fits.Column(name='LOGDIST_ALPHA',format='D',array=fpmock["logdist_alpha"].to_numpy())
col19 = fits.Column(name='LOGDIST_CORR',format='D',array=fpmock["logdist_corr"].to_numpy())
col20 = fits.Column(name='LOGDIST_CORR_ERR',format='D',array=fpmock["logdist_corr_err"].to_numpy())
col21 = fits.Column(name='LOGDIST_CORR_ALPHA',format='D',array=fpmock["logdist_corr_alpha"].to_numpy())
hdulist = fits.BinTableHDU.from_columns([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21], header=hdr)
hdulist.writeto(outfile, overwrite=True)