# From Cullan's notebook plot_DESI_FP_mocks.ipynb
# Only taking functions, not plotting 

'''
# DESI FP mocks
This notebook reads in the FP mocks created with 
the code `make_DESI_FP_mocks.py` and makes some 
summary plots, downsamples each mock so that the 
average n(z) matches the data, and converts them 
to clustering mocks. It also produces a random 
catalogue for the FP mocks by downsampling Chris 
Blake's BGS random catalogue.
'''

import os
import time
import h5py
import numpy as np
import scipy as sp
import pandas as pd
from calc_kcor import *
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as colours
import astropy.units as u
from astropy.io import fits
from astropy.cosmology import Planck15, FlatLambdaCDM
from astropy.table import Table
from k_correction import GAMA_KCorrection
from sklearn.neighbors import KDTree
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

LightSpeed = 299792.458
zmin = 0.01        # minimum redshift for selection
zmax = 0.1         # maximum redshift for selection
rfact = 20         # size of final random catalogue relative to data
ngrid = 128        # grid size for number density
nzbin = 36         # number of redshift bins for plotting
nrealran = 1       # number of Abacus random realisations used by Chris to compute the n(z)
n_reals = 27          # number of sub-samples
n_phases = 25          # number of phases
survey_area = 7739.58 * (np.pi/180.0)**2
cosmo = FlatLambdaCDM(H0=100, Om0=0.3151)

version = 3
mockversion = 0.5
comp_field = 'Y3_COMP'

pv_path = '/global/cfs/cdirs/desi/science/td/pv'

#- data
bgs_clus_data = pv_path+f'/bgsclustering/BGS_BRIGHT_clustering_forPV_data.fits'
bgs_clus_rand = pv_path+f'/bgsclustering/BGS_BRIGHT_clustering_forPV_rand.fits'
fp_clus_data = pv_path+f'/fpgalaxies/Y1/v{version}/FP_clustering_data_v{version}.fits'
fp_clus_rand = pv_path+f'/fpgalaxies/Y1/v{version}/FP_clustering_random_v{version}.fits'

#- mocks
mock_path = pv_path+f'/mocks/DR2'
bgs_base_path = mock_path+f'/BGS_base/v{mockversion}'
bgs_clus_path = mock_path+f'/BGS_clustering/v{mockversion}'
full_path = pv_path+f'/FP_mocks/full_mocks/v{mockversion}'
clus_path = pv_path+f'/FP_mocks/clustering_mocks/v{mockversion}'
mock_bgs_clus_data = bgs_clus_path+'/BGS_PV_AbacusSummit_clustering_ph{phase:03d}_r{real:03d}_{comp_field}.fits'
mock_bgs_clus_rand = bgs_clus_path+'/BGS_PV_AbacusSummit_clustering_random_{comp_field}.fits'

mock_fp_full_data = full_path+'/FP_AbacusSummit_c000_ph{phase:03d}_r{real:03d}.fits'
mock_fp_clus_data = clus_path+'/FP_AbacusSummit_c000_ph{phase:03d}_r{real:03d}.fits'
mock_fp_clus_rand = clus_path+'/FP_AbacusSummit_random.fits'
mock_bgs_base_rand = bgs_base_path+'/randoms/BGS_PV_AbacusSummit_base_c000_ph{phase:03d}_r{real:03d}_z0.11.ran.hdf5'
os.makedirs(clus_path, exist_ok=True)


def weighted_avg_and_std(values, weights, axis=None):
    average = np.average(values, weights=weights, axis=axis)
    average_err = np.std(values)*np.sqrt(np.sum((weights/np.sum(weights))**2))
    variance = np.average((values-average)**2, weights=weights, axis=axis)
    return (average, average_err, np.sqrt(variance))


def reweight(x, err):

    weight = 1.0 / err**2
    meanx = np.sum(x)/len(x)
    lamb = (np.sum(x*weight) - meanx*np.sum(weight))/(np.sum(x**2) - len(x)*meanx**2)
    newweight = weight - lamb*(x - meanx)
    newerr = np.sqrt(1.0/newweight)
    
    return newerr - np.mean(newerr) + np.mean(err)
    
    
if 1==1:
    # Read in the BGS clustering data and randoms
    data = Table.read(bgs_clus_data).to_pandas() 
    rand = Table.read(bgs_clus_rand).to_pandas()
    
    nzdat, zlims = np.histogram(data["Z"],
                                bins=nzbin,range=[zmin,zmax], weights=data["WEIGHT"])
    nzrand = np.histogram(rand["Z"],
                          bins=nzbin,range=[zmin,zmax], weights=rand["WEIGHT"])[0]
    print(data, rand)

    # Read in the FP clustering data and randoms
    fpdata = Table.read(fp_clus_data).to_pandas()
    fprand = Table.read(fp_clus_rand).to_pandas()
    
    fpdata["LOGDIST_GAUSS_ERR"] = reweight(fpdata["LOGDIST"], fpdata["LOGDIST_ERR"])
    nzfpdat = np.histogram(fpdata["Z"],
                           bins=nzbin,range=[zmin,zmax], weights=fpdata["WEIGHT"])[0]
    nzfprand = np.histogram(fprand["Z"],
                            bins=nzbin,range=[zmin,zmax], weights=fprand["WEIGHT"])[0]

    


# Read in all the mocks and compute the average n(z) for both the original density field mocks, and the FP mocks
ngals, meanpull, stdpull = 0.0, 0.0, 0.0
mock_count, fpmock_count = 0, 0
zvals, logdists_true, logdists_obs, logdists_err, logdists_corr, logdists_corr_err, logdists_gauss_err = [], [], [], [], [], [], []
nzmock, nzmockerr, nzfpmock, nzfpmockerr = np.zeros(nzbin), np.zeros(nzbin), np.zeros(nzbin), np.zeros(nzbin)
pullmock, logdistmock, logdisterrmock, logdisterrmock_gauss = np.zeros(nzbin), np.zeros(nzbin), np.zeros(nzbin), np.zeros(nzbin)
for phase in range(n_phases):
    print(phase)
    for real in range(n_reals):
        mock_bgs_clus_filename = mock_bgs_clus_data.format(phase=phase, real=real)
        mock_fp_full_filename = mock_fp_full_data.format(phase=phase, real=real)
        try:
            mock = Table.read(mock_bgs_clus_filename).to_pandas()
            nzmock += np.histogram(mock["Z"], bins=nzbin,range=[zmin,zmax], weights=mock["WEIGHT"])[0]
            nzmockerr += np.histogram(mock["Z"], bins=nzbin,range=[zmin,zmax], weights=mock["WEIGHT"])[0]**2
            mock_count += 1
        except:
            print(f"Can't process ", mock_bgs_clus_filename)
        
        try:
            mock = Table.read(mock_fp_full_filename).to_pandas()
            mock = mock[(mock["ZOBS"] >= zmin) & (mock["ZOBS"] <= zmax)]
            ngals += len(mock)
            
            # Gaussianise the logdistance ratios after malmquist bias correction
            mock["LOGDIST_GAUSS_ERR"] = reweight(mock["LOGDIST_CORR"], mock["LOGDIST_CORR_ERR"])
            
            # Zero-point the mock by forcing the mean measured logdistance ratio 
            # to be equal to the mean true log-distance ratio
            logdist_avg = weighted_avg_and_std(mock["LOGDIST_CORR"].to_numpy()-mock["LOGDIST_TRUE"].to_numpy(), 
                                               1.0/mock["LOGDIST_GAUSS_ERR"].to_numpy()**2)[0]
            mock["LOGDIST_CORR"] -= logdist_avg

            zvals.append(mock["ZOBS"].to_numpy())
            logdists_true.append(mock["LOGDIST_TRUE"].to_numpy())
            logdists_obs.append(mock["LOGDIST"].to_numpy())
            logdists_err.append(mock["LOGDIST_ERR"].to_numpy())
            logdists_corr.append(mock["LOGDIST_CORR"].to_numpy())
            logdists_corr_err.append(mock["LOGDIST_CORR_ERR"].to_numpy())
            logdists_gauss_err.append(mock["LOGDIST_GAUSS_ERR"].to_numpy())

            # Some histograms
            nzfpmock += np.histogram(mock["ZOBS"], bins=nzbin,range=[zmin,zmax])[0]
            nzfpmockerr += np.histogram(mock["ZOBS"], bins=nzbin,range=[zmin,zmax])[0]**2
            logdistmock += np.histogram(mock["LOGDIST_CORR"], bins=nzbin,range=[-0.3, 0.3])[0]
            logdisterrmock += np.histogram(mock["LOGDIST_CORR_ERR"], bins=nzbin,range=[0.08, 0.30])[0]
            logdisterrmock_gauss += np.histogram(mock["LOGDIST_GAUSS_ERR"], bins=nzbin,range=[0.08, 0.30])[0]
            pullmock += np.histogram((mock["LOGDIST_CORR"] - mock["LOGDIST_TRUE"])/mock["LOGDIST_GAUSS_ERR"], bins=nzbin,range=[-4.0, 4.0])[0]
            meanpull += np.sum((mock["LOGDIST_CORR"] - mock["LOGDIST_TRUE"])/mock["LOGDIST_GAUSS_ERR"])
            stdpull += np.sum(((mock["LOGDIST_CORR"] - mock["LOGDIST_TRUE"])/mock["LOGDIST_GAUSS_ERR"])**2)
            
            fpmock_count += 1
        except:
            print(f"Can't process ", mock_fp_full_filename)

zvals = np.concatenate(zvals)
logdists_true = np.concatenate(logdists_true)
logdists_obs = np.concatenate(logdists_obs)
logdists_err = np.concatenate(logdists_err)
logdists_corr = np.concatenate(logdists_corr)
logdists_corr_err = np.concatenate(logdists_corr_err)
logdists_gauss_err = np.concatenate(logdists_gauss_err)

nzmock /= mock_count
nzmockerr = np.sqrt(nzmockerr / mock_count - nzmock**2)
nzfpmock /= fpmock_count
nzfpmockerr = np.sqrt(nzfpmockerr / fpmock_count - nzfpmock**2)
logdistmock /= fpmock_count
logdisterrmock /= fpmock_count
logdisterrmock_gauss /= fpmock_count
pullmock /= fpmock_count
meanpull /= ngals
stdpull = np.sqrt(stdpull / ngals - meanpull**2)
print(meanpull, stdpull)
print(nzmock, nzmockerr, nzfpmock, nzfpmockerr)



# Now subsample the mocks to match the data (as best they can)

# Sub-sampling fraction of each mock versus redshift so that it as closely resembles the data as possible
print('\nSub-sampling mocks...')
subfracz = nzfpdat / nzfpmock 
# Normalise to 1.0 to account for cases where the mock n(z) is lower than the data n(z)
subfracz = np.where(subfracz > 1.0, 1.0, subfracz)
subfracz /= np.amax(subfracz)
# Smooth beyond where the subsampling is 1 to get something for the mock mean that is more homogeneous
subfracz = np.where(subfracz == 1.0, 1.0, savgol_filter(subfracz, 15, 1))
print('Sub-sampling fraction =',subfracz)



bins = np.linspace(zmin, zmax, nzbin)
ngals = np.zeros(nzbin-1)
logdistmean, logdistmean_err, logdiststd = np.zeros(nzbin-1), np.zeros(nzbin-1), np.zeros(nzbin-1)
logdistmean_corr, logdistmean_corr_err, logdiststd_corr = np.zeros(nzbin-1), np.zeros(nzbin-1), np.zeros(nzbin-1)
logdistmean_gauss, logdistmean_gauss_err, logdiststd_gauss = np.zeros(nzbin-1), np.zeros(nzbin-1), np.zeros(nzbin-1)
logdistmean_unweighted, logdistmean_err_unweighted, logdiststd_unweighted = np.zeros(nzbin-1), np.zeros(nzbin-1), np.zeros(nzbin-1)
logdistmean_corr_unweighted, logdistmean_corr_err_unweighted, logdiststd_corr_unweighted = np.zeros(nzbin-1), np.zeros(nzbin-1), np.zeros(nzbin-1)
for k in range(len(bins)-1):
    index = np.where(np.logical_and(zvals > bins[k], zvals <= bins[k+1]))[0]
    ngals[k] = len(index)
    if len(index) > 2:
        logdistmean[k], logdistmean_err[k], logdiststd[k] = weighted_avg_and_std(logdists_obs[index]-logdists_true[index], 1.0/logdists_err[index]**2)
        logdistmean_corr[k], logdistmean_corr_err[k], logdiststd_corr[k] = weighted_avg_and_std(logdists_corr[index]-logdists_true[index], 1.0/logdists_corr_err[index]**2)
        logdistmean_gauss[k], logdistmean_gauss_err[k], logdiststd_gauss[k] = weighted_avg_and_std(logdists_corr[index]-logdists_true[index], 1.0/logdists_gauss_err[index]**2)
        logdistmean_unweighted[k], logdistmean_err_unweighted[k], logdiststd_unweighted[k] = weighted_avg_and_std(logdists_obs[index]-logdists_true[index], np.ones(len(index)))
        logdistmean_corr_unweighted[k], logdistmean_corr_err_unweighted[k], logdiststd_corr_unweighted[k] = weighted_avg_and_std(logdists_corr[index]-logdists_true[index], np.ones(len(index)))
        print(logdistmean[k], logdistmean_unweighted[k], logdistmean_corr[k], logdistmean_corr_unweighted[k], logdistmean_gauss[k])

midvals = (bins[0:-1]+bins[1:])/2.0
logdistfix = CubicSpline(midvals, logdistmean_gauss)
logdists_obs_fix = logdists_corr - logdistfix(zvals)
ngals_corr = np.zeros(nzbin-1)


# Copy Chris' method/code to produce clustering mocks and randoms for the FP subsamples

# First read in all the random catalogues
mockrasran,mockdecran,mockredran,mocknran = np.array([]),np.array([]),np.array([]),np.array([],dtype='int')

# Loop over Abacus random realisations and sub-samples
for ireal in range(nrealran):
    for phase in range(n_phases):

        # Read in mock random catalogue
        mockranfile = mock_bgs_base_rand.format(phase=phase,real=ireal)
        f = h5py.File(mockranfile,'r')
        mockrasran1 = f['ra'][...]
        mockdecran1 = f['dec'][...]
        mockredran1 = f['zobs'][...]
        mockcompran1 = f[comp_field][...]
        f.close()
        nran = len(mockrasran1)
        
        # Cut mock random catalogue
        cut = (mockredran1 >= zmin) & (mockredran1 <= zmax) & (np.random.uniform(size=nran) < mockcompran1)
        nran = len(mockrasran1[cut])
        mocknran = np.concatenate((mocknran,np.array([nran])))
        mockrasran = np.concatenate((mockrasran,mockrasran1[cut]))
        mockdecran = np.concatenate((mockdecran,mockdecran1[cut]))
        mockredran = np.concatenate((mockredran,mockredran1[cut]))
        
# Set data weights to 1
mockweiran = np.ones(len(mockrasran))

# Randomize the order of the random sources
print('')
print('Randomizing order of randoms...')
cut = np.arange(len(mockrasran))
np.random.shuffle(cut)
mockrasran, mockdecran, mockredran, mockweiran = \
    mockrasran[cut],mockdecran[cut],mockredran[cut],mockweiran[cut]
print('\nTotal random points =', len(mockredran))




# Apply sub-sampling to random catalogue so that it matches the average mock n(z), 
# after that has also been subsampled to match the data as best it can (see above).
# This assumes the distribution of FP data follows Y1 completeness mask.

# Sub-sampling fraction versus redshift
nzold = np.histogram(mockredran, bins=nzbin, range=[zmin,zmax], weights=mockweiran)[0]
print('\nSub-sampling random catalogue...')
subfraczran = nzfpmock.astype(float)*subfracz/nzold.astype(float)
# Normalise to 1.0 to maximize size of random catalogue
subfraczran /= np.amax(subfraczran)
print('Sub-sampling fraction =',subfraczran)
izs = np.digitize(mockredran,np.linspace(zmin,zmax,nzbin+1)) - 1
cut = subfraczran[izs] > np.random.uniform(size=len(mockredran))
mockrasran,mockdecran,mockredran,mockweiran = mockrasran[cut],mockdecran[cut],mockredran[cut],mockweiran[cut]
print(len(mockrasran),'randoms after sub-sampling')

# New FP Random n(z)
nzfpmockrand = np.histogram(mockredran, bins=nzbin,range=[zmin,zmax], weights=mockweiran)[0]





# Now use all the above to produce FP clustering mocks and randoms 
# with n(z) columns and in the same format as the data

# Box enclosing data
distmax = cosmo.comoving_distance(zmax).value
nx,ny,nz = ngrid,ngrid,ngrid
lx,ly,lz = 2.*distmax,2.*distmax,2.*distmax
dx,dy,dz = lx/nx,ly/ny,lz/nz
x0,y0,z0 = distmax,distmax,distmax
dvol = dx*dy*dz

xlims = np.linspace(0.,lx,nx+1) - x0
ylims = np.linspace(0.,ly,ny+1) - y0
zlims = np.linspace(0.,lz,nz+1) - z0

# Construct 3D number density from random catalogues for density
mockrandoms = Table.read(mock_bgs_clus_rand).to_pandas()
dens_ra = np.radians(mockrandoms["RA"].to_numpy())
dens_dec = np.radians(mockrandoms["DEC"].to_numpy())
dens_z = mockrandoms["Z"].to_numpy()
dens_we = mockrandoms["WEIGHT"].to_numpy()
dist = cosmo.comoving_distance(dens_z).value
mockxran = dist*np.cos(dens_dec)*np.cos(dens_ra)
mockyran = dist*np.cos(dens_dec)*np.sin(dens_ra)
mockzran = dist*np.sin(dens_dec)
pos = np.vstack([mockxran+x0,mockyran+y0,mockzran+z0]).transpose()
winweigrid, edges = np.histogramdd(pos, 
                                   bins=(nx,ny,nz),
                                   range=((0.,lx),(0.,ly),(0.,lz)),
                                   density=False,
                                   weights=dens_we)
ndensweigrid = (np.sum(nzmock)/dvol)*(winweigrid/np.sum(winweigrid))

# Construct 3D number density from random catalogues for PV
dist = cosmo.comoving_distance(mockredran).value
mockxran = dist*np.cos(np.radians(mockdecran))*np.cos(np.radians(mockrasran))
mockyran = dist*np.cos(np.radians(mockdecran))*np.sin(np.radians(mockrasran))
mockzran = dist*np.sin(np.radians(mockdecran))
pos = np.vstack([mockxran+x0,mockyran+y0,mockzran+z0]).transpose()
winweigrid,edges = np.histogramdd(pos,
                                  bins=(nx,ny,nz),
                                  range=((0.,lx),(0.,ly),(0.,lz)),
                                  density=False,
                                  weights=mockweiran)
npvweigrid = (np.sum(nzfpmock*subfracz)/dvol)*(winweigrid/np.sum(winweigrid))
print(np.sum(nzfpmock), np.sum(nzfpmock*subfracz))

# Sub-sample random catalogue
cut = np.random.choice(len(mockrasran), rfact*np.sum(nzfpmock).astype(int),replace=False)
mockrasran,mockdecran,mockredran,mockweiran,mockxran,mockyran,mockzran = mockrasran[cut],mockdecran[cut],mockredran[cut],mockweiran[cut],mockxran[cut],mockyran[cut],mockzran[cut]
print('\nRandom catalogue cut to',len(mockrasran),'sources')





# Now loop over all the mocks, and process them into clustering mocks
mockxdat, mockydat, mockzdat, mocklogdisterr, mockndens, mocknpv = [], [], [], [], [], []
for phase in range(n_phases):
    print(phase)
    for real in range(n_reals):
        mock = Table.read(mock_fp_full_data.format(phase,real)).to_pandas()
        mock = mock[(mock["ZOBS"] >= zmin) & (mock["ZOBS"] <= zmax)]
        
        # Downsample the mock to match the data n(z)
        izs = np.digitize(mock["ZOBS"],np.linspace(zmin,zmax,nzbin+1)) - 1
        cut = subfracz[izs] > np.random.uniform(size=len(mock))
        mock = mock.iloc[cut]

        # Gaussianise the logdistance ratios after malmquist bias correction
        mock["LOGDIST_GAUSS_ERR"] = reweight(mock["LOGDIST_CORR"], mock["LOGDIST_CORR_ERR"])
        
        # Zero-point the mock by forcing the mean measured logdistance ratio 
        # to be equal to the mean true log-distance ratio
        logdist_avg = weighted_avg_and_std(mock["LOGDIST_CORR"].to_numpy()-mock["LOGDIST_TRUE"].to_numpy(), 
                                           1.0/mock["LOGDIST_GAUSS_ERR"].to_numpy()**2)[0]
        mock["LOGDIST_CORR"] -= logdist_avg + logdistfix(mock["ZOBS"])
        
        # Use the v1 estimator from Carreres et al., 2023 to estimate the PV from the logdistance ratio
        mock["PV"] = (LightSpeed * np.log(10.0) * mock["LOGDIST_CORR"] / 
                      ((LightSpeed*(1.0 + mock["ZOBS"]) / 
                        cosmo.comoving_distance(mock["ZOBS"]).value/cosmo.H(mock["ZOBS"]).value) - 1.0))
        mock["PV_ERR"] = (LightSpeed * np.log(10.0) * mock["LOGDIST_GAUSS_ERR"] / 
                          ((LightSpeed*(1.0 + mock["ZOBS"]) / 
                            cosmo.comoving_distance(mock["ZOBS"]).value/cosmo.H(mock["ZOBS"]).value) - 1.0))
        mock["PV_TRUE"] = LightSpeed * ((1.0 + mock["ZOBS"])/(1.0 + mock["ZCOS"]) - 1.0)
        
        # Sample both the density field and PV field number density at PV galaxy and random positions
        dist = cosmo.comoving_distance(mock["ZOBS"]).value
        mock_ra = np.radians(mock["RA"])
        mock_dec = np.radians(mock['DEC'])
        mockxdat_ij = dist*np.cos(mock_dec)*np.cos(mock_ra)
        mockydat_ij = dist*np.cos(mock_dec)*np.sin(mock_ra)
        mockzdat_ij = dist*np.sin(mock_dec)
        ix = np.digitize(mockxdat_ij, xlims) - 1
        iy = np.digitize(mockydat_ij, ylims) - 1
        iz = np.digitize(mockzdat_ij, zlims) - 1
        mockndensdat = ndensweigrid[ix, iy, iz]
        mocknpvdat = npvweigrid[ix, iy, iz]
        
        # Store the mock positions and errors to assign fake errors to the randoms later
        mockxdat.append(mockxdat_ij)
        mockydat.append(mockydat_ij)
        mockzdat.append(mockzdat_ij)
        mocklogdisterr.append(mock["LOGDIST_GAUSS_ERR"].to_numpy())
        mockndens.append(mockndensdat)
        mocknpv.append(mocknpvdat)

        # Save the new mock
        col1 = fits.Column(name='RA',format='D',array=mock["RA"].to_numpy())
        col2 = fits.Column(name='DEC',format='D',array=mock["DEC"].to_numpy())
        col3 = fits.Column(name='Z',format='D',array=mock["ZOBS"].to_numpy())
        col4 = fits.Column(name='WEIGHT',format='D',array=np.ones(len(mock)))
        col5 = fits.Column(name='NPV',format='D',array=mocknpvdat)
        col6 = fits.Column(name='NDENS',format='D',array=mockndensdat)
        col7 = fits.Column(name='LOGDIST',format='D',array=mock["LOGDIST_CORR"].to_numpy())
        col8 = fits.Column(name='LOGDIST_ERR',format='D',array=mock["LOGDIST_GAUSS_ERR"].to_numpy())
        col9 = fits.Column(name='LOGDIST_TRUE',format='D',array=mock["LOGDIST_TRUE"].to_numpy())
        col10 = fits.Column(name='PV',format='D',array=mock["PV"].to_numpy())
        col11 = fits.Column(name='PV_ERR',format='D',array=mock["PV_ERR"].to_numpy())
        col12 = fits.Column(name='PV_TRUE',format='D',array=mock["PV_TRUE"].to_numpy())
        hdulist = fits.BinTableHDU.from_columns([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12])
        outfile = mock_fp_clus_data.format(phase=phase, real=real)
        hdulist.writeto(outfile, overwrite=True)


# -- For the PV clustering the FKP-style weights also have a <v^2> term, 
# -- so we need a value for this for each random point. 
# -- Let's generate one by assigning an error based on the nearest real galaxy
tree_data = KDTree(np.c_[np.concatenate(mockxdat), np.concatenate(mockydat), np.concatenate(mockzdat)])
nn = tree_data.query(np.c_[mockxran, mockyran, mockzran], return_distance=False, dualtree=True)
print(nn[:,0])
mockranlogdisterr = np.concatenate(mocklogdisterr)[nn[:,0]]
mockranpverr = LightSpeed * np.log(10.0) * mockranlogdisterr / ((LightSpeed*(1.0 + mockredran)/cosmo.comoving_distance(mockredran).value/cosmo.H(mockredran).value) - 1.0)

# Output random catalogue
xlims = np.linspace(0.,lx,nx+1) - x0
ylims = np.linspace(0.,ly,ny+1) - y0
zlims = np.linspace(0.,lz,nz+1) - z0
ix = np.digitize(mockxran,xlims) - 1
iy = np.digitize(mockyran,ylims) - 1
iz = np.digitize(mockzran,zlims) - 1
mockndensran = ndensweigrid[ix,iy,iz]
mocknpvran = npvweigrid[ix,iy,iz]

print('\nNumber density from grid:')
print('Data dens =',np.mean(np.concatenate(mockndens)),'+/-',np.std(np.concatenate(mockndens)))
print('Random dens =',np.mean(mockndensran),'+/-',np.std(mockndensran))
print('Data pv =',np.mean(np.concatenate(mocknpv)),'+/-',np.std(np.concatenate(mocknpv)))
print('Random pv =',np.mean(mocknpvran),'+/-',np.std(mocknpvran))

col1 = fits.Column(name='RA',format='D',array=mockrasran)
col2 = fits.Column(name='DEC',format='D',array=mockdecran)
col3 = fits.Column(name='Z',format='D',array=mockredran)
col4 = fits.Column(name='WEIGHT',format='D',array=mockweiran)
col5 = fits.Column(name='NPV',format='D',array=mocknpvran)
col6 = fits.Column(name='NDENS',format='D',array=mockndensran)
col7 = fits.Column(name='LOGDIST_ERR',format='D',array=mockranlogdisterr)
col8 = fits.Column(name='PV_ERR',format='D',array=mockranpverr)
hdulist = fits.BinTableHDU.from_columns([col1,col2,col3,col4,col5,col6,col7,col8])
print('\nWriting out mock random catalogue...')
print(mock_fp_clus_rand)
hdulist.writeto(mock_fp_clus_rand, overwrite=True)


