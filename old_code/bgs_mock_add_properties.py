#!/usr/bin/env python
# coding: utf-8

# # Add other DR9 photometric and FastSpecFit properties to the BGS mocks by crossmatching to Y1 data

# In[1]:


import fitsio
import h5py
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from k_correction import GAMA_KCorrection
from astropy.cosmology import Planck15


# In[2]:


# Set up things for k-correction code. Make sure the cosmology matches the mock you are using!!
cosmology = Planck15
k_r = GAMA_KCorrection(cosmology, "./k_corr_rband_z01.dat")

# Read in Caitlin's Iron + Fullsweep cross matched catalogue, and her accumulated FastSpectFit data.
#iron = pd.read_csv("/global/cfs/cdirs/desicollab/science/td/pv/redshift_data/Y1/iron_fullsweep_catalogue_z012.csv")
iron = pd.read_csv("/global/cfs/cdirs/desi/science/td/pv/redshift_data/Y1/specprod_iron_healpix_z015.csv")
print(len(iron))
iron = iron.drop(iron[iron["deltachi2"] < 30.0].index) #drop entries with bad z fits
print(len(iron))
print([key for key in iron.keys()])

# Drop data that doesn't pass the photometric cuts
#iron = iron.drop(iron[(iron["inbasiccuts"] == 0) | (iron["has_corrupt_phot"] == 1)].index)
#print(len(iron))
#keep_flag = (iron["inBGS"] == 1) | (iron["inlocalbright"] == 1) | (iron["inSGA"] == 1)
#iron = iron[keep_flag == True]
#print(len(iron))
iron = iron.drop(iron[(iron["z"] > 0.11)].index)
print(len(iron))
iron = iron.drop(iron[(iron["flux_g"] <= 0.0) | (iron["flux_r"] <= 0.0) | (iron["flux_z"] <= 0.0)].index)
print(len(iron))
iron["col"] = iron["mag_g"] - iron["mag_r"]
iron = iron.drop(iron[(iron["col"] < -0.5) | (iron["col"] > 1.5)].index)
print(len(iron))

# Compute some absolute magnitudes in the same way as the mocks
iron["abs_mag_r"] = k_r.absolute_magnitude(iron["mag_r"], iron["z"], iron["col"])
print(np.amin(iron["z"]), np.amax(iron["z"]))
print(np.amin(iron["abs_mag_r"]), np.amax(iron["abs_mag_r"]))
print(np.amin(iron["col"]), np.amax(iron["col"]))
print(np.where(np.isnan(iron["z"])), np.where(np.isnan(iron["abs_mag_r"])), np.where(np.isnan(iron["col"])))

# Get fastspecfit data and merge into the iron data
#fastspecfit = pd.read_csv("/global/cfs/cdirs/desicollab/science/td/pv/redshift_data/Y1/fastspec_iron_healpix.csv")
#iron = iron.merge(fastspecfit, how='inner', left_on=['targetid', 'survey', 'program', 'healpix'], right_on=['targetid', 'survey', 'program', 'healpix'])
#for key in iron.keys():
#    print(key)


# # Plot the BGS mock redshift and magnitude distribution against the Y1 data file to check for consistency

# In[3]:


#data_abacus = pd.DataFrame(fitsio.read("./v0.1/AbacusSummit_base_c000_ph000_r000_z0.11.dat.fits"))
#print(np.amax(data_abacus["R_MAG_APP"]))
#print(data_abacus.keys())

#data_abacus = h5py.File("./v0.2/BGS_PV_AbacusSummit_base_c000_ph000_r000_z0.11.dat.hdf5", 'r')
#print(data_abacus.keys())

data_abacus = h5py.File("./v0.5/data/BGS_PV_AbacusSummit_base_c000_ph000_r000_z0.11.dat.hdf5", 'r')
print(data_abacus.keys())



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

iron_rcut = iron.drop(iron[iron["mag_r"] > 19.7].index)
print(len(iron), len(iron_rcut))

# Specify the transformations we are applying to the data (and mocks)
znorm = 0.11
magnorm = np.std(iron["abs_mag_r"])
colournorm = np.std(iron["col"])
magnorm_mock = np.std(np.array(data_abacus["abs_mag"]))
colournorm_mock = np.std(np.array(data_abacus["col_obs"]))

iron_z = np.array(iron_rcut["z"])
iron_mag = np.array(iron_rcut["abs_mag_r"])
iron_col = np.array(iron_rcut["col"])
z_renorm = np.amax(iron_z) - np.amin(iron_z)
mag_renorm = np.amax(iron_mag) - np.amin(iron_mag)
col_renorm = np.amax(iron_col) - np.amin(iron_col)

abacus_index = np.where(np.array(data_abacus["app_mag"]) < 19.7)[0]
mock_z = np.array(data_abacus["zobs"])
mock_mag = np.array(data_abacus["abs_mag"])# - np.array(data_abacus["zobs"]) - 0.8
mock_col = np.array(data_abacus["col_obs"])# * (0.85 + np.array(data_abacus["zobs"])) - 0.6*np.array(data_abacus["zobs"]) + 0.1
mock_app_mag = np.array(data_abacus["app_mag"])# - np.array(data_abacus["zobs"]) - 0.8

z_bins = np.linspace(0.0, 0.1, 6)
app_mag_bins = np.linspace(12.0, 20.0, 41)
#abs_mag_bins = np.linspace(-24.0, -10.0, 41)
#col_bins = np.linspace(-0.1, 1.1, 41)
abs_mag_bins = np.linspace(0.0, 1.0, 41)
col_bins = np.linspace(0.0, 1.0, 41)

cmap = plt.get_cmap('viridis')
new_cmap2 = truncate_colormap(cmap, 0.0, 1.0)
mycolors = []
for j in range(len(z_bins)):
    ival = j*((1.0 - 0.0)/(len(z_bins)-1.0)) + 0.0
    mycolors.append(cmap(ival))

fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.82, 0.82])
for i, (z_bin_low, z_bin_high) in enumerate(zip(z_bins[:-1],z_bins[1:])):
    dataindex = np.where(np.logical_and(iron_rcut["z"].to_numpy() > z_bin_low, iron_rcut["z"].to_numpy() <= z_bin_high))[0]
    mockindex = np.where(np.logical_and(np.array(data_abacus["zobs"])[abacus_index] > z_bin_low, np.array(data_abacus["zobs"])[abacus_index] <= z_bin_high))[0]
    data_hist = np.histogram((iron_mag[dataindex]-np.amin(iron_mag))/mag_renorm, bins=abs_mag_bins, density=True)[0]
    mock_hist = np.histogram((np.array(data_abacus["abs_mag"])[abacus_index][mockindex]-np.amin(iron_mag))/mag_renorm, bins=abs_mag_bins, density=True)[0]
    plt.step(abs_mag_bins[:-1], data_hist, where='pre', color=mycolors[i])
    plt.errorbar((abs_mag_bins[:-1]+abs_mag_bins[1:])/2.0, mock_hist, color=mycolors[i], ls='None', marker='o', alpha=0.4)
plt.show()

fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.82, 0.82])
for i, (z_bin_low, z_bin_high) in enumerate(zip(z_bins[:-1],z_bins[1:])):
    dataindex = np.where(np.logical_and(iron_rcut["z"].to_numpy() > z_bin_low, iron_rcut["z"].to_numpy() <= z_bin_high))[0]
    mockindex = np.where(np.logical_and(np.array(data_abacus["zobs"])[abacus_index] > z_bin_low, np.array(data_abacus["zobs"])[abacus_index] <= z_bin_high))[0]
    data_hist = np.histogram(np.array(iron_rcut["mag_r"])[dataindex], bins=app_mag_bins, density=True)[0]
    mock_hist = np.histogram(np.array(data_abacus["app_mag"])[abacus_index][mockindex], bins=app_mag_bins, density=True)[0]
    mock_hist2 = np.histogram(mock_app_mag[abacus_index][mockindex], bins=app_mag_bins, density=True)[0]
    plt.step(app_mag_bins[:-1], data_hist, where='pre', color=mycolors[i])
    plt.errorbar((app_mag_bins[:-1]+app_mag_bins[1:])/2.0, mock_hist, color=mycolors[i], ls='None', marker='o', alpha=0.4)
plt.show()

fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.82, 0.82])
for i, (z_bin_low, z_bin_high) in enumerate(zip(z_bins[:-1],z_bins[1:])):
    dataindex = np.where(np.logical_and(iron_rcut["z"].to_numpy() > z_bin_low, iron_rcut["z"].to_numpy() <= z_bin_high))[0]
    mockindex = np.where(np.logical_and(np.array(data_abacus["zobs"])[abacus_index] > z_bin_low, np.array(data_abacus["zobs"])[abacus_index] <= z_bin_high))[0]
    data_hist = np.histogram((iron_col[dataindex]-np.amin(iron_col))/col_renorm, bins=col_bins, density=True)[0]
    mock_hist = np.histogram((np.array(data_abacus["col_obs"])[abacus_index][mockindex]-np.amin(iron_col))/col_renorm, bins=col_bins, density=True)[0]
    plt.step(col_bins[:-1], data_hist, where='pre', color=mycolors[i])
    plt.errorbar((col_bins[:-1]+col_bins[1:])/2.0, mock_hist, color=mycolors[i], ls='None', marker='o', alpha=0.4)
plt.show();


# # Seems okay. So build a k-d tree to find the nearest Iron galaxy for each mock galaxy


from scipy.spatial import KDTree 
from astropy.table import Table

z_renorm = np.amax(iron["z"]) - np.amin(iron["z"])
mag_renorm = np.amax(iron["abs_mag_r"]) - np.amin(iron["abs_mag_r"])
col_renorm = np.amax(iron["col"]) - np.amin(iron["col"])

tree = KDTree(np.c_[(iron["z"] - np.amin(iron["z"]))/z_renorm, (iron["abs_mag_r"] - np.amin(iron["abs_mag_r"]))/mag_renorm, (iron["col"] - np.amin(iron["col"]))/col_renorm])

# What properties do we want for each galaxy?
keys = ['targetid', 'survey', 'program', 'healpix'] 

for i in range(8,9):
    for j in range(26,27):
        infile = str("./v0.5/data/BGS_PV_AbacusSummit_base_c000_ph%03d_r%03d_z0.11.dat.hdf5" % (i, j))
        outfile = str("./v0.5/iron/BGS_PV_AbacusSummit_base_c000_ph%03d_r%03d_z0.11.dat.hdf5" % (i, j))
        #data_abacus = h5py.File(infile, 'r')
        #with h5py.File(outfile, 'w') as out:
        #    for key in data_abacus.keys():        
        #        data_abacus.copy(key, out)
        #    distance, neighbour = tree.query(np.c_[(np.array(data_abacus["zobs"]) - np.amin(iron["z"]))/z_renorm, (np.array(data_abacus["abs_mag"]) - np.amin(iron["abs_mag_r"]))/mag_renorm, (np.array(data_abacus["col_obs"]) - np.amin(iron["col"]))/col_renorm])
        #    for key in keys:
        #        out[key] = iron[key].iloc[neighbour].to_numpy()
        #print(i, j, infile, outfile)




# Check we can read in mocks and recover the ancillary properties
iron_keys = ['targetid', 'survey', 'program', 'healpix', 'morphtype', 'z', 'deltachi2', 'mag_r', 'mag_err_r', 'flux_g', 'flux_r', 'flux_z', 'mag_g', 'mag_z',
             'sersic', 'sersic_ivar', 'shape_e1', 'shape_e1_err', 'shape_e2', 'shape_e2_err',
             'shape_epsilon', 'shape_epsilon_err', 'BA_ratio', 'BA_ratio_err','circ_radius', 'circ_radius_err', 'pos_angle',
             'vdisp', 'vdisp_ivar', 'age', 'zzsun', 'logmstar', 'sfr', 'halpha_ew', 'halpha_ew_ivar']

iron = pd.read_csv("/global/cfs/cdirs/desi/science/td/pv/redshift_data/Y1/specprod_iron_healpix_z015.csv", usecols=iron_keys)
iron = iron.drop(iron[iron["deltachi2"] < 30.0].index) #drop entries with bad z fits
iron = iron.drop(iron[(iron["z"] > 0.11)].index)
iron = iron.drop(iron[(iron["flux_g"] <= 0.0) | (iron["flux_r"] <= 0.0) | (iron["flux_z"] <= 0.0)].index)
iron["col"] = iron["mag_g"] - iron["mag_r"]
iron = iron.drop(iron[(iron["col"] < -0.5) | (iron["col"] > 1.5)].index)
iron["abs_mag_r"] = k_r.absolute_magnitude(iron["mag_r"], iron["z"], iron["col"])
iron = iron.drop(iron[iron["mag_r"] > 19.7].index)




magbins = np.linspace(-24.0, -12.0, 51)
colbins = np.linspace(-0.25, 1.3, 51)
massbins = np.linspace(6.0, 12.0, 31)
ssfrbins = np.linspace(-14.0, -8.0, 31)
counts, counts2 = np.zeros((50,50)), np.zeros((30,30))
for i in range(1):
    for j in range(27):
        mock = {}
        infile = str("./v0.5/iron/BGS_PV_AbacusSummit_base_c000_ph%03d_r%03d_z0.11.dat.hdf5" % (i, j))
        f = h5py.File(infile, 'r')
        for key in f.keys():
            if key == 'vel':
                mock['vx'] = f['vel'][:,0]
                mock['vy'] = f['vel'][:,1]
                mock['vz'] = f['vel'][:,2]
            else:
                mock[key] = f[key][()]
            if key == 'survey' or key == 'program':
                mock[key] = mock[key].astype('U')
        f.close()
        mock = pd.DataFrame.from_dict(mock)
        mock = mock.merge(iron, how='inner', on=['targetid', 'survey', 'program', 'healpix'])

        counts += np.histogram2d(mock["abs_mag"].to_numpy(), mock["col_obs"].to_numpy(), bins=(magbins, colbins))[0]        
        counts2 += np.histogram2d(mock["logmstar"].to_numpy(), np.log10(mock["sfr"].to_numpy()) - mock["logmstar"].to_numpy(), bins=(massbins, ssfrbins))[0]        

        print(j, len(mock))

counts /= np.sum(counts)
counts2 /= np.sum(counts2)


# Plot the data against the mock
t = np.linspace(0, counts.max(), 1000)
integral = ((counts >= t[:, None, None]) * counts).sum(axis=(1,2))
f = sp.interpolate.interp1d(integral, t)
counts_contours = f(np.array([0.997, 0.985, 0.95, 0.875, 0.70, 0.40, 0.10, 0.02]))

cmap = plt.get_cmap('gray_r')
new_cmap2 = truncate_colormap(cmap, 0.2, 1.0)
mycolors = []
for i in range(len(counts_contours)):
    ival = i*((1.0 - 0.2)/(len(counts_contours)-1.0)) + 0.2
    mycolors.append(cmap(ival))

fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.82, 0.82])
cax = ax.hexbin(iron["abs_mag_r"].to_numpy(), iron["col"].to_numpy(), mincnt=50.0, gridsize=50, cmap=truncate_colormap(plt.get_cmap('viridis'), 0.0, 0.95), reduce_C_function = np.sum)
ax.contour(counts.T, levels=counts_contours, colors=mycolors,extent=[magbins.min(),magbins.max(),colbins.min(),colbins.max()],linewidths=2,alpha=0.9)
ax.set_xlabel(r"$M_{r}$", fontsize=14)
ax.set_ylabel(r"$g-r$", fontsize=14, labelpad=0)
ax.tick_params(width=1.3)
ax.tick_params('both',length=10, which='major')
ax.tick_params('both',length=5, which='minor')
for axis in ['top','left','bottom','right']:
    ax.spines[axis].set_linewidth(1.3)
for tick in ax.xaxis.get_ticklabels():
    tick.set_fontsize(12)
for tick in ax.yaxis.get_ticklabels():
    tick.set_fontsize(12)
ax.set_xlim(-24.0, -12.0)
ax.set_ylim(-0.25, 1.3)
plt.savefig("/global/cfs/cdirs/desi/science/td/pv/mocks/BGS_base/BGS_PV_AbacusSummit_M_vs_gr.png", dpi=300)

#plt.show()

t = np.linspace(0, counts2.max(), 1000)
integral = ((counts2 >= t[:, None, None]) * counts2).sum(axis=(1,2))
f = sp.interpolate.interp1d(integral, t)
counts_contours = f(np.array([0.997, 0.985, 0.95, 0.875, 0.70, 0.40, 0.10, 0.02]))

cmap = plt.get_cmap('gray_r')
new_cmap2 = truncate_colormap(cmap, 0.2, 1.0)
mycolors = []
for i in range(len(counts_contours)):
    ival = i*((1.0 - 0.2)/(len(counts_contours)-1.0)) + 0.2
    mycolors.append(cmap(ival))

fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.82, 0.82])
cax = ax.hexbin(iron["logmstar"].to_numpy(), np.log10(iron["sfr"].to_numpy()) - iron["logmstar"].to_numpy(), bins='log', mincnt=10.0, gridsize=80, cmap=truncate_colormap(plt.get_cmap('viridis'), 0.0, 0.95), reduce_C_function = np.sum)
ax.contour(counts2.T, levels=counts_contours, colors=mycolors,extent=[massbins.min(),massbins.max(),ssfrbins.min(),ssfrbins.max()],linewidths=2,alpha=0.9)
ax.set_xlabel(r"$\log(M_{*}/M_{\odot})$", fontsize=14)
ax.set_ylabel(r"$\log(sSFR/yr)$", fontsize=14, labelpad=0)
ax.tick_params(width=1.3)
ax.tick_params('both',length=10, which='major')
ax.tick_params('both',length=5, which='minor')
for axis in ['top','left','bottom','right']:
    ax.spines[axis].set_linewidth(1.3)
for tick in ax.xaxis.get_ticklabels():
    tick.set_fontsize(12)
for tick in ax.yaxis.get_ticklabels():
    tick.set_fontsize(12)
ax.set_xlim(6.0, 12.0)
ax.set_ylim(-14.0, -8.0)
plt.savefig("/global/cfs/cdirs/desi/science/td/pv/mocks/BGS_base/BGS_PV_AbacusSummit_logM_vs_sSFR.png", dpi=300)

#plt.show()







