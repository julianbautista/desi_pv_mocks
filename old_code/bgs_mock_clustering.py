# source /global/cfs/cdirs/desi/software/desi_environment.sh
import os
import numpy as np
import h5py
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

def main():
    # Initialisations
    comp_field = 'Y3_COMP'         # 1) Y1 2) Y3 3) Y5
    zmin = 0.01        # minimum redshift for selection
    zmax = 0.1         # maximum redshift for selection
    appmaglim = 20.    # faint apparent magnitude limit for selection
    absmaglim = -18.1  # minimum luminosity for selection
    nrealdat = 25      # number of Abacus data realisations
    nrealran = 3       # number of Abacus random realisations
    nsub = 27          # number of sub-samples
    ngrid = 128        # grid size for number density
    bgs_base_path = '/global/cfs/cdirs/desi/science/td/pv/mocks/BGS_base/v0.5/'
    mockranfile_base = 'randoms/BGS_PV_AbacusSummit_base_c000_ph{phase:03d}_r{real:03d}_z0.11.ran.hdf5'
    mockdatfile_base = 'data/BGS_PV_AbacusSummit_base_c000_ph{phase:03d}_r{real:03d}_z0.11.dat.hdf5'
    bgs_clust_path = '/global/cfs/cdirs/desi/science/td/pv/mocks/BGS_clustering/v0.5/'
    os.path.makedirs(bgs_clust_path, exist_ok=True)
    mockdatoutfile_base = 'BGS_PV_AbacusSummit_clustering_c000_ph{phase:03d}_r{real:03d}_z0.11_{comp_field}.fits'
    mockranoutfile_base = f'BGS_PV_AbacusSummit_clustering_random_{comp_field}.fits'

    outfile = f'ndens_denssample_mock_{comp_field}.dat'

    # Box enclosing data
    cosmo = FlatLambdaCDM(H0=100,Om0=0.3151)
    distmax = cosmo.comoving_distance(zmax).value
    nx, ny, nz = ngrid, ngrid, ngrid
    lx, ly, lz = 2.*distmax, 2.*distmax, 2.*distmax
    dx, dy, dz = lx/nx, ly/ny, lz/nz
    x0, y0, z0 = distmax, distmax, distmax
    dvol = dx*dy*dz
    xlims = np.linspace(0., lx, nx+1) - x0
    ylims = np.linspace(0., ly, ny+1) - y0
    zlims = np.linspace(0., lz, nz+1) - z0

    # First build the number density grid using all the random catalogues
    mockrasran, mockdecran, mockredran, mocknran = np.array([]), np.array([]), np.array([]), np.array([], dtype='int')

    # Loop over Abacus random realisations and sub-samples
    for real in range(nrealran):

        for phase in range(nsub):

            # Read in mock random catalogue
            #mockranfile = 'randoms/BGS_PV_AbacusSummit_base_c000' + creal + csub + '_z0.11.ran.hdf5'
            mockranfile = mockranfile_base.format(phase=phase,real=real)
            print('\nReading in mock random catalogue...')
            print(bgs_base_path+mockranfile)
            f = h5py.File(bgs_base_path+mockranfile,'r')
            mockrasran1 = f['ra'][...]
            mockdecran1 = f['dec'][...]
            mockredran1 = f['zobs'][...]
            mockabsran1 = f['abs_mag'][...]
            mockappran1 = f['app_mag'][...]
            mockcompran1 = f[comp_field][...]
            f.close()
            nran = len(mockrasran1)
            print('Read in',nran,'mock random galaxies')
            
            # Cut mock random catalogue
            cut = ((mockredran1 > zmin) & (mockredran1 < zmax) & 
                   (mockappran1 < appmaglim) & (mockabsran1 < absmaglim) & 
                   (np.random.uniform(size=nran) < mockcompran1))
            nran = len(mockrasran1[cut])
            print(nran,'mock random galaxies with', zmin,'< z <',zmax,'and r <',appmaglim,'and M_r <',absmaglim,'and completeness sub-sampling')
            mocknran = np.concatenate((mocknran, np.array([nran])))
            mockrasran = np.concatenate((mockrasran, mockrasran1[cut]))
            mockdecran = np.concatenate((mockdecran, mockdecran1[cut]))
            mockredran = np.concatenate((mockredran, mockredran1[cut]))

    # Set data weights to 1
    nran = len(mockrasran)
    print('\nTotal random points =',nran)
    mockweiran = np.ones(nran)

    # Convert to (x,y,z) positions
    dist = cosmo.comoving_distance(mockredran).value
    mockxran = dist*np.cos(np.radians(mockdecran))*np.cos(np.radians(mockrasran))
    mockyran = dist*np.cos(np.radians(mockdecran))*np.sin(np.radians(mockrasran))
    mockzran = dist*np.sin(np.radians(mockdecran))

    # Create number density catalogue
    wingrid, edges = np.histogramdd(np.vstack([mockxran+x0,mockyran+y0,mockzran+z0]).transpose(),
                                    bins=(nx, ny, nz),
                                    range=((0.,lx), (0.,ly), (0.,lz)))
    
    ndat = np.mean(mocknran.astype(float))
    print('Average number of randoms =',ndat)
    ndensgrid = (ndat/dvol)*(wingrid/np.sum(wingrid))

    # Output number density grid
    outfile = f'ndens_denssample_mock_{comp_field}.dat'
    f = open(outfile,'w')
    f.write('{} {}'.format(zmin,zmax) + '\n')
    f.write('{} {} {} {} {} {} {} {} {}'.format(nx,ny,nz,lx,ly,lz,x0,y0,z0) + '\n')
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                f.write('{}'.format(ndensgrid[ix,iy,iz]) + '\n')
    f.close()

    # Sample number density at random positions
    ix = np.digitize(mockxran, xlims) - 1
    iy = np.digitize(mockyran, ylims) - 1
    iz = np.digitize(mockzran, zlims) - 1
    mockndensran = ndensgrid[ix, iy, iz]
    
    # Output random catalogue
    col1 = fits.Column(name='RA',format='D',array=mockrasran)
    col2 = fits.Column(name='DEC',format='D',array=mockdecran)
    col3 = fits.Column(name='Z',format='D',array=mockredran)
    col4 = fits.Column(name='WEIGHT',format='D',array=mockweiran)
    col5 = fits.Column(name='NDENS',format='D',array=mockndensran)
    hdulist = fits.BinTableHDU.from_columns([col1,col2,col3,col4,col5])
    print('\nWriting out mock random catalogue...')
    print(bgs_clust_path+mockranoutfile_base)
    hdulist.writeto(bgs_clust_path+mockranoutfile_base)

    
    # Loop over Abacus data realisations and sub-samples
    for real in range(nrealdat):
        for phase in range(nsub):
            # Read in mock data catalogue
            #mockdatfile = 'data/BGS_PV_AbacusSummit_base_c000' + creal + csub + '_z0.11.dat.hdf5'
            mockdatfile = mockdatfile_base.format(phase=phase,real=real)
            print('\nReading in mock data catalogue...')
            print(bgs_base_path+mockdatfile)
            f = h5py.File(bgs_base_path+mockdatfile,'r')
            mockrasdat = f['ra'][...]
            mockdecdat = f['dec'][...]
            mockreddat = f['zobs'][...]
            #mockvx = f['vel'][:,0]
            #mockvy = f['vel'][:,1]
            #mockvz = f['vel'][:,2]
            mockabsdat = f['abs_mag'][...]
            mockappdat = f['app_mag'][...]
            mockcompdat = f[comp_field][...]
            f.close()
            ndat = len(mockrasdat)
            print('Read in',ndat,'mock data galaxies')

            # Cut mock data catalogue
            cut = ((mockreddat > zmin) & (mockreddat < zmax) & 
                   (mockappdat < appmaglim) & (mockabsdat < absmaglim) & 
                   (np.random.uniform(size=ndat) < mockcompdat))
            mockrasdat, mockdecdat, mockreddat = mockrasdat[cut], mockdecdat[cut], mockreddat[cut]
            #mockvx,mockvy,mockvz = mockvx[cut],mockvy[cut],mockvz[cut]
            ndat = len(mockrasdat) 
            print(ndat,'mock data galaxies with',zmin,'< z <',zmax,'and r <',appmaglim,'and M_r <',absmaglim,'and completeness sub-sampling')

            # Set data weights to 1
            mockweidat = np.ones(ndat)

            # Sample number density at galaxy positions
            dist = cosmo.comoving_distance(mockreddat).value
            mockxdat = dist*np.cos(np.radians(mockdecdat))*np.cos(np.radians(mockrasdat))
            mockydat = dist*np.cos(np.radians(mockdecdat))*np.sin(np.radians(mockrasdat))
            mockzdat = dist*np.sin(np.radians(mockdecdat))
            ix = np.digitize(mockxdat,xlims) - 1
            iy = np.digitize(mockydat,ylims) - 1
            iz = np.digitize(mockzdat,zlims) - 1
            mockndensdat = ndensgrid[ix,iy,iz]

            # Output data catalogue
            col1 = fits.Column(name='RA',format='D',array=mockrasdat)
            col2 = fits.Column(name='DEC',format='D',array=mockdecdat)
            col3 = fits.Column(name='Z',format='D',array=mockreddat)
            col4 = fits.Column(name='WEIGHT',format='D',array=mockweidat)
            col5 = fits.Column(name='NDENS',format='D',array=mockndensdat)
            hdulist = fits.BinTableHDU.from_columns([col1,col2,col3,col4,col5])
            #outfile = 'BGS_PV_AbacusSummit_clustering' + creal + csub + '_data' + ext + '.fits'
            outfile = mockdatoutfile_base.format(phase=phase,real=real)
            print('\nWriting out mock data catalogue...')
            print(bgs_clust_path+outfile)
            hdulist.writeto(bgs_clust_path+outfile)



main()