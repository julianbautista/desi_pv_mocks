"""
Tools for cosmological distance conversions and (redshift-space) power
spectrum estimation from galaxy/momentum catalogues.

Written by Fei Qin 2026. Modified by Claude. Updated by Julian Bautista.

This module is organized into two sections:

    Sec 1. Cosmological distance utilities, peculiar-velocity estimators,
           and galaxy number-density estimators.
    Sec 2. Power spectrum estimation (density, momentum, and density-momentum
           cross power spectra), including Yamamoto-style multipole (l=0..4)
           estimators on a Cartesian FFT grid.

Note: the two power-spectrum "workhorse" functions, `estimate_power_spectrum` (for real
survey/random catalogues) and `estimate_power_spectrum_simbox` (for periodic simulation
boxes), share the majority of their multipole-projection logic. They are
kept separate here since they were not refactored into a shared helper,
but see the docstring notes on each for how they differ.
"""

import sys
import pandas as pd
import numpy as np
import scipy as sp
import scipy.fft
from scipy import integrate
from scipy.interpolate import splev

LIGHT_SPEED = 299792.458  # km/s


def save_pickle(file_dir, x, pic=4):
    """Pickle-dump `x` to `file_dir` using protocol `pic`."""
    import pickle
    with open(file_dir, 'wb') as f:
        pickle.dump(x, f, protocol=pic)
    return []


###############################################################################
######                                                                  ######
######   Sec 1. Functions used to calculate cosmological distances      ######
######                                                                  ######
###############################################################################

def hubble_parameter(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap):
    """Dimensionless Hubble parameter E(z) = H(z) / H0.

    Supports a CPL-style dark-energy equation of state
    w(a) = w0 + wa * (ap - a), with curvature inferred from
    omega_k = 1 - omega_m - omega_lambda - omega_rad.
    """
    fz = (1.0 + redshift) ** (3 * (1.0 + w0 + wa * ap)) * np.exp(-3 * wa * (redshift / (1.0 + redshift)))
    omega_k = 1.0 - omega_m - omega_lambda - omega_rad
    return np.sqrt(
        omega_rad * (1.0 + redshift) ** 4
        + omega_m * (1.0 + redshift) ** 3
        + omega_k * (1.0 + redshift) ** 2
        + omega_lambda * fz
    )


def comoving_distance_integrand(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap):
    """Integrand 1/E(z) for the comoving distance integral."""
    return 1.0 / hubble_parameter(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap)


def comoving_distance(redshift, omega_m, omega_lambda, omega_rad, hubble_constant, w0, wa, ap):
    """Comoving distance to `redshift`, in Mpc (assuming flat/curved FRW)."""
    integral = integrate.quad(
        comoving_distance_integrand, 0.0, redshift,
        args=(omega_m, omega_lambda, omega_rad, w0, wa, ap),
    )[0]
    return (LIGHT_SPEED / hubble_constant) * integral


def build_distance_spline(omega_m, omega_lambda, hubble_constant, nbin=10000, redmax=2.5):
    """Build a spline of comoving distance vs. redshift (flat LCDM, w=-1).

    Tabulates `nbin` points between z=0 and z=redmax and returns a
    B-spline representation suitable for `scipy.interpolate.splev`.
    """
    red = np.array([j * redmax / nbin for j in range(nbin)])
    dist = np.array([
        comoving_distance(z, omega_m, omega_lambda, 0.0, hubble_constant, -1.0, 0.0, 0.0) for z in red
    ])
    return sp.interpolate.splrep(red, dist, s=0)


def redshift_to_distance(xdt, omega_m, omega_lambda, hubble_constant, nbin=10000, redmax=2.5):
    """Convert redshift(s) `xdt` to comoving distance via a cached spline."""
    spl_fun = build_distance_spline(omega_m, omega_lambda, hubble_constant, nbin, redmax)
    return splev(xdt, spl_fun)


def sky_to_cartesian(ra, dec, rsft, omega_m, omega_lambda, hubble_constant, nbin=3000, redmax=2.5):
    """Convert (RA, Dec, redshift) [degrees, degrees, z] to Cartesian (X, Y, Z) in Mpc."""
    disz = redshift_to_distance(rsft, omega_m, omega_lambda, hubble_constant, nbin, redmax)
    dec_rad = dec / 180.0 * np.pi
    ra_rad = ra / 180.0 * np.pi
    x = disz * np.cos(dec_rad) * np.cos(ra_rad)
    y = disz * np.cos(dec_rad) * np.sin(ra_rad)
    z = disz * np.sin(dec_rad)
    return x, y, z


# --- Peculiar velocity estimators -------------------------------------------
# Following Watkins & Feldman 2015: https://arxiv.org/abs/1411.6665

def velocity_to_logdist(vpec, rsf, omega_m):
    """Convert a peculiar velocity `vpec` to a log-distance-ratio estimate."""
    deccel = 3.0 * omega_m / 2.0 - 1.0
    v_mod = rsf * LIGHT_SPEED * (
        1.0 + 0.5 * (1.0 - deccel) * rsf
        - (2.0 - deccel - 3.0 * deccel * deccel) * rsf * rsf / 6.0
    )
    return vpec / (np.log(10.0) * v_mod / (1.0 + v_mod / LIGHT_SPEED))


def logdist_to_velocity(logd, rsf, omega_m):
    """Convert a log-distance-ratio `Logd` to a peculiar velocity estimate."""
    deccel = 3.0 * omega_m / 2.0 - 1.0
    v_mod = rsf * LIGHT_SPEED * (
        1.0 + 0.5 * (1.0 - deccel) * rsf
        - (2.0 - deccel - 3.0 * deccel * deccel) * rsf * rsf / 6.0
    )
    return np.log(10.0) * v_mod / (1.0 + v_mod / LIGHT_SPEED) * logd


def random_peculiar_velocities(logd, cz, czR, n_cz_bin, omega_m):
    """Generate random peculiar velocities for a random catalogue.

    Estimates the peculiar-velocity error as a function of redshift from
    the data (`cz`, `logd`) by binning into `NczBin` bins, fitting a linear
    trend, and evaluating it at the random catalogue's redshifts `czR`.
    Random Gaussian peculiar velocities are then drawn with that scatter.
    """
    pv = logdist_to_velocity(logd, cz / LIGHT_SPEED, omega_m)
    Num, bin_edges = np.histogram(cz, n_cz_bin)
    ev = []
    for i in range(len(Num)):
        ind = (cz >= bin_edges[i]) & (cz <= bin_edges[i + 1])
        ev.append(np.std(pv[ind]))
    ev = np.array(ev)

    x = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
    k, b = np.polyfit(x, ev, deg=1)
    epvR = k * czR + b

    np.random.seed(326)
    pvR = np.random.normal(loc=0.0, scale=epvR, size=len(epvR))
    return pvR, epvR


# --- Galaxy number density ---------------------------------------------------

def radial_number_density(delt_z, zhR, weit, survey_area, omega_m, omega_lambda, hubble_constant, nbin=10000, redmax=2.3):
    """Estimate n(z) (radial-selection number density) for a random catalogue.

    Bins the random catalogue `zhR` (optionally weighted by `weit`) in
    redshift shells of width `delt_z`, converts shell volumes using the
    given cosmology, and returns a smoothed n(z) evaluated at `zhR`.

    `survey_area` is expected in units of steradians/pi (i.e. solid angle
    fraction such that volume = survey_area * (Rout^3 - Rin^3) / 3).
    """
    Nz = int(np.abs(np.max(zhR) - np.min(zhR)) / delt_z)
    bin_edges_z = np.array([np.min(zhR) + i * delt_z for i in range(Nz + 2)])

    if not isinstance(weit, np.ndarray):
        weit = np.ones(len(zhR)) * weit * 1.0

    counts, bin_edges = np.histogram(zhR, bins=bin_edges_z, weights=weit)
    dist_edges = redshift_to_distance(bin_edges, omega_m, omega_lambda, hubble_constant, nbin, redmax)

    spl_nbar = np.zeros(len(dist_edges) - 1)
    for i in range(len(dist_edges) - 1):
        r_in, r_out = dist_edges[i], dist_edges[i + 1]
        volume = survey_area * (r_out ** 3 - r_in ** 3) / 3.0
        spl_nbar[i] = counts[i] / volume

    if len(spl_nbar) == 1:
        return np.ones(len(zhR)) * spl_nbar

    bin_centers = dist_edges[:-1] + np.abs(dist_edges[1] - dist_edges[2]) / 2.0
    if 2 <= len(spl_nbar) <= 3:
        nbar_spline = sp.interpolate.splrep(bin_centers, spl_nbar, k=len(spl_nbar) - 1)
    else:
        nbar_spline = sp.interpolate.splrep(bin_centers, spl_nbar, s=0)

    dist_at_points = redshift_to_distance(zhR, omega_m, omega_lambda, hubble_constant, nbin=10000, redmax=0.3)
    return sp.interpolate.splev(dist_at_points, nbar_spline, der=0)


def assign_radial_number_density(rsf, zmR, nbR_Snorm):
    """Interpolate a redshift-binned n(z) (`nbR_Snorm` at redshifts `zmR`) onto `rsf`."""
    ind = np.argsort(zmR)
    return np.interp(rsf, zmR[ind], nbR_Snorm[ind])


def grid_number_density(ra, dec, rsf, weit, nx, ny, nz, omega_m, omega_lambda, hubble_constant):
    """Estimate a 3D number-density grid from a random catalogue and sample it.

    Builds an (nx, ny, nz) Cartesian grid covering a cube of side
    2 * max(comoving distance), bins the (RA, Dec, z) catalogue onto it,
    and returns the number density sampled at each input point plus the
    grid itself (for later use with `nbarGAsign_Fun`).
    """
    rsfmax = np.max(rsf)
    distmax = redshift_to_distance(rsfmax, omega_m, omega_lambda, hubble_constant)
    lx = ly = lz = 2.0 * distmax
    x0 = y0 = z0 = distmax
    dx, dy, dz = lx / nx, ly / ny, lz / nz
    dvol = dx * dy * dz

    xlims = np.linspace(0.0, lx, nx + 1) - x0
    ylims = np.linspace(0.0, ly, ny + 1) - y0
    zlims = np.linspace(0.0, lz, nz + 1) - z0

    ndat = len(ra) * 1.0
    x, y, z = sky_to_cartesian(ra, dec, rsf, omega_m, omega_lambda, hubble_constant)

    pts = np.vstack([x + x0, y + y0, z + z0]).transpose()
    grid_range = ((0.0, lx), (0.0, ly), (0.0, lz))
    if not isinstance(weit, np.ndarray):
        Num, _ = np.histogramdd(pts, bins=(nx, ny, nz), range=grid_range)
    else:
        Num, _ = np.histogramdd(pts, bins=(nx, ny, nz), range=grid_range, weights=weit)

    ndensgrid = (ndat / dvol) * (Num / np.sum(Num))

    ix = np.digitize(x, xlims) - 1
    iy = np.digitize(y, ylims) - 1
    iz = np.digitize(z, zlims) - 1
    nb = ndensgrid[ix, iy, iz]

    return nb, ndensgrid, xlims, ylims, zlims


def assign_grid_number_density(ra, dec, rsf, nbRgrid_norm, xlims, ylims, zlims, omega_m, omega_lambda, hubble_constant):
    """Sample a precomputed number-density grid (`grid_number_density`) at new points."""
    x, y, z = sky_to_cartesian(ra, dec, rsf, omega_m, omega_lambda, hubble_constant)
    ix = np.digitize(x, xlims) - 1
    iy = np.digitize(y, ylims) - 1
    iz = np.digitize(z, zlims) - 1
    return nbRgrid_norm[ix, iy, iz]

#########################     End of Sec 1.    ###############################
###############################################################################


###############################################################################
#####                                                                    #####
#####  Sec 2. Functions used to calculate the measured power spectrum    #####
#####                                                                    #####
###############################################################################

def grid_correction(dx, dy, dz, fx, fy, fz):
    """NGP/CIC-style grid deconvolution correction (inverse sinc window).

    Given grid spacings (dx, dy, dz) and frequency arrays (fx, fy, fz),
    returns 1 / (sinc(fx*dx*pi) * sinc(fy*dy*pi) * sinc(fz*dz*pi)) to
    correct for the mass-assignment window function.
    """
    sincx, sincy, sincz = np.ones(len(fx)), np.ones(len(fy)), np.ones(len(fz))
    indx, indy, indz = (fx != 0.0), (fy != 0.0), (fz != 0.0)
    sincx[indx] = np.sin(fx[indx] * dx * np.pi) / (fx[indx] * dx * np.pi)
    sincy[indy] = np.sin(fy[indy] * dy * np.pi) / (fy[indy] * dy * np.pi)
    sincz[indz] = np.sin(fz[indz] * dz * np.pi) / (fz[indz] * dz * np.pi)
    return 1.0 / (sincx * sincy * sincz)


def estimate_power_spectrum(kmin, kmax, nk, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax,
              longi=0, lati=0, rsf=0, nb=0, fkp=1600.,
              longir=0, latir=0, rsfr=0, nbr_norm=0,
              longiv=0, lativ=0, rsfv=0, vp=0, evp=0, nbmom=0, fkp_v=5. * 10. ** 9,
              omega_m=0.3, omega_lambda=0.7, hubble_constant=100., sigv=300.0,
              bulk_vel=np.array([0., 0., 0.]),
              ps_type='mom', ps_multi='yes', file_dir='PSestDir', wd=1., wdR=1., wv=1.):
    """Estimate the (Yamamoto-style) power spectrum multipoles l=0..4.

    Observer is assumed to sit at the Cartesian origin [0, 0, 0].

    Parameters
    ----------
    ps_type : {'den', 'mom', 'crs'}
        'den' = galaxy density power spectrum, 'mom' = momentum/peculiar
        velocity power spectrum, 'crs' = density-momentum cross spectrum.
    ps_multi : {'yes', 'no', 'all'}
        'no' computes only the monopole; 'yes' computes the non-vanishing
        multipoles for the given ps_type (l=0,2,4 for 'den'/'mom'; l=1,3
        for 'crs'); 'all' computes every multipole l=0..4 regardless of
        whether it is expected to vanish.
    longi, lati, rsf, nb : array-like
        Galaxy catalogue: RA, Dec, redshift, and number density (for the
        'den'/'crs' spectra).
    longir, latir, rsfr, nbr_norm : array-like
        Random catalogue counterpart of the above.
    longiv, lativ, rsfv, vp, evp, nbmom : array-like
        Peculiar-velocity tracer catalogue: RA, Dec, redshift, peculiar
        velocity, velocity error, and number density (for the 'mom'/'crs'
        spectra).
    bulk_vel : array-like, shape (3,)
        Bulk flow velocity (Cartesian) to subtract from the velocity field
        before estimating the momentum power spectrum.
    file_dir : str
        Output path for the resulting power spectrum table.

    Returns
    -------
    file_dir : str
        Path to the written power spectrum table.
    weifkp : list of arrays
        fkp-type weights used, depending on ps_type.
    Norm : float
        Normalization factor applied to the power spectrum.
    """
    # -- 1. Initial checks and settings --------------------------------------
    if nx % 2 != 0:
        sys.exit('\n Error: nx should be set to an even number.\n')
    if ny % 2 != 0:
        sys.exit('\n Error: ny should be set to an even number.\n')
    if nz % 2 != 0:
        sys.exit('\n Error: nz should be set to an even number.\n')
    #print(' Please make sure that in the catalogue, the observer is at the '
    #      'coordinate origin [0, 0, 0].\n')

    if ps_type in ('den', 'crs'):
        print(' ', ps_type, ' Ngal=', len(longi))
    if ps_type in ('mom', 'crs'):
        print(' ', ps_type, ' Npv =', len(vp))
    #print()

    lx, ly, lz = np.abs(xmax - xmin), np.abs(ymax - ymin), np.abs(zmax - zmin)
    dk = (kmax - kmin) / nk
    n_modes = np.zeros(nk, dtype=int)

    p0 = np.zeros(nk)
    if ps_multi != 'no':
        p1 = np.zeros(nk)
        p2 = np.zeros(nk)
        p3 = np.zeros(nk)
        p4 = np.zeros(nk)

    # -- Cartesian coordinates and fkp weights -------------------------------
    if ps_type in ('den', 'crs'):
        nbr = nbr_norm
        x, y, z = sky_to_cartesian(longi, lati, rsf, omega_m, omega_lambda, hubble_constant)
        xr, yr, zr = sky_to_cartesian(longir, latir, rsfr, omega_m, omega_lambda, hubble_constant)
        w = 1.0 / (1.0 + nb * fkp)
        wr = 1.0 / (1.0 + nbr * fkp)
    if ps_type in ('mom', 'crs'):
        xpv, ypv, zpv = sky_to_cartesian(longiv, lativ, rsfv, omega_m, omega_lambda, hubble_constant)
        wmom = 1.0 / (evp ** 2 + sigv ** 2 + nbmom * fkp_v)
        wrho = 1.0 / (1.0 + nbmom * fkp)
        if fkp_v == 0:
            wmom = np.ones(len(xpv))

    # -- Remove bulk flow velocity -------------------------------------------
    if ps_type in ('mom', 'crs'):
        if bulk_vel[0] != 0.0 or bulk_vel[1] != 0.0 or bulk_vel[2] != 0.0:
            print('  Bulk velocity removed.\n')
            dist = np.sqrt(xpv ** 2 + ypv ** 2 + zpv ** 2)
            hatx, haty, hatz = xpv / dist, ypv / dist, zpv / dist
            vp = vp - (bulk_vel[0] * hatx + bulk_vel[1] * haty + bulk_vel[2] * hatz)

    # -- 2. Grid the catalogue(s) ---------------------------------------------
    dx, dy, dz = (xmax - xmin) / nx, (ymax - ymin) / ny, (zmax - zmin) / nz
    grid_range = ((-lx / 2., lx / 2.), (-ly / 2., ly / 2.), (-lz / 2., lz / 2.))

    if ps_type in ('den', 'crs'):
        grid, _ = np.histogramdd(np.vstack([x, y, z]).transpose(), bins=(nx, ny, nz),
                                 range=grid_range, weights=w * wd)
        alpha = np.sum(w) / np.sum(wr)
        tmp, _ = np.histogramdd(np.vstack([xr, yr, zr]).transpose(), bins=(nx, ny, nz),
                                 range=grid_range, weights=alpha * wr * wdR)
        grid = grid - tmp
    if ps_type in ('mom', 'crs'):
        if ps_type == 'mom':
            grid, _ = np.histogramdd(np.vstack([xpv, ypv, zpv]).transpose(), bins=(nx, ny, nz),
                                     range=grid_range, weights=wmom * vp * wv)
        if ps_type == 'crs':
            mom_grid, _ = np.histogramdd(np.vstack([xpv, ypv, zpv]).transpose(), bins=(nx, ny, nz),
                                        range=grid_range, weights=wmom * vp * wv)

    # -- Nyquist frequency of the grid ---------------------------------------
    fx_nqu, fy_nqu, fz_nqu = np.pi / dx, np.pi / dy, np.pi / dz
    min_nqu = min(fx_nqu, fy_nqu, fz_nqu)

    # -- Grid-cell positions (for multipole projection) ----------------------
    if ps_multi != 'no':
        grid_x = np.linspace(xmin, xmax, nx + 1)[:-1]
        grid_y = np.linspace(ymin, ymax, ny + 1)[:-1]
        grid_z = np.linspace(zmin, zmax, nz + 1)[:-1]
        grid_mesh = np.meshgrid(grid_x, grid_y, grid_z)
        pos_vect = np.zeros((nx * ny * nz, 3))
        # vect = [vect_Z, vect_x, vect_y]; Z-component is stored first.
        pos_vect[:, 1] = grid_mesh[0].flatten()
        pos_vect[:, 2] = grid_mesh[1].flatten()
        pos_vect[:, 0] = grid_mesh[2].flatten()
        vec_len_sq = pos_vect[:, 0] ** 2 + pos_vect[:, 1] ** 2 + pos_vect[:, 2] ** 2
        vec_len_sq[vec_len_sq == 0.0] = 1.0

    # -- k-space grid ----------------------------------------------------------
    k_vect = np.zeros((nx * ny * nz // 2, 3))
    k_vals_x = np.concatenate((
        np.linspace(0., 2. * np.pi * (nx // 2) / (nx * dx), nx // 2 + 1),
        np.linspace(2. * np.pi * (nx // 2 + 1 - nx) / (nx * dx), 0., nx // 2)[:-1],
    ))
    k_vals_y = np.concatenate((
        np.linspace(0., 2. * np.pi * (ny // 2) / (ny * dy), ny // 2 + 1),
        np.linspace(2. * np.pi * (ny // 2 + 1 - ny) / (ny * dy), 0., ny // 2)[:-1],
    ))
    k_vals_z = np.linspace(0., 2. * np.pi * (nz // 2 - 1) / (nz * dz), nz // 2)
    k_mesh = np.meshgrid(k_vals_x, k_vals_y, k_vals_z)
    # kvect = [kvect_Z, kvect_x, kvect_y]; Z-component is stored first.
    k_vect[:, 1] = k_mesh[0].flatten()
    k_vect[:, 2] = k_mesh[1].flatten()
    k_vect[:, 0] = k_mesh[2].flatten()

    grid_cor = grid_correction(dx, dy, dz, k_vect[:, 1] / (2. * np.pi),
                             k_vect[:, 2] / (2. * np.pi), k_vect[:, 0] / (2. * np.pi))
    k_mag = np.sqrt(k_vect[:, 0] ** 2 + k_vect[:, 1] ** 2 + k_vect[:, 2] ** 2)

    ik_bin_idx = np.array((k_mag - kmin) / dk, dtype=int)
    ind_nqu = np.where((ik_bin_idx >= 0) & (ik_bin_idx < nk) & (k_mag < 0.5 * min_nqu))[0]
    ind_use_list = [ind_nqu[np.where(ik_bin_idx[ind_nqu] == ikbin)[0]] for ikbin in range(nk)]

    # -- 3. Normalization factor and shot noise -------------------------------
    if ps_type == 'den':
        pnoise = np.sum(w * w * wd * wd)
        pnoise_r = alpha * alpha * np.sum(wr * wr * wdR * wdR)
        norm = np.sum(nb * w * w * wd * wd)
        psn = (pnoise + pnoise_r) / norm
    if ps_type == 'mom':
        pvnoise = np.sum(wmom * wmom * vp * vp * wv * wv)
        norm = np.sum(nbmom * wmom * wmom * wv * wv)
        psn = pvnoise / norm
    if ps_type == 'crs':
        pnoise_c = np.sum(wrho * wmom * vp * wv)
        norm = np.sqrt(np.sum(nb * w * w * wd * wd)) * np.sqrt(np.sum(nbmom * wmom * wmom * wv * wv))
        psn = 0.0

    # -- 4. l=0 ----------------------------------------------------------------
    grid_fft = scipy.fft.fftn(grid)[:, :, :nz // 2].flatten()
    if ps_type == 'crs':
        mom_grid_fft = scipy.fft.fftn(mom_grid)[:, :, :nz // 2].flatten()
    if ps_type == 'crs' and ps_multi == 'all':
        tmp1 = (0.5 * (np.real(grid_fft) * np.imag(mom_grid_fft) - np.imag(grid_fft) * np.real(mom_grid_fft)
                        + np.real(mom_grid_fft) * np.imag(grid_fft) - np.imag(mom_grid_fft) * np.real(grid_fft))
                - pnoise_c) * grid_cor * grid_cor
    if ps_type == 'den':
        tmp2 = (np.real(grid_fft) ** 2 + np.imag(grid_fft) ** 2 - (pnoise + pnoise_r)) * grid_cor * grid_cor
    if ps_type == 'mom':
        tmp3 = (np.real(grid_fft) ** 2 + np.imag(grid_fft) ** 2 - pvnoise) * grid_cor * grid_cor
    if ps_multi in ('yes', 'all'):
        if ps_type != 'crs':
            tmp4 = (np.real(grid_fft) ** 2 + np.imag(grid_fft) ** 2) * grid_cor * grid_cor
        if ps_type == 'crs' and ps_multi == 'all':
            tmp5 = 0.5 * (np.real(grid_fft) * np.imag(mom_grid_fft) - np.imag(grid_fft) * np.real(mom_grid_fft)
                           + np.real(mom_grid_fft) * np.imag(grid_fft) - np.imag(mom_grid_fft) * np.real(grid_fft)
                           ) * grid_cor * grid_cor

    for ikbin in range(nk):
        ind_use = ind_use_list[ikbin]
        if ps_type == 'crs' and ps_multi == 'all':
            p0[ikbin] = np.sum(tmp1[ind_use])
        if ps_type == 'den':
            p0[ikbin] = np.sum(tmp2[ind_use])
        if ps_type == 'mom':
            p0[ikbin] = np.sum(tmp3[ind_use])
        n_modes[ikbin] = len(ik_bin_idx[ind_use])

        if ps_multi in ('yes', 'all'):
            if ps_type != 'crs':
                tmp = np.sum(tmp4[ind_use])
                p2[ikbin] = -0.5 * tmp
                p4[ikbin] = 0.375 * tmp
            if ps_type == 'crs' and ps_multi == 'all':
                tmp = np.sum(tmp5[ind_use])
                p2[ikbin] = -0.5 * tmp
                p4[ikbin] = 0.375 * tmp

    # Save the un-transformed density/velocity grid and its FFT for reuse below.
    grid_saved = grid.copy()
    grid = np.zeros((nx, ny, nz))
    grid_fft_saved = grid_fft.copy()
    grid_fft = 0.
    tmp1 = tmp2 = tmp3 = tmp4 = tmp5 = 0.
    if ps_type == 'crs':
        mom_grid_saved = mom_grid.copy()
        mom_grid = np.zeros((nx, ny, nz))
        mom_grid_fft_saved = mom_grid_fft.copy()
        mom_grid_fft = 0.

    # ============  Power spectrum multipoles (skipped if ps_multi == 'no') ====
    if ps_multi != 'no':
        # -- 5. l=1 -------------------------------------------------------------
        if ps_type == 'crs' or ps_multi == 'all':
            for ii in range(3):
                if ps_type != 'crs':
                    grid = grid_saved * (pos_vect[:, ii] / np.sqrt(vec_len_sq)).reshape(nx, ny, nz)
                if ps_type == 'crs':
                    grid = grid_saved * (pos_vect[:, ii] / np.sqrt(vec_len_sq)).reshape(nx, ny, nz)
                    mom_grid = mom_grid_saved * (pos_vect[:, ii] / np.sqrt(vec_len_sq)).reshape(nx, ny, nz)

                if ps_type != 'crs' and ps_multi == 'all':
                    grid_fft = scipy.fft.fftn(grid)[:, :, :nz // 2].flatten()
                if ps_type == 'crs':
                    grid_fft = scipy.fft.fftn(grid)[:, :, :nz // 2].flatten()
                    mom_grid_fft = scipy.fft.fftn(mom_grid)[:, :, :nz // 2].flatten()

                if ps_type != 'crs' and ps_multi == 'all':
                    tmp1 = k_vect[:, ii] * ((np.real(grid_fft_saved) * np.imag(grid_fft)
                                             - np.imag(grid_fft_saved) * np.real(grid_fft)) * grid_cor * grid_cor)
                if ps_type == 'crs':
                    tmp2 = k_vect[:, ii] * (0.5 * (-np.imag(grid_fft) * np.real(mom_grid_fft_saved)
                                                   + np.real(grid_fft) * np.imag(mom_grid_fft_saved)
                                                   + np.imag(mom_grid_fft) * np.real(grid_fft_saved)
                                                   - np.real(mom_grid_fft) * np.imag(grid_fft_saved)) * grid_cor * grid_cor)

                if ps_type == 'crs' or ps_multi == 'all':
                    for ikbin in range(nk):
                        kprefac = 1.0
                        ind_use = ind_use_list[ikbin]
                        k_sel = k_mag[ind_use]
                        valid_k = (k_sel > 0)
                        if ps_type != 'crs' and ps_multi == 'all':
                            p1[ikbin] += np.sum((1.0 * kprefac * tmp1[ind_use])[valid_k] / k_sel[valid_k])
                            p3[ikbin] += np.sum((-1.5 * kprefac * tmp1[ind_use])[valid_k] / k_sel[valid_k])
                        if ps_type == 'crs':
                            p1[ikbin] += np.sum((1.0 * kprefac * tmp2[ind_use])[valid_k] / k_sel[valid_k])
                            p3[ikbin] += np.sum((-1.5 * kprefac * tmp2[ind_use])[valid_k] / k_sel[valid_k])

            tmp1 = tmp2 = 0.
            if ps_type != 'crs' and ps_multi == 'all':
                grid = np.zeros((nx, ny, nz))
                grid_fft = 0.
            if ps_type == 'crs':
                mom_grid = np.zeros((nx, ny, nz))
                grid = np.zeros((nx, ny, nz))
                grid_fft = 0.
                mom_grid_fft = 0.

        # -- 6. l=2 ---------------------------------------------------------------
        if ps_type != 'crs' or ps_multi == 'all':
            for ii in range(3):
                for jj in range(3):
                    if jj < ii:
                        continue
                    if ps_type != 'crs':
                        grid = grid_saved * (pos_vect[:, ii] * pos_vect[:, jj] / vec_len_sq).reshape(nx, ny, nz)
                    if ps_type == 'crs':
                        grid = grid_saved * (pos_vect[:, ii] * pos_vect[:, jj] / vec_len_sq).reshape(nx, ny, nz)
                        mom_grid = mom_grid_saved * (pos_vect[:, ii] * pos_vect[:, jj] / vec_len_sq).reshape(nx, ny, nz)

                    grid_fft = scipy.fft.fftn(grid)[:, :, :nz // 2].flatten()
                    if ps_type == 'crs' and ps_multi == 'all':
                        mom_grid_fft = scipy.fft.fftn(mom_grid)[:, :, :nz // 2].flatten()

                    if ps_type != 'crs':
                        tmp1 = k_vect[:, ii] * k_vect[:, jj] * (
                            (np.real(grid_fft) * np.real(grid_fft_saved)
                             + np.imag(grid_fft) * np.imag(grid_fft_saved)) * grid_cor * grid_cor)
                    if ps_type == 'crs' and ps_multi == 'all':
                        tmp2 = k_vect[:, ii] * k_vect[:, jj] * (
                            0.5 * (np.real(grid_fft) * np.imag(mom_grid_fft_saved)
                                   - np.imag(grid_fft) * np.real(mom_grid_fft_saved)
                                   + np.real(mom_grid_fft) * np.imag(grid_fft_saved)
                                   - np.imag(mom_grid_fft) * np.real(grid_fft_saved)) * grid_cor * grid_cor)

                    for ikbin in range(nk):
                        kprefac = 2.0 if ii != jj else 1.0
                        ind_use = ind_use_list[ikbin]
                        k_sel = k_mag[ind_use]
                        valid_k = (k_sel > 0)
                        if ps_type != 'crs':
                            p2[ikbin] += np.sum((1.5 * kprefac * tmp1[ind_use])[valid_k] / k_sel[valid_k] ** 2)
                            p4[ikbin] += np.sum((-3.75 * kprefac * tmp1[ind_use])[valid_k] / k_sel[valid_k] ** 2)
                        if ps_type == 'crs' and ps_multi == 'all':
                            p2[ikbin] += np.sum((1.5 * kprefac * tmp2[ind_use])[valid_k] / k_sel[valid_k] ** 2)
                            p4[ikbin] += np.sum((-3.75 * kprefac * tmp2[ind_use])[valid_k] / k_sel[valid_k] ** 2)

            grid = np.zeros((nx, ny, nz))
            grid_fft = 0.
            if ps_type == 'crs':
                mom_grid = np.zeros((nx, ny, nz))
                grid_fft = 0.
                mom_grid_fft = 0.
            tmp1 = tmp2 = 0.

        # -- 7. l=3 -----------------------------------------------------------------
        # Only the 10 index-permutation classes with jusm not in the skip-set below
        # correspond to distinct (ii, jj, kk) triples for a symmetric rank-3 tensor.
        _L3_SKIP = {3, 4, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23}
        if ps_type == 'crs' or ps_multi == 'all':
            combo_idx = 0
            for ii in range(3):
                for jj in range(3):
                    for kk in range(3):
                        if combo_idx not in _L3_SKIP:
                            if ps_type != 'crs':
                                grid = grid_saved * (pos_vect[:, ii] * pos_vect[:, jj] * pos_vect[:, kk]
                                                   / (vec_len_sq * np.sqrt(vec_len_sq))).reshape(nx, ny, nz)
                            if ps_type == 'crs':
                                grid = grid_saved * (pos_vect[:, ii] * pos_vect[:, jj] * pos_vect[:, kk]
                                                   / (vec_len_sq * np.sqrt(vec_len_sq))).reshape(nx, ny, nz)
                                mom_grid = mom_grid_saved * (pos_vect[:, ii] * pos_vect[:, jj] * pos_vect[:, kk]
                                                         / (vec_len_sq * np.sqrt(vec_len_sq))).reshape(nx, ny, nz)

                            if ps_type != 'crs' and ps_multi == 'all':
                                grid_fft = scipy.fft.fftn(grid)[:, :, :nz // 2].flatten()
                            if ps_type == 'crs':
                                grid_fft = scipy.fft.fftn(grid)[:, :, :nz // 2].flatten()
                                mom_grid_fft = scipy.fft.fftn(mom_grid)[:, :, :nz // 2].flatten()

                            if ps_type != 'crs' and ps_multi == 'all':
                                tmp1 = k_vect[:, ii] * k_vect[:, jj] * k_vect[:, kk] * (
                                    (np.real(grid_fft_saved) * np.imag(grid_fft)
                                     - np.imag(grid_fft_saved) * np.real(grid_fft)) * grid_cor * grid_cor)
                            if ps_type == 'crs':
                                tmp2 = k_vect[:, ii] * k_vect[:, jj] * k_vect[:, kk] * (
                                    0.5 * (-np.imag(grid_fft) * np.real(mom_grid_fft_saved)
                                           + np.real(grid_fft) * np.imag(mom_grid_fft_saved)
                                           + np.imag(mom_grid_fft) * np.real(grid_fft_saved)
                                           - np.real(mom_grid_fft) * np.imag(grid_fft_saved)) * grid_cor * grid_cor)

                            for ikbin in range(nk):
                                if ii != kk:
                                    kprefac = 6.0 if ii != jj else 3.0
                                else:
                                    kprefac = 1.0
                                ind_use = ind_use_list[ikbin]
                                k_sel = k_mag[ind_use]
                                valid_k = (k_sel > 0)
                                if ps_type != 'crs' and ps_multi == 'all':
                                    p3[ikbin] += np.sum((2.5 * kprefac * tmp1[ind_use])[valid_k] / k_sel[valid_k] ** 3)
                                if ps_type == 'crs':
                                    p3[ikbin] += np.sum((2.5 * kprefac * tmp2[ind_use])[valid_k] / k_sel[valid_k] ** 3)
                        combo_idx += 1

            if ps_type != 'crs' and ps_multi == 'all':
                grid = np.zeros((nx, ny, nz))
                grid_fft = 0.
            if ps_type == 'crs':
                mom_grid = np.zeros((nx, ny, nz))
                grid = np.zeros((nx, ny, nz))
                grid_fft = 0.
                mom_grid_fft = 0.
            tmp1 = tmp2 = 0.

        # -- 8. l=4 -----------------------------------------------------------------
        _L4_SKIP = {4, 7, 8, 10, 11, 16, 17, 19, 21, 22, 23, 24}
        if ps_type != 'crs' or ps_multi == 'all':
            combo_idx = 1
            for ii in range(3):
                for jj in range(3):
                    for kk in range(3):
                        if combo_idx not in _L4_SKIP:
                            if ps_type != 'crs':
                                grid = grid_saved * (pos_vect[:, ii] * pos_vect[:, ii] * pos_vect[:, jj] * pos_vect[:, kk]
                                                   / vec_len_sq ** 2).reshape(nx, ny, nz)
                            if ps_type == 'crs':
                                grid = grid_saved * (pos_vect[:, ii] * pos_vect[:, ii] * pos_vect[:, jj] * pos_vect[:, kk]
                                                   / vec_len_sq ** 2).reshape(nx, ny, nz)
                                mom_grid = mom_grid_saved * (pos_vect[:, ii] * pos_vect[:, ii] * pos_vect[:, jj] * pos_vect[:, kk]
                                                         / vec_len_sq ** 2).reshape(nx, ny, nz)

                            grid_fft = scipy.fft.fftn(grid)[:, :, :nz // 2].flatten()
                            if ps_type == 'crs' and ps_multi == 'all':
                                mom_grid_fft = scipy.fft.fftn(mom_grid)[:, :, :nz // 2].flatten()

                            if ps_type != 'crs':
                                tmp1 = k_vect[:, ii] * k_vect[:, ii] * k_vect[:, jj] * k_vect[:, kk] * (
                                    (np.real(grid_fft) * np.real(grid_fft_saved)
                                     + np.imag(grid_fft) * np.imag(grid_fft_saved)) * grid_cor * grid_cor)
                            if ps_type == 'crs' and ps_multi == 'all':
                                tmp2 = k_vect[:, ii] * k_vect[:, ii] * k_vect[:, jj] * k_vect[:, kk] * (
                                    0.5 * (np.real(grid_fft) * np.imag(mom_grid_fft_saved)
                                           - np.imag(grid_fft) * np.real(mom_grid_fft_saved)
                                           + np.real(mom_grid_fft) * np.imag(grid_fft_saved)
                                           - np.imag(mom_grid_fft) * np.real(grid_fft_saved)) * grid_cor * grid_cor)

                            for ikbin in range(nk):
                                if ii == jj:
                                    kprefac = 4.0 if ii != kk else 1.0
                                else:
                                    kprefac = 6.0 if jj == kk else 12.0
                                ind_use = ind_use_list[ikbin]
                                k_sel = k_mag[ind_use]
                                valid_k = (k_sel > 0)
                                if ps_type != 'crs':
                                    p4[ikbin] += np.sum((4.375 * kprefac * tmp1[ind_use])[valid_k] / k_sel[valid_k] ** 4)
                                if ps_type == 'crs' and ps_multi == 'all':
                                    p4[ikbin] += np.sum((4.375 * kprefac * tmp2[ind_use])[valid_k] / k_sel[valid_k] ** 4)
                        combo_idx += 1
            tmp1 = tmp2 = 0.
    # ====================  End of power-spectrum multipoles  ===================

    # -- 9. Normalize and save -------------------------------------------------
    for ik in range(nk):
        if n_modes[ik] > 0.0:
            p0[ik] = p0[ik] / (n_modes[ik] * norm)
            if ps_multi in ('yes', 'all'):
                p1[ik] = p1[ik] * 3.0 / (n_modes[ik] * norm)
                p2[ik] = p2[ik] * 5.0 / (n_modes[ik] * norm)
                p3[ik] = p3[ik] * 7.0 / (n_modes[ik] * norm)
                p4[ik] = p4[ik] * 9.0 / (n_modes[ik] * norm)

    n_modes[-1] = 0  # force the last k-bin to zero (edge effects)

    '''
    with open(file_dir, 'w') as outfile:
        outfile.write("# No.          k            P0            P1           P2           P3"
                       "           P4            Nk           Norm           SNois  \n")
        for i in range(nk):
            if i == nk - 1:
                outfile.write("  %7d     %17.10lf     %7d     %7d     %7d     %7d     %7d     %7d     %7d     %7d \n"
                               % (i + 1, kmin + (i + 0.5) * dk, 0, 0, 0, 0, 0, 0, 0, 0))
            elif ps_multi != 'no':
                outfile.write("  %7d     %17.10lf     %17.10lf     %17.10lf     %17.10lf     %17.10lf     "
                               "%17.10lf    %7d     %30.20lf     %30.20lf \n"
                               % (i + 1, kmin + (i + 0.5) * dk, np.real(p0[i]), np.real(p1[i]), np.real(p2[i]),
                                  np.real(p3[i]), np.real(p4[i]), n_modes[i], norm, psn))
            else:
                outfile.write("  %7d     %17.10lf     %17.10lf     %7d     %7d     %7d     %7d     %7d     "
                               "%30.20lf     %30.20lf \n"
                               % (i + 1, kmin + (i + 0.5) * dk, np.real(p0[i]), 0, 0, 0, 0, n_modes[i], norm, psn))
    '''
    # Collect the multipoles; P1-P4 are zero if they were not computed.
    multipoles = np.zeros((5, nk))
    multipoles[0] = np.real(p0)
    if ps_multi != 'no':
        multipoles[1] = np.real(p1)
        multipoles[2] = np.real(p2)
        multipoles[3] = np.real(p3)
        multipoles[4] = np.real(p4)

    ps_table = pd.DataFrame({
        'No': np.arange(1, nk + 1),
        'k': kmin + (np.arange(nk) + 0.5) * dk,
        'P0': multipoles[0],
        'P1': multipoles[1],
        'P2': multipoles[2],
        'P3': multipoles[3],
        'P4': multipoles[4],
        'Nk': n_modes,
        'Norm': norm,
        'SNois': psn,
    })

    # The last k-bin is forced to zero (edge effects), keeping only its k value.
    ps_table.loc[nk - 1, ['P0', 'P1', 'P2', 'P3', 'P4', 'Nk', 'Norm', 'SNois']] = 0.
    ps_table.to_csv(file_dir, index=False)

    if ps_type == 'den':
        weights_fkp = [w, wr]
    if ps_type == 'mom':
        weights_fkp = [wmom, wrho]
    if ps_type == 'crs':
        weights_fkp = [w, wr, wmom, wrho]

    return file_dir, weights_fkp, norm


def estimate_random_power_spectrum(kmin, kmax, nk, nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax,
                longir, latir, rsfr, epv, nbr_norm, fkp, omega_m, omega_lambda, hubble_constant, sigv,
                ps_type, file_dir='PSestDir', wdt=1.):
    """Estimate the monopole power spectrum of a random catalogue alone.

    A simpler, monopole-only counterpart to `estimate_power_spectrum`'s random-catalogue
    handling, useful e.g. for checking the random catalogue's own shot-noise
    dominated power spectrum. Loops explicitly over the FFT grid (rather
    than the vectorized binning used in `estimate_power_spectrum`).
    """
    if nx % 2 != 0:
        sys.exit('\n Error: nx should be set to an even number.\n')
    if ny % 2 != 0:
        sys.exit('\n Error: ny should be set to an even number.\n')
    if nz % 2 != 0:
        sys.exit('\n Error: nz should be set to an even number.\n')
    print('\n  Random-PS=', ps_type, '\n')

    xr, yr, zr = sky_to_cartesian(longir, latir, rsfr, omega_m, omega_lambda, hubble_constant)

    # NOTE: nbr_norm is the number density normalized to the *galaxy* catalogue,
    # not the random catalogue itself.
    nbr = nbr_norm
    if ps_type == 'den':
        wr = 1.0 / (1.0 + nbr * fkp)
    if ps_type == 'mom':
        wr = 1.0 / (epv ** 2 + sigv ** 2 + nbr * fkp)
    if fkp == 0:
        wr = np.ones(len(xr))

    # -- 1. Grid the catalogue -------------------------------------------------
    dx, dy, dz = (xmax - xmin) / nx, (ymax - ymin) / ny, (zmax - zmin) / nz
    lx, ly, lz = np.abs(xmax - xmin), np.abs(ymax - ymin), np.abs(zmax - zmin)
    grid_range = ((-lx / 2., lx / 2.), (-ly / 2., ly / 2.), (-lz / 2., lz / 2.))
    grid, _ = np.histogramdd(np.vstack([xr, yr, zr]).transpose(), bins=(nx, ny, nz),
                             range=grid_range, weights=wr * wdt)

    # -- 2. Shot noise ----------------------------------------------------------
    pnoise_r = np.sum(wr * wr * wdt)

    # -- 3. Nyquist frequency ----------------------------------------------------
    fx_nqu, fy_nqu, fz_nqu = np.pi / dx, np.pi / dy, np.pi / dz
    min_nqu = min(fx_nqu, fy_nqu, fz_nqu)

    # -- 4. FFT -------------------------------------------------------------------
    grid_fft = scipy.fft.fftn(grid)

    # -- 5. l=0 monopole ------------------------------------------------------------
    dk = (kmax - kmin) / nk
    n_modes = np.zeros(nk, dtype=int)
    p0 = np.zeros(nk)
    norm_r = None
    for i in range(nx):
        fx = i / (nx * dx) if i <= nx // 2 else (i - nx) / (nx * dx)
        for j in range(ny):
            fy = j / (ny * dy) if j <= ny // 2 else (j - ny) / (ny * dy)
            for k in range(nz // 2):
                fz = k / (nz * dz)
                k_mag = 2.0 * np.pi * np.sqrt(fx * fx + fy * fy + fz * fz)
                ik_bin_idx = int((k_mag - kmin) / dk)
                if 0 <= ik_bin_idx < nk and k_mag < 0.5 * min_nqu:
                    sincx = np.sin(fx * dx * np.pi) / (fx * dx * np.pi) if fx != 0.0 else 1.0
                    sincy = np.sin(fy * dy * np.pi) / (fy * dy * np.pi) if fy != 0.0 else 1.0
                    sincz = np.sin(fz * dz * np.pi) / (fz * dz * np.pi) if fz != 0.0 else 1.0
                    grid_cor = 1.0 / (sincx * sincy * sincz)
                    p0[ik_bin_idx] += (np.real(grid_fft[i, j, k]) ** 2 + np.imag(grid_fft[i, j, k]) ** 2
                                - pnoise_r) * grid_cor * grid_cor
                    if ik_bin_idx == 0:
                        norm_r = np.real(grid_fft[i, j, k]) ** 2 + np.imag(grid_fft[i, j, k]) ** 2 - pnoise_r
                    n_modes[ik_bin_idx] += 1

    # -- 6. Normalize and save ---------------------------------------------------
    for ik in range(nk):
        if n_modes[ik] > 0.0:
            p0[ik] = p0[ik] / (n_modes[ik] * norm_r)
    n_modes[-1] = 0

    with open(file_dir, 'w') as outfile:
        for i in range(nk):
            if np.real(p0[i]) > 0.0:
                if i == nk - 1:
                    outfile.write("  %7d     %17.10lf     %7d     %7d \n" % (i + 1, kmin + (i + 0.5) * dk, 0, 0))
                else:
                    outfile.write("  %7d     %17.10lf     %17.10lf     %7d \n"
                                   % (i + 1, kmin + (i + 0.5) * dk, np.real(p0[i]), n_modes[i]))

    return file_dir, wr


def estimate_power_spectrum_simbox(kmin, kmax, nk, nx, ny, nz, Lbox,
                      x, y, z, vx, vy, vz,
                      omega_m=0.3, omega_lambda=0.7, hubble_constant=100.,
                      ps_type='mom', ps_multi='yes', file_dir='PSestDir', wd=1., wv=1.):
    """Estimate power spectrum multipoles from a periodic N-body simulation box.

    Same multipole (l=0..4) machinery as `estimate_power_spectrum`, but for a cubic,
    periodic simulation box of side `Lbox` centered on the origin, with no
    survey mask / random catalogue needed (density contrast is computed
    directly against the mean number density n = Ngal / Lbox**3), and the
    plane-of-sky velocity is replaced by the line-of-sight velocity
    projected along the (simulation-frame) position vector,
    vp = (vx*x + vy*y + vz*z) / |r|.
    """
    if nx % 2 != 0:
        sys.exit('\n Error: nx should be set to an even number.\n')
    if ny % 2 != 0:
        sys.exit('\n Error: ny should be set to an even number.\n')
    if nz % 2 != 0:
        sys.exit('\n Error: nz should be set to an even number.\n')
    #print(' Please make sure the coordinate origin [0, 0, 0] is at the center of the sim box.\n')

    ndata = len(x)
    print(' ', ps_type, ' Ngal=', ndata)

    lx = ly = lz = Lbox
    dx, dy, dz = Lbox / nx, Lbox / ny, Lbox / nz
    xmin, xmax = -lx / 2., lx / 2.
    ymin, ymax = -ly / 2., ly / 2.
    zmin, zmax = -lz / 2., lz / 2.

    dk = (kmax - kmin) / nk
    n_modes = np.zeros(nk, dtype=int)

    p0 = np.zeros(nk)
    if ps_multi != 'no':
        p1 = np.zeros(nk)
        p2 = np.zeros(nk)
        p3 = np.zeros(nk)
        p4 = np.zeros(nk)

    # -- 2. Grid the box ----------------------------------------------------------
    grid_range = ((-lx / 2., lx / 2.), (-ly / 2., ly / 2.), (-lz / 2., lz / 2.))
    if ps_type in ('den', 'crs'):
        grid, _ = np.histogramdd(np.vstack([x, y, z]).transpose(), bins=(nx, ny, nz),
                                 range=grid_range, weights=wd * np.ones(len(x)))
        grid = grid - ndata / (nx * ny * nz)
    if ps_type in ('mom', 'crs'):
        vp = (vx * x + vy * y + vz * z) / np.sqrt(x * x + y * y + z * z)
        if ps_type == 'mom':
            grid, _ = np.histogramdd(np.vstack([x, y, z]).transpose(), bins=(nx, ny, nz),
                                     range=grid_range, weights=vp * wv)
        if ps_type == 'crs':
            mom_grid, _ = np.histogramdd(np.vstack([x, y, z]).transpose(), bins=(nx, ny, nz),
                                        range=grid_range, weights=vp * wv)

    fx_nqu, fy_nqu, fz_nqu = np.pi / dx, np.pi / dy, np.pi / dz
    min_nqu = min(fx_nqu, fy_nqu, fz_nqu)

    if ps_multi != 'no':
        grid_x = np.linspace(xmin, xmax, nx + 1)[:-1]
        grid_y = np.linspace(ymin, ymax, ny + 1)[:-1]
        grid_z = np.linspace(zmin, zmax, nz + 1)[:-1]
        grid_mesh = np.meshgrid(grid_x, grid_y, grid_z)
        pos_vect = np.zeros((nx * ny * nz, 3))
        pos_vect[:, 1] = grid_mesh[0].flatten()
        pos_vect[:, 2] = grid_mesh[1].flatten()
        pos_vect[:, 0] = grid_mesh[2].flatten()
        vec_len_sq = pos_vect[:, 0] ** 2 + pos_vect[:, 1] ** 2 + pos_vect[:, 2] ** 2
        vec_len_sq[vec_len_sq == 0.0] = 1.0

    k_vect = np.zeros((nx * ny * nz // 2, 3))
    k_vals_x = np.concatenate((
        np.linspace(0., 2. * np.pi * (nx // 2) / (nx * dx), nx // 2 + 1),
        np.linspace(2. * np.pi * (nx // 2 + 1 - nx) / (nx * dx), 0., nx // 2)[:-1],
    ))
    k_vals_y = np.concatenate((
        np.linspace(0., 2. * np.pi * (ny // 2) / (ny * dy), ny // 2 + 1),
        np.linspace(2. * np.pi * (ny // 2 + 1 - ny) / (ny * dy), 0., ny // 2)[:-1],
    ))
    k_vals_z = np.linspace(0., 2. * np.pi * (nz // 2 - 1) / (nz * dz), nz // 2)
    k_mesh = np.meshgrid(k_vals_x, k_vals_y, k_vals_z)
    k_vect[:, 1] = k_mesh[0].flatten()
    k_vect[:, 2] = k_mesh[1].flatten()
    k_vect[:, 0] = k_mesh[2].flatten()

    grid_cor = grid_correction(dx, dy, dz, k_vect[:, 1] / (2. * np.pi),
                             k_vect[:, 2] / (2. * np.pi), k_vect[:, 0] / (2. * np.pi))
    k_mag = np.sqrt(k_vect[:, 0] ** 2 + k_vect[:, 1] ** 2 + k_vect[:, 2] ** 2)

    ik_bin_idx = np.array((k_mag - kmin) / dk, dtype=int)
    ind_nqu = np.where((ik_bin_idx >= 0) & (ik_bin_idx < nk) & (k_mag < 0.5 * min_nqu))[0]
    ind_use_list = [ind_nqu[np.where(ik_bin_idx[ind_nqu] == ikbin)[0]] for ikbin in range(nk)]

    # -- 3. Normalization / shot noise ---------------------------------------------
    norm = ndata ** 2 / Lbox ** 3
    if ps_type == 'den':
        pnoise = float(ndata)
        psn = pnoise / norm
    if ps_type == 'mom':
        pvnoise = np.sum(vp * vp * wv * wv)
        psn = pvnoise / norm
    if ps_type == 'crs':
        pnoise_c = np.sum(vp * wv)
        psn = 0.0

    # -- 4. l=0 ---------------------------------------------------------------------
    grid_fft = scipy.fft.fftn(grid)[:, :, :nz // 2].flatten()
    if ps_type == 'crs':
        mom_grid_fft = scipy.fft.fftn(mom_grid)[:, :, :nz // 2].flatten()
    if ps_type == 'crs' and ps_multi == 'all':
        tmp1 = (0.5 * (np.real(grid_fft) * np.imag(mom_grid_fft) - np.imag(grid_fft) * np.real(mom_grid_fft)
                        + np.real(mom_grid_fft) * np.imag(grid_fft) - np.imag(mom_grid_fft) * np.real(grid_fft))
                - pnoise_c) * grid_cor * grid_cor
    if ps_type == 'den':
        tmp2 = (np.real(grid_fft) ** 2 + np.imag(grid_fft) ** 2 - pnoise) * grid_cor * grid_cor
    if ps_type == 'mom':
        tmp3 = (np.real(grid_fft) ** 2 + np.imag(grid_fft) ** 2 - pvnoise) * grid_cor * grid_cor
    if ps_multi in ('yes', 'all'):
        if ps_type != 'crs':
            tmp4 = (np.real(grid_fft) ** 2 + np.imag(grid_fft) ** 2) * grid_cor * grid_cor
        if ps_type == 'crs' and ps_multi == 'all':
            tmp5 = 0.5 * (np.real(grid_fft) * np.imag(mom_grid_fft) - np.imag(grid_fft) * np.real(mom_grid_fft)
                           + np.real(mom_grid_fft) * np.imag(grid_fft) - np.imag(mom_grid_fft) * np.real(grid_fft)
                           ) * grid_cor * grid_cor

    for ikbin in range(nk):
        ind_use = ind_use_list[ikbin]
        if ps_type == 'crs' and ps_multi == 'all':
            p0[ikbin] = np.sum(tmp1[ind_use])
        if ps_type == 'den':
            p0[ikbin] = np.sum(tmp2[ind_use])
        if ps_type == 'mom':
            p0[ikbin] = np.sum(tmp3[ind_use])
        n_modes[ikbin] = len(ik_bin_idx[ind_use])

        if ps_multi in ('yes', 'all'):
            if ps_type != 'crs':
                tmp = np.sum(tmp4[ind_use])
                p2[ikbin] = -0.5 * tmp
                p4[ikbin] = 0.375 * tmp
            if ps_type == 'crs' and ps_multi == 'all':
                tmp = np.sum(tmp5[ind_use])
                p2[ikbin] = -0.5 * tmp
                p4[ikbin] = 0.375 * tmp

    grid_saved = grid.copy()
    grid = np.zeros((nx, ny, nz))
    grid_fft_saved = grid_fft.copy()
    grid_fft = 0.
    tmp1 = tmp2 = tmp3 = tmp4 = tmp5 = 0.
    if ps_type == 'crs':
        mom_grid_saved = mom_grid.copy()
        mom_grid = np.zeros((nx, ny, nz))
        mom_grid_fft_saved = mom_grid_fft.copy()
        mom_grid_fft = 0.

    if ps_multi != 'no':
        # -- 5. l=1 ---------------------------------------------------------------
        if ps_type == 'crs' or ps_multi == 'all':
            for ii in range(3):
                if ps_type != 'crs':
                    grid = grid_saved * (pos_vect[:, ii] / np.sqrt(vec_len_sq)).reshape(nx, ny, nz)
                if ps_type == 'crs':
                    grid = grid_saved * (pos_vect[:, ii] / np.sqrt(vec_len_sq)).reshape(nx, ny, nz)
                    mom_grid = mom_grid_saved * (pos_vect[:, ii] / np.sqrt(vec_len_sq)).reshape(nx, ny, nz)

                if ps_type != 'crs' and ps_multi == 'all':
                    grid_fft = scipy.fft.fftn(grid)[:, :, :nz // 2].flatten()
                if ps_type == 'crs':
                    grid_fft = scipy.fft.fftn(grid)[:, :, :nz // 2].flatten()
                    mom_grid_fft = scipy.fft.fftn(mom_grid)[:, :, :nz // 2].flatten()

                if ps_type != 'crs' and ps_multi == 'all':
                    tmp1 = k_vect[:, ii] * ((np.real(grid_fft_saved) * np.imag(grid_fft)
                                             - np.imag(grid_fft_saved) * np.real(grid_fft)) * grid_cor * grid_cor)
                if ps_type == 'crs':
                    tmp2 = k_vect[:, ii] * (0.5 * (-np.imag(grid_fft) * np.real(mom_grid_fft_saved)
                                                   + np.real(grid_fft) * np.imag(mom_grid_fft_saved)
                                                   + np.imag(mom_grid_fft) * np.real(grid_fft_saved)
                                                   - np.real(mom_grid_fft) * np.imag(grid_fft_saved)) * grid_cor * grid_cor)

                if ps_type == 'crs' or ps_multi == 'all':
                    for ikbin in range(nk):
                        kprefac = 1.0
                        ind_use = ind_use_list[ikbin]
                        k_sel = k_mag[ind_use]
                        valid_k = (k_sel > 0)
                        if ps_type != 'crs' and ps_multi == 'all':
                            p1[ikbin] += np.sum((1.0 * kprefac * tmp1[ind_use])[valid_k] / k_sel[valid_k])
                            p3[ikbin] += np.sum((-1.5 * kprefac * tmp1[ind_use])[valid_k] / k_sel[valid_k])
                        if ps_type == 'crs':
                            p1[ikbin] += np.sum((1.0 * kprefac * tmp2[ind_use])[valid_k] / k_sel[valid_k])
                            p3[ikbin] += np.sum((-1.5 * kprefac * tmp2[ind_use])[valid_k] / k_sel[valid_k])

            tmp1 = tmp2 = 0.
            if ps_type != 'crs' and ps_multi == 'all':
                grid = np.zeros((nx, ny, nz))
                grid_fft = 0.
            if ps_type == 'crs':
                mom_grid = np.zeros((nx, ny, nz))
                grid = np.zeros((nx, ny, nz))
                grid_fft = 0.
                mom_grid_fft = 0.

        # -- 6. l=2 -----------------------------------------------------------------
        if ps_type != 'crs' or ps_multi == 'all':
            for ii in range(3):
                for jj in range(3):
                    if jj < ii:
                        continue
                    if ps_type != 'crs':
                        grid = grid_saved * (pos_vect[:, ii] * pos_vect[:, jj] / vec_len_sq).reshape(nx, ny, nz)
                    if ps_type == 'crs':
                        grid = grid_saved * (pos_vect[:, ii] * pos_vect[:, jj] / vec_len_sq).reshape(nx, ny, nz)
                        mom_grid = mom_grid_saved * (pos_vect[:, ii] * pos_vect[:, jj] / vec_len_sq).reshape(nx, ny, nz)

                    grid_fft = scipy.fft.fftn(grid)[:, :, :nz // 2].flatten()
                    if ps_type == 'crs' and ps_multi == 'all':
                        mom_grid_fft = scipy.fft.fftn(mom_grid)[:, :, :nz // 2].flatten()

                    if ps_type != 'crs':
                        tmp1 = k_vect[:, ii] * k_vect[:, jj] * (
                            (np.real(grid_fft) * np.real(grid_fft_saved)
                             + np.imag(grid_fft) * np.imag(grid_fft_saved)) * grid_cor * grid_cor)
                    if ps_type == 'crs' and ps_multi == 'all':
                        tmp2 = k_vect[:, ii] * k_vect[:, jj] * (
                            0.5 * (np.real(grid_fft) * np.imag(mom_grid_fft_saved)
                                   - np.imag(grid_fft) * np.real(mom_grid_fft_saved)
                                   + np.real(mom_grid_fft) * np.imag(grid_fft_saved)
                                   - np.imag(mom_grid_fft) * np.real(grid_fft_saved)) * grid_cor * grid_cor)

                    for ikbin in range(nk):
                        kprefac = 2.0 if ii != jj else 1.0
                        ind_use = ind_use_list[ikbin]
                        k_sel = k_mag[ind_use]
                        valid_k = (k_sel > 0)
                        if ps_type != 'crs':
                            p2[ikbin] += np.sum((1.5 * kprefac * tmp1[ind_use])[valid_k] / k_sel[valid_k] ** 2)
                            p4[ikbin] += np.sum((-3.75 * kprefac * tmp1[ind_use])[valid_k] / k_sel[valid_k] ** 2)
                        if ps_type == 'crs' and ps_multi == 'all':
                            p2[ikbin] += np.sum((1.5 * kprefac * tmp2[ind_use])[valid_k] / k_sel[valid_k] ** 2)
                            p4[ikbin] += np.sum((-3.75 * kprefac * tmp2[ind_use])[valid_k] / k_sel[valid_k] ** 2)

            grid = np.zeros((nx, ny, nz))
            grid_fft = 0.
            if ps_type == 'crs':
                mom_grid = np.zeros((nx, ny, nz))
                grid_fft = 0.
                mom_grid_fft = 0.
            tmp1 = tmp2 = 0.

        # -- 7. l=3 -------------------------------------------------------------------
        _L3_SKIP = {3, 4, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23}
        if ps_type == 'crs' or ps_multi == 'all':
            combo_idx = 0
            for ii in range(3):
                for jj in range(3):
                    for kk in range(3):
                        if combo_idx not in _L3_SKIP:
                            if ps_type != 'crs':
                                grid = grid_saved * (pos_vect[:, ii] * pos_vect[:, jj] * pos_vect[:, kk]
                                                   / (vec_len_sq * np.sqrt(vec_len_sq))).reshape(nx, ny, nz)
                            if ps_type == 'crs':
                                grid = grid_saved * (pos_vect[:, ii] * pos_vect[:, jj] * pos_vect[:, kk]
                                                   / (vec_len_sq * np.sqrt(vec_len_sq))).reshape(nx, ny, nz)
                                mom_grid = mom_grid_saved * (pos_vect[:, ii] * pos_vect[:, jj] * pos_vect[:, kk]
                                                         / (vec_len_sq * np.sqrt(vec_len_sq))).reshape(nx, ny, nz)

                            if ps_type != 'crs' and ps_multi == 'all':
                                grid_fft = scipy.fft.fftn(grid)[:, :, :nz // 2].flatten()
                            if ps_type == 'crs':
                                grid_fft = scipy.fft.fftn(grid)[:, :, :nz // 2].flatten()
                                mom_grid_fft = scipy.fft.fftn(mom_grid)[:, :, :nz // 2].flatten()

                            if ps_type != 'crs' and ps_multi == 'all':
                                tmp1 = k_vect[:, ii] * k_vect[:, jj] * k_vect[:, kk] * (
                                    (np.real(grid_fft_saved) * np.imag(grid_fft)
                                     - np.imag(grid_fft_saved) * np.real(grid_fft)) * grid_cor * grid_cor)
                            if ps_type == 'crs':
                                tmp2 = k_vect[:, ii] * k_vect[:, jj] * k_vect[:, kk] * (
                                    0.5 * (-np.imag(grid_fft) * np.real(mom_grid_fft_saved)
                                           + np.real(grid_fft) * np.imag(mom_grid_fft_saved)
                                           + np.imag(mom_grid_fft) * np.real(grid_fft_saved)
                                           - np.real(mom_grid_fft) * np.imag(grid_fft_saved)) * grid_cor * grid_cor)

                            for ikbin in range(nk):
                                if ii != kk:
                                    kprefac = 6.0 if ii != jj else 3.0
                                else:
                                    kprefac = 1.0
                                ind_use = ind_use_list[ikbin]
                                k_sel = k_mag[ind_use]
                                valid_k = (k_sel > 0)
                                if ps_type != 'crs' and ps_multi == 'all':
                                    p3[ikbin] += np.sum((2.5 * kprefac * tmp1[ind_use])[valid_k] / k_sel[valid_k] ** 3)
                                if ps_type == 'crs':
                                    p3[ikbin] += np.sum((2.5 * kprefac * tmp2[ind_use])[valid_k] / k_sel[valid_k] ** 3)
                        combo_idx += 1

            if ps_type != 'crs' and ps_multi == 'all':
                grid = np.zeros((nx, ny, nz))
                grid_fft = 0.
            if ps_type == 'crs':
                mom_grid = np.zeros((nx, ny, nz))
                grid = np.zeros((nx, ny, nz))
                grid_fft = 0.
                mom_grid_fft = 0.
            tmp1 = tmp2 = 0.

        # -- 8. l=4 -------------------------------------------------------------------
        _L4_SKIP = {4, 7, 8, 10, 11, 16, 17, 19, 21, 22, 23, 24}
        if ps_type != 'crs' or ps_multi == 'all':
            combo_idx = 1
            for ii in range(3):
                for jj in range(3):
                    for kk in range(3):
                        if combo_idx not in _L4_SKIP:
                            if ps_type != 'crs':
                                grid = grid_saved * (pos_vect[:, ii] * pos_vect[:, ii] * pos_vect[:, jj] * pos_vect[:, kk]
                                                   / vec_len_sq ** 2).reshape(nx, ny, nz)
                            if ps_type == 'crs':
                                grid = grid_saved * (pos_vect[:, ii] * pos_vect[:, ii] * pos_vect[:, jj] * pos_vect[:, kk]
                                                   / vec_len_sq ** 2).reshape(nx, ny, nz)
                                mom_grid = mom_grid_saved * (pos_vect[:, ii] * pos_vect[:, ii] * pos_vect[:, jj] * pos_vect[:, kk]
                                                         / vec_len_sq ** 2).reshape(nx, ny, nz)

                            grid_fft = scipy.fft.fftn(grid)[:, :, :nz // 2].flatten()
                            if ps_type == 'crs' and ps_multi == 'all':
                                mom_grid_fft = scipy.fft.fftn(mom_grid)[:, :, :nz // 2].flatten()

                            if ps_type != 'crs':
                                tmp1 = k_vect[:, ii] * k_vect[:, ii] * k_vect[:, jj] * k_vect[:, kk] * (
                                    (np.real(grid_fft) * np.real(grid_fft_saved)
                                     + np.imag(grid_fft) * np.imag(grid_fft_saved)) * grid_cor * grid_cor)
                            if ps_type == 'crs' and ps_multi == 'all':
                                tmp2 = k_vect[:, ii] * k_vect[:, ii] * k_vect[:, jj] * k_vect[:, kk] * (
                                    0.5 * (np.real(grid_fft) * np.imag(mom_grid_fft_saved)
                                           - np.imag(grid_fft) * np.real(mom_grid_fft_saved)
                                           + np.real(mom_grid_fft) * np.imag(grid_fft_saved)
                                           - np.imag(mom_grid_fft) * np.real(grid_fft_saved)) * grid_cor * grid_cor)

                            for ikbin in range(nk):
                                if ii == jj:
                                    kprefac = 4.0 if ii != kk else 1.0
                                else:
                                    kprefac = 6.0 if jj == kk else 12.0
                                ind_use = ind_use_list[ikbin]
                                k_sel = k_mag[ind_use]
                                valid_k = (k_sel > 0)
                                if ps_type != 'crs':
                                    p4[ikbin] += np.sum((4.375 * kprefac * tmp1[ind_use])[valid_k] / k_sel[valid_k] ** 4)
                                if ps_type == 'crs' and ps_multi == 'all':
                                    p4[ikbin] += np.sum((4.375 * kprefac * tmp2[ind_use])[valid_k] / k_sel[valid_k] ** 4)
                        combo_idx += 1
            tmp1 = tmp2 = 0.
    # ====================  End of power-spectrum multipoles  ===================

    # -- 9. Normalize and save ---------------------------------------------------
    for ik in range(nk):
        if n_modes[ik] > 0.0:
            p0[ik] = p0[ik] / (n_modes[ik] * norm)
            if ps_multi in ('yes', 'all'):
                p1[ik] = p1[ik] * 3.0 / (n_modes[ik] * norm)
                p2[ik] = p2[ik] * 5.0 / (n_modes[ik] * norm)
                p3[ik] = p3[ik] * 7.0 / (n_modes[ik] * norm)
                p4[ik] = p4[ik] * 9.0 / (n_modes[ik] * norm)

    n_modes[-1] = 0

        # Collect the multipoles; P1-P4 are zero if they were not computed.
    multipoles = np.zeros((5, nk))
    multipoles[0] = np.real(p0)
    if ps_multi != 'no':
        multipoles[1] = np.real(p1)
        multipoles[2] = np.real(p2)
        multipoles[3] = np.real(p3)
        multipoles[4] = np.real(p4)

    ps_table = pd.DataFrame({
        'No': np.arange(1, nk + 1),
        'k': kmin + (np.arange(nk) + 0.5) * dk,
        'P0': multipoles[0],
        'P1': multipoles[1],
        'P2': multipoles[2],
        'P3': multipoles[3],
        'P4': multipoles[4],
        'Nk': n_modes,
        'Norm': norm,
        'SNois': psn,
    })

    # The last k-bin is forced to zero (edge effects), keeping only its k value.
    ps_table.loc[nk - 1, ['P0', 'P1', 'P2', 'P3', 'P4', 'Nk', 'Norm', 'SNois']] = 0.

    ps_table.to_csv(file_dir, index=False)

    '''
    with open(file_dir, 'w') as outfile:
        outfile.write("# No.          k            P0            P1           P2           P3"
                       "           P4            Nk           Norm           SNois  \n")
        for i in range(nk):
            if i == nk - 1:
                outfile.write("  %7d     %17.10lf     %7d     %7d     %7d     %7d     %7d     %7d     %7d     %7d \n"
                               % (i + 1, kmin + (i + 0.5) * dk, 0, 0, 0, 0, 0, 0, 0, 0))
            elif ps_multi != 'no':
                outfile.write("  %7d     %17.10lf     %17.10lf     %17.10lf     %17.10lf     %17.10lf     "
                               "%17.10lf    %7d     %30.20lf     %30.20lf \n"
                               % (i + 1, kmin + (i + 0.5) * dk, np.real(p0[i]), np.real(p1[i]), np.real(p2[i]),
                                  np.real(p3[i]), np.real(p4[i]), n_modes[i], norm, psn))
            else:
                outfile.write("  %7d     %17.10lf     %17.10lf     %7d     %7d     %7d     %7d     %7d     "
                               "%30.20lf     %30.20lf \n"
                               % (i + 1, kmin + (i + 0.5) * dk, np.real(p0[i]), 0, 0, 0, 0, n_modes[i], norm, psn))
    '''

    return file_dir, norm

#########################     End of Sec 2.    ################################
###############################################################################
