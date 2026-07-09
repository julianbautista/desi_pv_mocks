from locale import normalize

import numpy as np

LIGHT_SPEED = 299_792.458  # km/s

def radec_to_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Convert (RA, Dec, comoving distance) → Cartesian (x, y, z). Returns (3, N)."""
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    x = dist * np.cos(dec) * np.cos(ra)
    y = dist * np.cos(dec) * np.sin(ra)
    z = dist * np.sin(dec)
    return np.stack([x, y, z])

def build_grid_box(distmax, ngrid):
    """Return grid parameters for a cube enclosing the survey volume."""
    side = 2.0 * distmax
    d = side / ngrid
    origin = distmax          # offset so coordinates are centred at the origin
    lims = np.linspace(0.0, side, ngrid + 1) - origin
    return dict(ngrid=ngrid, side=side, d=d, origin=origin, dvol=d**3, lims=lims)
 
def safe_digitize(values, edges):
    """Return grid indices clipped to [0, n-1] to guard against boundary objects."""
    n = edges.size - 1
    return np.clip(np.digitize(values, edges) - 1, 0, n - 1)
 

def build_density_mesh(positions, box, weights=None, normalize=False):
    ''' Build a 3D density mesh from positions and weights 
    
    '''
    ngrid = box["ngrid"]
    s = box["side"]

    wingrid, _ = np.histogramdd(
        positions.T, 
        weights=weights,
        bins=(ngrid, ngrid, ngrid),
        range=((-s/2, s/2), (-s/2, s/2), (-s/2, s/2)),
    )
    ndensgrid = (wingrid/ box["dvol"]) 
    if normalize:
        ndensgrid /= wingrid.sum()

    return ndensgrid

def build_mesh(pos, quantity, ngrid, side, weights=None):

    if weights is None:
        weights = np.ones(len(pos[0]))

    values, _ = np.histogramdd(
        pos.T, 
        weights=weights*quantity,
        bins=(ngrid, ngrid, ngrid),
        range=((-side/2, side/2), 
               (-side/2, side/2), 
               (-side/2, side/2)),
    )

    counts, _ = np.histogramdd(
        pos.T, 
        weights=weights,
        bins=(ngrid, ngrid, ngrid),
         range=((-side/2, side/2), 
               (-side/2, side/2), 
               (-side/2, side/2)),
    )

    mesh = np.zeros_like(values)
    w = counts > 0
    mesh[w] = values[w] / counts[w]

    return mesh

def get_mesh_value(mesh, positions, bins):
    # Sample 3d density grid at galaxy positions 
    ix = safe_digitize(positions[0], bins)
    iy = safe_digitize(positions[1], bins)
    iz = safe_digitize(positions[2], bins)
    mesh_value = mesh[ix, iy, iz]
    return mesh_value

def compute_nz(z, zbins, cosmo, frac_sky, weights=None, normalize=False):
    """Compute n(z) in units of [Mpc/h]^-3 from redshifts and weights."""

    zvol = (cosmo.comoving_volume(zbins[1:]).value-cosmo.comoving_volume(zbins[:-1]).value)
    nz, _ = np.histogram(z, bins=zbins, weights=weights)
    if normalize:
        nz = nz / nz.sum()
    return nz/zvol/frac_sky 

def pv_from_logdist(logdist, z, cosmo):
    """
    Carreres et al. (2023) v1 estimator:  pv = c ln(10) η / (c(1+z)/χH(z) − 1)
    """
    
    denom = (
        LIGHT_SPEED * (1.0 + z)
        / (cosmo.comoving_distance(z).value * cosmo.H(z).value)
        - 1.0
    )
    return LIGHT_SPEED * np.log(10.0) * logdist / denom
