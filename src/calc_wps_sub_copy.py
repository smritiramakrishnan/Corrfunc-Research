#!/usr/bin/env python
import os
import numpy as np
import fitsio
from Corrfunc._countpairs import countpairs_rp_pi
# Deleting imports and path settings that are not accessible/used from shuleic/...
# Full code with those imports included in calc_wps_sub.py
import yaml

# returns a string like 'z0p2500' for redshift 0.25
def get_zname(redshift):
    return f"z{str(redshift).replace('.', 'p')}00"

# makes sure points are within boxsize using periodic boundary conditions (no halving)
def pbc(xg, yg, zg, boxsize):
    xg = np.mod(xg, boxsize)
    yg = np.mod(yg, boxsize)
    zg = np.mod(zg, boxsize)
    return xg, yg, zg

# computes Alcock-Paczynski scaling factors (differenc between simulation and fiducial cosmology))
# return S_M(transverse scaling factor), and S_C(LOS scaling factor)
def _ap_scalings(cosmo_sim, cosmo_fid, h_sim, h_fid, z):
    """
    S_M (transverse) and S_C (LOS) using Colossus distances (h^-1 Mpc).
    Since Mpc = (h^-1 Mpc)/h, the *Mpc* ratio carries (h_sim/h_fid).
    """
    # returns comoving distances inferred by redshift (hubble, along LOS)
    D_C_sim_h = cosmo_sim.comovingDistance(0.0, z, transverse=False)  # h^-1 Mpc
    D_C_fid_h = cosmo_fid.comovingDistance(0.0, z, transverse=False)  # h^-1 Mpc
    # returns comoving distances inferred by angle and redshift (transverse)
    D_M_sim_h = cosmo_sim.comovingDistance(0.0, z, transverse=True)   # h^-1 Mpc
    D_M_fid_h = cosmo_fid.comovingDistance(0.0, z, transverse=True)   # h^-1 Mpc

    # Mpc ratios: (D_fid_h / h_fid) / (D_sim_h / h_sim) = (h_sim/h_fid) * (D_fid_h / D_sim_h)
    # finds scaling factors to convert simulation distances to fiducial distances
    S_C = (h_sim / h_fid) * (D_C_fid_h / D_C_sim_h)
    S_M = (h_sim / h_fid) * (D_M_fid_h / D_M_sim_h)
    return S_M, S_C

# Takes 3D simulation coordinates and applies redshift-space distortions (mimics real observations).
# returns transformed coordinates in comoving Mpc/h units.
"""
data(position data), los(line of sight axis), redshift, cosmo_fid(fiducial cosmology)
pec_vel(whether to include peculiar velocity: bool), boxsize, mask(bool[]: subset selection)
data_type: 'galaxy' or 'cluster'
"""
def transform_coordinates(
    data,
    los,
    redshift,
    cosmo_fid,
    pec_vel,
    boxsize,
    mask=None,
    data_type='galaxy'
):
    """
    Apply only RSD (if enabled) to comoving Mpc/h positions.
    Positions remain in comoving Mpc/h (simulation native).
    """
    # uses all data if no mask provided
    if mask is None:
        mask = np.ones(len(data['px']), dtype=bool)

    # Positions: keep in comoving Mpc/h
    px = data['px'][mask]
    py = data['py'][mask]
    pz = data['pz'][mask]

    # if cluster, then red-shift space distortions are not applied
    if data_type == 'cluster':
        if boxsize is None:
            return px, py, pz
        else:
            return pbc(px, py, pz, boxsize)
        
    # Velocity factor to convert km/s to cMpc/h at given redshift in fiducial cosmology
    Ez_fid = cosmo_fid.Ez(redshift) # gives dimensionless hubble parameter (E(z)) at redshift z (h(z)/h0)
    velocity_factor = (1 + redshift) / Ez_fid / 100.  # cMpc/h / (km/s)
    
    #rotates so that los is along z-axis, applies RSD, then rotates back
    # gets velocity along los, converts using velocity factor, adds to position along los
    if los == 'z':
        vz = data.get('vz', np.zeros_like(px))[mask] if pec_vel else 0  # km/s, default to 0 if not present
        v_disp_z = velocity_factor * vz
        x_trans = px
        y_trans = py
        z_trans = pz + v_disp_z
    elif los == 'x':
        vx = data.get('vx', np.zeros_like(px))[mask] if pec_vel else 0
        v_disp_x = velocity_factor * vx
        x_trans = py
        y_trans = pz
        z_trans = px + v_disp_x
    elif los == 'y':
        vy = data.get('vy', np.zeros_like(px))[mask] if pec_vel else 0
        v_disp_y = velocity_factor * vy
        x_trans = pz
        y_trans = px
        z_trans = py + v_disp_y
    else:
        raise ValueError(f"Invalid line-of-sight direction: {los}. Must be 'x', 'y', or 'z'.")

    if boxsize is None:
        return x_trans, y_trans, z_trans
    else:
        return pbc(x_trans, y_trans, z_trans, boxsize)

# ---------------------------
# Pair counting / RR / wp
# ---------------------------
class MeasureWp(object):

    """
    Rmin: minimum bin radius (cMpc/h)
    Rmax: maximum bin radius (cMpc/h)
    pimax: maximum line-of-sight separation (cMpc/h)
    out_loc: output location for binfile
    nthreads: number of threads for Corrfunc
    binfile_path: path to save binfile
    """
    def __init__(self, Rmin, Rmax, pimax, out_loc, nthreads, binfile_path):
        self.out_loc = out_loc
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.pimax = pimax
        self.nthreads = nthreads

        # Changed to 20 for homogeneity
        self.n_rp = 20
        self.binfile = binfile_path
        self.write_bin_file()
    
    # writes binfile for Corrfunc
    def write_bin_file(self):
        rp = np.logspace(np.log10(self.Rmin), np.log10(self.Rmax), self.n_rp + 1)
        with open(self.binfile, 'w') as outfile:
            #n_rp = 15
            for ir in range(self.n_rp):
                outfile.write('%g %g \n' % (rp[ir], rp[ir+1]))

    # Calculates uniform 
    def analytical_RR_projected(self, rp_bins, pi_bins, boxsize):
        # Shell-by-bin volume: 2Δπ * π(r_out^2 - r_in^2)
        dA = np.pi * (rp_bins[1:]**2 - rp_bins[:-1]**2)[:, None]     # shape (nrp, 1)
        dPi = (pi_bins[1:] - pi_bins[:-1])[None, :] * 2.0            # shape (1, npi)
        dV = dA * dPi                                                # (nrp, npi)
        Vbox = boxsize**3
        return dV / Vbox

    def measure_wp(self, x1, y1, z1, x2, y2, z2, boxsize, autocorr=True):
        rp_bins = np.logspace(np.log10(self.Rmin), np.log10(self.Rmax), self.n_rp + 1)
        pi_bins = np.linspace(0, self.pimax, int(self.pimax) + 1)  # width 1 up to pimax (assumed integer)

        kwargs = {
            "autocorr": int(autocorr),
            "nthreads": self.nthreads,
            "pimax": int(self.pimax),
            "binfile": self.binfile,
            "X1": x1.astype(float),
            "Y1": y1.astype(float),
            "Z1": z1.astype(float),
            "boxsize": (np.float64(boxsize), np.float64(boxsize), np.float64(boxsize)),
            "output_rpavg": True,
            "periodic": True,
            "verbose": False,
        }
        if not autocorr:
            kwargs["X2"] = x2.astype(float)
            kwargs["Y2"] = y2.astype(float)
            kwargs["Z2"] = z2.astype(float)

        results_DD = countpairs_rp_pi(**kwargs)[0]
        DD = np.array([item[4] for item in results_DD]).reshape((self.n_rp, int(self.pimax)))
        rpavg = np.array([item[2] for item in results_DD])  # unused, but kept

        RR = self.analytical_RR_projected(rp_bins, pi_bins, boxsize)

        if autocorr:
            DD_norm = DD / (len(x1) * len(x1))
        else:
            DD_norm = DD / (len(x1) * len(x2))

        wp_rpavg = 2.0 * np.sum((DD_norm / RR - 1.0), axis=1)
        rp_mid = 0.5 * (rp_bins[:-1] + rp_bins[1:])
        return rp_mid, wp_rpavg
    
    def measure_DD(self, x1, y1, z1, x2, y2, z2, boxsize, autocorr=True, periodic=False):

        rp_bins = np.logspace(np.log10(self.Rmin), np.log10(self.Rmax), self.n_rp + 1)

        kwargs = {
            "autocorr": int(autocorr),
            "nthreads": self.nthreads,
            "pimax": self.pimax,
            "binfile": self.binfile,
            "X1": x1.astype(float),
            "Y1": y1.astype(float),
            "Z1": z1.astype(float),
            "output_rpavg": True,
            "periodic": periodic,
            "verbose": False
        }

        if not autocorr:
            kwargs["X2"] = x2.astype(float)
            kwargs["Y2"] = y2.astype(float)
            kwargs["Z2"] = z2.astype(float)
        if periodic:
            kwargs["boxsize"] = (np.float64(boxsize), np.float64(boxsize), np.float64(boxsize))
        else:
            kwargs["boxsize"] = (-1., -1., -1.) 

        results_DD = countpairs_rp_pi(**kwargs)[0]
        #print(results_DD)
        DD = np.array([item[4] for item in results_DD]).reshape((self.n_rp, self.pimax))

        rp_mid = 0.5 * (rp_bins[:-1] + rp_bins[1:])

        return rp_mid, DD

    def wp_cg(self, xh, yh, zh, xg, yg, zg, boxsize):
        # applies periodic boundary conditions
        xh, yh, zh = pbc(xh, yh, zh, boxsize)
        xg, yg, zg = pbc(xg, yg, zg, boxsize)
        return self.measure_wp(xh, yh, zh, xg, yg, zg, boxsize, autocorr=False)

    def wp_auto(self, xg, yg, zg, boxsize):
        xg, yg, zg = pbc(xg, yg, zg, boxsize)
        return self.measure_wp(xg, yg, zg, None, None, None, boxsize, autocorr=True)

    def save_results_to_fits(self, path_base, rp, wp, tag=""):
        output_filename = f"{path_base}_wp_results_{tag}.fits"
        temp_filename = output_filename.replace('.fits', '_temp.fits')
        data = np.array(list(zip(rp, wp)), dtype=[('rp', 'f8'), ('wp', 'f8')])
        fitsio.write(temp_filename, data, clobber=True)
        os.rename(temp_filename, output_filename)
        print(f"Results saved to {output_filename}")

    def compute_and_save(self, xh, yh, zh, xg, yg, zg, galaxy_path_base, cluster_path_base, boxsize):
        out_g = f"{galaxy_path_base}_wp_results_gg.fits"
        if not os.path.exists(out_g):
            rp_gg, wp_gg = self.wp_auto(xg, yg, zg, boxsize)
            self.save_results_to_fits(galaxy_path_base, rp_gg, wp_gg, tag="gg")

        out_cg = f"{cluster_path_base}_wp_results_cg.fits"
        if not os.path.exists(out_cg):
            rp_cg, wp_cg = self.wp_cg(xh, yh, zh, xg, yg, zg, boxsize)
            self.save_results_to_fits(cluster_path_base, rp_cg, wp_cg, tag="cg")

def load_parameters(yml_file):
    with open(yml_file, 'r') as stream:
        try:
            para = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Error loading YAML file: {exc}")
    return para

# Note: Deleted get_readcat() and if __name__ == "__main__": blocks: use imports from inaccessible path 
