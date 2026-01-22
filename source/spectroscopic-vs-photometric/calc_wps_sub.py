#!/usr/bin/env python
import os
import numpy as np
import fitsio
from scipy.interpolate import interp1d
from Corrfunc._countpairs import countpairs_rp_pi
import sys
sys.path.append('/users/shuleic/scripts/pythonscripts/Unify_HOD_pipeline/python_scripts/utils/')
from arg_parser_utils import parse_arguments
args = parse_arguments()
import shutil

from periodic_boundary_condition import periodic_boundary_condition
from astropy.io import fits
from astropy.table import Table
import yaml
from concurrent.futures import ProcessPoolExecutor
import argparse
from colossus.cosmology import cosmology

def get_zname(redshift):
    return f"z{str(redshift).replace('.', 'p')}00"

def pbc(xg, yg, zg, boxsize):
    xg = np.mod(xg, boxsize)
    yg = np.mod(yg, boxsize)
    zg = np.mod(zg, boxsize)
    return xg, yg, zg

def _ap_scalings(cosmo_sim, cosmo_fid, h_sim, h_fid, z):
    """
    S_M (transverse) and S_C (LOS) using Colossus distances (h^-1 Mpc).
    Since Mpc = (h^-1 Mpc)/h, the *Mpc* ratio carries (h_sim/h_fid).
    """
    D_C_sim_h = cosmo_sim.comovingDistance(0.0, z, transverse=False)  # h^-1 Mpc
    D_C_fid_h = cosmo_fid.comovingDistance(0.0, z, transverse=False)  # h^-1 Mpc
    D_M_sim_h = cosmo_sim.comovingDistance(0.0, z, transverse=True)   # h^-1 Mpc
    D_M_fid_h = cosmo_fid.comovingDistance(0.0, z, transverse=True)   # h^-1 Mpc

    # Mpc ratios: (D_fid_h / h_fid) / (D_sim_h / h_sim) = (h_sim/h_fid) * (D_fid_h / D_sim_h)
    S_C = (h_sim / h_fid) * (D_C_fid_h / D_C_sim_h)
    S_M = (h_sim / h_fid) * (D_M_fid_h / D_M_sim_h)
    return S_M, S_C

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
    if mask is None:
        mask = np.ones(len(data['px']), dtype=bool)

    # Positions: keep in comoving Mpc/h
    px = data['px'][mask]
    py = data['py'][mask]
    pz = data['pz'][mask]

    if data_type == 'cluster':
        if boxsize is None:
            return px, py, pz
        else:
            return pbc(px, py, pz, boxsize)

    Ez_fid = cosmo_fid.Ez(redshift)
    velocity_factor = (1 + redshift) / Ez_fid / 100.  # cMpc/h / (km/s)
    
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
    def __init__(self, Rmin, Rmax, pimax, out_loc, nthreads, binfile_path):
        self.out_loc = out_loc
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.pimax = pimax
        self.nthreads = nthreads

        self.n_rp = 15
        self.binfile = binfile_path
        self.write_bin_file()

    def write_bin_file(self):
        rp = np.logspace(np.log10(self.Rmin), np.log10(self.Rmax), self.n_rp + 1)
        with open(self.binfile, 'w') as outfile:
            for ir in range(self.n_rp):
                outfile.write('%g %g \n' % (rp[ir], rp[ir+1]))

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
#         print(results_DD)
        DD = np.array([item[4] for item in results_DD]).reshape((self.n_rp, self.pimax))

        rp_mid = 0.5 * (rp_bins[:-1] + rp_bins[1:])

        return rp_mid, DD

    def wp_cg(self, xh, yh, zh, xg, yg, zg, boxsize):
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

def get_readcat(nbody, nbody_loc, redshift):
    if nbody == 'mini_uchuu':
        from read_mini_uchuu import ReadMiniUchuu
        return ReadMiniUchuu(nbody_loc, redshift)
    elif nbody == 'uchuu':
        from read_uchuu import ReadUchuu
        return ReadUchuu(nbody_loc, redshift)
    elif nbody == 'abacus_summit':
        # if args.use_HOD:
        #     from read_abacus_summit_HOD import ReadAbacusSummit
        # else:
        from read_abacus_summit import ReadAbacusSummit
        return ReadAbacusSummit(nbody_loc, redshift)
    elif nbody == 'tng_dmo':
        from read_tng_dmo import ReadTNGDMO
        halofinder = para.get('halofinder', 'rockstar')
        return ReadTNGDMO(nbody_loc, halofinder, redshift)
    else:
        raise ValueError(f"Unsupported nbody value: {nbody}")

if __name__ == "__main__":
    para = load_parameters(args.yml_file)
    para_name  = args.para_name
    h_sim = para['hubble']

    # Simulation cosmology (used as "true")
    params_LCDM = {'de_model': 'lambda', 'Om0': para['OmegaM'], 'H0': 100*h_sim,
                   'Ob0': para['OmegaB'], 'sigma8': para['sigma8'], 'ns': para['ns'], 'flat': True}
    cosmology.addCosmology('LCDM', **params_LCDM)
    params = {'de_model': 'w0wa', 'Om0': para['OmegaM'], 'w0': para['w0'], 'wa': para['wa'],
              'H0': 100*h_sim, 'Ob0': para['OmegaB'], 'sigma8': para['sigma8'],
              'ns': para['ns'], 'flat': True}
    cosmology.addCosmology('w0waCDM', **params)
    cosmo_sim = cosmology.setCosmology('w0waCDM')  # use this as "true" unless you want LCDM

    # Fiducial cosmology (used to "observe")
    # if args.use_fid:
    h_fid = args.h
    params_fid = {'de_model': 'w0wa', 'Om0': args.Om, 'w0': args.w0, 'wa': args.wa,
                    'H0': 100*h_fid, 'Ob0': 0.0493, 'sigma8': 0.8111, 'ns': 0.9649, 'flat': True}
    cosmology.addCosmology('FidCosmo', **params_fid)
    cosmo_fid = cosmology.setCosmology('FidCosmo')
    # else:
    #     # if not using a separate fid, observe in the sim cosmology itself
    #     cosmo_fid = cosmo_sim
    #     h_fid = h_sim
    boxsize = para['boxsize']

    depth = args.depth
    pec_vel    = args.pec_vel
    pec_suffix = "_pecvel" if pec_vel else ""
    redshift   = para['redshift']
    zname      = get_zname(redshift)
    phase_name = args.phase_name
    is_sub     = args.is_sub
    sub_index  = args.sub_index
    HOD_index  = args.HOD_index
    use_200m   = args.use_200m
    los        = args.los
    los_suffix = '' if los == 'z' else f'_LOS{los}'
    new_suffix = '_new' if args.new else ''

    output_loc = os.path.join(para['output_loc'], f'HOD_z{redshift}_{phase_name}_{HOD_index}_{para_name}')
    if is_sub:
        output_loc = os.path.join(para['output_loc'], f'HOD_z{redshift}_{phase_name}_{HOD_index}_{para_name}_sub{sub_index}')
    os.makedirs(output_loc, exist_ok=True)
    out_path = output_loc

    model_name = para['model_name']
    seed = args.seed

    cluster_path = os.path.join(out_path, f'model_{model_name}_d{depth}{pec_suffix}/matched_richness_d{depth}{pec_suffix}_seed{seed}{los_suffix}_redMaPPer_lg20.fit')
    galaxy_path = os.path.join(out_path, f'model_{model_name}/gals_seed{seed}.fit')
    if is_sub:
        galaxy_random_path = os.path.join(output_loc, f'model_{model_name}/gals_seed{seed}_randoms.fit')
        cluster_random_path = os.path.join(output_loc, f'model_{model_name}_d{depth}{pec_suffix}/matched_richness_d{depth}{pec_suffix}_seed{seed}_redMaPPer_lg20_randoms.fit')
    else:
        galaxy_random_path = None
        cluster_random_path = None 
    def read_data_and_compute(redshift, galaxy_path, cluster_path, out_path, is_sub, sub_index, 
                              galaxy_random_path, cluster_random_path, pec_vel=False, los='z', 
                              unit='hicMpc', new_suffix=''):
        unit_wp_suffix = f'_{unit}'
        cluster_base = os.path.splitext(cluster_path)[0]
        galaxy_base  = os.path.splitext(galaxy_path)[0]
        galaxy_base  = f"{galaxy_base}{pec_suffix}{los_suffix}{unit_wp_suffix}{new_suffix}"

        out_cg = f"{cluster_base}{unit_wp_suffix}{new_suffix}_wp_results_cg.fits"
        out_gg = f"{galaxy_base}_wp_results_gg.fits"
        if os.path.exists(out_cg) and os.path.exists(out_gg):
            return

        with fits.open(galaxy_path) as hdul:
            galaxies = hdul[1].data
        with fits.open(cluster_path) as hdul:
            clusters = hdul[1].data

        xg, yg, zg = transform_coordinates(galaxies, los, redshift, cosmo_fid, pec_vel, boxsize, data_type='galaxy')
        xh, yh, zh = transform_coordinates(clusters, los, redshift, cosmo_fid, False, boxsize, data_type='cluster')

        if galaxy_random_path is not None:
            with fits.open(galaxy_random_path) as hdul: 
                xgr, ygr, zgr = transform_coordinates(hdul[1].data, los, redshift, cosmo_fid, False, None if is_sub else boxsize)

        if cluster_random_path is not None:
            with fits.open(cluster_random_path) as hdul:
                xcr, ycr, zcr = transform_coordinates(hdul[1].data, los, redshift, cosmo_fid, False, None if is_sub else boxsize)
        
        if pec_vel:
            galaxy_path = os.path.splitext(galaxy_path)[0] + "_pecvel.fits"

        rp_min = args.rp_min  # cMpc
        rp_max = args.rp_max  # cMpc
        pi_max = args.pi_max  # cMpc
        S_M, S_C = _ap_scalings(cosmo_sim, cosmo_fid, h_sim, h_fid, redshift)
        rp_min_sim = rp_min * h_sim / S_M if unit == 'hicMpc' else rp_min / S_M # only cMpc or cMpc/h
        rp_max_sim = rp_max * h_sim / S_M if unit == 'hicMpc' else rp_max / S_M # only cMpc or cMpc/h
        pi_max_sim = pi_max * h_sim / S_C if unit == 'hicMpc' else pi_max / S_C # only cMpc or cMpc/h
        pi_max_sim_int = int(np.ceil(pi_max_sim))
        binfile_path = os.path.join(out_path, f'rp_bins_wp{unit_wp_suffix}_SM{S_M:.6f}_pi{pi_max_sim_int}{new_suffix}.dat')

        wps = MeasureWp(Rmin=rp_min_sim, Rmax=rp_max_sim, pimax=pi_max_sim_int, out_loc=out_path, 
                        nthreads=args.nthreads, binfile_path=binfile_path)

        if not is_sub:
            wps.compute_and_save(xh, yh, zh, xg, yg, zg,f"{galaxy_base}", f"{cluster_base}{unit_wp_suffix}{new_suffix}", boxsize)
        else:
            rp_base, DD_base = wps.measure_DD(xg, yg, zg, None, None, None, boxsize, autocorr=True, periodic=False)
            _, RD_base = wps.measure_DD(xgr, ygr, zgr, xg, yg, zg, boxsize, autocorr=False, periodic=False)
            _, DR_base = wps.measure_DD(xg, yg, zg, xgr, ygr, zgr, boxsize, autocorr=False, periodic=False)
            _, RR_base = wps.measure_DD(xgr, ygr, zgr, None, None, None, boxsize, autocorr=True, periodic=False)
            DDs_cross = []
            DRs_cross = []
            RDs_cross = []
            RRs_cross = []
            for index in range(27):
                if index != sub_index:
                    with fits.open(galaxy_path.replace(f'sub{sub_index}',f'sub{index}')) as hdul:
                        xgn, ygn, zgn = transform_coordinates(hdul[1].data, los, redshift, cosmo_fid, pec_vel, boxsize, data_type='galaxy')
                        _, DD_cross = wps.measure_DD(xg, yg, zg, xgn, ygn, zgn, boxsize, autocorr=False, periodic=True)
                        DDs_cross.append(DD_cross)
                        _, RD_cross = wps.measure_DD(xgr, ygr, zgr, xgn, ygn, zgn, boxsize, autocorr=False, periodic=True)
                        RDs_cross.append(RD_cross)
                    with fits.open(galaxy_random_path.replace(f'sub{sub_index}',f'sub{index}')) as hdul:
                        xgn, ygn, zgn = transform_coordinates(hdul[1].data, los, redshift, cosmo_fid, False, boxsize, data_type='galaxy')
                        _, DR_cross = wps.measure_DD(xg, yg, zg, xgn, ygn, zgn, boxsize, autocorr=False, periodic=True)
                        DRs_cross.append(DR_cross)
                        _, RR_cross = wps.measure_DD(xgr, ygr, zgr, xgn, ygn, zgn, boxsize, autocorr=False, periodic=True)
                        RRs_cross.append(RR_cross)
                    del xgn, ygn, zgn, DD_cross, RD_cross, DR_cross, RR_cross

            DD_normalized = 1/(len(xg)*len(xg))*(DD_base + 0.5*np.sum(DDs_cross, axis=0))
            DR_normalized = 1/(len(xg)*len(xgr))*(DR_base + 0.5*np.sum(DRs_cross, axis=0))
            RD_normalized = 1/(len(xgr)*len(xg))*(RD_base + 0.5*np.sum(RDs_cross, axis=0))
            RR_normalized = 1/(len(xgr)*len(xgr))*(RR_base + 0.5*np.sum(RRs_cross, axis=0))
            wp_rpavg = 2.0 * np.sum((DD_normalized - DR_normalized - RD_normalized) / RR_normalized + 1, axis=1)
            wps.save_results_to_fits(f"{galaxy_base}", rp_base, wp_rpavg, tag='gg')
            
            rp_base, DD_base = wps.measure_DD(xh, yh, zh, xg, yg, zg, boxsize, autocorr=False, periodic=False)
            _, RD_base = wps.measure_DD(xcr, ycr, zcr, xg, yg, zg, boxsize, autocorr=False, periodic=False)
            _, DR_base = wps.measure_DD(xh, yh, zh, xgr, ygr, zgr, boxsize, autocorr=False, periodic=False)
            _, RR_base = wps.measure_DD(xcr, ycr, zcr, xgr, ygr, zgr, boxsize, autocorr=False, periodic=False)
            DDs_cross = []
            DRs_cross = []
            RDs_cross = []
            RRs_cross = []
            for index in range(27):
                if index != sub_index:
                    with fits.open(galaxy_path.replace(f'sub{sub_index}',f'sub{index}')) as hdul:
                        xgn, ygn, zgn = transform_coordinates(hdul[1].data, los, redshift, cosmo_fid, pec_vel, boxsize, data_type='galaxy')
                        _, DD_cross = wps.measure_DD(xh, yh, zh, xgn, ygn, zgn, boxsize, autocorr=False, periodic=True)
                        DDs_cross.append(DD_cross)
                        _, RD_cross = wps.measure_DD(xcr, ycr, zcr, xgn, ygn, zgn, boxsize, autocorr=False, periodic=True)
                        RDs_cross.append(RD_cross)
                    with fits.open(galaxy_random_path.replace(f'sub{sub_index}',f'sub{index}')) as hdul:
                        xgn, ygn, zgn = transform_coordinates(hdul[1].data, los, redshift, cosmo_fid, False, boxsize, data_type='galaxy')
                        _, DR_cross = wps.measure_DD(xh, yh, zh, xgn, ygn, zgn, boxsize, autocorr=False, periodic=True)
                        DRs_cross.append(DR_cross)
                        _, RR_cross = wps.measure_DD(xcr, ycr, zcr, xgn, ygn, zgn, boxsize, autocorr=False, periodic=True)
                        RRs_cross.append(RR_cross)
                    del xgn, ygn, zgn, DD_cross, RD_cross, DR_cross, RR_cross

            DD_normalized = 1/(len(xh)*len(xg))*(DD_base + 0.5*np.sum(DDs_cross, axis=0))
            DR_normalized = 1/(len(xh)*len(xgr))*(DR_base + 0.5*np.sum(DRs_cross, axis=0))
            RD_normalized = 1/(len(xcr)*len(xg))*(RD_base + 0.5*np.sum(RDs_cross, axis=0))
            RR_normalized = 1/(len(xcr)*len(xgr))*(RR_base + 0.5*np.sum(RRs_cross, axis=0))
            wp_rpavg = 2.0 * np.sum((DD_normalized - DR_normalized - RD_normalized) / RR_normalized + 1, axis=1)
            wps.save_results_to_fits(f"{cluster_base}{unit_wp_suffix}{new_suffix}", rp_base, wp_rpavg, tag='cg')

    read_data_and_compute(redshift, galaxy_path, cluster_path, out_path, is_sub, sub_index, 
                          galaxy_random_path, cluster_random_path, pec_vel=pec_vel, los=args.los, 
                          unit=args.unit_wp, new_suffix=new_suffix)