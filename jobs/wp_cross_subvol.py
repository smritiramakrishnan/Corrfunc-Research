import numpy as np
from Corrfunc.theory import wp
from Corrfunc.theory.DDrppi import DDrppi
import fitsio
import sys
from calc_wps_sub_copy import MeasureWp
from wp_functions import WpCalculations

# returns x, y, z positions arrays from fits file
def create_data(fname, var = '', min_ = 0):
    data = fitsio.read(fname)
    if (var == ''): 
        x = np.array(data['px'], dtype=np.float64, order='C')
        y = np.array(data['py'], dtype=np.float64, order='C')
        z = np.array(data['pz'], dtype=np.float64, order='C')
    else: 
        x = np.array(data['px'], dtype=np.float64, order='C')[data[var] > min_]
        y = np.array(data['py'], dtype=np.float64, order='C')[data[var] > min_]
        z = np.array(data['pz'], dtype=np.float64, order='C')[data[var] > min_]
    return x, y, z

# divides the data into subvolumes and returns the positions, section limits, and bin ids
def subvolume_calc(boxsize = 0, fname = '', x = None, y = None, z = None, sub_ = 4, var = '', min_ = 0):

    if fname != '':
        x, y, z = create_data(fname, var = var, min_ = min_)
        boxsize = np.max(x)
    elif (boxsize == 0 or (x is None or y is None or z is None)):
        raise ValueError("boxsize or xyz coords must be given")
    
    subvol_boxsize = boxsize / sub_
    bin_id = []

    d_one = np.minimum((x//subvol_boxsize).astype(np.int64), sub_ - 1)
    d_two = np.minimum((y//subvol_boxsize).astype(np.int64), sub_ - 1)
    d_three = np.minimum((z//subvol_boxsize).astype(np.int64), sub_ - 1)

    bin_id = (d_one * sub_ * sub_) + (d_two * sub_) + d_three

    sections = np.cumsum(np.bincount(bin_id, minlength = sub_**3))
    sections = np.concatenate([[0], sections])
    bin_id = np.argsort(bin_id)
    
    return x, y, z, sections, bin_id

#Creates random data and subdivides it
def rand_subvolume_calc(boxsize, len_, scale_ = 50, sub_ = 4):

    num = int(len_ * scale_)
    data = np.random.uniform(0, boxsize, size = (3, num))
    xr, yr, zr, sections, bin_id = subvolume_calc(boxsize = boxsize, x = data[0], y = data[1], z = data[2], sub_ = sub_)
    
    return xr, yr, zr, sections, bin_id

#Returns up to 26 boxes surrounding chosen box
def get_surr(current, sub_):
    surrounding = []
    x = current // (sub_ * sub_)
    y = (current // sub_) % sub_
    z = current % sub_
    
    for xrel in range(-1, 2):
        for yrel in range(-1, 2):
            for zrel in range(-1, 2):
                x_fin = x + xrel
                y_fin = y + yrel
                z_fin = z + zrel
                if((not(xrel == yrel == zrel == 0)) and (0 <= x_fin < sub_) and (0 <= y_fin < sub_) and (0 <= z_fin < sub_)):
                    surrounding.append((x_fin * sub_ * sub_) + (y_fin * sub_) + z_fin)
    return surrounding
        
def wp_cross_box(galaxies, halos, sub_ = 4, h_var = '', h_min = 0, 
                 wps = MeasureWp(Rmin = 0.1, Rmax = 20, pimax = 100, out_loc = None, nthreads = 4, binfile_path = "rp_bins.txt")):
    
    all_xg, all_yg, all_zg, sections_g, bin_id_g = subvolume_calc(fname = galaxies, sub_ = sub_)
    all_xh, all_yh, all_zh, sections_h, bin_id_h = subvolume_calc(fname = halos, sub_ = sub_, var = h_var, min_ = h_min)

    boxsize = np.max(all_xg)

    all_xgr, all_ygr, all_zgr, sections_gr, bin_id_gr = rand_subvolume_calc(boxsize = boxsize, len_ = len(all_xg))
    all_xcr, all_ycr, all_zcr, sections_cr, bin_id_cr = rand_subvolume_calc(boxsize = boxsize, len_ = len(all_xh))


    rp_all = []
    wp_all = []

    for sub_index in range(0, sub_**3):

        # get subvolume data for galaxies
        idx_gd = bin_id_g[sections_g[sub_index]: sections_g[sub_index + 1]]
        xg = all_xg[idx_gd]
        yg = all_yg[idx_gd]
        zg = all_zg[idx_gd]

        idx_gr = bin_id_gr[sections_gr[sub_index]: sections_gr[sub_index + 1]]
        xgr = all_xgr[idx_gr]
        ygr = all_ygr[idx_gr]
        zgr = all_zgr[idx_gr]
                
        ###  cross correlation starts here

        idx_hd = bin_id_h[sections_h[sub_index]: sections_h[sub_index + 1]]
        xh = all_xh[idx_hd]
        yh = all_yh[idx_hd]
        zh = all_zh[idx_hd]

        idx_cr = bin_id_cr[sections_cr[sub_index]: sections_cr[sub_index + 1]]
        xcr = all_xcr[idx_cr]
        ycr = all_ycr[idx_cr]
        zcr = all_zcr[idx_cr]

        rp_base, DD_base = wps.measure_DD(xh, yh, zh, xg, yg, zg, boxsize, autocorr=False, periodic=False)
        _, RD_base = wps.measure_DD(xcr, ycr, zcr, xg, yg, zg, boxsize, autocorr=False, periodic=False)
        _, DR_base = wps.measure_DD(xh, yh, zh, xgr, ygr, zgr, boxsize, autocorr=False, periodic=False)
        _, RR_base = wps.measure_DD(xcr, ycr, zcr, xgr, ygr, zgr, boxsize, autocorr=False, periodic=False)
        
        DDs_cross = []
        DRs_cross = []
        RDs_cross = []
        RRs_cross = []

        surr_vols = get_surr(sub_index, sub_ = sub_)
        for index in (surr_vols):
            if index != sub_index: # a different subvol
                idx_gd = bin_id_g[sections_g[index]: sections_g[index + 1]]
                xgn = all_xg[idx_gd]
                ygn = all_yg[idx_gd]
                zgn = all_zg[idx_gd]

                #with fits.open(galaxy_path.replace(f'sub{sub_index}',f'sub{index}')) as hdul:
                    #xgn, ygn, zgn = transform_coordinates(hdul[1].data, los, redshift, cosmo_fid, pec_vel, boxsize, data_type='galaxy')
                _, DD_cross = wps.measure_DD(xh, yh, zh, xgn, ygn, zgn, boxsize, autocorr=False, periodic=True)
                DDs_cross.append(DD_cross)
                _, RD_cross = wps.measure_DD(xcr, ycr, zcr, xgn, ygn, zgn, boxsize, autocorr=False, periodic=True)
                RDs_cross.append(RD_cross)
                #with fits.open(galaxy_random_path.replace(f'sub{sub_index}',f'sub{index}')) as hdul:
                    #xgn, ygn, zgn = transform_coordinates(hdul[1].data, los, redshift, cosmo_fid, False, boxsize, data_type='galaxy')
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

        rp_all.append(rp_base)
        wp_all.append(wp_rpavg)
        #wps.save_results_to_fits(f"{cluster_base}{unit_wp_suffix}{new_suffix}", rp_base, wp_rpavg, tag='cg')

    return rp_all, wp_all

if __name__ == "__main__":
    i = int(sys.argv[1])
    galaxies = f"/projects/hywu/cluster_sims/cluster_finding/data/emulator_data/base_c000_ph{i:03d}/z0p300/model_hod000000/gals.fit"
    halos = f"/projects/hywu/cluster_sims/cluster_finding/data/AbacusSummit_base/base_c000/base_c000_ph{i:03d}/z0p300/halos_3e+12.fit"
    rp_sim, wp_sim = wp_cross_box(galaxies, halos, h_var = 'mass', h_min = 10**14)
    np.savez(f"wp_ph{i:03d}.npz", rp_avg = rp_sim, wp_ = wp_sim)
