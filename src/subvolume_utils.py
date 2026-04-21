import fitsio
import numpy as np

import os
import sys

sys.path.append(os.path.abspath(os.path.join('..')))
from src.calc_wps_sub_copy import MeasureWp
from Corrfunc.theory import wp
from Corrfunc.theory.DDrppi import DDrppi
from src.correlation_tools import create_data

# returns x, y, z, and count_var
def create_data_counts(fname, count_var = 'mass'):
    data = fitsio.read(fname)
    x = np.array(data['px'], dtype=np.float64, order='C')
    y = np.array(data['py'], dtype=np.float64, order='C')
    z = np.array(data['pz'], dtype=np.float64, order='C')
    count_var_data = np.array(data[count_var], dtype=np.float64, order='C')
    return x, y, z, count_var_data

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

# Getting indexes and boundaries for subvolumes
def subvolume_calc(boxsize = 0, fname = '', x = None, y = None, z = None, sub_ = 4,  var = 'mass', min_ = 0, counts = 0, hist = False):

    if fname != '':
        if(counts > 0):
            #count_var could be mass or richness
            x, y, z, count_var = create_data_counts(fname, var)
        else:
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

    if (counts > 0):
    # gives indeces in order of bin ([2, 7, 5, 6: bin 1, 8, 3, 4: bin 2...)
        full_bin_id = np.argsort(bin_id)
        
        halo_idx = []
        for i in range(sub_**3):
            # subvol is indeces of halos in current subvol
            subvol = full_bin_id[sections[i]: sections[i+1]]
            # idx is indeces of highest 1831 halos in current subvol
            idx = (np.argpartition(count_var[subvol], -counts))[-counts:]
            # count_var is 83 mill in size, count_var[subvol] is the size of subvolume
            # appends indeces of top 1831 halos within subvol to index tracker
            halo_idx.append(subvol[idx])
        
        # creates 1d array
        halo_idx = np.concatenate(halo_idx)

        halo_idx = halo_idx.astype(np.int64)
        x = x[halo_idx]
        y = y[halo_idx]
        z = z[halo_idx]

        bin_id = bin_id[halo_idx]

        sections = np.cumsum(np.bincount(bin_id, minlength = sub_**3))
        sections = np.concatenate([[0], sections])

        bin_id = np.argsort(bin_id)

        if(hist):
            return x, y, z, halo_idx, sections, bin_id
        else:
            return x, y, z, sections, bin_id
    
    bin_id = np.argsort(bin_id)
    
    return x, y, z, sections, bin_id

class WpCalculations():

    def __init__(self):
        self.sub_ = 4

    # Uses semester 1 code (modified slightly) to get wp rp (mostly used for plotting histograms)
    def error_wp_calc_oldv(self, galaxies, cluster, clusters = True, mass = False):
        wp = []
        rpavg = []
        autocorr=0
        nthreads=4
        galaxy_data = fitsio.read(galaxies)
        X1 = np.array(galaxy_data['px'], dtype=np.float64)#, order='C')
        Y1 = np.array(galaxy_data['py'], dtype=np.float64)#, order='C')
        Z1 = np.array(galaxy_data['pz'], dtype=np.float64)#, order='C')

        cluster_data = fitsio.read(cluster)
        if (clusters):
            if (mass):
                richness = cluster_data['mass_host']
            else:
                richness = cluster_data['lambda']
        else:
            richness = cluster_data['mass']

        temp_counts = -1831 * 64
        idx = np.argsort(richness)[temp_counts:]
        X2 = np.array((cluster_data['px'])[idx], dtype=np.float64)#, order='C')
        Y2 = np.array((cluster_data['py'])[idx], dtype=np.float64)#, order='C')
        Z2 = np.array((cluster_data['pz'])[idx], dtype=np.float64)#, order='C')

        if (clusters):
            mass = cluster_data['mass_host'][idx]
        else:
            mass = richness[idx]
        print('%e'%np.mean(mass))

        N1 = len(X1)
        N2 = len(X2)

        nbins = 20
        rmin = 0.1
        rmax = 20.0
        rbins = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)

        boxsize = np.max(X1)
        pimax = 100

        DD_counts = DDrppi(autocorr, nthreads, pimax, rbins, X1, Y1, Z1,
                X2=X2, Y2=Y2, Z2=Z2,periodic = True, boxsize = boxsize, 
                verbose=True, output_rpavg=True)
        
        for n in range(0, int(len(DD_counts)/int(pimax))):

            wp_ = 0
            rpavg_ = 0
            total_pairs = 0
            for m in range(0, int(pimax)):
                index = n*int(pimax) + m
                DD_ = DD_counts[index][4]
                RR_ = N1*N2/(boxsize**3.) * 2 * np.pi * (DD_counts[index][1]**2. - DD_counts[index][0]**2.)
                rpavg_ += DD_counts[index][2] * DD_
                total_pairs += DD_
                wp_ += 2.0 * ( DD_ / RR_ - 1)
                
            wp.append(wp_)
            rpavg.append(rpavg_/total_pairs)
        return wp, rpavg, mass 
   
    def wp_cross_calc(self, galaxies, halos, h_var = '', h_min = 0, counts = False,
                    wps = MeasureWp(Rmin = 0.1, Rmax = 20, pimax = 100, out_loc = None, nthreads = 4, binfile_path = "rp_bins.txt")):
        
        all_xg, all_yg, all_zg, sections_g, bin_id_g = subvolume_calc(fname = galaxies, sub_ = self.sub_)

        if (counts):
            all_xh, all_yh, all_zh, sections_h, bin_id_h = subvolume_calc(fname = halos, sub_ = self.sub_, counts = 1831, var = h_var)
        else:
            all_xh, all_yh, all_zh, sections_h, bin_id_h = subvolume_calc(fname = halos, sub_ = self.sub_, var = h_var, min_ = h_min)

        boxsize = np.max(all_xg)

        all_xgr, all_ygr, all_zgr, sections_gr, bin_id_gr = rand_subvolume_calc(boxsize = boxsize, len_ = len(all_xg))
        all_xcr, all_ycr, all_zcr, sections_cr, bin_id_cr = rand_subvolume_calc(boxsize = boxsize, len_ = len(all_xh))

        rp_all = []
        wp_all = []

        for sub_index in range(0, self.sub_**3):

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

            surr_vols = get_surr(sub_index, sub_ = self.sub_)
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
    