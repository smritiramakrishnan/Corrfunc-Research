import numpy as np
from Corrfunc.theory import wp
from Corrfunc.theory.DDrppi import DDrppi
import fitsio
import statistics

from dotenv import load_dotenv
import os

load_dotenv()

# creates and returns np arrays positions from fitsio data
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

# Semester 1: getting auto-correlated rp and wp of data
def generate_results(x, y, z, nbins = 20, print_res = False):
    
    boxsize = max(x)
    pimax = 100
    nthreads = 4

    rmin = 0.1
    rmax = 20.0

    rbins = np.logspace(np.log10(0.1), np.log10(rmax), nbins + 1)
    wp_results = wp(boxsize, pimax, nthreads, rbins, x, y, z, verbose = True, output_rpavg = True)

    if (print_res):
        print("#############################################################################")
        print("  rmin           rmax            rpavg             wp            npairs")
        print("#############################################################################")
        print(wp_results)
        
    rpavg_res = []
    wp_res = []
    for result in wp_results:
        rpavg_res.append(result[2])
        wp_res.append(result[3])
    return wp_res, rpavg_res

# Semester 1 (used for comparing Sem 2): getting cross-correlated wp and rp
def wp_pairs_cross(X1, Y1, Z1, X2, Y2, Z2, pimax, bins, boxsize):
    wp = []
    rpavg = []
    autocorr=0
    nthreads=4
    N1 = len(X1)
    N2 = len(X2)

    # Produces rppi 
    DD_counts = DDrppi(autocorr, nthreads, pimax, bins, X1, Y1, Z1,
               X2=X2, Y2=Y2, Z2=Z2,periodic = True, boxsize = boxsize, 
               verbose=True, output_rpavg=True)
    
    # Looping through dd_counts/pimax: b/c pimax indicates max z axis comp length, only 
    # looking at range of 1 pimax/bin per iteration
    for n in range(0, int(len(DD_counts)/int(pimax))):

        # var to store calculated wp value
        wp_ = 0
        rpavg_ = 0
        total_pairs = 0

        # In order to properly access each DD_ count
        for m in range(0, int(pimax)):

            # Which DD_count point to get; If above loop were just DD_counts, index = current DD_count
            index = n*int(pimax) + m

            # DD_ = npairs
            DD_ = DD_counts[index][4]


            # N1 * N2 = total num pairs in xy plane of first dataset
            # /(boxsize**3) = total area of 3d map
            # * 2 ??? 
            # pi * (DD_counts[index][1] ** 2 - DD_counts[index][0] ** 2) 
            # Overall: Calculating npairs random
            RR_ = N1*N2/(boxsize**3.) * 2 * np.pi * (DD_counts[index][1]**2. - DD_counts[index][0]**2.)

            # added to get weighted rpavg
            rpavg_ += DD_counts[index][2] * DD_
            total_pairs += DD_

            # (rp, pi)*dpi = (DD/RR) - 1 = npairs simulation/npairs random - 1
            # In Corrunc research paper: wp(rp) = 2 * [0 to pimax]integral ((rp, pi)*dpi) ==> wp(rp) = 2 * [0 to pimax]integral
            # I think according to formula that wp_ should be multiplied by 2 after summation
            wp_ += 2.0 * ( DD_ / RR_ - 1)
            
        wp.append(wp_)
        rpavg.append(rpavg_/total_pairs)
    return wp, rpavg

# getting mean and stdev of multiple simulation box wp_rp results
def error_data(rpavg_, wp_):
    wp_mean = []
    rpavg_mean = []
    stdev_ = []
    for i in range(len(wp_[0])):
        all_wp = []
        all_rpavg = []
        for arr in range(len(wp_)):
            all_wp.append(wp_[arr][i])
            all_rpavg.append(rpavg_[arr][i])
        wp_mean.append(statistics.mean(all_wp))
        rpavg_mean.append(statistics.mean(all_rpavg))
        stdev_.append(statistics.stdev(all_wp))
    return rpavg_mean, wp_mean, stdev_

# Semester 1: getting all wp and rp values across 20 simualtion boxes
def simulate(rbins, scale_, max_ = 20, range_ = False, var = '', min = 0, gal_sub = 1, g_base = os.getenv('galaxy_base'), h_base = os.getenv('halo_base')): 
    all_wp = []
    all_rpavg = []
    n_gal = []
    for j in range(0, max_):
        galaxies = f"{g_base}base_c000_ph{j:03}/z0p300/model_hod000000/gals.fit"
        halos = f"{h_base}base_c000_ph{j:03}/z0p300/halos_3e+12.fit"
        g_x, g_y, g_z = create_data(galaxies)
        boxsize = max(max(g_z),max(g_y), max(g_x))
        if (gal_sub < 1):
            subg_x = []
            subg_y = []
            subg_z = []
            for i in range (len(g_x)):
                prob = np.random.rand()
                if (prob < gal_sub):
                    subg_x.append(g_x[i])
                    subg_y.append(g_y[i])
                    subg_z.append(g_z[i])
            g_x, g_y, g_z = np.array(subg_x, dtype=np.float64, order='C'), np.array(subg_y, dtype=np.float64, order='C'), np.array(subg_z, dtype=np.float64, order='C')
            n_gal.append(len(g_x)/(boxsize**3))
        if range_: 
            h_x, h_y, h_z = create_data(halos, var = var, min_ = min)
        else: 
            h_x, h_y, h_z = create_data(halos)
        norm = np.random.normal(size = len(g_z), scale = scale_)
        g_z = g_z + norm
        g_z = g_z % boxsize
        wp_current, rpavg_current = wp_pairs_cross(g_x, g_y, g_z, h_x, h_y, h_z, pimax = 100, bins = rbins, boxsize = boxsize)
        all_wp.append(wp_current)
        all_rpavg.append(rpavg_current)
    if (gal_sub < 1):
        return all_wp, all_rpavg, n_gal
    return all_wp, all_rpavg
