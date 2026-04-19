import sys
from wp_functions import WpCalculations
import numpy as np

if __name__ == "__main__":
    i = int(sys.argv[1])
    galaxies = f"/projects/hywu/cluster_sims/cluster_finding/data/emulator_data/base_c000_ph{i:03d}/z0p300/model_hod000000/gals.fit"
    halos = f"/projects/hywu/cluster_sims/cluster_finding/data/AbacusSummit_base/base_c000/base_c000_ph{i:03d}/z0p300/halos_3e+12.fit"
    current = WpCalculations()
    wp_old, rp_old = current.error_wp_calc_oldv(galaxies, halos)
    np.savez(f"wp_testing_old_halos{i:03d}.npz", rp_old = rp_old, wp_old = wp_old)