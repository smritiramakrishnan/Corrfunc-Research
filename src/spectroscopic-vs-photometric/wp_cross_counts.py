from wp_functions import WpCalculations
import sys
import numpy as np

if __name__ == "__main__":
    i = int(sys.argv[1])
    galaxies = f"/projects/hywu/cluster_sims/cluster_finding/data/emulator_data/base_c000_ph{i:03d}/z0p300/model_hod000000/gals.fit"
    halos = f"/projects/hywu/cluster_sims/cluster_finding/data/AbacusSummit_base/base_c000/base_c000_ph{i:03d}/z0p300/halos_3e+12.fit"
    current = WpCalculations()
    rp_sim, wp_sim = current.wp_cross_calc(galaxies, halos, counts = True)
    np.savez(f"wp_counts_ph{i:03d}.npz", rp_avg = rp_sim, wp_ = wp_sim)