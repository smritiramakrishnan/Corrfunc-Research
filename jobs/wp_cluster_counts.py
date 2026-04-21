import sys
from wp_functions import WpCalculations
import numpy as np
if __name__ == "__main__":
    i = int(sys.argv[1])
    galaxies = f"/projects/hywu/cluster_sims/cluster_finding/data/emulator_data/base_c000_ph{i:03d}/z0p300/model_hod000000/gals.fit"
    clusters = f"/projects/hywu/cluster_sims/cluster_finding/data/emulator_data/base_c000_ph{i:03d}/z0p300/model_hod000000/richness_q180_bg.fit"
    current = WpCalculations()
    rp_sim, wp_sim = current.wp_cross_calc(galaxies, clusters, h_var = 'lambda', counts = True)
    np.savez(f"wp_cluster_counts_ph{i:03d}.npz", rp=rp_sim, wp=wp_sim)
