import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
from src.subvolume_utils import WpCalculations

from dotenv import load_dotenv
load_dotenv()

g_base = os.getenv('galaxy_base')
c_base = os.getenv('cluster_base')

import numpy as np
if __name__ == "__main__":
    i = int(sys.argv[1])
    galaxies = f"{g_base}base_c000_ph{i:03d}/z0p300/model_hod000000/gals.fit"
    clusters = f"{c_base}base_c000_ph{i:03d}/z0p300/model_hod000000/richness_q180_bg.fit"
    current = WpCalculations()
    rp_sim, wp_sim = current.wp_cross_calc(galaxies, clusters, h_var = 'lambda', counts = True)
    np.savez(f"wp_cluster_counts_ph{i:03d}.npz", rp=rp_sim, wp=wp_sim)
