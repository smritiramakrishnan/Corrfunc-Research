import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
from src.subvolume_utils import WpCalculations

from dotenv import load_dotenv
load_dotenv()

g_base = os.getenv('galaxy_base')
h_base = os.getenv('halo_base')

import numpy as np

if __name__ == "__main__":
    i = int(sys.argv[1])
    galaxies = f"{g_base}base_c000_ph{i:03d}/z0p300/model_hod000000/gals.fit"
    halos = f"{h_base}base_c000_ph{i:03d}/z0p300/halos_3e+12.fit"
    current = WpCalculations()
    rp_sim, wp_sim = current.wp_cross_calc(galaxies, halos, counts = True)
    np.savez(f"wp_counts_ph{i:03d}.npz", rp_avg = rp_sim, wp_ = wp_sim)