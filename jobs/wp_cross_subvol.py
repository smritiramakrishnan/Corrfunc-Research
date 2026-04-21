import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))
from dotenv import load_dotenv
load_dotenv()

from src.subvolume_utils import WpCalculations
g_base = os.getenv('galaxy_base')
h_base = os.getenv('halo_base')

if __name__ == "__main__":
    i = int(sys.argv[1])
    galaxies = f"{g_base}base_c000_ph{i:03d}/z0p300/model_hod000000/gals.fit"
    halos = f"{h_base}base_c000_ph{i:03d}/z0p300/halos_3e+12.fit"
    rp_sim, wp_sim = WpCalculations.wp_cross_calc(galaxies, halos, h_var = 'mass', h_min = 10**14)
    np.savez(f"wp_ph{i:03d}.npz", rp_avg = rp_sim, wp_ = wp_sim)
