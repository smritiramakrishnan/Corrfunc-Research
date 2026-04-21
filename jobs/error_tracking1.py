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
    wp_old, rp_old = current.error_wp_calc_oldv(galaxies, halos)
    np.savez(f"wp_testing_old_halos{i:03d}.npz", rp_old = rp_old, wp_old = wp_old)