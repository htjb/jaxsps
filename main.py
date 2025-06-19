from jaxsps.isochrones import read_isochrones
import pandas as pd
import matplotlib.pyplot as plt


iso = read_isochrones('data/MIST_v1.2_vvcrit0.4_full_isos/' + 
                     'MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.4_full.iso')

for f in iso._fields:
    print(f)
plt.plot(iso.log_Teff, iso.log_L, marker='o', linestyle='None')
plt.gca().invert_xaxis()
plt.show()