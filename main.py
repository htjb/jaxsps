from jaxsps.isochrones import read_collection_of_isochrones
import pandas as pd
import matplotlib.pyplot as plt


iso = read_collection_of_isochrones('data/MIST_v1.2_vvcrit0.4_full_isos/')

metalicity = 2.5
age = 9.0
isochrone = iso.isochrone[iso.metalicity == metalicity]
idx = (isochrone.log10_isochrone_age_yr == age)
plt.plot(isochrone.log_L[idx], isochrone.log_Teff[idx], 
         label=f'Metallicity: {metalicity} Zsun',
         marker='o', markersize=0.5)
plt.xlabel('Log Luminosity (L/Lsun)')
plt.ylabel('Log Effective Temperature (Teff)')
plt.title(f'Isochrone at Age {age} Gyr')
plt.show()