from jaxsps.ssp_model import ssp
from jaxsps.spectra import read_collection_of_spectra
import jax.numpy as jnp
import matplotlib.pyplot as plt


metalicity = -0.25
age = 5.0

spectra = read_collection_of_spectra('data/MILES_library_v9.1_ASCII/')

spectra, lam = ssp(metalicity, age)
print(spectra)

plt.plot(lam, spectra)
plt.loglog()
plt.show()

