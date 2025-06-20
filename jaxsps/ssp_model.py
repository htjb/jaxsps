from jaxsps.isochrones import read_collection_of_isochrones
from jaxsps.spectra import read_collection_of_spectra
from jaxsps.imf import salpeter_imf
import jax.numpy as jnp
from scipy.spatial import KDTree
import jax

def ssp(metalicity: float, age: float):

    def interp(Teff, logg):
        target = [10**Teff, logg, metalicity]
        dists, idxs = tree.query(target, k=5)
        weights = 1/(dists + 1e-6)
        weights = weights/weights.sum()

        interp_spectrum = jnp.sum(weights[:, None]*spectra.flux[idxs], axis=0)
        return interp_spectrum

    
    iso = read_collection_of_isochrones('data/MIST_v1.2_vvcrit0.4_full_isos/')

    isochrone = iso.isochrone[iso.metalicity == metalicity]
    idx = (isochrone.log10_isochrone_age_yr == age)

    spectra = read_collection_of_spectra('data/MILES_library_v9.1_ASCII/')

    tree = KDTree(jnp.array([spectra.Teff,
                                         spectra.logg,
                                         spectra.FeH]).T)

    
    pop_spectra = jnp.array([interp(Teff, logg) 
                   for Teff, logg in zip(isochrone.log_Teff[idx],
                                        isochrone.log_g[idx])])
    
    imf = jax.vmap(salpeter_imf, in_axes=0)
    imf_weight = imf(isochrone.initial_mass[idx])

    population_spectrum = jnp.average(pop_spectra, 
                                      weights=imf_weight, axis=0)

    return population_spectrum, spectra.wavelength[0]