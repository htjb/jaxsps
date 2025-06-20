from collections import namedtuple
import jax.numpy as jnp

def read_individual_spectrum(file_path: str) -> namedtuple:
    """
    Reads spectrum data from a text file and returns it as a namedtuple.
    
    Parameters:
    file_path (str): The path to the text file containing spectrum data.
    
    Returns:
    namedtuple: A namedtuple containing the spectrum data with fields 'wavelength' and 'flux'.
    """

    spectrum = jnp.loadtxt(file_path)
    Spectrum = namedtuple('Spectrum', ['wavelength', 'flux'])
    spectrum_data = Spectrum(spectrum[:, 0], spectrum[:, 1])
    return spectrum_data
