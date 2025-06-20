from collections import namedtuple
import jax.numpy as jnp
import numpy as np
import pandas as pd
import glob

def read_individual_spectrum(file_path: str) -> namedtuple:
    """
    Reads spectrum data from a text file and returns it as a namedtuple.
    
    Parameters:
    file_path (str): The path to the text file containing spectrum data.
    
    Returns:
    namedtuple: A namedtuple containing the spectrum data with fields 'wavelength' and 'flux'.
    """

    spectrum = np.loadtxt(file_path)
    parameters = np.loadtxt('/'.join(file_path.split('/')[:2]) + '/parameters.csv',
                             skiprows=3, dtype=str, usecols=(0, 1, 2, 3))
    col_names = ["Name", "MILES", "CaT", "SpT", "Teff", "logg", "FeH", "Refs", "L1",
                  "L2", "L3", "L4", "L5"]
    parameters = pd.read_fwf('/'.join(file_path.split('/')[:2]) + '/parameters.csv', skiprows=3,
                     names=col_names)  # skip header decorations
    
    tag = file_path.split('/')[-1].split('s')[-1].split('.')[0]
    
    row = parameters['MILES'] == int(tag)
    Spectrum = namedtuple('Spectrum', ['wavelength', 'flux', 'Teff', 'logg', 'Fe_H'])
    try:
        spectrum_data = Spectrum(
            wavelength=spectrum[:, 0],
            flux=spectrum[:, 1],
            Teff=float(parameters.loc[row, 'Teff'].values[0]),
            logg=float(parameters.loc[row, 'logg'].values[0]),
            Fe_H=float(parameters.loc[row, 'FeH'].values[0])
        )
        return spectrum_data
    except ValueError:
        pass

def read_collection_of_spectra(file_path: str) -> namedtuple:
    """
    Reads a collection of spectrum files from a directory and returns them as a namedtuple.
    
    Parameters:
    file_path (str): The path to the directory containing spectrum files.
    
    Returns:
    namedtuple: A namedtuple containing the spectra with 
        fields 'Teff', 'logg', 'Fe/H', and 'spectrum'.
    """
    
    spectrum_files = glob.glob(file_path + '/s*')
    spectrum_files.sort()
    SpectrumCollection = namedtuple('SpectrumCollection', 
                ['name', 'wavelength', 'flux', 'Teff', 'logg', 'FeH'])
    
    spectrum_data, name = [], []    
    for file in spectrum_files:
        spec = read_individual_spectrum(file)
        if spec is not None:
            spectrum_data.append(spec)
        name.append(file.split('/')[-1])
    
    return SpectrumCollection(name, 
                              jnp.array([s.wavelength for s in spectrum_data]),
                              jnp.array([s.flux for s in spectrum_data]),
                              jnp.array([s.Teff for s in spectrum_data]),
                              jnp.array([s.logg for s in spectrum_data]),
                              jnp.array([s.Fe_H for s in spectrum_data]))