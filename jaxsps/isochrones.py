from collections import namedtuple
import jax.numpy as jnp


def read_isochrones(file_path: str) -> namedtuple:
    """
    Reads isochrone data from a .iso file and returns it as a pandas DataFrame.

    Parameters:
    file_path (str): The path to the .iso file containing isochrone data.

    Returns:
    namedtuple: A namedtuple containing the isochrone data with 
        fields corresponding to the columns in the file.
    
    """
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        columns = lines[10].split(' ')[1:]
        columns = [col.strip() for col in columns if col.strip()]
        for line in lines[11:]:
            if line.strip() and line[0] != '#':
                values = line.split(' ')
                values = [float(val) for val in values if val.strip()]
                data.append(values)
    data = jnp.array(data)
    Isochrone = namedtuple('Isochrone', columns)
    isochrones = Isochrone(*data.T)
    return isochrones
        
