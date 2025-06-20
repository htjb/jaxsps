from collections import namedtuple
import jax.numpy as jnp
import glob


def read_individual_isochrones(file_path: str) -> namedtuple:
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
                values = [val for val in values if val.strip()]
                data.append(values)
    data = jnp.array(data, dtype=jnp.float32)
    Isochrone = namedtuple('Isochrone', columns)
    isochrones = Isochrone(*data.T)
    return isochrones
        
def read_collection_of_isochrones(file_path: str) -> namedtuple:

    isochrone_files = glob.glob(file_path + '/*.iso')
    isochrones = namedtuple('Isochrones', ['metalicity', 'isochrone'])
    isochrones_list = []
    for file in isochrone_files:
        isochrone = read_individual_isochrones(file)
        # need to check that the _m and _p actually correspond to
        # positive and minus
        try:
            metalicity = float(file.split('feh_m')[1].split('_')[0])
        except ValueError:
            metalicity = float(file.split('feh_p')[1].split('_')[0])
        isochrones_list.append((metalicity, isochrone))
    isochrones = isochrones(*zip(*isochrones_list))
    return isochrones
