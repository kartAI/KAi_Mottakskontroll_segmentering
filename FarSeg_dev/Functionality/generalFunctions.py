# FarSeg_dev/Functionality/generalFunctions.py

# Libraries:

import os
import geopandas as gpd
from shapely.validation import make_valid
import shutil

# Functions:

def validInput(ans, input):
    """
    Checks if the user answer is valid input

    Args:
        ans (string): input answer from user
        input (list[string]): valid input
    Returns:
        A boolean value
    """
    ans = ans.lower()
    if ans in input:
        return True
    else:
        return False

def yesNo(ans):
    """
    Checks if an answer is either yes or no
    
    Args:
        ans (string): input answer from user
    Returns:
        A boolean value
    """
    ans = ans.lower()
    if ans.lower() == "y":
        return True
    elif ans == "n":
        return False

def load_geopackages(folder):
    """
    Load geometries for buildings and roads from multiple GeoPackages in a folder.

    Args:
        folder (string): File path to the folder containing the geopackages
    
    Returns:
        geodata (dict): Dictionary containing all the GeoDataFrames for buildings and roads
    """
    geopackages = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.gpkg')]
    types = ["buildings", "roads"]
    geodata = {}

    for i in range(len(geopackages)):
        if i > len(types) - 1:
            break
        gdf = gpd.read_file(geopackages[i])
        gdf['geometry'] = gdf['geometry'].apply(make_valid)
        gdf = gdf[gdf['geometry'].notnull() & ~gdf['geometry'].is_empty]
        geodata[types[i]] = gdf
    
    return geodata

def emptyFolder(folder):
    """
    Deletes the folder, if it exists, and creates a new, empty one.

    Args:
        folder (string): Path to the new folder

    Returns:
        bool: True if operation succeeds, False otherwise
    """
    try:
        if os.path.exists(folder):
            if not os.access(folder, os.W_OK): # Sjekk skrivetilgang
                print(f"Mangler skrivetilgang til '{folder}'.")
                return False
            shutil.rmtree(folder)
        os.makedirs(folder)
        return True
    except Exception as e:
        print(f"En feil oppsto under sletting/oppretting av mappen '{folder}': {e}")
        return False

def log_info(logfile, message):
    """
    Writes the message to the logfile.

    Args:
        logfile (string): Path to the log file
        message (string): Text that should be written
    """
    with open(logfile, 'a') as f:
        f.write(message + '/n')
