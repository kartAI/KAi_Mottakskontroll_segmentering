# FarSeg_dev/Functionality/generalFunctions.py

# Libraries:

import os
import geopandas as gpd
from shapely.validation import make_valid
import shutil

# Functions:

def get_valid_input(prompt, validator):
    """
    A general function checking that user input is valid.

    Args:
        prompt (string): Text showed to the user when asked for input
        validator (callable): A function taking a string, returning True for valid input, otherwise False
    
    Returns:
        string: Valid input from user
    """
    while True:
        user_input = input(prompt)
        if validator(user_input):
            return user_input
        print("Invalid input, try again!")

def yesNo(ans):
    """
    Checks if an answer is either yes or no.

    Args:
        ans (string): Input answer from user.
    
    Returns:
        bool: True for 'yes' or 'y', False for 'no' or 'n', None for invalid input.
    """
    ans = ans.lower()
    if ans in ["y", "yes"]:
        return True
    elif ans in ["n", "no"]:
        return False
    return None  # Return None explicitly for invalid input

def doesPathExists(path):
    """
    Checks if a path exists.

    Args:
        path (string): Path to a folder or a file
    
    Returns:
        bool: Wether or not the path exists
    """
    return os.path.exists(path)

def positiveNumber(text):
    """
    Checks if the input is a positive integer.

    Args:
        text (string): Input from the user
    
    Returns:
        bool: Wether or not the input is a positive integer
    """
    if text.isdigit():
        return int(text) > 0
    return False

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
            if not os.access(folder, os.W_OK): # Checking writing access
                print(f"Mangler skrivetilgang til '{folder}'.")
                return False
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
        return True
    except Exception as e:
        print(f"A failure occured during deleting / creation of the folder '{folder}': {e}")
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

def log_info(logfile, message):
    """
    Writes the message to the logfile.

    Args:
        logfile (string): Path to the log file
        message (string): Text that should be written
    """
    with open(logfile, 'a') as f:
        f.write(message + '/n')
