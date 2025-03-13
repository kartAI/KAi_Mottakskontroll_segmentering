# FarSeg/Functionality/generalFunctions.py

# Libraries:

import geopandas as gpd
import os
import pandas as pd
from shapely.validation import make_valid
import shutil
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Functions:

def get_valid_input(prompt, validator, default=None):
    """
    A general function checking that user input is valid.

    Arguments:
        prompt (string): Text showed to the user when asked for input
        validator (callable): A function taking a string, returning True for valid input, otherwise False
    
    Returns:
        string: Valid input from user
    """
    while True:
        user_input = input(prompt)
        if user_input == "" and default != None:
            return default
        if validator(user_input):
            return user_input
        print("Invalid input, try again!")

def yesNo(input):
    """
    Checks if an input is either yes or no.

    Argument:
        input (string): Input from user.
    
    Returns:
        bool: True for 'yes' or 'y', False for 'no' or 'n', None for invalid input.
    """
    input = input.lower()
    if input in ["y", "yes"]:
        return True
    elif input in ["n", "no"]:
        return False
    return None # Return None explicitly for invalid input

def doesPathExists(input):
    """
    Checks if the given user input is a path that exists.

    Argument:
        input (string): Input from the user
    
    Returns:
        bool: Wether or not the input is a path that exists
    """
    return os.path.exists(input)

def positiveNumber(input):
    """
    Checks if the input is a positive integer.

    Argument:
        input (string): Input from the user
    
    Returns:
        bool: Wether or not the input is a positive integer
    """
    try:
        return float(input) > 0
    except:
        return False

def emptyFolder(input):
    """
    Deletes the folder given as an input string from the user, if it exists, and creates a new, empty one.

    Argument:
        input (string): Input path to the new folder given by the user

    Returns:
        bool: True if operation succeeds, False otherwise
    """
    try:
        if os.path.exists(input):
            if not os.access(input, os.W_OK): # Checking writing access
                print(f"Missing write access to '{input}'.")
                return False
            shutil.rmtree(input)
        os.makedirs(input, exist_ok=True)
        return True
    except Exception as e:
        print(f"An error occurred during deletion/creation of the folder '{input}': {e}")
        return False

def resetFile(input):
    """
    Deletes the file given as an input string from the user, if it exists, and creates a new , empty one.

    Argument:
        input (string): Input path to the new file given by the user
    
    Returns:
        bool: True if operation succeeds, False otherwise
    """
    try:
        if os.path.exists(input):
            if not os.access(input, os.W_OK):
                print(f"Missing write access to '{input}'.")
                return False
            os.remove(input)
        with open(input, 'w') as f:
            pass
        return True
    except Exception as e:
        print(f"An error occurred during deletion/creation of the file '{input}': {e}")

def correctUTMZone(zone):
    """
    Checks if the input zone is a valid zone for UTM in Norway.

    Argument:
        zone (string): UTM zone given as user input

    Returns:
        bool: True is valid zone, False otherwise
    """
    valid_zones = ["32N", "33N", "35N"]
    return zone in valid_zones

def load_geopackages(folder):
    """
    Load geometries for relevant map objects from multiple GeoPackages in a folder.

    Argument:
        folder (string): File path to the folder containing the geopackages
    
    Returns:
        geodata (dict): Dictionary containing all the GeoDataFrames for relevant map objects
    """
    geopackages = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.gpkg')]
    gdfs = []

    for i, filepath in enumerate(geopackages):
        gdf = gpd.read_file(filepath)
        gdf['geometry'] = gdf['geometry'].apply(make_valid)
        gdf = gdf[gdf['geometry'].notnull() & ~gdf['geometry'].is_empty]
        if i == 0:
            target_crs = gdf.crs
        if gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)
        gdfs.append(gdf)
    
    if gdfs:
        geodata = {'data': gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))}
    else:
        geodata = {'data': None}
    
    return geodata

def log_info(logfile, message):
    """
    Writes the message to the logfile.

    Arguments:
        logfile (string): Path to the log file
        message (string): Text that should be written
    """
    with open(logfile, 'a') as f:
        f.write(message + '\n')
