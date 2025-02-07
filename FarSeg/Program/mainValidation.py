# FarSeg/Program/mainValidation.py

# Libraries:

import os
import shutil
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Functionality import generalFunctions as gf
from Functionality.preProcessing import geotiff_to_geopackage
from Functionality.validation import validation

# Function:

def mainValidation():
    """
    Performs the main part of validating the results from a FarSeg model.
    """
    # Fetches input from the user:
    print()
    result_folder = gf.get_valid_input("Where are the segmentation results stored(?): ", gf.doesPathExists)
    geodata_folder = gf.get_valid_input("Where are the geographic data stored (the solution)(?): ", gf.doesPathExists)
    mask_folder = gf.get_valid_input("Where would you store temporarly masks(?): ", gf.emptyFolder)
    log_file = gf.get_valid_input("Write the path of the log file that will contain the results: ", gf.resetFile)
    print()
    # Loads the geopackages:
    geodata_gpkg = [f for f in os.listdir(geodata_folder) if f.endswith('.gpkg')]
    geodata_tif = [f for f in os.listdir(geodata_folder) if f.endswith('.tif') and f.replace('.tif', '.gpkg') not in geodata_gpkg]
    # If some of the training data is stored as GeoTIFF format:
    if len(geodata_tif) > 0:
        for file in geodata_tif:
            file = os.path.join(geodata_folder, file)
            geotiff_to_geopackage(
                file,
                file.replace(".tif", ".gpkg"),
                file.split('.')[0].split('/')[-1],
                log_file
            )
    # Validates the data:
    validator = validation(result_folder, geodata_folder)
    validator.validate(mask_folder, log_file)
    # Deletes unnecessary data:
    if os.path.exists(mask_folder):
        shutil.rmtree(mask_folder)
