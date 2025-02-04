# FarSeg/Program/mainValidation.py

# Libraries:

import os
import shutil
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Functionality import generalFunctions as gf
from Functionality.validation import validation

# Function:

def mainValidation():
    """
    Performs the main part of validating the results from a FarSeg model.
    """
    # Fetches input from the user:
    result_folder = gf.get_valid_input("Where are the segmentation results stored(?): ", gf.doesPathExists)
    geopackage_folder = gf.get_valid_input("Where are the relevant geopackages stored(?): ", gf.doesPathExists)
    mask_folder = gf.get_valid_input("Where would you store temporarly masks(?): ", gf.emptyFolder)
    log_file = gf.get_valid_input("Write the path of the log file that will contain the results: ", gf.resetFile)
    # Validates the data:
    validator = validation(result_folder, geopackage_folder)
    validator.validate(mask_folder, log_file)
    # Deletes unnecessary data:
    if os.path.exists(mask_folder):
        shutil.rmtree(mask_folder)
