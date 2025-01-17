# FarSeg_dev/Program/mainValidation.py

# Libraries:

from Functionality import generalFunctions as gf
from Functionality.validation import validation

def mainValidation():
    """
    Performs the main part of validating the results from a FarSeg model.
    """
    result_folder = gf.get_valid_input("Where are the segmentation results stored(?): ", gf.doesPathExists)
    geopackage_folder = gf.get_valid_input("Where are the relevant geopackages stored(?): ", gf.doesPathExists)
    mask_folder = gf.get_valid_input("Where would you store temporarly masks(?): ", gf.emptyFolder)
    log_file = gf.get_valid_input("Write the path of the log file that will contain the results: ", gf.resetFile)
    validator = validation(result_folder, geopackage_folder)
    validator.validate(mask_folder, log_file)