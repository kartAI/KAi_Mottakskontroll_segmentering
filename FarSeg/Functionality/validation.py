# FarSeg/Functionality/validation.py

# Libraries:

import glob
import numpy as np
import os
import rasterio
from shapely.geometry import box
from tqdm import tqdm

import generalFunctions as gf
from geoTIFFandJPEG import imageSaver

# Classes:

class tileValidation():
    """
    Instance validating tiles of images to use for DL training.

    Attributes:
        folder (string): Path to the folder containing relevant geopackage data
    """

    def __init__(self, folder):
        """
        Creates a new instance of tileValidation.

        Args:
            folder (string): Path to the folder containing relevant geopackage data
        """
        self.geopackages = gf.load_geopackages(folder)
    
    def validate(self, tile_folder, validate):
        """
        Validates the tiles depending on overlap with geopackages.

        Args:
            tile_folder (string): Path to the folder containing the GeoTIFFs
            validate (bool): Wether or not to validate the tiles
        
        Returns:
            valid_tiles (list[string]): A list with the file path to all valid tiles if requested
        """
        tile_paths = [os.path.join(tile_folder, f) for f in os.listdir(tile_folder) if f.endswith('.tif')]
        valid_tiles = []

        for path in tqdm(tile_paths, desc="Validated tiles"):
            if validate:
                with rasterio.open(path) as tile:
                    # Fetches the bounding box of the tile in coordinates:
                    bounds = tile.bounds
                    tile_box = box(*bounds)
                    # Checks if any buildings or roads overlaps with the tile:
                    for layer in self.geopackages:
                        if self.geopackages[layer].intersects(tile_box).any():
                            valid_tiles.append(path)
                            break
            else:
                valid_tiles.append(path)
        
        return valid_tiles

class validation():
    """
    Instance validating the final results from the DL inference.

    Attributes:
        folder (string): Path to the result folder were original and segmented images are stored
    """

    def __init__(self, folder, geopackage_folder):
        """
        Creates a new instance of validation.

        Args:
            folder (string): Path to the result folder were original and segmented images are stored
            geopakage_folder (string): Path to the geopackage data used in the validation
        """
        predictions = glob.glob(folder + '/*.tif')
        self.originals = [pred for pred in predictions if 'original' in pred]
        self.segmentations = [pred for pred in predictions if 'segmented' in pred]
        self.geopackages = geopackage_folder

    def validate(self, mask_folder, log_file):
        """
        Performs the validation of the predicted segmentations.
        Writes the validation results to a specified log file.

        Args:
            mask_folder (string): Path to a new folder where the masks are temporarly stored
            log_file (string): Path to a new logfile to be generated
        """
        gf.emptyFolder(mask_folder)

        imageHandler = imageSaver(self.geopackages)

        if len(self.originals) > 1:
            merged_original = os.path.join(os.path.dirname(self.originals[0]), "merged_original.tif")
            imageHandler.mergeGeoTIFFs(
                self.originals,
                merged_original
            )
        else:
            merged_original = self.originals[0]
        if len(self.segmentations) > 1:
            merged_segmented = os.path.join(os.path.dirname(self.segmentations[0]), "merged_segmented.tif")
            imageHandler.mergeGeoTIFFs(
                self.segmentations,
                merged_segmented
            )
        else:
            merged_segmented = self.segmentations[0]

        if check_geographic_overlap(merged_original, merged_segmented):
            imageHandler.createMaskGeoTIFF(merged_original, mask_folder)
            mask = glob.glob(mask_folder + '/*.tif')[0]
            if check_geographic_overlap(merged_segmented, mask):
                IoU = calculate_IoU_between_masks(mask, merged_segmented)
                gf.log_info(log_file, f"Original file: {merged_original}")
                gf.log_info(log_file, f"Segmented file: {merged_segmented}")
                gf.log_info(log_file, f"Mask file: {mask}")
                if IoU:
                    gf.log_info(
                        log_file,
                        f"""
##############
Total results:
#############

IoU score: {IoU}
"""
)
        gf.emptyFolder(mask_folder)

# Helper functions:

def check_geographic_overlap(original_file, segmented_file):
    """
    Checks if two GeoTIFFs overlaps 100%.

    Args:
        original_file (string): File path to aerial image
        segmented_file (string): File path to segmented image
    
    Returns:
        bool: True if overlap, False otherwise
    """
    try:
        with rasterio.open(original_file) as original:
            original_bounds = original.bounds
        with rasterio.open(segmented_file) as segmented:
            segmented_bounds = segmented.bounds
        if original_bounds == segmented_bounds: # Checks for complete overlap
            return True
        else:
            return False
    except Exception as e:
        print(f"An error occured: {e}")
        return False

def get_segmented_pixels(file):
    """
    Fetches the segmented pixels in the image.

    Args:
        file (string): File path to the image

    Returns:
        segmented (ndarray): A numpy array representing the segmented pixels
    """
    with rasterio.open(file) as dataset:
        rgb = dataset.read([1, 2, 3]).transpose([1, 2, 0])
        segmented = np.all(rgb == [255, 255, 255], axis = -1)
    
    return segmented

def compute_IoU(mask1, mask2):
    """
    Calculates the IoU value of the image represented as
    mask1 and mask2 - the predicted segmentations and
    created, correct mask.

    Args:
        mask1 (ndarray): An numpy-representation of an image with one class
        mask2 (ndarray): An numpy-representation of an image with one class

    Returns:
        float: Float value of the IoU score
    """
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    if union == 0:
        return 0
    else:
        return intersection / union

def calculate_IoU_between_masks(mask, prediction):
    """
    Calculates the IoU value of the segmented area between the mask
    and prediction of the same area.

    Args:
        mask (string): File path to the GeoTIFF representing the mask
        prediction (string): File path to the GeoTIFF representing the prediction
    
    Returns:
        iou (None, float): Float value of the IoU score of the segmentations, default None
    """
    segmented1 = get_segmented_pixels(mask)
    segmented2 = get_segmented_pixels(prediction)
    iou = None

    if np.any(segmented1):
        iou = compute_IoU(segmented1, segmented2)
    
    return iou
