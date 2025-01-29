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
    
    def validate(self, tile_folder):
        """
        Validates the tiles depending on overlap with geopackages.

        Args:
            tile_folder (string): Path to the folder containing the GeoTIFFs
        
        Returns:
            valid_tiles (list[string]): A list with the file path to all valid tiles
        """
        tile_paths = [os.path.join(tile_folder, f) for f in os.listdir(tile_folder) if f.endswith('.tif')]
        valid_tiles = []

        for path in tqdm(tile_paths, desc="Validated tiles"):
            with rasterio.open(path) as tile:
                # Fetches the bounding box of the tile in coordinates:
                bounds = tile.bounds
                tile_box = box(*bounds)
                # Checks if any buildings or roads overlaps with the tile:
                for layer in self.geopackages:
                    if self.geopackages[layer].intersects(tile_box).any():
                        valid_tiles.append(path)
                        break
        
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

        count_buildings = 0
        count_roads = 0
        mIoU_buildings = 0
        mIoU_roads = 0
        
        imageHandler = imageSaver(self.geopackages)

        for i in range(len(self.originals)):
            gf.log_info(log_file, f"Image set: {i + 1}")
            if check_geographic_overlap(self.originals[i], self.segmentations[i]):
                imageHandler.createMaskGeoTIFF(self.originals[i], mask_folder)
                mask = glob.glob(mask_folder + '/*.tif')[0]
                if check_geographic_overlap(self.segmentations[i], mask):
                    IoU_buildings, IoU_roads = calculate_IoU_between_masks(mask, self.segmentations[i])
                    gf.log_info(log_file, f"Original file: {self.originals[i]}")
                    gf.log_info(log_file, f"Segmented file: {self.segmentations[i]}")
                    gf.log_info(log_file, f"Mask file: {mask}")
                    if IoU_buildings != None:
                        count_buildings += 1
                        mIoU_buildings += IoU_buildings
                        gf.log_info(log_file, f"IoU for buildings: {IoU_buildings}")
                    if IoU_roads != None:
                        count_roads += 1
                        mIoU_roads += IoU_roads
                        gf.log_info(log_file, f"IoU for roads: {IoU_roads}")
            gf.emptyFolder(mask_folder)
        gf.log_info(log_file, "\n#############\nTotal results:\n#############\n")
        if count_buildings > 0:
            gf.log_info(log_file, f"mIoU buildings: {mIoU_buildings / count_buildings}")
        if count_roads > 0:
            gf.log_info(log_file, f"mIoU roads: {mIoU_roads / count_roads}")

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
        if original_bounds == segmented_bounds:
            # Complete overlap
            return True
        else:
            # Not overlapping
            return False
    except Exception as e:
        print(f"An error occured: {e}")
        return False

def get_build_and_road_pixels(file):
    """
    Divides the pixels in the image marked as buildings and roads.

    Args:
        file (string): File path to the image

    Returns:
        buildings (ndarray): A numpy array representing the building rasters
        roads (ndarray): A numpy array representing the road rasters
    """
    with rasterio.open(file) as dataset:
        rgb = dataset.read([1, 2, 3]).transpose([1, 2, 0])
        buildings = np.all(rgb == [255, 0, 0], axis = -1)
        roads = np.all(rgb == [255, 255, 0], axis = -1)
    
    return buildings, roads

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
    Calculates the IoU value of buildings and roads between the mask
    and prediction of the same area.

    Args:
        mask (string): File path to the GeoTIFF representing the mask
        prediction (string): File path to the GeoTIFF representing the prediction
    
    Returns:
        iou_buildings (None, float): Float value of the IoU score for buildings, default None
        iou_roads (None, float): Float value of the IoU score for roads, default None
    """
    building1, road1 = get_build_and_road_pixels(mask)
    building2, road2 = get_build_and_road_pixels(prediction)
    iou_buildings = None
    iou_roads = None

    if np.any(building1):
        iou_buildings = compute_IoU(building1, building2)
    if np.any(road1):
        iou_roads = compute_IoU(road1, road2)
    
    return iou_buildings, iou_roads
