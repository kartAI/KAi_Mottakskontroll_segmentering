# FarSeg/Functionality/validation.py

# Libraries:

import glob
import numpy as np
import os
import rasterio
from tqdm import tqdm

import generalFunctions as gf
from geoTIFFandJPEG import imageSaver
import vectorization as V

# Classes:

class validation():
    """
    Instance validating the final results from the DL inference.

    Attribute:
        folder (string): Path to the result folder were original and segmented images are stored
    """

    def __init__(self, folder, geopackage_folder):
        """
        Creates a new instance of validation.

        Arguments:
            folder (string): Path to the result folder were original and segmented images are stored
            geopakage_folder (string): Path to the geopackage data used in the validation
        """
        predictions = glob.glob(folder + '/*.tif')
        self.originals = [pred for pred in predictions if 'original' in pred]
        self.segmentations = [pred for pred in predictions if 'segmented' in pred]
        self.geopackages = geopackage_folder

    def validate(self, mask_folder, log_file, isRoad, save, output_geojson, zone, utmOrLatLon):
        """
        Performs the validation of the predicted segmentations.
        Writes the validation results to a specified log file.

        Arguments:
            mask_folder (string): Path to a new folder where the masks are temporarly stored
            log_file (string): Path to a new logfile to be generated
            isRoad (bool): If True, the segmentation data is for roads, False otherwise
            save (bool): If True, the function saves the vector data in GeoJSON files
            output_geojson (string): Path to the folder where the GeoJSON files will be saved
            zone (string): The UTM zone of the GeoTIFFs
            utmOrLatLon (bool): If True, the coordinates are converted to latlon, otherwise UTM coordinates are kept
        """
        gf.emptyFolder(mask_folder)
        gf.emptyFolder(mask_folder + "_final")

        imageHandler = imageSaver(self.geopackages)

        tp, tn, fp, fn = 0, 0, 0, 0
        mask_lines, mask_boundaries, segmented_lines, segmented_boundaries = 0, 0, 0, 0

        if len(self.originals) == len(self.segmentations):
            for i in tqdm(range(len(self.originals)), desc="Calculating statistic", colour="yellow"):
                imageHandler.createMaskGeoTIFF(self.originals[i], mask_folder)
                mask = glob.glob(mask_folder + '/*.tif')[0]
                v1, v2, v3, v4 = imageHandler.generate_comparison_GeoTIFF(
                    self.segmentations[i],
                    mask,
                    os.path.join(mask_folder + "_final", f"Compared_mask_{i+1}.tif")
                )
                tp += v1
                tn += v2
                fp += v3
                fn += v4
                if isRoad:
                    mask_lines += V.createCenterLines(mask, False, output_geojson, zone, utmOrLatLon, log_file) # Does not save GeoJSON of solution
                    segmented_lines += V.createCenterLines(self.segmentations[i], save, output_geojson, zone, utmOrLatLon, log_file, count=i)
                mask_boundaries += V.createBoundaries(mask, False, output_geojson, zone, utmOrLatLon, log_file) # Does not save GeoJSON of solution
                segmented_boundaries += V.createBoundaries(self.segmentations[i], save, output_geojson, zone, utmOrLatLon, log_file, count=i)
                gf.emptyFolder(mask_folder)

        total = tp + tn + fp + fn

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
            print("Merged GeoTIFFs overlap.")
            imageHandler.createMaskGeoTIFF(merged_original, mask_folder)
            mask = glob.glob(mask_folder + '/*.tif')[0]
            if check_geographic_overlap(merged_segmented, mask):
                print("Segmentations and masks overlap.")
                IoU = calculate_IoU_between_masks(mask, merged_segmented)
                print("IoU calculated.")
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
True positives: {tp/total}
True negatives: {tn/total}
False positives: {fp/total}
False negatives: {fn/total}
Precision: {tp/(tp + fp)} (How many retrieved pixels are relevant?)
Recall: {tp/(tp+fn)} (How many relevant pixels are retrieved?)
F1: {2 * tp /(2 * tp + fp + fn)} (Harmonic mean of precision and recall)
Total length of boundaries (segmented / correct): {segmented_boundaries} / {mask_boundaries} = {segmented_boundaries / mask_boundaries}
{f'Total length of centerlines (segmented / correct): {segmented_lines} / {mask_lines} = {segmented_lines / mask_lines}' if isRoad else ''}
"""
)

# Helper functions:

def check_geographic_overlap(original_file, segmented_file):
    """
    Checks if two GeoTIFFs overlaps 100%.

    Arguments:
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

    Argument:
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

    Arguments:
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

    Arguments:
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
