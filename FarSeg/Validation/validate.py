# Validation/validate.py

# Imports libraries:

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Data.pre_processing as pre
from Data.createMasks import create_mask_for_single_geotiff
from Data.post_processing import clear_output_directory

import glob
import rasterio
import numpy as np

# Functions:

def check_geographic_overlap(original_file, segmented_file):
    """
    Checks if two geotiffs overlaps 100%
    """

    try:
        with rasterio.open(original_file) as original:
            original_bounds = original.bounds
        
        with rasterio.open(segmented_file) as segmented:
            segmented_bounds = segmented.bounds
        
        if original_bounds == segmented_bounds:
            print("The boundaries are the same 8)")
            return True
        else:
            print("The boundaries are not the same 8)")
            return False
    except Exception as e:
        print(f"An error occured: {e}")
        return False

def get_build_and_road_pixels(filepath):
    """
    ...
    """

    with rasterio.open(filepath) as dataset:
        rgb = dataset.read([1, 2, 3]).transpose([1, 2, 0])
        buildings = np.all(rgb == [255, 0, 0], axis = -1)
        roads = np.all(rgb == [255, 255, 0], axis = -1)
    
    return buildings, roads

def compute_IoU(mask1, mask2):
    """
    ...
    """

    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)

    if union == 0:
        return 0
    else:
        return intersection / union

def calculate_IoU_between_masks(mask, prediction):
    """
    ...
    """

    building_mask1, road_mask1 = get_build_and_road_pixels(mask)
    building_mask2, road_mask2 = get_build_and_road_pixels(prediction)

    iou_buildings = None
    if np.any(building_mask1):
        iou_buildings = compute_IoU(building_mask1, building_mask2)
    
    iou_roads = None
    if np.any(road_mask1):
        iou_roads = compute_IoU(road_mask1, road_mask2)

    return iou_buildings, iou_roads

def log_info(logfile, message):
    """
    ...
    """
    with open(logfile, 'a') as f:
        f.write(message + '\n')

# Variables:

mask_folder = './FarSeg/Data/Masks'
predicted_folder = './FarSeg/Inference/Final result/'

# Program:

os.makedirs(mask_folder, exist_ok=True)
clear_output_directory(mask_folder)

predictions = glob.glob(predicted_folder + '/*.tif')
predictions_original = [pred for pred in predictions if 'original' in pred]
predictions_segmented = [pred for pred in predictions if 'segmented' in pred]

geopackage_folder = 'C:/Users/jshjelse/Documents/Prosjektoppgave/Geopackage/Ver2'
geopackages = pre.load_geopackages(geopackage_folder) # [Buildings, Roads]

logfile = "./validation_log.txt"
if os.path.exists(logfile):
    os.remove(logfile)

count_buildings = 0
count_roads = 0
mIoU_buildings = 0
mIoU_roads = 0

for i in range(len(predictions_original)):
    log_info(logfile, f"Image set: {i + 1}")
    if check_geographic_overlap(predictions_original[i], predictions_segmented[i]):
        create_mask_for_single_geotiff(predictions_original[i], geopackages, mask_folder)
        mask = glob.glob(mask_folder + '/*.tif')[0]

        if check_geographic_overlap(predictions_segmented[i], mask):
            iou_buildings, iou_roads = calculate_IoU_between_masks(mask, predictions_segmented[i])

            log_info(logfile, f"Original file: {predictions_original[i]}")
            log_info(logfile, f"Segmented file: {predictions_segmented[i]}")
            log_info(logfile, f"Mask file: {mask}")

            if iou_buildings != None:
                count_buildings += 1
                mIoU_buildings += iou_buildings
                log_info(logfile, f"IoU for building: {iou_buildings}")
            
            if iou_roads != None:
                count_roads += 1
                mIoU_roads += iou_roads
                log_info(logfile, f"IoU for roads: {iou_roads}")
        else:
            print("Mask is not overlapping! Continues to next.")
    clear_output_directory(mask_folder)

if count_buildings > 0:
    log_info(logfile, f"mIoU buildings: {mIoU_buildings / count_buildings}")

if count_roads > 0:
    log_info(logfile, f"mIoU roads: {mIoU_roads / count_roads}")
