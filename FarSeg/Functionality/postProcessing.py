# FarSeg/Functionality/postProcessing.py

# Libraries:

import cv2
import numpy as np
import os
import re

from geoTIFFandJPEG import imageSaver

#  Class:

class postProcessor():
    """
    A post-processor element that performs all the post-processing of
    the GeoTIFF data after inferencing.

    Attributes:
        geotiff_folder (string): Path to the folder containing all the original aerial imageas as GeoTIFFs
        segmented_folder (string): Path to the folder containing all the segmented images as GeoTIFFs
    """
    def __init__(self, geotiff_folder, segmented_folder):
        """
        Creates an instance of postProcessor.

        Arguments:
            geotiff_folder (string): Path to the folder containing all the original aerial imageas as GeoTIFFs
            segmented_folder (string): Path to the folder containing all the segmented images as GeoTIFFs
        """
        self.geotiff_folder = geotiff_folder
        self.segmented_folder = segmented_folder
    
    def merge_images(self, original, segmented, original_size, jpeg=False):
        """
        Combines all GeoTIFF files from a folder into one single .tif file,
        crops the merged image to the specified original size,
        and converts it to .jpg as well, if requested.

        Arguments:
            original (string): File path and name of the merged, original output file (aerial image) as .tif file
            segmented (string): File path and name of the merged, segmented output file (segmented image) as .tif file
            original_size (tuple): (height, width) in pixels for cropping the final image
            jpeg (bool): Boolean value telling wether or not to save image as JPEG as well as GeoTIFF, default False
        
        Note:
            The tif_folder and segmented_folder must contain the same number of files
        """
        # List of tiles in GeoTIFF format (sorted for consistent positioning):
        geotiffs = sorted([f for f in os.listdir(self.geotiff_folder) if f.endswith('.tif')], key=lambda x: parse_tile_filename(x))
        segmented_geotiffs = sorted([f for f in os.listdir(self.segmented_folder) if f.endswith('.tif')], key=lambda x: parse_tile_filename(x, True))
        # Checks if both directories contains the same number of files:
        if len(geotiffs) != len(segmented_geotiffs):
            raise ValueError("Mismatch in tile counts between GeoTIFF and segmented folders!")
        # Parse filenames to extract grid positions:
        tile_positions = [parse_tile_filename(f) for f in geotiffs]
        # Determine grid dimensions:
        max_row = max(row for row, _ in tile_positions)
        max_col = max(col for _, col in tile_positions)
        rows, cols = max_row + 1, max_col + 1
        # Use the first tile to define dimensions and metadata:
        imageHandler = imageSaver()
        image_data, metadata = imageHandler.readGeoTIFF(os.path.join(self.geotiff_folder, geotiffs[0]))
        tile_height, tile_width = metadata["height"], metadata["width"]
        profile = metadata["profile"]
        # Updates profile for full image size:
        profile.update({
            "height": rows * tile_height,
            "width": cols * tile_width,
            "count": 3, # 3 bands for RGB
            "dtype": "uint8",
            "photometric": "RGB"
        })
        # Prepare large arrays for combined image:
        full_original_image = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=profile["dtype"])
        full_segmented_image = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=profile["dtype"])
        for geotiff_file, segmented_file in zip(geotiffs, segmented_geotiffs):
            row, col = parse_tile_filename(geotiff_file)
            # Merge original image by fetching data:
            image_data, _ = imageHandler.readGeoTIFF(os.path.join(self.geotiff_folder, geotiff_file))
            start_row, start_col = row * tile_height, col * tile_width
            end_row, end_col = start_row + tile_height, start_col + tile_width
            adjust_row, adjust_col = 0, 0
            if end_row > original_size[0]:
                adjust_row = end_row - original_size[0]
                end_row = original_size[0]
            if end_col > original_size[1]:
                adjust_col = end_col - original_size[1]
                end_col = original_size[1]
            for band in range(3):
                full_original_image[start_row:end_row, start_col:end_col, band] = image_data[adjust_row:, adjust_col:, band]
            # Merge segmented image by fetching data:
            image_data, _ = imageHandler.readGeoTIFF(os.path.join(self.segmented_folder, segmented_file))
            tile_h, tile_w, _ = image_data.shape
            start_row, start_col = row * tile_height, col * tile_width
            end_row, end_col = start_row + tile_h, start_col + tile_w
            for band in range(3):
                full_segmented_image[start_row:end_row, start_col:end_col, band] = image_data[:, :, band]
        # Crop to the specified original size:
        crop_height, crop_width = original_size
        cropped_original_image = full_original_image[:crop_height, :crop_width, :]
        cropped_segmented_image = full_segmented_image[:crop_height, :crop_width, :]
        # Update profile to match cropped size:
        profile.update({
            "height": crop_height,
            "width": crop_width
        })
        # Save the merged original image as GeoTIFF:
        imageHandler.createGeoTIFF(original, profile, cropped_original_image)
        # Save the merged segmented image as GeoTIFF:
        imageHandler.createGeoTIFF(segmented, profile, cropped_segmented_image)
        if jpeg:
            # Convert final GeoTIFF to JPG:
            imageHandler.saveGeoTIFFasJPEG(original, os.path.dirname(original))
            imageHandler.saveGeoTIFFasJPEG(segmented, os.path.dirname(segmented))

# Helper functions:

def parse_tile_filename(filename, segmented=False):
    """
    Checks for consistency in file name
    and returns the row and column of the tile in the image.

    Arguments:
        filename (string): Filename to be checked
        segmented (bool): Check if it is segmented or original file, default False
    
    Returns:
        row (int): The integer value of the row position of the tile in the complete image
        col (int): The integer value of the column position of the tile in the complete image
    """
    if segmented:
        match = re.match(r'tile_(\d+)_(\d+)_segmented\.tif', filename)
    else:
        match = re.match(r'tile_(\d+)_(\d+)\.tif', filename)
    if match:
        row, col = int(match.group(1)), int(match.group(2))
        return row, col
    else:
        if segmented:
            raise ValueError(f"Filename  {filename} does not match expected pattern 'tile_<row>_<col>_segmented.tif")
        raise ValueError(f"Filename  {filename} does not match expected pattern 'tile_<row>_<col>.tif")

def remove_noise(mask):
        """
        ...
        """
        if np.all(mask == 0):
            return mask
        gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned_mask = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) >= 750:
                cv2.drawContours(cleaned_mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        return cleaned_mask