# Data/post_processing.py

# Imports libraries:

import os
import re
import rasterio
from PIL import Image
import numpy as np
import glob
from rasterio.plot import reshape_as_raster

# Increase the image size limit to avoid DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = None

# Functions:

def clear_output_directory(output_dir):
    """
    Deletes all .tif and .jpg files in the output directory

    Args:
        output_dir (string): The file path to the output folder to empty
    """

    tif_files = glob.glob(os.path.join(output_dir, '*.tif'))
    for tif_file in tif_files:
        os.remove(tif_file)
    jpg_files = glob.glob(os.path.join(output_dir, '*.jpg'))
    for jpg_file in jpg_files:
        os.remove(jpg_file)

def geotiff_to_jpg(geotiff_path):
    """
    Converts a GeoTIFF to a .jpg file

    Args:
        geotiff_path (string): The file path to the GeoTIFF
    """
    
    # Gets the folder and filename from the GeoTIFF path:
    folder, filename = os.path.split(geotiff_path)
    
    # Creates the output filename by replacing .tif or .tiff with .jpg:
    output_filename = os.path.splitext(filename)[0] + '.jpg'
    output_path = os.path.join(folder, output_filename)
    
    # Opens the GeoTIFF file:
    with rasterio.open(geotiff_path) as src:
        image_data = src.read()

        # Convert the image data to a format that PIL can work with
        # Reshape from (bands, height, width) to (height, width, bands)
        if image_data.shape[0] == 1:  # If it's a single-band image
            image_data = np.squeeze(image_data, axis=0)  # Remove the band axis
            mode = 'L'  # Grayscale
        else:  # Multiband image (RGB or similar)
            image_data = reshape_as_raster(image_data)
            mode = 'RGB'

        if mode == 'RGB':
            image_data = image_data.transpose((2, 0, 1))
        
        # Ensure the output is uint8, which is needed for JPEG
        if image_data.dtype != np.uint8:
            image_data = (image_data / np.max(image_data) * 255).astype(np.uint8)

        img = Image.fromarray(image_data, mode)
        img.save(output_path, 'JPEG')

def merge_images(tif_folder, segmented_folder, output_original, output_segmented):
    """
    Combines all GeoTIFF files from a folder into one single .tif file, and converts it to .jpg as well

    Arg:
        tif_folder (string): File path to the folder containing all the original aerial images as GeoTIFFs
        segmented_folder (string): File path to the folder containing all the segmented images as GeoTIFFs
        output_original (string): File path and name of the merged, original output file (aerial image) as .tif file
        output_segmented (string): File path and name of the merged, segmented output file (segmented image) as .tif file
    
    Note:
        The tif_folder and segmented_folder must contain the same number of files.
    """
    
    # List of tiles in TIFF format (sorted for consistent positioning)
    tif_files = sorted([f for f in os.listdir(tif_folder) if f.endswith('.tif')], key=lambda x: parse_tile_filename(x))
    segmented_tif_files = sorted([f for f in os.listdir(segmented_folder) if f.endswith('.tif')], key=lambda x: parse_tile_filename_segmented(x))

    # Check if both directories contain the same number of files
    if len(tif_files) != len(segmented_tif_files):
        raise ValueError("Mismatch in tile counts between TIFF and segmented folders.")

    # Parse filenames to extract grid positions
    tile_positions = [parse_tile_filename(f) for f in tif_files]

    # Determine grid dimensions
    max_row = max(row for row, col in tile_positions)
    max_col = max(col for row, col in tile_positions)
    rows, cols = max_row + 1, max_col + 1

    # Use the first tile to define dimensions and metadata
    first_tile_path = os.path.join(tif_folder, tif_files[0])
    with rasterio.open(first_tile_path) as src:
        tile_height, tile_width = src.height, src.width
        profile = src.profile

    # Update profile for full image size
    profile.update({
        "height": rows * tile_height,
        "width": cols * tile_width,
        "count": 3,  # 3 bands for RGB output
        "dtype": "uint8"
    })

    # Prepare large arrays for combined images
    full_original_image = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=profile["dtype"])
    full_segmented_image = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=profile["dtype"])

    for tif_file, segmented_file in zip(tif_files, segmented_tif_files):
        row, col = parse_tile_filename(tif_file)

        # Merge original images
        original_path = os.path.join(tif_folder, tif_file)
        with rasterio.open(original_path) as src:
            original_data = src.read()  # Shape: (bands, height, width)

        for band in range(3):
            full_original_image[
                row * tile_height : (row + 1) * tile_height,
                col * tile_width  : (col + 1) * tile_width,
                band
            ] = original_data[band]

        # Merge segmented images
        segmented_path = os.path.join(segmented_folder, segmented_file)
        with rasterio.open(segmented_path) as src:
            segment_data = src.read()  # Shape: (bands, height, width)

        for band in range(3):
            full_segmented_image[
                row * tile_height : (row + 1) * tile_height,
                col * tile_width  : (col + 1) * tile_width,
                band
            ] = segment_data[band]

    # Save the merged original image as GeoTIFF
    with rasterio.open(output_original, "w", **profile) as dst:
        for band in range(3):
            dst.write(full_original_image[:, :, band], band + 1)

    # Save the merged segmented image as GeoTIFF
    with rasterio.open(output_segmented, "w", **profile) as dst:
        for band in range(3):
            dst.write(full_segmented_image[:, :, band], band + 1)

    # Convert final GeoTIFF to JPG
    geotiff_to_jpg(output_original)
    geotiff_to_jpg(output_segmented)
    print(f"Combined GeoTIFF saved at {output_original}")
    print(f"Combined GeoTIFF saved at {output_segmented}")

def parse_tile_filename(filename):
    """
    Checks for consistency in file name and returns the row and column of the tile in the image

    Args:
        filename (string): The filename to be checked
    
    Returns:
        row (int): The integer value of the row position of the tile in the complete image
        col (int): The integer value of the column position of the tile in the complete image
    """

    match = re.match(r'tile_(\d+)_(\d+)\.tif', filename)
    if match:
        row, col = int(match.group(1)), int(match.group(2))
        return row, col
    else:
        raise ValueError(f"Filename {filename} does not match expected pattern 'tile_<row>_<col>.tif'")

def parse_tile_filename_segmented(filename):
    """
    Checks for consistency in file name and returns the row and column of the tile in the image

    Args:
        filename (string): The filename to be checked
    
    Returns:
        row (int): The integer value of the row position of the tile in the complete image
        col (int): The integer value of the column position of the tile in the complete image
    """
    
    match = re.match(r'tile_(\d+)_(\d+)_segmented\.tif', filename)
    if match:
        row, col = int(match.group(1)), int(match.group(2))
        return row, col
    else:
        raise ValueError(f"Filename {filename} does not match expected pattern 'tile_<row>_<col>.tif'")