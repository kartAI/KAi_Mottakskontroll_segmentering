# Data/createMasks.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import Data.pre_processing as pre

import glob
import rasterio
from rasterio.features import rasterize
import numpy as np
from PIL import Image
from tqdm import tqdm

def create_masks_and_save_images(geotiff_folder, geopackage_data, output_folder):
    """
    Generate masks and save both the masks and the original GeoTIFFs as JPGs.

    Parameters:
    - geotiff_folder: Folder containing GeoTIFF files.
    - geopackage_data: Dictionary with GeoDataFrames for 'buildings' and 'roads'.
    - output_folder: Folder to save the mask JPG and original GeoTIFF JPG files.
    """
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all GeoTIFF files
    tif_files = glob.glob(geotiff_folder + '/*.tif')
    tif_files = [tif for tif in tif_files if 'original' in tif]

    for tif_file in tqdm(tif_files, desc="Processing GeoTIFFs"):
        with rasterio.open(tif_file) as src:
            # Read the GeoTIFF data
            image_data = src.read()
            image_data = np.transpose(image_data, (1, 2, 0))  # Rearrange to (height, width, bands)

            # Normalize data if required
            if image_data.dtype != np.uint8:  # If the data is not 8-bit
                image_min = image_data.min()
                image_max = image_data.max()
                image_data = ((image_data - image_min) / (image_max - image_min) * 255).astype('uint8')

            # Save the original GeoTIFF as JPG
            original_jpg_filename = os.path.join(output_folder, os.path.basename(tif_file).replace('.tif', '_original.jpg'))
            original_image = Image.fromarray(image_data)
            original_image.save(original_jpg_filename, format='JPEG')

            # Save the original GeoTIFF as TIFF
            original_tif_filename = os.path.join(output_folder, os.path.basename(tif_file).replace('.tif', '_original.tif'))
            with rasterio.open(
                original_tif_filename,
                'w',
                driver='GTiff',
                height=src.height,
                width=src.width,
                count=image_data.shape[2],
                dtype=image_data.dtype,
                crs=src.crs,
                transform=src.transform
            ) as dst:
                dst.write(np.transpose(image_data, [2, 0, 1])) # Back to [bands, height, width]

            # Create mask RGB (only two layers: roads and buildings)
            rgb_mask = np.zeros((src.height, src.width, 3), dtype='uint8')  # RGB mask image

            # Rasterize geometries for roads (yellow) and buildings (red)
            for layer, color in zip(['roads', 'buildings'], [(255, 255, 0), (255, 0, 0)]):  # Yellow, Red
                if layer in geopackage_data:
                    shapes = [(geom, 1) for geom in geopackage_data[layer].geometry if geom.is_valid]
                    layer_mask = rasterize(
                        shapes,
                        out_shape=(src.height, src.width),
                        transform=src.transform,
                        fill=0,
                        dtype='uint8',
                        all_touched=True
                    )
                    # Apply color to the corresponding pixels
                    for channel, value in enumerate(color):  # R, G, B values
                        rgb_mask[:, :, channel] += (layer_mask * value).astype('uint8')

            # Save the mask as JPG
            mask_jpg_filename = os.path.join(output_folder, os.path.basename(tif_file).replace('.tif', '_mask.jpg'))
            mask_image = Image.fromarray(rgb_mask)
            mask_image.save(mask_jpg_filename, format='JPEG')

            # Save the mask as TIFF
            mask_tif_filename = os.path.join(output_folder, os.path.basename(tif_file).replace('.tif', '_mask.tif'))
            with rasterio.open(
                mask_tif_filename,
                'w',
                driver='GTiff',
                height=src.height,
                width=src.width,
                count=3,
                dtype=image_data.dtype,
                crs=src.crs,
                transform=src.transform
            ) as dst:
                dst.write(np.transpose(rgb_mask, [2, 0, 1])) # Back to [bands, height, width]

def create_mask_for_single_geotiff(geotiff, geopackage_data, output_folder):
    """
    Generate mask and save it as a GeoTIFF

    Parameters:
    - geotiff (string): Folder containing GeoTIFF files.
    - geopackage_data: Dictionary with GeoDataFrames for 'buildings' and 'roads'.
    - output_folder: Folder to save the mask JPG and original GeoTIFF JPG files.
    """

    with rasterio.open(geotiff) as src:
        # Read the GeoTIFF data
        image_data = src.read()
        image_data = np.transpose(image_data, (1, 2, 0))  # Rearrange to (height, width, bands)

        # Normalize data if required
        if image_data.dtype != np.uint8:  # If the data is not 8-bit
            image_min = image_data.min()
            image_max = image_data.max()
            image_data = ((image_data - image_min) / (image_max - image_min) * 255).astype('uint8')

        # Create mask RGB (only two layers: roads and buildings)
        rgb_mask = np.zeros((src.height, src.width, 3), dtype='uint8')  # RGB mask image

        # Rasterize geometries for roads (yellow) and buildings (red)
        for layer, color in zip(['roads', 'buildings'], [(255, 255, 0), (255, 0, 0)]):  # Yellow, Red
            if layer in geopackage_data:
                shapes = [(geom, 1) for geom in geopackage_data[layer].geometry if geom.is_valid]
                layer_mask = rasterize(
                    shapes,
                    out_shape=(src.height, src.width),
                    transform=src.transform,
                    fill=0,
                    dtype='uint8',
                    all_touched=True
                )
                # Apply color to the corresponding pixels
                for channel, value in enumerate(color):  # R, G, B values
                    rgb_mask[:, :, channel] += (layer_mask * value).astype('uint8')

        # Save the mask as TIFF
        mask_tif_filename = os.path.join(output_folder, os.path.basename(geotiff).replace('.tif', '_mask.tif'))
        with rasterio.open(
            mask_tif_filename,
            'w',
            driver='GTiff',
            height=src.height,
            width=src.width,
            count=3,
            dtype=image_data.dtype,
            crs=src.crs,
            transform=src.transform
        ) as dst:
            dst.write(np.transpose(rgb_mask, [2, 0, 1])) # Back to [bands, height, width]

def main():
    # Paths to data
    # Folder with three geopackage files: buildings, roads and water
    geopackage_folder = 'C:/Users/jshjelse/Documents/Prosjektoppgave/Geopackage/Ver2'
    # Folder with hundreds of different, small GeoTIFFs
    geotiff_folder = 'C:/Users/jshjelse/Documents/Prosjektoppgave/FarSeg/Inference/Final result'

    output_folder = './FarSeg/Data/Masks'

    geopackages = pre.load_geopackages(geopackage_folder) # [Buildings, Roads]

    # Create masks
    create_masks_and_save_images(geotiff_folder, geopackages, output_folder)

#main()