# Data/pre_processing.py

# Imports libraries:

import os
import geopandas as gpd
import rasterio
from rasterio.windows import Window
import numpy as np
from rasterio.features import geometry_mask
from post_processing import clear_output_directory
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import random
from shapely.validation import make_valid

# Class:

class MapSegmentationDataset(Dataset):
    """
    Creates a Dataset corresponding to the PyTorch system wit Dataset and Dataloaders

    Attributes:
        geotiff_list (list of strings): List of GeoTIFF file paths
        geodata (dict): Dictionary with 'buildings' and 'roads' GeoDataFrames
        transform (dict): Optional, any data augmentation or transformations
    """

    def __init__(self, geotiff_list, geodata, transform=None):
        """
        Creates a new instance of a MapSegmentationDataset

        Args:
            geotiff_list (list of strings): List of GeoTIFF file paths
            geodata (dict): Dictionary with 'buildings' and 'roads' GeoDataFrames
            transform (dict): Optional, any data augmentation or transformations
        """
        
        self.geotiff_list = geotiff_list
        self.geodata = geodata
        self.transform = transform  # Optional: For data augmentation

    def __len__(self):
        return len(self.geotiff_list)
    
    def __getitem__(self, idx):
        """
        Fetches one of the elements in the data set

        Args:
            idx (int): The index of the object in the data set to fetch

        Returns:
            image_tensor (ndarray): A ndarray representation of the image
            label_mask_tensor (ndarray): A ndarray representation of the mask of the image
        """
        
        geotiff_path = self.geotiff_list[idx]
        
        # Load the image
        with rasterio.open(geotiff_path) as src:
            image = src.read()
            transform = src.transform
            out_shape = (src.height, src.width)
        
        # Reproject geometries to match the CRS of the current GeoTIFF
        building_geometries = self.geodata['buildings'].to_crs(src.crs)['geometry'].values
        road_geometries = self.geodata['roads'].to_crs(src.crs)['geometry'].values

        # Rasterize the geometries for each class
        building_mask = geometry_mask(building_geometries, transform=transform, invert=True, out_shape=out_shape)
        road_mask = geometry_mask(road_geometries, transform=transform, invert=True, out_shape=out_shape)
        
        # Combine the masks into a single-channel label (0: background, 1: buildings, 2: roads)
        label_mask = np.zeros(out_shape, dtype=np.uint8)  # Initialize background as class 0
        label_mask[building_mask] = 1   # Class 1 for buildings
        label_mask[road_mask] = 2       # Class 2 for roads

        # Convert image and masks to PyTorch tensors
        image_tensor = torch.from_numpy(image).float()  # Convert image to float tensor
        label_mask_tensor = torch.from_numpy(label_mask)  # Multi-channel label mask
        
        if self.transform:
            # Apply any data transformations (like augmentation)
            image_tensor, label_mask_tensor = self.transform(image_tensor, label_mask_tensor)
        
        return image_tensor, label_mask_tensor

# Functions:

def get_random_geotiff(folder_path):
    """
    Fetches a random GeoTIFF file

    Args:
        folder_path (string): The file path to the folder containing the GeoTIFFs

    Returns:
        string: The file path to the GeoTIFF, if one exists
    """

    # Get a list of all files in the folder
    all_files = os.listdir(folder_path)
    total_files = len(all_files)

    if total_files == 0:
        print("The folder is empty.")
        return None

    # Generate a random index within the range of files
    random_index = random.randint(0, total_files - 1)

    # Loop through files starting from the random index until the first GeoTIFF is found
    for i in range(total_files):
        # Calculate the current index using modulus to wrap around
        current_index = (random_index + i) % total_files
        file_name = all_files[current_index]

        # Check if the file is a GeoTIFF (file extension: .tif or .tiff)
        if file_name.lower().endswith(('.tif', '.tiff')):
            return os.path.join(folder_path, file_name)

    print("No GeoTIFF files found in the folder.")
    return None

def load_geopackages(geopackage_folder):
    """
    Load geometries for buildings and roads from multiple GeoPackages in a folder

    Args:
        geopackage_folder (string): File path to the folder containing the geopackages
    
    Returns:
        geodata (dict): Dictionary containing all the GeoDataFrames for buildings and roads
    """

    geopackage_files = [os.path.join(geopackage_folder, f) for f in os.listdir(geopackage_folder) if f.endswith('.gpkg')]
    
    types = ["buildings", "roads"]

    geodata = {}

    for i in tqdm(range(len(geopackage_files)), "Files"):
        
        if i > len(types) - 1:
            break

        gdf = gpd.read_file(geopackage_files[i])
        
        gdf['geometry'] = gdf['geometry'].apply(make_valid)

        gdf = gdf[gdf['geometry'].notnull() & ~gdf['geometry'].is_empty]

        geodata[types[i]] = gdf

    return geodata  # Returns a dictionary with 'building' and 'road' GeoDataFrames

def split_data(geotiff_dir, split_ratio=0.7):
    """
    Split geotiffs into training and validation sets
    
    Args:
        geotiff_dir (string): File path to the folder containing all the GeoTIFFs
        split_ratio (float): A float number showing the split ratio between training and validation set, default is 0.7
    
    Returns:
        lists: two lists (one training and one validation) with 70 / 30 split containing file paths to the GeoTIFF files
    """
    geotiff_files = [os.path.join(geotiff_dir, f) for f in os.listdir(geotiff_dir) if f.endswith('.tif')]
    np.random.shuffle(geotiff_files)
    split_idx = int(len(geotiff_files) * split_ratio)
    return geotiff_files[:split_idx], geotiff_files[split_idx:]

def save_tile(src, window, transform, output_path, nodata_value, tile_width=1000, tile_height=1000):
    """
    Save a tile to a new GeoTIFF file

    Args:
        src (rasterio extension): Rasterio extension describing the resolution of the image / tile
        window (rasterio Window): Rasterio Window object containing the height and width of the image
        transform (rasterio): Rasterio transform fetched from the origin image
        output_path (string): file path to the folder with file name of the saved tile
        nodata_value (int): Nodata value read by rasterio (default = 0)
        tile_width (int): Width of the tile (default = 1000)
        tile_height (int): Height of the tile (default = 1000)
    """
    
    # Read the tile data, handling cases where the window exceeds the bounds
    tile_data = src.read(window=window, boundless=True, fill_value=nodata_value)

    # Get the actual size of the window (it might be smaller at the edges)
    actual_height, actual_width = tile_data.shape[1], tile_data.shape[2]

    # If the tile is smaller than 1024x1024, we need to pad it
    if actual_width < tile_width or actual_height < tile_height:
        # Create a padded array filled with the nodata value
        padded_tile = np.full((src.count, tile_height, tile_width), nodata_value, dtype=tile_data.dtype)
        
        # Paste the actual tile data into the top-left corner of the padded array
        padded_tile[:, :actual_height, :actual_width] = tile_data

        # Update the tile data to be the padded version
        tile_data = padded_tile

    # Update the profile (size and transform) to reflect the tile size
    profile = src.profile
    profile.update({
        'height': tile_height,
        'width': tile_width,
        'transform': transform
    })

    # Write the padded tile to a new GeoTIFF
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(tile_data)

def tile_contains_valid_data(tile_data, nodata_value):
    """
    Checks if not all pixels in the tile contains nodata value (none of the pixels having valid data / pixels)

    Args:
        tile_data (rasterio): The content of the image read by rasterio
        nodata_value (int): The nodata value of the image read by rasterio
    
    Returns:
        boolean: Wether or not some of the tiles are valid (all pixels equal nodata value = False, otherwise = True)
    """

    if nodata_value is not None:
        # Check if all pixel is nodata; if so, return False
        return not np.all(tile_data == nodata_value)
    else:
        # If no nodata value is defined, assume all pixels have valid data
        return True

def generate_tiles(input_geotiff, output_dir, tile_width=1000, tile_height=1000):
    """
    Splits a GeoTIFF into fixed-size tiles without overlap, padding edges as necessary
    
    Args:
        input_geotiff (string): File path to the GeoTIFF
        output_dir (string): File path to the output folder
        tile_width (int): Width of the tile (default = 1000)
        tile_height (int): Height of the tile (default = 1000)
    """
    
    # Clear any existing .tif files in the output directory
    clear_output_directory(output_dir)

    # Open the input GeoTIFF
    with rasterio.open(input_geotiff) as src:
        img_width = src.width
        img_height = src.height
        nodata_value = src.nodata  # Get the nodata value if defined, or default to 0

        # Calculate the number of tiles in both dimensions, without adjusting for overlap
        num_tiles_x = (img_width + tile_width - 1) // tile_width
        num_tiles_y = (img_height + tile_height - 1) // tile_height

        # Iterate over the grid without overlap
        for i in tqdm(range(num_tiles_y), 'Tile rows'):
            for j in range(num_tiles_x):
                # Calculate the window position without overlap
                x_off = j * tile_width
                y_off = i * tile_height

                # Define the window for this tile with fixed dimensions
                window = Window(x_off, y_off, tile_width, tile_height)
                new_transform = src.window_transform(window)

                # Read the tile data, and pad if the window extends beyond image bounds
                tile_data = src.read(window=window, boundless=True, fill_value=nodata_value)

                # Check if the tile contains any valid data
                if tile_contains_valid_data(tile_data, nodata_value):
                    # Define output filename
                    output_filename = os.path.join(output_dir, f"tile_{i}_{j}.tif")

                    # Save the tile with padding if necessary
                    save_tile(src, window, new_transform, output_filename, nodata_value, tile_width, tile_height)
