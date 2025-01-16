# FarSeg_dev/Functionality/preProcessing.py

# Libraries:

import glob
import numpy as np
import os
import rasterio
from rasterio.features import geometry_mask
import torch
from torch.utils.data  import Dataset

import generalFunctions as gf
from geoTIFFandJPEG import imageSaver

#  Classes:

class MapSegmentationDataset(Dataset):
    """
    Creates a Dataset corresponding to the PyTorch system
    with Dataset and Dataloaders.

    Attributes:
        geotiff_list (list[string]): List of GeoTIFF file paths
        geodata (string): Path to the folder containing geopackages
        transform (None, callable): A function or callable object that takes two arguments (image_tensor, label_mask_tensor) and applies transformations. Default is None
    """

    def __init__(self, geotiff, geodata, transform=None):
        """
        Creates a new instance of MapSegmentationDataset.

        Args:
            geotiff (string): Path to the folder containing geotiffs
            geodata (string): Path to the folder containing geopackages
            transform (None, dict): Optional, any data augmentations or transformations, default None
        """
        self.geotiff_list = glob.glob(geotiff + '/*.tif')
        self.geodata = gf.load_geopackages(geodata)
        self.transform = transform # Optional: For data augmentations

    def __len__(self):
        """
        Gives the length of GeoTIFF list.

        Returns:
            int: Number of GeoTIFFs
        """
        return len(self.geotiff_list)
    
    def __getitem__(self, idx):
        """
        Fetches one of the elements in the data set.

        Args:
            idx (int): The index of the object in the data set to fetch
        
        Returns:
            image_tensor (ndarray): A ndarray representation of the image
            label_mask_tensor (ndarray): A ndarray representation of the mask of the image
        """
        path = self.geotiff_list[idx]
        # Load the image:
        with rasterio.open(path) as src:
            image = src.read()
            transform = src.transform
            out_shape = (src.height, src.width)
            geometries = {}
            masks = {}
            # Combine the masks into a single-channel label (0: background, 1: buildings, 2: roads):
            label_mask = np.zeros(out_shape, dtype=np.uint8) # Initialize background as class 0
            for objtype, val in enumerate(self.geodata):
                # Reproject geometries to math the CRS of the current GeoTIFF:
                geometries[objtype] = self.geodata[objtype].to_crs(src.crs)['geometry'].values
                # Rasterize the geometries for each class:
                masks[objtype] = geometry_mask(geometries[objtype], transform=transform, invert=True, out_shape=out_shape)
                # Insert class in the labels:
                label_mask[masks[objtype]] = val + 1
            # Convert image and masks to PyTorch tensors:
            image_tensor = torch.from_numpy(image).float() # Converts image to float tensor
            label_mask_tensor = torch.from_numpy(label_mask) # Multi-channel label mask
            if self.transform:
                # Apply data augmentations if available:
                image_tensor, label_mask_tensor = self.transform(image_tensor, label_mask_tensor)
        return image_tensor, label_mask_tensor

class preProcessor():
    """
    A pre-processor element that performs all the pre-processing of
    the GeoTIFF data before training and inferencing.

    Attributes:
        split (float): Split ratio for training and validation
        output (string): Path to output folder to store tiles
        width (int): Image width of the tile, default 1024
        height (int): Image height of the tile, default 1024
    """

    def __init__(self, split, output, width=1024, height=1024):
        """
        Creates an instance of preProcessor.

        Args:
            split (float): Split ratio for training and validation
            output (string): Path to output folder to store tiles
            width (int): Image width of the tile, default 1024
            height (int): Image height of the tile, default 1024
        """
        self.split_ratio = split
        self.output = output
        self.width = width
        self.height = height

    def generate_tiles(self, geopackage_folder, geotiff):
        """
        Splits a GeoTIFF into tiles and saves it in specified folder.

        Args:
            geopackage_folder (string): Path to folder containing geopackage data
            geotiff (string): Path to the GeoTIFF to split
        """
        gf.emptyFolder(self.output)
        imageHandler = imageSaver(geopackage_folder)
        data, metadata = imageHandler.readGeoTIFF(geotiff)
        count_x = (metadata["width"] + self.width - 1) // self.width
        count_y = (metadata["height"] + self.height - 1) // self.height
        # Iterates over the grid without overlap:
        for i in range(count_y):
            for j in range(count_x):
                # Calculates the window position without overlap:
                x_start = j * self.width
                y_start = i * self.height
                x_end = min(x_start + self.width, metadata["width"])
                y_end = min(y_start + self.height, metadata["height"])
                # Extracts the tile data from the numpy array:
                tile_data = data[y_start:y_end, x_start:x_end]
                # Handles padding if the tile is smaller than the specified size:
                if tile_data.shape[0] < self.height or tile_data.shape[1] < self.width:
                    padded_tile = np.full(
                        (self.height, self.width, tile_data.shape[2]),
                        metadata.get("nodata", 0),
                        dtype=tile_data.dtype
                    )
                    padded_tile[:tile_data.shape[0], :tile_data.shape[1], :] = tile_data
                    tile_data = padded_tile
                # Adjust the transform for the current tile:
                new_transform = metadata["transform"] * rasterio.Affine.translation(x_start, y_start)
                # Checks if the tile contains any valid data:
                if tile_contains_valid_data(tile_data, metadata.get("nodata", 0)):
                    # Defines output filename:
                    filename = os.path.join(self.output, f"tile_{i}_{j}.tif")
                    # Saves the tile:
                    self.save_tile(tile_data, new_transform, metadata, filename)
        return count_x * count_y

    def save_tile(self, tile_data, transform, metadata, filename):
        """
        Save a tile to a new GeoTIFF.

        Args:
        tile_data (ndarray): ndarray representation of the tile (image)
        transform (Affine): An Affain transformation object that defines the georeferencing of the tile
        metadata (dict): Dictionary with all the metadata of the new tile
        filename (string): Path to the new GeoTIFF
        """
        profile = metadata["profile"]
        profile.update({
            'height': tile_data.shape[0],
            'width': tile_data.shape[1],
            'transform': transform
        })
        # Write the tile to a new GeoTIFF:
        with rasterio.open(filename, 'w', **profile) as dst:
            dst.write(tile_data)

    def split_data(self, folder=None, liste=None):
        """
        Splits the GeoTIFFs from the folder in training and validation sets.

        Args:
            folder (string): Path to the folder containing the GeoTIFFs, default = None
            liste (list[string]): List of GeoTIFF paths, default = None
        
        Returns:
            list[string]: Lists of file paths, training and validation sets
            If no GeoTIFF files were found, return None
        """
        if folder != None:
            files = glob.glob(folder + '/*.tif')
        elif liste != None:
            files = liste
        if len(files) == 0:
            return None
        np.random.shuffle(files)
        split_idx = int(len(files) * self.split_ratio)
        return files[:split_idx], files[split_idx:]

# Helper functions:

def tile_contains_valid_data(tile_data, nodata):
    """
    Checks if not all pixels in the tile contains nnodata value
    (Ensure some valid data / pixels)

    Args:
        tile_data (ndarray): ndarray representation of the image
        nodata (int): The nodata value of the image
    
    Returns:
        bool: Wether or not some of the tiles are valid
    """
    if nodata is not None:
        # Checks if all pixels are nodata: if so, return False
        return not np.all(tile_data == nodata)
    else:
        # If no nodata value is defined, assume all pixels have valid data
        return True
