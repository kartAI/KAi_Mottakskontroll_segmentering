# FarSeg/Functionality/preProcessing.py

# Libraries:

import geopandas as gpd
import numpy as np
import os
from pyproj import CRS
import rasterio
from rasterio.features import geometry_mask, shapes
from shapely.geometry import shape
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

    def __init__(self, geotiffs, geodata, transform=None):
        """
        Creates a new instance of MapSegmentationDataset.

        Arguments:
            geotiffs (list[string]): List of path to the GeoTIFFs used for training
            geodata (dict): Dictionary with 'buildings' and 'roads' as key, refering to geographic data
            transform (dict): Optional, any data augmentations or transformations, default None
        """
        self.geotiff_list = geotiffs
        self.geodata = geodata
        self.transform = transform

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

        Argument:
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
            # Combine the masks into a single-channel label:
            # (0: background, 1: object type)
            label_mask = np.zeros(out_shape, dtype=np.uint8) # Initialize background as class 0
            for val, objtype in enumerate(self.geodata):
                # Reproject geometries to match the CRS of the current GeoTIFF:
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
        output (string): Path to output folder to store tiles
        width (int): Image width of the tile, default 1024
        height (int): Image height of the tile, default 1024
    """

    def __init__(self, output, width=1024, height=1024):
        """
        Creates an instance of preProcessor.

        Arguments:
            output (string): Path to output folder to store tiles
            width (int): Image width of the tile, default 1024
            height (int): Image height of the tile, default 1024
        """
        self.output = output
        self.width = width
        self.height = height

    def generate_tiles_overlap(self, geotiff, remove=True, count=False):
        """
        Splits a GeoTIFF into tiles with 25% overlap in width and height and saves them in specified folder.

        Arguments:
            geotiff (string): Path to the GeoTIFF to split
            remove (bool): Telling if the tile folder should be emptied before new ones are generated, default True
            count (int): Integer used for file names of saved tiles, default False

        Returns:
            int: Number of generated tiles (tiles_x-direction * tiles_y-direction)
        """
        if remove:
            gf.emptyFolder(self.output)
        imageHandler = imageSaver()
        data, metadata = imageHandler.readGeoTIFF(geotiff)
        jump = int(self.width * 0.75)
        x, y = self.width, self.height
        i, j = 0, 0
        # Iterates over the grid without overlap:
        while y <= data.shape[0]:
            while x <= data.shape[1]:
                # Calculates the window position without overlap:
                x_end = x
                y_end = y
                x_start = x_end - self.width
                y_start = y_end - self.height
                # Extracts the tile data from the numpy array:
                tile_data = data[y_start:y_end, x_start:x_end]
                # Adjust the transform for the current tile:
                new_transform = metadata["transform"] * rasterio.Affine.translation(x_start, y_start)
                # Defines output filename:
                if count:
                    filename = os.path.join(self.output, f"tile_{count}_{i}_{j}.tif")
                else:
                    filename = os.path.join(self.output, f"tile_{i}_{j}.tif")
                # Saves the tile:
                self.save_tile(tile_data, new_transform, metadata, filename)
                j += 1
                if x != data.shape[1]:
                    x += jump
                if x > data.shape[1]:
                    x = data.shape[1]
                elif x == data.shape[1]:
                    x = np.inf
            x = self.width
            j = 0
            i += 1
            if y != data.shape[0]:
                y += jump
            if y > data.shape[0]:
                y = data.shape[0]
            elif y == data.shape[0]:
                y = np.inf
        return i * j
    
    def generate_tiles_no_overlap(self, geotiff, remove=True, count=False):
        """
        Splits a GeoTIFF into tiles and saves it in specified folder. Here it is no overlap except for the last row and column to avoid no-data value in the tile.

        Arguments:
            geotiff (string): Path to the GeoTIFF to split
            remove (bool): Telling if the tile folder should be emptied before new ones are generated, default True
            count (int): Integer used for file names of saved tiles, default False

        Returns:
            int: Number of generated tiles (tiles_x-direction * tiles_y-direction)
        """
        if remove:
            gf.emptyFolder(self.output)
        imageHandler = imageSaver()
        data, metadata = imageHandler.readGeoTIFF(geotiff)
        count_x = (metadata["width"] + self.width - 1) // self.width
        count_y = (metadata["height"] + self.height - 1) // self.height
        # Iterates over the grid without overlap:
        for i in range(count_y):
            for j in range(count_x):
                # Calculates the window position without overlap:
                x_end = min((j+1) * self.width, metadata["width"])
                y_end = min((i+1) * self.height, metadata["height"])
                x_start = x_end - self.width
                y_start = y_end - self.height
                # Extracts the tile data from the numpy array:
                tile_data = data[y_start:y_end, x_start:x_end]
                # Adjust the transform for the current tile:
                new_transform = metadata["transform"] * rasterio.Affine.translation(x_start, y_start)
                # Defines output filename:
                if count:
                    filename = os.path.join(self.output, f"tile_{count}_{i}_{j}.tif")
                else:
                    filename = os.path.join(self.output, f"tile_{i}_{j}.tif")
                # Saves the tile:
                self.save_tile(tile_data, new_transform, metadata, filename)
        return count_x * count_y

    def save_tile(self, tile_data, transform, metadata, filename):
        """
        Save a tile to a new GeoTIFF.

        Arguments:
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
        # Check for correct order:
        if tile_data.shape == (1024, 1024, 3):
            tile_data = np.transpose(tile_data, (2, 0, 1))
        # Write the tile to a new GeoTIFF:
        with rasterio.open(filename, 'w', **profile) as dst:
            dst.write(tile_data)

# Helper functions:

def geotiff_to_geopackage(input_tiff, output_gpkg, layer_name, log_file):
    """
    Converts a binary GeoTIFF (1 = True, 0 = False) to a GeoPackage.

    Arguments:
        input_tiff (string): Path to the GeoTIFF
        output_gpkg (string): Path to save the GeoPackage
        layer_name (string): Name of the layer in the geopackage
    """

    def is_epsg(crs):
        try:
            return CRS.from_user_input(crs).to_epsg() is not None
        except:
            return False

    # Open the GeoTIFF:
    with rasterio.open(input_tiff) as src:
        image = src.read(1) # Reads the first band
        transform = src.transform
        crs = src.crs

    if crs == None or not is_epsg(crs):
        user_crs = gf.get_valid_input("Write the EPSG-code here: ", gf.positiveNumber)
        crs = CRS.from_epsg(int(user_crs))

    # Filter the object from the layer (values = 1):
    mask = image == 1

    # Convert raster to vector (polygons):
    shapes_gen = shapes(image, mask=mask, transform=transform)

    # Create a list of geometries:
    geoms = [
        {"geometry": shape(geom), "properties": {"value": value}}
        for geom, value in shapes_gen if value == 1
    ]

    if not geoms:
        gf.log_info(log_file, f"No geometries found in {input_tiff}. Skips.")
        return

    # Convert to GeoDataFrame:
    gdf = gpd.GeoDataFrame.from_features(geoms, crs=crs)

    if gdf.crs is None:
        gdf.set_crs(crs, inplace=True)

    # Save as GeoPackage:
    gdf.to_file(output_gpkg, layer=layer_name, driver="GPKG")
    gf.log_info(log_file, f"GeoTIFF {input_tiff} saved as geopackage in {output_gpkg}")
