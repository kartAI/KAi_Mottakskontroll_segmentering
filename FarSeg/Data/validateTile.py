# Data/validateTile.py

# Imports libraries:

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import geopandas as gpd
from shapely.geometry import box
import rasterio

# Class:

class tileValidation():
    """
    Creates an instance that can check if a GeoTIFF file is intersecting with the geopackages - contains relevant information for the training.

    Attributes:
        geopackages (list of strings): A list containing the file paths to the geopackages
    """

    def __init__(self, geopackages):
        """
        Creates a new instance of tileValidation

        Args:
            geopackages (list of strings): A list containing the file paths to the geopackages
        """
        buildings = geopackages[0]
        roads = geopackages[1]

        self.buildings = gpd.read_file(buildings)
        self.roads = gpd.read_file(roads)
    
    def validate(self, tile_folder):
        """
        Validates the tiles depending on overlap with geopackages

        Args:
            tilefolder (string): Path to all the GeoTIFF tiles
        
        Returns:
            valid_tiles (list of strings): A list with the file path to all the valid tiles
        """
        
        tile_paths = [os.path.join(tile_folder, f) for f in os.listdir(tile_folder) if f.endswith('.tif')]
        valid_tiles = []

        for path in tile_paths:
            with rasterio.open(path) as tile:
                # Fetches the bounding box of the tile in coordinates:
                bounds = tile.bounds
                tile_box = box(*bounds)

                # Check if any buildings or roads overlaps with the tile:
                if self.buildings.intersects(tile_box).any():
                    valid_tiles.append(path)
                elif self.roads.intersects(tile_box).any():
                    valid_tiles.append(path)
        
        return valid_tiles