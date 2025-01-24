# FarSeg_dev/Functionality/dataStatistics.py

# Libraries:

from collections import Counter
import glob
import matplotlib.pyplot as plt
import rasterio
from shapely.geometry import box
from tqdm import tqdm

import generalFunctions as gf
from preProcessing import preProcessor

# Classes:

class statistics():
    """
    Instance calculating and creating statistics about the used data

    Attributes:
        total_tiles (int): Total number of tiles
        valid_tiles (int): Number of valid tiles
        tiles_with_buildings (int): Number of valid tiles with buildings
        tiles_with_roads (int): Number of valid tiles with roads
        building_counts (Counter): Counter object calculating total number of building elements
        road_counts (Counter): Counter object calculating total number of road elements
        geopackages (string): Path to the folder containing geopackages
    """

    def __init__(self, geopackages):
        """
        Creates a new instance of statistics.

        Args:
            geopackages (string): Path to the folder containing geopackages
        """
        self.total_tiles = 0
        self.valid_tiles = 0
        self.tiles_with_buildings = 0
        self.tiles_with_roads = 0
        self.building_count = Counter()
        self.road_count = Counter()
        self.geopackages = geopackages
    
    def main(self):
        """
        Runs the actual statistic function and creates the data.
        """
        # Fetches and generates folders:
        geotiff_folder = gf.get_valid_input("Where are the folder containing the relevant geotiffs(?): ", gf.doesPathExists)
        tile_folder = gf.get_valid_input("Where should the temporarly tiles be saved(?): ", gf.emptyFolder)
        # Creates statistic:
        statistic = self.createStatistic(geotiff_folder, tile_folder)
        # Prints the statistics:
        for key, value in statistic.items():
            print(f"{key}: {value}")
        # Plots data:
        buildings, roads = statistic["Building_distribution"], statistic["Road_distribution"]
        # Buildings
        plt.figure(figsize=(14, 6))
        plt.bar(buildings.keys(), buildings.values(), color='skyblue', alpha=0.7)
        plt.title('Buildings per Tile', fontsize=16)
        plt.xlabel('Number of Buildings per Tile', fontsize=12)
        plt.ylabel('Number of Tiles', fontsize=12)
        plt.xticks(list(buildings.keys()), rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        # Roads
        plt.figure(figsize=(14, 6))
        plt.bar(roads.keys(), roads.values(), color='skyblue', alpha=0.7)
        plt.title('Roads per Tile', fontsize=16)
        plt.xlabel('Number of Roads per Tile', fontsize=12)
        plt.ylabel('Number of Tiles', fontsize=12)
        plt.xticks(list(roads.keys()), rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def createStatistic(self, geotiffs, tile_folder):
        """
        Creates the relevant statistics.

        Args:
            geotiffs (string): Path to the folder containing all the GeoTIFFs used in the process
            tile_folder (string): Path to the folder for the temporarly tiles
        """
        geodata = gf.load_geopackages(self.geopackages)
        geotiffs = glob.glob(geotiffs + '/*.tif')
        for tif in tqdm(geotiffs, "GeoTIFF files"):
            gf.emptyFolder(tile_folder)
            tileGenerator = preProcessor(0.7, tile_folder)
            self.total_tiles += tileGenerator.generate_tiles(tif)
            valid_tiles_list = glob.glob(tile_folder + '/*.tif')
            self.valid_tiles += len(valid_tiles_list)
            # Analyzes valid tiles:
            for tilepath in valid_tiles_list:
                with rasterio.open(tilepath) as tile:
                    # Fetches the bounding box of the tile in coordinates:
                    bounds = tile.bounds
                    tile_box = box(*bounds)
                    # Filters buildings and roads that overlaps with the tile:
                    tile_buildings = geodata["buildings"][geodata["buildings"].intersects(tile_box)]
                    tile_roads = geodata["roads"][geodata["roads"].intersects(tile_box)]
                    # Updates statistics:
                    if not tile_buildings.empty:
                        self.tiles_with_buildings += 1
                        self.building_count[len(tile_buildings)] += 1
                    if not tile_roads.empty:
                        self.tiles_with_roads += 1
                        self.road_count[len(tile_roads)] += 1
        gf.emptyFolder(tile_folder)
        # Returns statistics:
        return {
            "Total_tiles": self.total_tiles,
            "Valid_tiles": self.valid_tiles,
            "Invalid_tiles": self.total_tiles - self.valid_tiles,
            "Tiles_with_buildings": self.tiles_with_buildings,
            "Tiles with roads": self.tiles_with_roads,
            "Building_distribution": dict(self.building_count),
            "Road_distribution": dict(self.road_count)
        }
