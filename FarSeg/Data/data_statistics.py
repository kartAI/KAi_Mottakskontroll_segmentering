# Data/data_statistics.py

# Imports libraries:

import os
from collections import Counter
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm
import rasterio
import glob
import matplotlib.pyplot as plt

import pre_processing as pre

# Function:

def analyze_tiles(geo_tiffs, building_layer, road_layer, tile_folder):
    """
    Calculates statistics about the given data sets

    Args:
        geo_tiffs (list of strings): List of all the file paths as strings
        building_layer (string): File path to the geopackage layer
        road_layer (string): File path to the geopackage layer
        tile_folder (string): File path to the folder where the temporarly tiles will be stored
    
    Returns:
        Dict: Dictionary with all the calculated statistics
    """
    
    # Initializes the statistics:
    total_tiles = 0
    valid_tiles = 0
    tiles_with_buildings = 0
    tiles_with_roads = 0
    building_counts = Counter()
    road_counts = Counter()

    # Loads building and road layer:
    buildings = gpd.read_file(building_layer)
    roads = gpd.read_file(road_layer)

    for tif in tqdm(geo_tiffs, 'TIFF files'):
        pre.clear_output_directory(tile_folder)
        pre.generate_tiles(tif, tile_folder)
        valid_tiles_list = [os.path.join(tile_folder, f) for f in os.listdir(tile_folder) if f.endswith('.tif')]
        valid_tiles += len(valid_tiles_list)
        total_tiles += 12 * 16 # Each GeoTIFF is divided into 12 x 16 pieces

        # Analyzes valid tiles:
        for tilepath in valid_tiles_list:
            with rasterio.open(tilepath) as tile:
                # Fetches the bounding box of the tile in coordinates:
                bounds = tile.bounds
                tile_box = box(*bounds)

                # Filters buildings and roads that overlaps with the tile:
                tile_buildings = buildings[buildings.intersects(tile_box)]
                tile_roads = roads[roads.intersects(tile_box)]

                # Updates statistics:
                if not tile_buildings.empty:
                    tiles_with_buildings += 1
                    building_counts[len(tile_buildings)] += 1
                if not tile_roads.empty:
                    tiles_with_roads += 1
                    road_counts[len(tile_roads)] += 1
    
    pre.clear_output_directory(tile_folder)
    
    # Estimates invalid tiles:
    invalid_tiles = total_tiles - valid_tiles

    # Returns statistics:
    return {
        "Total_tiles": total_tiles,
        "Valid_tiles": valid_tiles,
        "Invalid_tiles": invalid_tiles,
        "Tiles_with_buildings": tiles_with_buildings,
        "Tiles_with_roads": tiles_with_roads,
        "Building_distribution": dict(building_counts),
        "Road_distribution": dict(road_counts),
    }

# Program:

#These filepaths do you need to change to match your data:
geotiff_folder = "C:/images_mj"
geo_tiffs = glob.glob(geotiff_folder + '/*.tif')
building_layer = "C:/Users/jshjelse/Documents/Prosjektoppgave/Geopackage/Ver2/1_Farsund_Endelig_Bygning.gpkg"
road_layer = "C:/Users/jshjelse/Documents/Prosjektoppgave/Geopackage/Ver2/2_Veg_Aktuelle_Flater.gpkg"

# This folder is created automatically:
tile_folder = "~/KAi_Mottakskontroll_segmentering/FarSeg/Data/Tiles"
os.makedirs(tile_folder, exist_ok=True)

statistics = analyze_tiles(geo_tiffs, building_layer, road_layer, tile_folder)

# Write results:
for key, value in statistics.items():
    print(f"{key}: {value}")

buildings, roads = statistics["Building_distribution"], statistics["Road_distribution"]

# Plot results:

plt.figure(figsize=(14,6))
plt.bar(buildings.keys(), buildings.values(), color='skyblue', alpha=0.7)
plt.title('Buildings per Tile', fontsize=16)
plt.xlabel('Number of Buildings per Tile', fontsize=12)
plt.ylabel('Number of Tiles', fontsize=12)
plt.xticks(list(buildings.keys()), rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,6))
plt.bar(roads.keys(), roads.values(), color='skyblue', alpha=0.7)
plt.title('Roads per Tile', fontsize=16)
plt.xlabel('Number of Roads per Tile', fontsize=12)
plt.ylabel('Number of Tiles', fontsize=12)
plt.xticks(list(roads.keys()), rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
