# Data/data_statistics.py

import os
from collections import Counter
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm
import rasterio
import glob

import pre_processing as pre

def analyze_tiles(geo_tiffs, building_layer, road_layer, tile_folder):
    # Initializing of statistics:
    total_tiles = 0
    valid_tiles = 0
    tiles_with_buildings = 0
    tiles_with_roads = 0
    building_counts = Counter()
    road_counts = Counter()

    # Load building and road layer:
    buildings = gpd.read_file(building_layer)
    roads = gpd.read_file(road_layer)

    for tif in tqdm(geo_tiffs, 'TIFF files'):
        pre.clear_output_directory(tile_folder)
        pre.generate_tiles(tif, tile_folder)
        valid_tiles_list = [os.path.join(tile_folder, f) for f in os.listdir(tile_folder) if f.endswith('.tif')]
        valid_tiles += len(valid_tiles_list)
        total_tiles += 12 * 16 # Each GeoTIFF is divided into 12 x 16 pieces

        # Analyzing valid tiles:
        for tilepath in valid_tiles_list:
            with rasterio.open(tilepath) as tile:
                # Fetch the bounding box of the tile in coordinates:
                bounds = tile.bounds
                tile_box = box(*bounds)

                # Filter buildings and roads that overlaps with the tile:
                tile_buildings = buildings[buildings.intersects(tile_box)]
                tile_roads = roads[roads.intersects(tile_box)]

                # Update statistics:
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

geotiff_folder = "C:/images_mj"
geo_tiffs = glob.glob(geotiff_folder + '/*.tif')
building_layer = "C:/Users/jshjelse/Documents/Prosjektoppgave/Geopackage/Ver2/1_Farsund_Endelig_Bygning.gpkg"
road_layer = "C:/Users/jshjelse/Documents/Prosjektoppgave/Geopackage/Ver2/2_Veg_Aktuelle_Flater.gpkg"
tile_folder = "C:/Users/jshjelse/Documents/Prosjektoppgave/FarSeg/data/Tiles"

statistics = analyze_tiles(geo_tiffs, building_layer, road_layer, tile_folder)

# Write results:
for key, value in statistics.items():
    print(f"{key}: {value}")
