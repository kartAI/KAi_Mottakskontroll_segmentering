# FarSeg/Functionality/vectorization.py

# Libraries:

import cv2
import json
import networkx as nx
import numpy as np
#from pyproj import Transformer
import rasterio
from shapely.geometry import LineString, Point
from shapely.ops import snap
from skimage.morphology import skeletonize, remove_small_objects
from tqdm import tqdm

import coordinates as C

# Program:

# Configurations
SAVE_AS_LATLON = True
FILE_PATH = "C:/Users/jshjelse/Documents/Results/merged_segmented_tif_8.tif"
OUTPUT_PATH = "C:/Users/jshjelse/Documents/cleaned_roads"

# Load raster and extract road mask
with rasterio.open(FILE_PATH) as src:
    img, utm_crs, data = src.read(), src.crs, src
binary_road = ((img[0] > 150) & (img[1] > 150) & (img[2] < 100)).astype(np.uint8)

# Clean up noise and skeletonize
binary_road = remove_small_objects(binary_road.astype(bool), min_size=50).astype(np.uint8)
binary_road = cv2.morphologyEx(binary_road, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
skeleton = skeletonize(binary_road)

# Vectorize skeleton and filter short segments
contours, _ = cv2.findContours(skeleton.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
road_lines = [LineString(cnt[:, 0, :]) for cnt in contours if len(cnt) > 1 and LineString(cnt[:, 0, :]).length > 50]

# Create graph and intersections
G = nx.Graph()
for line in tqdm(road_lines, desc="Processing roads"):
    for p1, p2 in zip(line.coords[:-1], line.coords[1:]):
        G.add_edge(p1, p2)
intersection_points = [Point(n) for n, d in G.degree() if d > 2]

# Snap poorly segmented roads
road_lines = [snap(line, line, 5) for line in road_lines]
intersection_points = [p for p in intersection_points if G.degree(tuple(p.coords[0])) >= 3]

# Convert to GeoJSON format
ZONE = "32N"

def convert_coords(geom, src, zone, save_as_latlon):
    """
    Converts coordinates based on the SAVE_AS_LATLON flag.
    
    - If SAVE_AS_LATLON is True: Converts from UTM to LatLon
    - if SAVE_AS_LATLON is False: Retrieves UTM coordinates but does not convert

    Args:
        geom (geometry): A shapely geometry (LineString or Point)
        src (rasterio.io.DatasetReader): The rasterio object to get UTM coordinates from
        zone (string): UTM zone for conversion
        save_as_latlon (bool): Flag indicating whether to convert to LatLon or keep in UTM

    Returns:
        geometry: Converted geometry with LatLon or  coordinates
    """
    if isinstance(geom, LineString):
        coords = []
        for x, y in geom.coords:
            utm_x, utm_y = src.xy(y, x)
            if save_as_latlon:
                lat, lon = C.UTMtoLatLon(utm_y, utm_x, zone)
                coords.append((lon, lat))
            else:
                coords.append((utm_y, utm_x))
        return LineString(coords)
    elif isinstance(geom, Point):
        utm_x, utm_y = src.xy(geom.y, geom.x)
        if save_as_latlon:
            lat, lon = C.UTMtoLatLon(utm_y, utm_x, zone)
            return Point(lon, lat)
        else:
            return Point(utm_y, utm_x)
    print("Ingen")
    return geom

road_lines = [convert_coords(line, data, ZONE, SAVE_AS_LATLON) for line in road_lines]
intersection_points = [convert_coords(p, data, ZONE, SAVE_AS_LATLON) for p in intersection_points]
road_data = [{"type": "Feature", "geometry": line.__geo_interface__, "properties": {}} for line in road_lines]
intersection_data = [{"type": "Feature", "geometry": p.__geo_interface__, "properties": {"intersection": True}} for p in intersection_points]

# Save GeoJSON files
roads_file = f"{OUTPUT_PATH}_roads_{'latlon' if SAVE_AS_LATLON else 'utm'}.geojson"
intersection_file = f"{OUTPUT_PATH}_intersections_{'latlon' if SAVE_AS_LATLON else 'utm'}.geojson"
with open(roads_file, "w") as f:
    json.dump({"type": "FeatureCollection", "features": road_data}, f, indent=2)
with open(intersection_file, "w") as f:
    json.dump({"type": "FeatureCollection", "features": intersection_data}, f, indent=2)

print(f"Roads saved to {roads_file}")
print(f"Intersections saved to {intersection_file}")
