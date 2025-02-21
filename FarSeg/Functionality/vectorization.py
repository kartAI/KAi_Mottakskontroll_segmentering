# FarSeg/Functionality/vectorization.py

# Libraries:

import cv2
import json
import networkx as nx
import numpy as np
import os
import rasterio
from rasterio.features import shapes
from scipy.ndimage import label
from shapely.geometry import LineString, Point, shape
from shapely.ops import snap, unary_union
from skimage.morphology import skeletonize, remove_small_objects
from tqdm import tqdm

import coordinates as C
import generalFunctions as gf

# Functions:

def convert_coords(geom, src, zone, utmOrLatLon):
    """
    Converts coordinates based on the utmOrLatLon flag.

    - If utmOrLatLon is True: Convert from UTM to LatLon
    - If utmOrLatLon is False: Retrieves UTM coordinates, but does not convert

    Arguments:
        geom (geometry): A shapely geometry (LineString or Point)
        src (rasterio.io.DatasetReader): The rasterio object to get UTM coordinates from
        zone (string): UTM zone for conversion
        utmOrLatLon (bool): Flag indicating whether to convert to LatLon or keep in UTM

    Returns:
        geometry: Converted geometry with LatLon or UTM coordinates
    """
    if isinstance(geom, LineString):
        coords = []
        for x, y in geom.coords:
            utm_x, utm_y = src.xy(y, x)
            if utmOrLatLon:
                lat, lon = C.UTMtoLatLon(utm_y, utm_x, zone)
                coords.append((lon, lat))
            else:
                coords.append((utm_y, utm_x))
        return LineString(coords)
    elif isinstance(geom, Point):
        utm_x, utm_y = src.xy(geom.y, geom.x)
        if utmOrLatLon:
            lat, lon = C.UTMtoLatLon(utm_y, utm_x, zone)
            return Point(lon, lat)
        else:
            return Point(utm_y, utm_x)
    return geom

def loadMask(input_geotiff):
    """
    Creates a ndarray of the mask from the given GeoTIFF.

    Arguments:
        input_geotiff (string): Path to the GeoTIFF to convert
    
    Returns:
        ndarray: A ndarray-representation of the mask from the GeoTIFF
        src: The metadata of the GeoTIFF
    """
    with rasterio.open(input_geotiff) as src:
        img = src.read(1)
        binary = (img >= 128).astype(np.uint8)
        return (binary == 1).astype(np.uint8), src

def save_JSON(elements, file):
    """
    Creates a JSON-element of the geographic data and saves it.
    
    Argumets:
        elements (list): A list of geographic elements
        file (string): Path to store the data in file
    """
    with open(file, 'w') as f:
        json.dump(
            {"type": "FeatureCollection", "features": [
                {"type": "Feature", "geometry": el.__geo_interface__, "properties": {}} for el in elements
            ]}, f, indent=2
        )

def createCenterLines(input_geotiff, save, output_geojson, zone, utmOrLatLon, log_file, count=None):
    """
    Creates centerlines of the segmented areas in input_geotiff, saves them in a GeoJSON file is required, and calculates and returns the total length of the center lines.

    Arguments:
        input_geotiff (string): File path to the GeoTIFF to be analysed
        save (bool): If True, the function saves the center lines as GeoJSON
        output_geojson (string): File path and name to the GeoJSON file to be created
        zone (string): The UTM zone of the GeoTIFF
        utmOrLatLon (bool): Boolean value to choose UTM or latlon cordinates in the GeoJSON file
        log_file (string): Path to log file to store data
        count (int): Number in the order of GeoTIFF when saved as GeoJSON, default None

    Returns:
        float: Total length of the center lines
    """

    # Step 1: load raster and extract mask
    mask, data = loadMask(input_geotiff)

    # Step 2: Clean up noise and skeletonize
    # All data in GeoTIFF consisting of less than 50 pixels is not used
    mask = remove_small_objects(mask.astype(bool), min_size=50).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    skeleton = skeletonize(mask)

    # Step 3: Vectorize skeleton and filter short segments
    contours, _ = cv2.findContours(skeleton.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lines = [LineString(cnt[:, 0, :]) for cnt in contours if len(cnt) > 1 and LineString(cnt[:, 0, :]).length > 5]

    # Step 4: Create graph and intersections
    G = nx.Graph()
    for line in tqdm(lines, desc="Processing lines", colour="green", leave=False):
        for p1, p2 in zip(line.coords[:-1], line.coords[1:]):
            G.add_edge(p1, p2)
    intersection_points = [Point(n) for n, d in G.degree() if d > 2]

    # Step 5: Snap poorly segmented areas
    all_lines = unary_union(lines)
    lines = [snap(line, all_lines, 5) for line in lines]

    # Step 6: Calculate total length
    total_distance = 0
    for line in tqdm(lines, desc="Calculate length", colour="green", leave=False):
        total_distance += line.length
    
    # Step 7 (optional): Convert to GeoJSON format and save to file
    if save:
        lines = [convert_coords(line, data, zone, utmOrLatLon) for line in lines]
        intersection_points = [convert_coords(p, data, zone, utmOrLatLon) for p in intersection_points]

        line_file = os.path.join(output_geojson, f"Segmented_lines_{count}_{'latlon' if utmOrLatLon else 'utm'}.geojson")
        intersection_file = os.path.join(output_geojson, f"Segmented_intersections_{count}_{'latlon' if utmOrLatLon else 'utm'}.geojson")

        save_JSON(lines, line_file)
        save_JSON(intersection_points, intersection_file)

        gf.log_info(log_file, f"Centerlines saved to {line_file}.")
        gf.log_info(log_file, f"Intersections saved to {intersection_file}.")
    
    return total_distance

def createBoundaries(input_geotiff, save, output_geojson, zone, utmOrLatLon, log_file, count=None):
    """
    Creates boundary lines of the segmented areas in input_geotiff, saves them in a GeoJSON file if required, and calculates and returns the total length of the boundaries.

    Arguments:
        input_geotiff (string): File path to the GeoTIFF to be analysed
        save (bool): If True, the function saves the boundaries as GeoJSON
        output_geojson (string): File path and name of the geojson to be created
        zone (string): The UTM zone of the GeoTIFF
        utmOrLatLon (bool): Boolean value to choose UTM or LatLon coordinates in the GeoJSON file
        log_file (string): Path to log file to store data
        count (int): Number in the order of GeoTIFF when saved as GeoJSON, default None
    
    Returns:
        float: Total length of the boundaries
    """

    min_size = 10

    # Step 1: Load raster and extract mask
    mask, data = loadMask(input_geotiff)
    labelled_array, _ = label(mask)

    if labelled_array.dtype != np.uint8:
        labelled_array = labelled_array.astype(np.uint8)

    # Step 2: Create boundaries (edges) of polygons
    boundaries = []
    for geom, value in tqdm(shapes(labelled_array, mask=mask), desc="Creating boundaries", colour="green", leave=False):
        if value > 0:
            poly = shape(geom)
            if poly.area > min_size:
                boundary = poly.exterior
                boundaries.append(boundary)
                for interior in poly.interiors:
                    boundaries.append(interior)
    
    # Step 3: Convert boundaries to LineString
    boundaries = [LineString(boundary.coords) for boundary in boundaries]

    # Step 4: Calculates total length of boundaries
    total_distance = 0
    for boundary in boundaries:
        total_distance += boundary.length

    # Step 5 (optional): Change coordinates for GeoJSON output, prepare GeoJSON data and save to file
    if save:
        boundaries = [convert_coords(boundary, data, zone, utmOrLatLon) for boundary in boundaries]

        boundary_file = os.path.join(output_geojson, f"Segmented_boundaries_{count}_{'latlon' if utmOrLatLon else 'utm'}.geojson")

        save_JSON(boundaries, boundary_file)

        gf.log_info(log_file, f"Boundaries saved to {boundary_file}")
    
    return total_distance
