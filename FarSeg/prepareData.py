# FarSeg/prepareData.py

# Libraries:

import geopandas as gpd
import math
import numpy as np
import os
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import shape, Polygon, MultiPolygon
from tqdm import tqdm

import Functionality.generalFunctions as gf

# Functions:

def validRatio(string):
    """
    Function ensuring correct split-ratio.

    Argument:
        string (string): User input to check for split-ratio

    Returns:
        bool: True if correct, otherwise False
    """
    string = string.replace(" ", "")
    try:
        numbers = string.split("-")
        if len(numbers) != 2:
            return False
        for num in numbers:
            if not int(num):
                return False
            if int(num) <= 0:
                return False
        return True
    except:
        return False

def correctDirecton(string):
    """
    Funtion checking correct split-direction.

    Argument:
        string (string): User input to check for direction
    
    Returns:
        bool: True if correct, otherwise False
    """
    if string.lower()[0] in ['v', 'h']:
        return True
    return False

def fetchDirection(string):
    """
    Function returning correct direction.

    Argument:
        string (string): User input
    
    Returns:
        string: Correct string
    """
    if string.lower()[0] == 'v':
        return 'vertical'
    else:
        return 'horizontal'

def getExteriors(geom):
    """
    """
    if isinstance(geom, Polygon):
        return [geom.exterior.coords]
    elif isinstance(geom, MultiPolygon):
        return [poly.exterior for poly in geom.geoms]
    else:
        return []

def angleBetweenVectors(v1, v2):
    """
    """
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0

    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = math.degrees(math.acos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def countSignificantCorners(polygon, threshold=30, length_threshold=2):
    """
    """
    coords = getExteriors(polygon)
    if len(coords) == 1:
        coords = coords[0].coords
    
    if len(coords) < 4:
        return 0
    
    significant_corners = 0

    for i in range(len(coords) - 1):
        p1 = np.array(coords[i - 1])
        p2 = np.array(coords[i])
        p3 = np.array(coords[(i + 1) % (len(coords) - 1)])
        
        v1 = p2 - p1
        v2 = p3 - p2

        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)

        angle = angleBetweenVectors(v1, v2)

        if angle > threshold and len_v1 > length_threshold and len_v2 > length_threshold:
            significant_corners += 1
    
    return significant_corners

def splitGeoTIFF(file):
    """
    Function that splits a GeoTIFF into a train and test part.

    Argument:
        file (string): Path to the GeoTIFF that are going to be splitted
    """
    ratio = gf.get_valid_input("Which ratio will you split the GeoTIFF in(?): ", validRatio)
    direction = fetchDirection(gf.get_valid_input("Will you split it vertically (v) or horizontally (h)(?): ", correctDirecton))

    ratio1, ratio2 = map(int, ratio.split('-'))
    total = ratio1 + ratio2
    frac1 = ratio1/total

    with rasterio.open(file) as src:
        width, height = src.width, src.height
        train_path = os.path.splitext(file)[0] + "_train.tif"
        test_path = os.path.splitext(file)[0] + "_test.tif"
        if direction == "vertical":
            split_col = int(width * frac1)
            window1 = Window(0, 0, split_col, height)
            window2 = Window(split_col, 0, width - split_col, height)
        else:
            split_row = int(height * frac1)
            window1 = Window(0, 0, width, split_row)
            window2 = Window(0, split_row, width, height - split_row)
        
        def write_tiff(dst_path, window):
            with rasterio.open(
                dst_path,
                'w',
                driver=src.driver,
                height=window.height,
                width=window.width,
                count=src.count,
                dtype=src.dtypes[0],
                crs=src.crs,
                transform=src.window_transform(window)
            ) as dst:
                dst.write(src.read(window=window))
        
        if ratio1 < ratio2:
            write_tiff(test_path, window1)
            write_tiff(train_path, window2)
        else:
            write_tiff(test_path, window2)
            write_tiff(train_path, window1)

def categorizeGeoTIFFBuilding(geodata, geotiff, output):
    """
    ...

    Arguments:
        geodata (string): Path to the folder to the GeoPackage
        geotiff (string): Path to the GeoTIFF
        output (string): Path to the output folder
    """
    gdf = gpd.read_file(geodata)

    with rasterio.open(geotiff) as src:
        transform = src.transform
        out_shape = [src.height, src.width]
        crs = src.crs
    
    red_band = np.zeros(out_shape, dtype=np.uint8)
    green_band = np.zeros(out_shape, dtype=np.uint8)
    blue_band = np.zeros(out_shape, dtype=np.uint8)

    building_classes = []

    for _, row in gdf.iterrows():
        poly = row.geometry
        num_sides = countSignificantCorners(poly)
        area = poly.area
        if num_sides <= 6:
            classification = "simple"
        elif num_sides >= 12:
            classification = "complex"
        else:
            classification = "medium"
        
        if area > 500:
            classification = "large"

        if classification == "large":
            color = (0, 255, 255)
        elif classification == "simple":
            color = (0, 255, 0)
        elif classification == "complex":
            color = (255, 0, 0)
        else:
            color = (255, 165, 0)
        
        building_classes.append((poly, color))
    
    red_band = rasterize([(poly, color[0]) for poly, color in building_classes], out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)
    green_band = rasterize([(poly, color[1]) for poly, color in building_classes], out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)
    blue_band = rasterize([(poly, color[2]) for poly, color in building_classes], out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)

    holes = [
        Polygon(interior)
        for poly, _ in building_classes
        for single_poly in (poly.geoms if poly.geom_type == "MultiPolygon" else [poly])
        for interior in single_poly.interiors
    ]

    buildings_inside_hole = []

    for building in building_classes:
        if any(building[0].within(hole) for hole in holes):
            buildings_inside_hole.append(building)

    if holes:
        hole_mask = rasterize([(hole, 255) for hole in holes], out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)
        red_band[hole_mask > 0] = 0
        green_band[hole_mask > 0] = 0
        blue_band[hole_mask > 0] = 0

    red_band |= rasterize([(poly, color[0]) for poly, color in buildings_inside_hole], out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)
    green_band |= rasterize([(poly, color[1]) for poly, color in buildings_inside_hole], out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)
    blue_band |= rasterize([(poly, color[2]) for poly, color in buildings_inside_hole], out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)

    with rasterio.open(output, 'w', driver="GTiff",
                       height=out_shape[0], width=out_shape[1], count=3,
                       dtype=np.uint8, crs=crs, transform=transform) as dst:
        dst.write(red_band, 1)
        dst.write(green_band, 2)
        dst.write(blue_band, 3)

# Program

geodata = "C:/Jakob_Marianne_2024_2025/Geopackage_Farsund/Test/Buildings.gpkg"
geotiff = "C:/Jakob_Marianne_2024_2025/Ortofoto/Training/Test/Training_area_urban.tif"
output = "C:/Users/jshjelse/Documents/dev/mask.tif"

categorizeGeoTIFFBuilding(geodata, geotiff, output)
