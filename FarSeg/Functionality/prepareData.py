# FarSeg/prepareData.py

# Libraries:

from collections import Counter
import geopandas as gpd
import glob
import math
import numpy as np
import os
import random
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from scipy.ndimage import label
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm

import generalFunctions as gf
from preProcessing import preProcessor

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
            if int(num) >= 100:
                return False
        if int(numbers[0]) + int(numbers[1]) != 100:
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
    Fetches the exterior geometries from the map layer.

    Argument:
        geom (geometry): The map objects of the layer

    Returns:
        list[geometry]: List of geometries for further analysis
    """
    if isinstance(geom, Polygon):
        return [geom.exterior]
    elif isinstance(geom, MultiPolygon):
        return [poly.exterior for poly in geom.geoms]
    else:
        return []

def angleBetweenVectors(v1, v2):
    """
    Calculates the angle between the two vectors.

    Arguments:
        v1, v2 (array): Numpy arrays representing the vectors

    Returns:
        float: Float number of the angle
    """
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0

    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = math.degrees(math.acos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def countSignificantCorners(polygon, angle_threshold=30, length_threshold=2):
    """
    Count number of building walls with a distinct enough characteristic.

    Arguments:
        polygon (geometry): Geometry of the polygon to be analysed
        angle_threshold (int): Minimum change of angle to count, default=10
        length_threshold (int): Minimum lenght of wall to be considered as a large enough wall, default=2
    
    Returns:
        int: Number of distinct walls / corners
    """
    coords = getExteriors(polygon)
    if len(coords) == 1:
        coords = coords[0].coords
    
    if len(coords) < 4:
        return 0
    
    significant_corners = 0

    for i in range(len(coords)):
        p1 = np.array(coords[i - 1])
        p2 = np.array(coords[i]) # It is this corner that counts
        p3 = np.array(coords[(i + 1) % len(coords)])
        p4 = np.array(coords[(i + 2) % len(coords)])
        
        v1 = p2 - p1
        v2 = p3 - p2 # It is this wall that counts
        v3 = p4 - p3

        len_v2 = np.linalg.norm(v2)

        angle_1 = angleBetweenVectors(v1, v2)
        angle_2 = angleBetweenVectors(v2, v3)

        if angle_threshold <= angle_1 or 360 - angle_threshold >= angle_1:
            if not 180 - angle_threshold <= angle_2 <= 180 + angle_threshold:
                if len_v2 > length_threshold:
                    significant_corners += 1
    
    return significant_corners

def getNumberOfBuildings(color, files):
    """
    """
    count = 0
    for file in files:
        counter = countColorAreas(file)
        count += counter[color]
    return count

def getRandomFilesWithGivenAmountOfBuildings(file_dictionary, limits, colors):
    """
    """
    def checkContinue(counter, limit, colors):
        """
        """
        count = 0
        valid = []
        for key in counter:
            if counter[key] >= limit:
                valid.append(colors[key])
                count += 1
        return valid, count == len(counter)

    limit = limits[0][1]
    counter = Counter({key: 0 for key in colors if file_dictionary[colors[key]][0] > 0})
    validated_colors = set()
    chosenFiles = []

    first = True
    
    for el in tqdm(limits, desc='Finding relevant files', colour='yellow'):
        if el[0] in validated_colors:
            continue
        if not first:
            validated, check = checkContinue(counter, limit, colors)
            validated_colors.update(validated)
            if check:
                break
        files = file_dictionary[el[0]][1:]
        random.shuffle(files)
        for file in files:
            if not first:
                validated, check = checkContinue(counter, limit, colors)
                validated_colors.update(validated)
                if check or el[0] in validated_colors:
                    break
            if file in chosenFiles:
                continue
            single_counter = countColorAreas(file)
            if len(single_counter) != 1 and single_counter[(0, 0, 0)]:
                del single_counter[(0, 0, 0)]
            counter.update(single_counter)
            chosenFiles.append(file)
        first = False

    return chosenFiles#, counter

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

def categorizeGeoTIFFBuilding(geodata, geotiff, output, classes):
    """
    Creates a new GeoTIFF with categorized raster depending on the complexity of the buildings

    Arguments:
        geodata (string): Path to the GeoPackage file with relevant data
        geotiff (string): Path to the GeoTIFF file
        output (string): Path to the file in the output folder
        classes (int): Number of classes to categorize the buildings into
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

    for _, row in tqdm(gdf.iterrows(), total=gdf.shape[0], desc="Geometries categorized", colour="yellow"):
        poly = row.geometry
        tolerance = 1
        simplified_poly = poly.simplify(tolerance, preserve_topology=True)
        num_sides = countSignificantCorners(simplified_poly)
        area = poly.area
        if classes == 3:
            if num_sides <= 6:
                classification = "simple"
            elif num_sides >= 12:
                classification = "complex"
            else:
                classification = "medium"
        elif classes == 6:
            if num_sides <= 6:
                classification = "simple"
            elif num_sides >= 12:
                classification = "complex"
            else:
                classification = "medium"
            if area > 500:
                if classification == "complex":
                    classification = "large complex"
                else:
                    classification = "large simple"
            elif area < 15:
                classification = "tiny"

        if classification == "large complex":
            color = (0, 255, 255)
        elif classification == "large simple":
            color = (255, 51, 255)
        elif classification == "tiny":
            color = (255, 255, 255)
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

    for building in tqdm(building_classes, desc="Finding buildings inside holes", colour="yellow"):
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

def getUniqueValues(filepath):
    """
    Fetches all unique RGB pixel values in the image

    Arguments:
        filepath (string): Path to the GeoTIFF to be analysed

    Returns:
        list[list[int]]: List of unique RGB combinations in the image
    """
    with rasterio.open(filepath) as src:
        image_data = src.read()
    
    if image_data.shape[0] == 3:
        rgb_image = np.dstack([image_data[0], image_data[1], image_data[2]])
    
    pixels = rgb_image.reshape(-1, 3)

    unique_colors = set(map(tuple, pixels))
    
    return unique_colors

def countColorAreas(filepath):
    """
    """
    color_counts = Counter()

    with rasterio.open(filepath) as src:
        image_data = src.read()
        rgb_image = np.dstack([image_data[0], image_data[1], image_data[2]])
        for color in getUniqueValues(filepath):
            mask = np.all(rgb_image == color, axis=-1)
            _, num_features = label(mask)
            color_counts[color] += num_features

    return color_counts

def main(geodata, geotiff, mask, tile_folder):
    """
    """
    """
    geodata = "C:/Jakob_Marianne_2024_2025/Geopackage_Farsund/Flater/Buildings.gpkg"
    geotiff = "C:/Jakob_Marianne_2024_2025/Ortofoto/Training/Test/Training_area_urban.tif"
    mask = "C:/Users/jshjelse/Documents/dev/mask.tif"
    tile_folder = "C:/Users/jshjelse/Documents/dev/Tiles"
    """

    categorizeGeoTIFFBuilding(geodata, geotiff, mask, 6)

    preProcessing = preProcessor(0.7, tile_folder)
    preProcessing.generate_tiles(mask)

    color_map = {
        (0, 0, 0): "Only background",
        (255, 255, 255): "tiny",
        (255, 51, 255): "large simple",
        (0, 255, 255): "large complex",
        (0, 255, 0): "simple",
        (255, 165, 0): "medium",
        (255, 0, 0): "complex"
    }

    paths = {
        "Only background": [0],
        "tiny": [0],
        "large simple": [0],
        "large complex": [0],
        "simple": [0],
        "medium": [0],
        "complex": [0]
    }

    for path in tqdm(glob.glob(tile_folder + '/*.tif'), desc='Processing tiles', colour='yellow'):
        counter = countColorAreas(path)
        if len(counter) == 1:
            if counter[(0, 0, 0)]:
                paths['Only background'].append(path)
                paths['Only background'][0] += 1
        else:
            for key in counter:
                if key == (0, 0, 0):
                    continue
                else:
                    paths[color_map[key]].append(path)
                    paths[color_map[key]][0] += counter[key]
    
    sorted_limits = []

    for key in paths:
        if paths[key][0] == 0:
            continue
        elif len(sorted_limits) == 0:
            sorted_limits.append([key, paths[key][0]])
        elif paths[key][0] >= sorted_limits[-1][1]:
            sorted_limits.append([key, paths[key][0]])
        else:
            for i in range(len(sorted_limits)):
                if paths[key][0] < sorted_limits[i][1]:
                    sorted_limits.append(sorted_limits[-1])
                    for j in range(len(sorted_limits) - 1, i, -1):
                        sorted_limits[j] = sorted_limits[j - 1]
                    sorted_limits[i] = [key, paths[key][0]]
                    break

    return getRandomFilesWithGivenAmountOfBuildings(paths, sorted_limits, color_map)
