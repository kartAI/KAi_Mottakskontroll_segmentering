# FarSeg/Functionality/prepareData.py

# Libraries:

from collections import Counter
import geopandas as gpd
import glob
import math
import numpy as np
import os
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from scipy.ndimage import label
from shapely.geometry import Polygon, MultiPolygon
import shutil
from tqdm import tqdm

import generalFunctions as gf
from geoTIFFandJPEG import imageSaver
from preProcessing import preProcessor

# Functions:

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

    if buildings_inside_hole:
        red_band |= rasterize([(poly, color[0]) for poly, color in buildings_inside_hole], out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)
        green_band |= rasterize([(poly, color[1]) for poly, color in buildings_inside_hole], out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)
        blue_band |= rasterize([(poly, color[2]) for poly, color in buildings_inside_hole], out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)

    with rasterio.open(output, 'w', driver="GTiff",
                       height=out_shape[0], width=out_shape[1], count=3,
                       dtype=np.uint8, crs=crs, transform=transform) as dst:
        dst.write(red_band, 1)
        dst.write(green_band, 2)
        dst.write(blue_band, 3)

def categorizeGeoTIFFRoads(geodata, geotiff, output):
    """
    Creates a new GeoTIFF with categorized raster depending on the type of the roads

    Arguments:
        geodata (string): Path to the GeoPackage file with relevant data
        geotiff (string): Path to the GeoTIFF file
        output (string): Path to the file in the output folder
    """
    gdf = gpd.read_file(geodata)

    with rasterio.open(geotiff) as src:
        transform = src.transform
        out_shape = [src.height, src.width]
        crs = src.crs

    road_classes = []
    color_map = {
        'S': (0, 255, 255),
        'P': (255, 51, 255),
        'K': (255, 165, 0),
        'F': (0, 255, 0)
    }

    for _, row in tqdm(gdf.iterrows(), total=gdf.shape[0], desc="Geometries categorized", colour="yellow"):
        poly = row.geometry
        category = row.vegkategori
        color = color_map[category]
        
        road_classes.append((poly, color))
    
    red_band = rasterize([(poly, color[0]) for poly, color in road_classes], out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)
    green_band = rasterize([(poly, color[1]) for poly, color in road_classes], out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)
    blue_band = rasterize([(poly, color[2]) for poly, color in road_classes], out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)

    holes = [
        Polygon(interior)
        for poly, _ in road_classes
        for single_poly in (poly.geoms if poly.geom_type == "MultiPolygon" else [poly])
        for interior in single_poly.interiors
    ]

    if holes:
        hole_mask = rasterize([(hole, 255) for hole in holes], out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8)
        red_band[hole_mask > 0] = 0
        green_band[hole_mask > 0] = 0
        blue_band[hole_mask > 0] = 0
    
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
    Count number of areas with specific colors in a raster.

    Argument:
        filepath (string): Path to the relevant file

    Returns:
        color_counts (Counter): Counter object with the number of each occurences of each building type
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

def fetchCategorizedTiles(geodata, geotiff, mask, tile_folder, count, object_type):
    """
    Function creating a new raster by categorizing all the buildings, and then split it strategically.
    When the categorized raster is splitted, it splits the aerial image into the same tiles.

    Arguments:
        geodata (string): Path to the relevant GeoPackage file
        geotiff (string): Path to the relevant GeoTIFF file
        mask (string): Path to the categorized GeoTIFF to be created
        tile_folder (string): Path to the folder where other folders with tiles (categorized and aerial images) are going to be created
        count (int): The number in the series of GeoTIFFs to be used in the training process
        object_type (string): String telling what kind of object type that is beeing analysed
    """

    tile_folder_categorized = tile_folder + f"/Categorized_{count}"
    tile_folder_original = tile_folder + f"/Original_{count}"
    
    gf.emptyFolder(tile_folder_categorized)
    gf.emptyFolder(tile_folder_original)

    if object_type == "buildings":
        categorizeGeoTIFFBuilding(geodata, geotiff, mask, 6)
    elif object_type == "roads":
        categorizeGeoTIFFRoads(geodata, geotiff, mask)

    color_map = {
        (0, 0, 0): "Only background",
        (255, 255, 255): "Simple",
        (255, 51, 255): "Complex",
        (0, 255, 255): "Complex",
        (0, 255, 0): "Simple",
        (255, 165, 0): "Simple",
        (255, 0, 0): "Simple"
    }

    def saveTile(x, y, jump, data, metadata, tile_folder):
        # Estimates the window:
        x_end, y_end = x + jump, y + jump
        if x_end > metadata["width"]: # Ensures no no data values
            x_end = metadata["width"]
        if y_end > metadata["height"]:
            y_end = metadata["height"]
        x_start, y_start = x_end - jump, y_end - jump # Ensures always 1024 x 1024 size
        # Extracts the tile data from the numpy array:
        tile_data = data[y_start:y_end, x_start:x_end]
        # Adjust the transform for the current tile:
        new_transform = metadata["transform"] * rasterio.Affine.translation(x_start, y_start)
        # Saves the tile:
        filename = os.path.join(tile_folder, f"tile_{1}_{x_start}_{y_start}.tif")
        preProcessing.save_tile(tile_data, new_transform, metadata, filename)
        return filename

    imageHandler = imageSaver()
    preProcessing = preProcessor(0, "")
    data, metadata = imageHandler.readGeoTIFF(mask)
    jump = 1024 # Each tile should be 1024 x 1024 pixels
    x, y = 0, 0
    with tqdm(total=int(metadata["height"]), desc="Creating tiles", colour="yellow") as pbar:
        while y < metadata["height"] - jump:
            while x < metadata["width"] - jump:
                filename = saveTile(x, y, jump, data, metadata, tile_folder_categorized)
                # Finds buildings types represented in the tile:
                colors = countColorAreas(filename)
                importance = 1
                for key in colors:
                    if color_map[key] == "Complex":
                        importance = 3
                    if color_map[key] == "Only background" and len(colors) == 1:
                        importance = 2
                if importance == 3:
                    saveTile(
                        x,
                        y - int(0.25 * jump) if (y - int(0.25 * jump)) >= 0 else 0,
                        jump, data, metadata, tile_folder_categorized
                    )
                    saveTile(
                        x,
                        y + int(0.25 * jump) if (y + int(0.25 * jump)) >= metadata["height"] else metadata["height"],
                        jump, data, metadata, tile_folder_categorized
                    )
                if importance == 3:
                    x += int(jump * 0.25)
                elif importance == 2:
                    x += jump
                else:
                    x += int(jump * 0.75)
            y += jump
            x = 0
            pbar.update(jump)
    
    with rasterio.open(geotiff) as src:
        for file in tqdm(glob.glob(tile_folder_categorized + "/*.tif"), desc="Saves tiles as aerial images", colour="yellow"):
            with rasterio.open(file) as tile:
                # Fetches the boundings of the categorized tile:
                bounds = tile.bounds
                # Defines window to crop image:
                window = from_bounds(*bounds, transform=src.transform)
                new_transform = src.window_transform(window)
                # Fetch the data:
                data = np.transpose(src.read(window=window), (1, 2, 0))
                # Update transform:
                metadata = src.meta.copy()
                metadata["profile"] = src.profile
                # Save the new tile of the aerial image:
                out_file = os.path.join(tile_folder_original, os.path.basename(file))
                preProcessing.save_tile(data, new_transform, metadata, out_file)

    if os.path.exists(tile_folder_categorized):
        shutil.rmtree(tile_folder_categorized)
    if os.path.exists(mask):
        os.remove(mask)

    return glob.glob(tile_folder_original + "/*.tif")
