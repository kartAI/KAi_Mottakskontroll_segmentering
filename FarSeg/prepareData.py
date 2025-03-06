# FarSeg/prepareData.py

# Libraries:

from collections import deque
import geopandas as gpd
import numpy as np
import os
import rasterio
from rasterio.windows import Window
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union, linemerge
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

def splitGeoTIFF(file):
    """
    Function that splits a GeoTIFF into a train and test part.

    Arguments:
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
