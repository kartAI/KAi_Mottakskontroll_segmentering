# Import libraries

import generalFunctions
import geoTIFFandJPEG
import preProcessing
import validation

import geopandas as gpd
import glob
import matplotlib.colors as mcolors
import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from shapely.geometry import box
from tqdm import tqdm

# Variables:

output = "C:/Users/jshjelse/Documents/Tiles"
masks = "C:/Users/jshjelse/Documents/Masks"
geotiff = "C:/Users/jshjelse/Documents/Segmented_semi-urban.tif"
geopackage = "C:/Jakob_Marianne_2024_2025/Geopackage_Farsund/Test"

output_gpkg = "C:/Users/jshjelse/Documents/validated.gpkg"
layer_name = "iou_tiles"
qml_path = "C:/Users/jshjelse/Documents/iou_tiles_style.qml"

# Functions:

def iou_to_color(iou):
    iou = max(0.0, min(1.0, iou))
    if iou <= 0.5:
      t = 1
      start = np.array([255, 0, 0]) # Red
      end = np.array([255, 0, 0]) # Red
    elif iou <= 0.75:
      t = iou / 0.5
      start = np.array([255, 0, 0]) # Red
      end = np.array([255, 165, 0]) # Orange
    else:
      t = (iou - 0.5) / 0.5
      start = np.array([255, 165, 0]) # Orange
      end = np.array([0, 255, 0]) # Green
    rgb = (1 - t) * start + t * end
    for i in range(len(rgb)):
      if rgb[i] < 0:
        rgb[i] = 0
      elif rgb[i] > 255:
         rgb[i] = 1
    return tuple(rgb)

def get_raster_bbox(tif):
    with rasterio.open(tif) as src:
        bounds: BoundingBox = src.bounds
        return (bounds.left, bounds.bottom, bounds.right, bounds.top)

def generate_qml(qml_path):
    """
    Createas a simple QGIS style file (.qml) that uses color_rgb("r", "g", "b") as color.

    Args:
        qml_path (str): Path name to write to (ends with '.qml')
    """
    qml_content = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="Symbology" version="3.28">
  <renderer-v2 type="singleSymbol">
    <symbols>
      <symbol name="0" type="fill" alpha="1" force_rhr="0">
        <layer pass="0" class="SimpleFill" locked="0">
          <prop k="color_expression" v="color_rgb(&quot;r&quot;, &quot;g&quot;, &quot;b&quot;)"/>
          <prop k="outline_color" v="0,0,0,255"/>
          <prop k="outline_width" v="0.26"/>
          <prop k="fill_opacity" v="0.5"/>
          <prop k="outline_color" v="255,0,0,255"/>
        </layer>
      </symbol>
    </symbols>
    <rotation/>
    <sizescale/>
    <data-defined-properties>
      <Option/>
    </data-defined-properties>
  </renderer-v2>
  <layerGeometryType>2</layerGeometryType> <!-- 2 = polygon -->
</qgis>
"""
    with open(qml_path, "w", encoding="utf-8") as f:
        f.write(qml_content)

# Program:

pre = preProcessing.preProcessor(output)
pre.generate_tiles_no_overlap(geotiff)
imageHandler = geoTIFFandJPEG.imageSaver(geopackage)

tiles = glob.glob(output + "/*.tif")

geoms = []
ious = []
rgb_list = []
hex_list = []
total = 1024**2

for i, tile in tqdm(enumerate(tiles), desc="Processing tiles", total=len(tiles), leave=False, colour="yellow"):
  if i == 0:
    _, meta_data = imageHandler.readGeoTIFF(tile)
    crs = meta_data["crs"]
  generalFunctions.emptyFolder(masks)
  imageHandler.createMaskGeoTIFF(tile, masks)
  mask = glob.glob(masks + "/*.tif")[0]
  iou = validation.calculate_IoU_between_masks(mask, tile)
  data = validation.get_segmented_pixels(tile)
  if iou == None:
    if np.any(data) == False:
      iou = 1
    if np.any(data) == True:
      iou = 0
  geoms.append(box(*get_raster_bbox(tile)))
  ious.append(iou)
  r, g, b = iou_to_color(iou)
  rgb_list.append((r, g, b))
  hex_list.append(mcolors.to_hex((r/255, g/255, b/255)))

gdf = gpd.GeoDataFrame({
    "iou": ious,
    "r": [r for r, _, _ in rgb_list],
    "g": [g for _, g, _ in rgb_list],
    "b": [b for _, _, b in rgb_list],
    "color_hex": hex_list,
    "geometry": geoms
}, crs=crs)

gdf.to_file(output_gpkg, layer=layer_name, driver="GPKG")
generate_qml(qml_path)