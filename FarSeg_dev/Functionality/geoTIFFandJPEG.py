# FarSeg_dev/Functionality/geoTIFFandJPEG.py

# Libraries:

import numpy as np
import os
from PIL import Image
import rasterio
from rasterio.features import rasterize

import generalFunctions as gf

# Class:

class imageSaver():
    """
    Instance handling saving of images as GeoTIFF and JPEG
    
    Attributes:
        folder (string): A string with the path to the folder containing relevant geopackage data
    """

    def __init__(self, folder):
        """
        Creates a new instance of imageSaver.

        Args:
            folder (string): A string with the path to the folder containing relevant geopackage data
        """
        self.geopackages = gf.load_geopackages(folder)
    
    def readGeoTIFF(self, tif_file):
        """
        Reads the GeoTIFF data and returns it for further work.

        Args:
            tif_file (string): The file path to the GeoTIFF

        Returns:
            image_data (ndarray): The content of the image, shape (height, width, bands) or (height, width) for single-band images
            metadata (dict): Metadata about the GeoTIFF, including projection, transform, etc.
        """
        with rasterio.open(tif_file) as src:
            # Read the GeoTIFF data
            image_data = src.read()
            
            # Handle single-band or multi-band images
            if image_data.shape[0] == 1:  # Single-band image
                image_data = image_data[0]  # Remove the band dimension
            else:  # Multi-band image
                image_data = np.transpose(image_data, (1, 2, 0))  # Rearrange to (height, width, bands)

            # Normalize data if required
            if image_data.dtype != np.uint8: # If the data is not 8-bit
                image_min = image_data.min()
                image_max = image_data.max()
                image_data = ((image_data - image_min) / (image_max - image_min) * 255).astype('uint8')
        
            metadata = src.meta

        return image_data, metadata

    def saveGeoTIFF(self, tif, output, mask=None, original=True):
        """
        Createas a copy of the input GeoTIFF and saves it.
        If the mask is going to be saved, the mask is given as input, and original = False.
        The function includes '_original' or '_mask' in the end of the file name depending on situation.

        Args:
            tif (string): File path to the GeoTIFF
            output (string): File path to the output folder
            mask (None, ndarray): The rgb mask saved in a ndarray, default None
            original (bool): A boolean telling what to be saved, default True
        """
        image_data, metadata = self.readGeoTIFF(tif)
        if original:
            filename = os.path.join(output, os.path.basename(tif).replace('.tif', '_original.tif'))
        else:
            image_data = mask
            filename = os.path.join(output, os.path.basename(tif).replace('.tif', '_mask.tif'))
        with rasterio.open(
            filename,
            'w',
            driver='GTiff',
            height=metadata["height"],
            width=metadata["width"],
            count=metadata["count"],
            dtype=metadata["dtype"],
            crs=metadata["crs"],
            transform=metadata["transform"]
        ) as dst:
            dst.write(np.transpose(image_data, [2, 0, 1])) # Back to [bands, height, width]

    def saveGeoTIFFasJPEG(self, tif, output, mask=None, original=True):
        """
        Creates a copy of the input GeoTIFF and saves it as a JPEG.
        If the mask is going to be saved, the mask is given as input, and original = False.
        The function includes '_original' or '_mask' in the end of the file name depending on situation.

        Args:
            tif (string): File path to the GeoTIFF
            output (string): File path to the output folder
            mask (None, ndarray): The rgb mask saved in a ndarray, default None
            original (bool): A boolean telling what to be saved, default True
        """
        if original:
            image_data, metadata = self.readGeoTIFF(tif)
            filename = os.path.join(output, os.path.basename(tif).replace('.tif', '_original.jpg'))
        else:
            image_data = mask
            filename = os.path.join(output, os.path.basename(tif).replace('.tif', '_mask.jpg'))
        original = Image.fromarray(image_data)
        original.save(filename, format='JPEG')

    def createMask(self, tif):
        """
        Creates the mask of the original GeoTIFF.

        Args:
            tif (string): File path to the GeoTIFF
        
        Returns:
            rgb_mask (ndarray): The mask of the GeoTIFF based upon geopackage layers
        """
        image_data, metadata = self.readGeoTIFF(tif)
        # Create RGB mask (only two layers: buildings and roads)
        rgb_mask =  np.zeros((metadata["height"], metadata["width"], 3), dtype='uint8')

        # Rasterize geometries for buildings (red) and roads(yellow)
        for layer, color in zip(['buildings', 'roads'], [(255, 0, 0), (255, 255, 0)]):
            if layer in self.geopackages:
                shapes = [(geom, 1) for geom in self.geopackages[layer].geometry if geom.is_valid]
                layer_mask = rasterize(
                    shapes,
                    out_shape=(metadata["height"], metadata["width"]),
                    transform=metadata["transform"],
                    fill=0,
                    dtype="uint8",
                    all_touched=True
                )
                # Apply color for the corresponding pixels:
                for channel, value in enumerate(color):
                    rgb_mask[:, :, channel] += (layer_mask * value).astype('uint8')
        return rgb_mask

    def createMaskGeoTIFF(self, tif, output):
        """
        Creates the mask of the GeoTIFF and saves it as a new GeoTIFF.

        Args:
            tif (string): File path to the GeoTIFF
            output (string): File path to the output folder
        """
        mask = self.createMask(tif)
        self.saveGeoTIFF(tif, output, mask, False)

    def createMaskJPEG(self, tif, output):
        """
        Creates the mask of the GeoTIFF and saves it as a new JPEG.

        Args:
            tif (string): File path to the GeoTIFF
            output (string): File path to the output folder
        """
        mask = self.createMask(tif)
        self.saveGeoTIFFasJPEG(tif, output, mask, False)
