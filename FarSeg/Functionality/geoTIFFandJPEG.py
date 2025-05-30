# FarSeg/Functionality/geoTIFFandJPEG.py

# Libraries:

import numpy as np
import os
from PIL import Image
import rasterio
from rasterio.features import rasterize
from rasterio.merge import merge
from shapely.geometry import Polygon

import generalFunctions as gf

# Class:

class imageSaver():
    """
    Instance handling saving of images as GeoTIFF and JPEG
    
    Attribute:
        folder (None, string): A string with the path to the folder containing relevant geopackage data, default None
    """

    def __init__(self, folder=None, small_objects=True):
        """
        Creates a new instance of imageSaver.

        Argument:
            folder (string): A string with the path to the folder containing relevant geopackage data
        """
        if folder != None:
            self.geopackages = gf.load_geopackages(folder, small_objects=small_objects)
        else:
            self.geopackages = None
    
    def readGeoTIFF(self, tif_file):
        """
        Reads the GeoTIFF data and returns it for further work.

        Argument:
            tif_file (string): The file path to the GeoTIFF

        Returns:
            image_data (ndarray): The content of the image, shape (height, width, bands) or (height, width) for single-band images
            metadata (dict): Metadata about the GeoTIFF, including projection, transform, etc.
        """
        with rasterio.open(tif_file) as src:
            # Read the GeoTIFF data:
            image_data = src.read()
            
            # Handle single-band or multi-band images:
            if image_data.shape[0] == 1:  # Single-band image
                image_data = image_data[0]  # Remove the band dimension
            else:  # Multi-band image
                image_data = np.transpose(image_data, (1, 2, 0))  # Rearrange to (height, width, bands)

            # Normalize data if required:
            if image_data.dtype != np.uint8: # If the data is not 8-bit
                image_min = image_data.min()
                image_max = image_data.max()
                image_data = ((image_data - image_min) / (image_max - image_min) * 255).astype('uint8')
        
            metadata = src.meta
            metadata["profile"] = src.profile

        return image_data, metadata

    def saveGeoTIFF(self, tif, output, mask=None, original=True):
        """
        Createas a copy of the input GeoTIFF and saves it.
        If the mask is going to be saved, the mask is given as input, and original = False.
        The function includes '_original' or '_mask' in the end of the file name depending on situation.

        Arguments:
            tif (string): File path to the GeoTIFF
            output (string): File path to the output folder
            mask (ndarray): The rgb mask saved in a ndarray, default None
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

        Arguments:
            tif (string): File path to the GeoTIFF
            output (string): File path to the output folder
            mask (ndarray): The rgb mask saved in a ndarray, default None
            original (bool): A boolean telling what to be saved, default True
        """
        if original:
            image_data, _ = self.readGeoTIFF(tif)
            filename = os.path.join(output, os.path.basename(tif).replace('.tif', '_original.jpg'))
        else:
            image_data = mask
            filename = os.path.join(output, os.path.basename(tif).replace('.tif', '_mask.jpg'))
        if image_data.shape[-1] == 4:
            image_data = image_data[:, :, :3]
        original = Image.fromarray(image_data)
        original.save(filename, format='JPEG')

    def createMask(self, tif):
        """
        Creates the mask of the original GeoTIFF.

        Arguments:
            tif (string): File path to the GeoTIFF
        
        Returns:
            rgb_mask (ndarray): The mask of the GeoTIFF based upon geopackage layers
        """

        _, metadata = self.readGeoTIFF(tif)
        out_shape = (metadata["height"], metadata["width"])
        transform = metadata["transform"]

        # Create RGB mask:
        rgb_mask = np.zeros((out_shape[0], out_shape[1], 3), dtype=np.uint8)

        # Rasterize geometries for segmented area (white)
        layer = list(self.geopackages.keys())
        
        shapes = [geom for geom in self.geopackages[layer[0]].geometry if geom.is_valid]

        holes = [
            Polygon(interior)
            for poly in shapes
            for single_poly in (poly.geoms if poly.geom_type == "MultiPolygon" else [poly])
            for interior in single_poly.interiors
        ]

        shapes_inside_holes = [shape for shape in shapes if any(shape.within(hole) for hole in holes)]

        # Mask for geometries:
        main_mask = rasterize(
            [(shape, 1) for shape in shapes],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True
        )

        # Mask for holes:
        if holes:
            hole_mask = rasterize(
                [(hole, 1) for hole in holes],
                out_shape=out_shape,
                transform=transform,
                fill=0,
                dtype=np.uint8,
                all_touched=True
            )
            # Removes holes:
            if hole_mask.any():
                main_mask[hole_mask > 0] = 0
        
        # Add shapes inside holes:
        if shapes_inside_holes:
            nested_shapes_mask = rasterize(
                [(shape, 1) for shape in shapes_inside_holes],
                out_shape=out_shape,
                transform=transform,
                fill=0,
                dtype=np.uint8,
                all_touched=True
            )

            final_mask = main_mask | nested_shapes_mask
        else:
            final_mask = main_mask

        # Creates white mask for shapes:
        rgb_mask[:, :, 0] = final_mask * 255
        rgb_mask[:, :, 1] = final_mask * 255
        rgb_mask[:, :, 2] = final_mask * 255

        return rgb_mask

    def createMaskGeoTIFF(self, tif, output):
        """
        Creates the mask of the GeoTIFF and saves it as a new GeoTIFF.

        Arguments:
            tif (string): File path to the GeoTIFF
            output (string): File path to the output folder
        """
        mask = self.createMask(tif)
        self.saveGeoTIFF(tif, output, mask, False)

    def createMaskJPEG(self, tif, output):
        """
        Creates the mask of the GeoTIFF and saves it as a new JPEG.

        Arguments:
            tif (string): File path to the GeoTIFF
            output (string): File path to the output folder
        """
        mask = self.createMask(tif)
        self.saveGeoTIFFasJPEG(tif, output, mask, False)
    
    def createGeoTIFF(self, output, profile, image_array):
        """
        Creates a new GeoTIFF based on a given array.

        Arguments:
            output (string): Path to the output file
            profile (dict): A dictionary containing metadata and configuration for the GeoTIFF file
            image_array (ndarray): An array with the image data
        """
        with rasterio.open(output, 'w', **profile) as dst:
            for band in range(3):
                dst.write(image_array[:, :, band], band + 1)

    def mergeGeoTIFFs(self, paths, output_path):
        """
        Merges all the GeoTIFFs into one bigger one.

        Arguments:
            paths (list[string]): A list containing the paths to all the GeoTIFFs to be merged
            output_path (string): The path to save the model
        """
        src_files = [rasterio.open(path) for path in paths]
        print(f"Number of files: {len(src_files)}.")
        # Merge raster
        mosaic, transform = merge(src_files)
        print("Files merged.")
        # Get the metadata of the first GeoTIFF
        meta = src_files[0].meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform
        })
        with rasterio.open(output_path, "w", **meta) as dest:
            dest.write(mosaic)
        for src in src_files:
            src.close()

    def generate_comparison_GeoTIFF(self, segmentation_path, mask_path, output_path):
        """
        Function that generates a new GeoTIFF with colors representing the precision of the segmentation and returns the number of true / false positives / negatives.

        Arguments:
            segmentation_path (string): Path to the segmented GeoTIFF
            mask_path (string): Path to the correct mask (solution)
            output_path (string): Path to save the new GeoTIFF

        Returns:
            tp, tn, fp, fn (int): Number of True Positives, True Negatives, False Positives, False Negatives
        """
        with rasterio.open(segmentation_path) as seg, rasterio.open(mask_path) as mask:
            seg_data = (seg.read(1) >= 140).astype(np.uint8)
            mask_data = (mask.read(1) > 0).astype(np.uint8)
            colors = { # [R,G,B,A]
                (1, 1): [  0, 255, 0, 255], # Green (True Positives)
                (1, 0): [255,   0, 0, 255], # Red (False Positives)
                (0, 1): [255, 165, 0, 255], # Orange (False Negatives)
                (0, 0): [  0,   0, 0, 255]  # Black (True Negatives)
            }
            tp_mask = (seg_data == 1) & (mask_data == 1)
            fp_mask = (seg_data == 1) & (mask_data == 0)
            fn_mask = (seg_data == 0) & (mask_data == 1)
            tn_mask = (seg_data == 0) & (mask_data == 0)

            # Fetches statistical values:
            tp = np.count_nonzero(tp_mask)
            fp = np.count_nonzero(fp_mask)
            fn = np.count_nonzero(fn_mask)
            tn = np.count_nonzero(tn_mask)

            output_image = np.zeros((4, seg.height, seg.width), dtype=np.uint8)
            for (s, m), color in colors.items():
                mask_indices = (seg_data == s) & (mask_data == m)
                for i in range(4):
                    output_image[i][mask_indices] = color[i]
            profile = seg.profile
            profile.update(count=4, dtype=np.uint8)
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(output_image)
        
        return tp, tn, fp, fn
