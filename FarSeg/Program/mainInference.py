# FarSeg/Program/mainInference.py

# Libraries:

import glob
import numpy as np
import os
import shutil
import sys
import torch
import torchvision.transforms as T
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Functionality import generalFunctions as gf
from Functionality.farSegModel import initialize_model
from Functionality.geoTIFFandJPEG import imageSaver
from Functionality.postProcessing import postProcessor, remove_noise
from Functionality.preProcessing import preProcessor, geotiff_to_geopackage

# Function:

def mainInference():
    """
    Performs the main part of inference with a trained FarSeg model.
    """
    # Paths to the data:
    print()
    log_file = gf.get_valid_input("Where will you log the process (.log file): ", gf.resetFile)
    model_path = gf.get_valid_input("Path to your trained model: ", gf.doesPathExists)
    geodata_folder = gf.get_valid_input("Where are the geographic data stored (the solution)(?): ", gf.doesPathExists)
    geotiff_folder = gf.get_valid_input("Path to your folder with orthophotos to be analyzed: ", gf.doesPathExists)
    tile_folder = gf.get_valid_input("Where would you like to store temporarly tiles(?): ", gf.emptyFolder)
    segmented_folder = gf.get_valid_input("Where would you like to store the segmented tiles(?): ", gf.emptyFolder)
    output_folder = gf.get_valid_input("Where would you like to store the final results(?): ", gf.emptyFolder)
    # Loads the GeoPackages:
    geodata_gpkg = [f for f in os.listdir(geodata_folder) if f.endswith('.gpkg')]
    geodata_tif = [f for f in os.listdir(geodata_folder) if f.endswith('.tif') and f.replace('.tif', '.gpkg') not in geodata_gpkg]
    # If some of the training data is stored as GeoTIFF format:
    if len(geodata_tif) > 0:
        for file in tqdm(geodata_tif, desc="GeoTIFFs to GeoPackage", colour="yellow"):
            file = os.path.join(geodata_folder, file)
            geotiff_to_geopackage(
                file,
                file.replace('.tif', '.gpkg'),
                file.split('.')[0].split('/')[-1],
                log_file
            )
    # If you want jpg or not:
    choice = gf.get_valid_input("Do you want to save the results as .jpg files as well(?)(y/n): ", lambda x: gf.yesNo(x) is not None)
    choice = gf.yesNo(choice)
    print()

    gf.log_info(
        log_file,
        f"""
####################
Segmentation started\n####################\n
Input data:\n
Model: {model_path}
GeoTIFF folder: {geotiff_folder}
Tile folder: {tile_folder}
Segmented folder: {segmented_folder}
Result folder: {output_folder}
Saves as jpg as well: {choice}
        """
    )

    # Fetches GPU or CPU device:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gf.log_info(log_file, f"\nDevice: {device}\n")
    geodata = gf.load_geopackages(geodata_folder, small_objects=False) # {"Building": [...], "Roads": [...], ...}
    # Load the trained model:
    num_classes = len(geodata) + 1
    model, _, _ = initialize_model(num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True)) # Loads the model from the saved file
    model = model.to(device)
    # Color map for background and segmented:
    color_map = {
        0: [  0,   0,   0], # Background - Black
        1: [255, 255, 255]  # Segmented  - White
    }
    # Converts the color map dictionary to a NumPy array:
    colors = np.array([color_map[i] for i in range(num_classes)], dtype=np.uint8) # Returns RGB colors
    # Runs inference on new GeoTIFFs:
    geotiff_paths = glob.glob(geotiff_folder + '/*.tif')
    gf.log_info(log_file, f"Number of geotiffs: {len(geotiff_paths)}\n")
    tileGenerator = preProcessor(tile_folder)
    imageCombiner = postProcessor(tile_folder, segmented_folder)
    imageHandler = imageSaver()
    for k, path in enumerate(tqdm(geotiff_paths, desc="GeoTIFFs segmented", colour="yellow")):
        # Step 0: Fetch original image size
        _, info = imageHandler.readGeoTIFF(path)
        original_size = (info["height"], info["width"])
        image_crs = info["crs"]
        # Step 1: Generate tiles from the input GeoTIFF
        tileGenerator.generate_tiles_no_overlap(path)
        splitted_geotiffs = [os.path.join(tile_folder, f) for f in os.listdir(tile_folder) if f.endswith('.tif')]
        # Step 2: Iterate over all the tiles
        for _, geotiff in enumerate(tqdm(splitted_geotiffs, desc=f"Tiles processed for GeoTIFF {k+1}", colour="green", leave=False)):
            image_data, metadata = imageHandler.readGeoTIFF(geotiff)
            # Step 3: Adjust the image
            # Check if the image has 3 bands (for RGB). If not, adapt the normalization dynamically:
            if image_data.shape[2] == 3: # Assuming RGB
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
            else:
                channels = image_data.shape[2]
                mean = [0.5] * channels
                std = [0.5] * channels
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
            # Step 4: Create image tensor
            # Apply the transformation (ToTensor converts image to shape: (bands, height, width)):
            image_tensor = transform(image_data).unsqueeze(0).to(device) # Add batch dimension for model input
            # Step 5: Performe inference
            with torch.no_grad():
                    output = model(image_tensor) # Forward pass through the model
                    predicted_segmented_mask = output.squeeze(0).argmax(0).cpu().numpy() # Shape: (height, width)
            # Step 6: Create an RGB image from the segmentation classes
            segmented_image_rgb = predicted_segmented_mask
            segmented_image_rgb = colors[predicted_segmented_mask]
            metadata["profile"].update({
                'count': 3, # 3 channels for RGB
                'dtype': 'uint8',
                'photometric': 'RGB',
                'crs': image_crs
            })
            # Crop output if it is the end of the image
            x, y = int(geotiff.split('_')[-1].split('.')[0]) == ((original_size[1] + 1023) // 1024) - 1, int(geotiff.split('_')[-2]) == ((original_size[0] + 1023) // 1024) - 1
            if x or y:
                x_start, x_end = 1024 - (original_size[1] - int(geotiff.split('_')[-1].split('.')[0]) * 1024) if x else 0, metadata["width"]
                y_start, y_end = 1024 - (original_size[0] - int(geotiff.split('_')[-2]) * 1024) if y else 0, metadata["height"]
                segmented_image_rgb = segmented_image_rgb[y_start:y_end, x_start:x_end]
                metadata["profile"].update({
                    "width": x_end - x_start,
                    "height": y_end - y_start
                })
            # Step 7: Save as GeoTIFF
            output_filename = os.path.splitext(os.path.basename(geotiff))[0] + '_segmented.tif'
            geotiff_output_filepath = os.path.join(segmented_folder, output_filename)
            imageHandler.createGeoTIFF(geotiff_output_filepath, metadata["profile"], segmented_image_rgb)
        # Step 8: Merge all tiles into a final combined image
        output_original = os.path.join(output_folder, f"merged_original_tif_{k+1}.tif")
        output_segmented = os.path.join(output_folder, f"merged_segmented_tif_{k+1}.tif")
        imageCombiner.merge_images(output_original, output_segmented, original_size, choice)
    
        remove_noise(output_segmented)

        gf.log_info(
            log_file,
            f"""
GeoTIFF #{k+1}:
Number of tiles: {len(splitted_geotiffs)}
Segmented results saved as:
{output_original}
{output_segmented}\n
            """
        )

        # Step 9: Prepare for next GeoTIFF by removing all the generated tiles
        gf.emptyFolder(tile_folder)
        gf.emptyFolder(segmented_folder)
    if os.path.exists(tile_folder):
        shutil.rmtree(tile_folder)
    if os.path.exists(segmented_folder):
        shutil.rmtree(segmented_folder)
