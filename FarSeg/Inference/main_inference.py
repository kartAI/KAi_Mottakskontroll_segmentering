# Inference/main_inference.py

# Imports libraries:

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Data.pre_processing import get_random_geotiff, generate_tiles
from Data.post_processing import clear_output_directory, merge_images
from Model.farseg_model import initialize_model

import torch
import torchvision.transforms as T
import rasterio
import numpy as np
from tqdm import tqdm
import glob

# Program:

# Path to model
modelPath = 'C:/Users/jshjelse/Documents/Prosjektoppgave/Model/trained_farseg_model_ByggVei_3.pth'
ortophoto_path = 'C:/images_mj'
tile_folder = './FarSeg/Inference/Tiles/tiles'
segmented_output_dir = './FarSeg/Inference/Tiles/segmented'

# Ensuring that the folders exists:
os.makedirs(tile_folder, exist_ok=True)
os.makedirs(segmented_output_dir, exist_ok=True)

# Prepare the folders for a new job:
clear_output_directory(tile_folder)
clear_output_directory(segmented_output_dir)

# Fetches GPU or CPU device:
device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using {device} device")

# Loads the trained model
num_classes = 3
model, _, _ = initialize_model(num_classes)
model.load_state_dict(torch.load(modelPath, weights_only=True))  # Load the model from the saved file
model = model.to(device)
print("Model successfully loaded")

# Generates a color map for the specified number of classes
color_map = {
            0: [0, 0, 0],        # Background - Black
            1: [255, 0, 0],      # Buildings - Red
            2: [255, 255, 0]     # Roads - Yellow
        }

# Convert the color map dictionary to a NumPy array
colors = np.array([color_map[i] for i in range(num_classes)], dtype=np.uint8) # Return RGB colors

# Define class names for the legend
class_names = ["Background", "Building", "Road"]

# Creates output folder:
output_folder = './FarSeg/Inference/Final result/'
os.makedirs(output_folder, exist_ok=True)

# Runs inference on a new geotiff
geotiff_paths = glob.glob(ortophoto_path + '/*.tif')
geotiff_paths = geotiff_paths[int(0.9 * len(geotiff_paths)):]

for k, path in enumerate(geotiff_paths):
    new_geotiff_path = get_random_geotiff(ortophoto_path)
    print("GeoTIFF fetched at: " + str(new_geotiff_path))

    # Generate tiles from the input GeoTIFF
    generate_tiles(new_geotiff_path, tile_folder)

    splitted_geotiffs = [os.path.join(tile_folder, f) for f in os.listdir(tile_folder) if f.endswith('.tif')]

    for i, geotiff in tqdm(enumerate(splitted_geotiffs), "GeoTIFFs"):
        with rasterio.open(geotiff) as src: 
            # Read image bands as a numpy array
            image = src.read().astype(np.float32)  # Shape: (bands, height, width)
            profile = src.profile

        # The input is a NumPy array of shape (bands, height, width).
        # Convert it to the shape expected by torchvision (height, width, bands).
        image = np.moveaxis(image, 0, -1)

        # Check if the image has 3 bands (for RGB). If not, adapt the normalization dynamically.
        if image.shape[2] == 3:  # Assuming RGB
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Use mean and std = 0.5 for each channel as a default for other cases
            channels = image.shape[2]
            mean = [0.5] * channels
            std = [0.5] * channels
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])

        # Apply the transform (ToTensor converts image to shape: (bands, height, width))
        image_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension for model input
        
        with torch.no_grad():
            output = model(image_tensor) # Forward pass through the model
            predicted_segmented_mask = output.squeeze(0).argmax(0).cpu().numpy() # Shape: (height, width)

        # Create an RGB image from the segmentation classes
        segmented_image_rgb = colors[predicted_segmented_mask]

        # Update the profile to save as RGB GeoTIFF
        profile.update({
            'count': 3, # 3 channels for RGB
            'dtype': 'uint8',
            'photometric': 'RGB'
        })

        # Save as GeoTIFF
        output_filename = os.path.splitext(os.path.basename(geotiff))[0] + '_segmented.tif'
        geotiff_output_filepath = os.path.join(segmented_output_dir, output_filename)
        with rasterio.open(
            geotiff_output_filepath,
            'w',
            **profile
        ) as dst:
            dst.write(segmented_image_rgb[:, :, 0], 1) # Red channel
            dst.write(segmented_image_rgb[:, :, 1], 2) # Green channel
            dst.write(segmented_image_rgb[:, :, 2], 3) # Blue channel

    # Merge all tiles into a final combined image
    output_original = f'C:/Users/jshjelse/Documents/Prosjektoppgave/FarSeg/inference/Final result/merged_original_tif_{k}.tif'
    output_segmented = f'C:/Users/jshjelse/Documents/Prosjektoppgave/FarSeg/inference/Final result/merged_segmented_tif_{k}.tif'

    merge_images(tile_folder, segmented_output_dir, output_original, output_segmented)

    # Clean up the output directories
    clear_output_directory(tile_folder)
    clear_output_directory(segmented_output_dir)
