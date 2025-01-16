# FarSeg_dev/Program/mainTrain.py

# Libraries:

import glob
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Functionality import generalFunctions as gf
from Functionality.preProcessing import MapSegmentationDataset, preProcessor
from Functionality.validation import tileValidation
from Functionality.farSegModel import initialize_model
from Functionality.train import train


# TODO: Needs to have: if __name__ == '__main__': !!!

def mainTrain():
    """
    Performs the main part of training a new FarSeg model.
    """
    # Folder with geopackage data (buildings and roads):
    geopackage_folder = gf.get_valid_input("Where are the geopackages stored(?): ", gf.doesFolderExists)
    # Folder with a lot of GeoTIFFs:
    geotiff_folder = gf.get_valid_input("Where are the GeoTIFFs stored(?): ", gf.doesFolderExists)
    # New folder to save all the tiles to be generated:
    tile_folder = gf.get_valid_input("Where should the tiles be saved(?): ", gf.emptyFolder)
    # Loads the geopackages:
    geopackages = gf.load_geopackages(geopackage_folder) # [Buildings, roads]
    # All GeoTIFF files in the training folder:
    tif_files = glob.glob(geotiff_folder + '/*.tif')
    # Validation element:
    tileContainer = tileValidation(geopackage_folder)
    # Initialize model, loss function and optimizer:
    num_classes = 3
    model, criterion, optimizer = initialize_model(num_classes, lr=1e-4)
    # Initializes the pre-processing element:
    pre_processing = preProcessor(0.7, tile_folder)
    # Values to use in the training:
    batches = gf.get_valid_input("Number of batches: ", gf.positiveNumber)
    epochs = gf.get_valid_input("Number of epochs: ", gf.positiveNumber)
    num_workers = gf.get_valid_input("Number of workers to use: ", gf.positiveNumber)
    # Loops through each GeoTIFF file:
    for tif in tqdm(tif_files, 'GeoTIFF files'):
        # Step 1: Generate tile for the current GeoTIFF
        pre_processing.generate_tiles(geopackage_folder, tif)
        valid_tiles = tileContainer.validate(tile_folder)
        if len(valid_tiles) == 0:
            continue
        # Step 2: Split tiles into training and validation sets
        train_files, val_files = pre_processing.split_data(liste=valid_tiles)
        # print(f"""GeoTIFF {tif} split into training and validation tiles.\nValid tiles: {len(valid_tiles)}\nTrain tiles: {len(train_files)}\nValidation tiles: {len(val_files)}""")
        if train_files == None or val_files == None:
            continue
        if len(train_files) == 0 or len(val_files) == 0:
            continue
        # Step 3: Prepare datasets and dataloaders for current tiles
        train_dataset = MapSegmentationDataset(train_files, geopackages)
        val_dataset = MapSegmentationDataset(val_files, geopackages)
        train_loader = DataLoader(train_dataset, batch_size=batches, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batches, shuffle=False, num_workers=num_workers, pin_memory=True)
        # Step 4: Train the modelon this batch of tiles
        train(model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs)
        # Step 5: Clear tiles in the folder to prepare for next GeoTIFF
        gf.emptyFolder(tile_folder)
        torch.cuda.empty_cache()
    # Save the model after training:
    model_path = gf.get_valid_input("Where will you save the model(?): ", gf.emptyFolder)
    model_name = input("Give the model a name (ends with '.pth'): ")
    torch.save(model.state_dict(), os.path.join(model_path, model_name))
