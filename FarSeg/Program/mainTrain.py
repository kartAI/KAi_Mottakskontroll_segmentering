# FarSeg/Program/mainTrain.py

# Libraries:

import glob
import os
import shutil
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Functionality import generalFunctions as gf
from Functionality.preProcessing import MapSegmentationDataset, preProcessor
from Functionality.validation import tileValidation
from Functionality.farSegModel import initialize_model
from Functionality.train import train

def mainTrain():
    """
    Performs the main part of training a new FarSeg model.
    """
    # Folder with geopackage data (buildings and roads):
    geopackage_folder = gf.get_valid_input("Where are the geopackages stored(?): ", gf.doesPathExists)
    # Folder with a lot of GeoTIFFs:
    geotiff_folder = gf.get_valid_input("Where are the GeoTIFFs stored(?): ", gf.doesPathExists)
    # New folder to save all the tiles to be generated:
    tile_folder = gf.get_valid_input("Where should the tiles be saved(?): ", gf.emptyFolder)
    # Loads the geopackages:
    geopackages = gf.load_geopackages(geopackage_folder) # [Buildings, roads]
    # All GeoTIFF files in the training folder:
    tif_files = glob.glob(geotiff_folder + '/*.tif')
    # Hyper-parameters to use in the training:
    batches = int(gf.get_valid_input("Number of batches (64): ", gf.positiveNumber))
    epochs = int(gf.get_valid_input("Number of epochs (30): ", gf.positiveNumber))
    num_workers = int(gf.get_valid_input("Number of workers to use (8): ", gf.positiveNumber))
    learning_rate = float(gf.get_valid_input("Float number to use as learning rate (0.001): ", gf.positiveNumber))
    num_classes = 3
    patience = int(gf.get_valid_input("Number of epochs to wait as patience (3): ", gf.positiveNumber))
    min_improvement = float(gf.get_valid_input("Float number to use as minimum improvement (0.01): ", gf.positiveNumber))
    val_split = 0.7
    # Validation element:
    tileContainer = tileValidation(geopackage_folder)
    # Initialize model, loss function and optimizer:
    model, criterion, optimizer = initialize_model(num_classes, lr=learning_rate)
    # Initializes the pre-processing element:
    pre_processing = preProcessor(val_split, tile_folder)
    # Give a name for the trained model:
    model_path = gf.get_valid_input("Where will you save the model(?): ", gf.emptyFolder)
    model_name = input("Give the model a name (ends with '.pth'): ")
    # Loops through each GeoTIFF file:
    for tif in tqdm(tif_files, 'GeoTIFF files'):
        # Step 1: Generate tile for the current GeoTIFF
        pre_processing.generate_tiles(tif)
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
        train(model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs, patience=patience, min_delta=min_improvement)
        # Step 5: Clear tiles in the folder to prepare for next GeoTIFF
        gf.emptyFolder(tile_folder)
        torch.cuda.empty_cache()
    # Removes the tile_folder after training:
    if os.path.exists(tile_folder):
        shutil.rmtree(tile_folder)
    # Save the model after training:
    torch.save(model.state_dict(), os.path.join(model_path, model_name))

def mainTrain2():
    """
    Performs the main part of training a new FarSeg model.
    """
    # Folder with geopackage data (buildings and roads):
    geopackage_folder = gf.get_valid_input("Where are the geopackages stored(?): ", gf.doesPathExists)
    # Folder with a lot of GeoTIFFs:
    geotiff_folder = gf.get_valid_input("Where are the GeoTIFFs stored(?): ", gf.doesPathExists)
    # New folder to save all the tiles to be generated:
    tile_folder = gf.get_valid_input("Where should the tiles be saved(?): ", gf.emptyFolder)
    # Loads the geopackages:
    geopackages = gf.load_geopackages(geopackage_folder) # [Buildings, roads]
    # All GeoTIFF files in the training folder:
    tif_files = glob.glob(geotiff_folder + '/*.tif')
    # Hyper-parameters to use in the training:
    batches = int(gf.get_valid_input("Number of batches (default: 64): ", gf.positiveNumber, default=64))
    epochs = int(gf.get_valid_input("Number of epochs (default: 30): ", gf.positiveNumber, default=30))
    num_workers = int(gf.get_valid_input("Number of workers to use (default: 8): ", gf.positiveNumber, default=8))
    learning_rate = float(gf.get_valid_input("Float number to use as learning rate (default: 0.001): ", gf.positiveNumber, default=0.001))
    num_classes = 3
    patience = int(gf.get_valid_input("Number of epochs to wait as patience (default: 3): ", gf.positiveNumber, default=3))
    min_improvement = float(gf.get_valid_input("Float number to use as minimum improvement (default: 0.01): ", gf.positiveNumber, default=0.01))
    val_split = 0.7
    # Validation element:
    tileContainer = tileValidation(geopackage_folder)
    # Initialize model, loss function and optimizer:
    model, criterion, optimizer = initialize_model(num_classes, lr=learning_rate)
    # Initializes the pre-processing element:
    pre_processing = preProcessor(val_split, tile_folder)
    # Give a name for the trained model:
    model_path = gf.get_valid_input("Where will you save the model(?): ", gf.emptyFolder)
    model_name = input("Give the model a name (ends with '.pth'): ")
    # Step 1: Generate tiles for all GeoTIFFs and fetch the valid ones:
    for count, tif in tqdm(enumerate(tif_files), desc="GeoTIFFs"):
        pre_processing.generate_tiles(tif, remove=False, count=count+1)
    valid_tiles = tileContainer.validate(tile_folder)
    if len(valid_tiles) == 0:
        return
    valid_set = set(valid_tiles)
    for tile in tqdm(glob.glob(tile_folder + "\\*.tif"), desc="Removed GeoTIFFs"):
        if tile not in valid_set:
            os.remove(tile)
    for i in range(4):
        tiles = valid_tiles[int(len(valid_tiles)/4)*i : int(len(valid_tiles)/4)*(i+1)]
        # Step 2: Split tiles into train and validation
        train_files, val_files = pre_processing.split_data(liste=tiles)
        if train_files == None or val_files == None:
            return
        if len(train_files) == 0 or len(val_files) == 0:
            return
        # Step 3: Prepare datasets and dataloaders for current tiles
        train_dataset = MapSegmentationDataset(train_files, geopackages)
        val_dataset = MapSegmentationDataset(val_files, geopackages)
        train_loader = DataLoader(train_dataset, batch_size=batches, shuffle=True, num_workers=num_workers) #, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batches, shuffle=False, num_workers=num_workers) #, pin_memory=True)
        # Step 4: Train the modelon this batch of tiles
        train(model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs, patience=patience, min_delta=min_improvement)
    # Step 5: Removes the tile_folder after training
    if os.path.exists(tile_folder):
        shutil.rmtree(tile_folder)
    # Save the model after training:
    torch.save(model.state_dict(), os.path.join(model_path, model_name))