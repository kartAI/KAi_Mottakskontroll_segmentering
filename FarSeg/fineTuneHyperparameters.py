# FarSeg/fineTuneHyperparameters.py

# Libraries:

import glob
import os
from sklearn.model_selection import ParameterGrid
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Functionality.farSegModel import initialize_model
import Functionality.generalFunctions as gf
from Functionality.prepareData import fetchCategorizedTiles
from Functionality.preProcessing import MapSegmentationDataset, preProcessor, geotiff_to_geopackage
from Functionality.train import train

# Functions:

def grid_search_training(hyperparameter_grid, train_files, val_files, geopackages, tile_folder, log_file):
    """
    Performs grid search for hyperparameter optimization.

    Arguments:
        hyperparameter_grid (dict): Dictionary containinghyperparameters to test
        train_files (string): Path to the folder with GeoTIFFs for training
        val_files (string): Path to the folder with GeoTIFFs for validation
        geopackages (string): Path to folder with geopackages
        tile_folder (string): Path to the folder where tiles are stored
        log_file (string): Path to the log_file
    
    Returns:
        dict: Best hyperparameters and their corresponding validation loss
    """
    # Define grid of hyperparameters to test:
    grid = ParameterGrid(hyperparameter_grid)
    best_params = None
    best_val_loss = float('inf')

    # Loads the GeoPackages:
    geodata_gpkg = [f for f in os.listdir(geopackages) if f.endswith('.gpkg')]
    geodata_tif = [f for f in os.listdir(geopackages) if f.endswith('.tif') and f.replace('.tif', '.gpkg') not in geodata_gpkg]
    # If some of the training data is stored as GeoTIFF format:
    if len(geodata_tif) > 0:
        for file in geodata_tif:
            file = os.path.join(geopackages, file)
            geotiff_to_geopackage(
                file,
                file.replace(".tif", ".gpkg"),
                file.split('.')[0].split('/')[-1],
                log_file
            )
    geodata = gf.load_geopackages(geopackages) # {"Object type": [...]}
    
    # Iterate over all the GeoTIFFs to create tiles:
    pre_processing = preProcessor(tile_folder)
    count = 1
    training_files = []
    for tif in glob.glob(train_files + '/*.tif'):
        name = os.path.basename(tif).split('.')[0]
        if '_' in name:
            name = name.split('_')[0]
        gpkg = [file for file in geodata_gpkg if file.split('.')[0] == name][0]
        training_files.extend(fetchCategorizedTiles(
            os.path.join(geopackages, gpkg),
            tif,
            f"{tile_folder}/mask_{count}.tif",
            tile_folder,
            count,
            "buildings"
        ))
        count += 1
    gf.emptyFolder(tile_folder + '_val')
    pre_processing = preProcessor(tile_folder + '_val')
    count = 1
    for tif in glob.glob(val_files + '/*.tif'):
        pre_processing.generate_tiles_no_overlap(tif, remove=False, count=count)
        count += 1
    val_files = glob.glob(tile_folder + '_val/*.tif')
    
    gf.log_info(
        log_file,
        f"""
################
Hyper parameters\n################\n
Number of tiles for training: {len(training_files)}
Number of tiles for validation: {len(val_files)}
     
        """
    )

    # Iterate through each combination of hyperparameters:
    for params in tqdm(grid, desc="Grid Search"):
        gf.log_info(log_file, f"Testing hyperparameters: {params}")
        # Initialize model, optimizer and loss function:
        model, criterion, optimizer = initialize_model(num_classes=len(geodata)+1, lr=params['lr'])
        # Set up DataSets and DataLoaders:
        train_dataset = MapSegmentationDataset(training_files, geodata)
        val_dataset = MapSegmentationDataset(val_files, geodata)
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        # Train the model with current hyperparameters:
        val_loss = train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            params['epochs'],
            patience=params['patience'],
            min_delta=params['min_improvement'],
            output=True
        )
        gf.log_info(log_file, f"Validation loss for params {params}: {val_loss}\n")
        if val_loss:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params
        # Clear CUDA cache after each run:
        torch.cuda.empty_cache()
    gf.log_info(log_file, f"Best hyperparameters: {best_params} with validation loss: {best_val_loss}")
    return best_params

# Program

if __name__ == '__main__':
    # Defines the hyperparameter grid:
    hyperparameter_grid = {
        'lr': [1e-4], # 'lr': [1e-3, 1e-4, 1e-5],
        'batch_size': [2], # 'batch_size': [2, 4],
        'epochs': [30], # 'epochs': [10, 20, 30],
        'patience': [5], # 'patience': [3, 5],
        'min_improvement': [0.001] # 'min_improvement': [0.1, 0.01, 0.001]
    }

    # Defines paths to the data:
    geopackages = 'C:/Jakob_Marianne_2024_2025/Combined/GeoPackages'
    train_files = 'C:/Jakob_Marianne_2024_2025/Combined/GeoTIFFs'
    val_files = 'C:/Users/jshjelse/Documents/Validation_tiles'
    tile_folder = 'C:/Users/jshjelse/Documents/Tiles'
    log_file = 'C:/Users/jshjelse/Documents/fineTuning.log'

    gf.emptyFolder(tile_folder)
    gf.resetFile(log_file)

    # Finds the best hyperparameters:
    best_hyperparameters = grid_search_training(
        hyperparameter_grid, train_files, val_files, geopackages, tile_folder, log_file
    )

    gf.log_info(log_file, f"Best hyperparameters found: {best_hyperparameters}")
