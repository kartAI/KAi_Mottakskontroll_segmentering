# FarSeg/fineTuneHyperparameters.py

# Libraries:

import glob
from sklearn.model_selection import ParameterGrid
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Functionality.farSegModel import initialize_model
import Functionality.generalFunctions as gf
from Functionality.preProcessing import MapSegmentationDataset, preProcessor
from Functionality.train import train, validate
from Functionality.validation import tileValidation

# Functions:

def grid_search_training(hyperparameter_grid, geotiffs, geopackages, tile_folder, log_file):
    """
    Performs grid search for hyperparameter optimization.

    Args:
        hyperparameter_grid (dict): Dictionary containinghyperparameters to test
        geotiffs (string): Path to GeoTIFFs for training
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

    # Load data and processes:
    tileValidator = tileValidation(geopackages)
    geopackages = gf.load_geopackages(geopackages)
    pre_processing = preProcessor(0.7, tile_folder)

    # Iterate over all the GeoTIFFs to create tiles:
    count = 1
    for tif in glob.glob(geotiffs + '/*.tif'):
        pre_processing.generate_tiles(tif, remove=False, count=count)
        count += 1
    geotiffs = tileValidator.validate(tile_folder)
    
    gf.log_info(
        log_file,
        f"""
################
Hyper parameters\n################\n
Number of tiles: {len(geotiffs)}
     
        """
    )

    train_files, val_files = pre_processing.split_data(liste=geotiffs)

    # Iterate through each combination of hyperparameters:
    for params in tqdm(grid, desc="Grid Search"):
        gf.log_info(log_file, f"Testing hyperparameters: {params}")
        # Initialize model, optimizer and loss function:
        model, criterion, optimizer = initialize_model(num_classes=3, lr=params['lr'])
        # Set up DataSets and DataLoaders:
        train_dataset = MapSegmentationDataset(train_files, geopackages)
        val_dataset = MapSegmentationDataset(val_files, geopackages)
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
        'lr': [1e-3],
        'batch_size': [16],
        'epochs': [1],
        'patience': [3],
        'min_improvement': [0.1]
    }

    # Defines paths to the data:
    geopackages = 'C:/Jakob_Marianne_2024_2025/Geopackage_Farsund/Flater'
    train_files = 'C:/Jakob_Marianne_2024_2025/Ortofoto/Training_3'
    tile_folder = 'C:/Users/jshjelse/Documents/Tiles'
    log_file = 'C:/Users/jshjelse/Documents/results.log'

    gf.emptyFolder(tile_folder)

    # Finds the best hyperparameters:
    best_hyperparameters = grid_search_training(
        hyperparameter_grid, train_files, geopackages, tile_folder, log_file
    )

    gf.log_info(log_file, f"Best hyperparameters found: {best_hyperparameters}")
