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
from Functionality.preProcessing import MapSegmentationDataset, preProcessor, geotiff_to_geopackage
from Functionality.validation import tileValidation
from Functionality.farSegModel import initialize_model, geotiffStopping
from Functionality.train import train

# Function:

def mainTrain():
    """
    Performs the main part of training a new FarSeg model.
    """
    torch.cuda.empty_cache()
    print()
    log_file = gf.get_valid_input("Where will you log the process (.log file): ", gf.resetFile)
    # Folder with geographic data (buildings and roads):
    geodata_folder = gf.get_valid_input("Where are the geographic data stored (the solution)(?): ", gf.doesPathExists)
    # Folder with a lot of GeoTIFFs:
    geotiff_folder = gf.get_valid_input("Where are the GeoTIFFs stored(?): ", gf.doesPathExists)
    # New folder to save all the tiles to be generated:
    tile_folder = gf.get_valid_input("Where should the tiles be saved(?): ", gf.emptyFolder)
    # Loads the GeoPackages:
    geodata_gpkg = [f for f in os.listdir(geodata_folder) if f.endswith('.gpkg')]
    geodata_tif = [f for f in os.listdir(geodata_folder) if f.endswith('.tif') and f.replace('.tif', '.gpkg') not in geodata_gpkg]
    # If some of the training data is stored as GeoTIFF format:
    if len(geodata_tif) > 0:
        for file in geodata_tif:
            file = os.path.join(geodata_folder, file)
            geotiff_to_geopackage(
                file,
                file.replace(".tif", ".gpkg"),
                file.split('.')[0].split('/')[-1],
                log_file
            )
    geopackages = gf.load_geopackages(geodata_folder) # {"Buildings": [...], "Roads": [...], ...}
    # All GeoTIFF files in the training folder:
    tif_files = glob.glob(geotiff_folder + '/*.tif')
    # Hyper-parameters to use in the training:
    batches = int(gf.get_valid_input("Number of batches (default: 2): ", gf.positiveNumber, default=2))
    epochs = int(gf.get_valid_input("Number of epochs (default: 30): ", gf.positiveNumber, default=30))
    num_workers = int(gf.get_valid_input("Number of workers to use (default: 8): ", gf.positiveNumber, default=8))
    learning_rate = float(gf.get_valid_input("Float number to use as learning rate (default: 0.0001): ", gf.positiveNumber, default=0.0001))
    num_classes = len(geopackages) + 1 # + 1 because of the background = nothing
    patience = int(gf.get_valid_input("Number of epochs to wait as patience (default: 5): ", gf.positiveNumber, default=5))
    min_improvement = float(gf.get_valid_input("Float number to use as minimum improvement (default: 0.001): ", gf.positiveNumber, default=0.001))
    val_split = 0.7
    # Validation element:
    tileContainer = tileValidation(geodata_folder)
    # Initialize model, loss function and optimizer:
    model, criterion, optimizer = initialize_model(num_classes, lr=learning_rate)
    # Initializes the pre-processing element:
    preProcessing = preProcessor(val_split, tile_folder)
    geotiffCounter = geotiffStopping(patience, min_improvement)
    # Give a name for the trained model:
    model_path = gf.get_valid_input("Where will you save the model(?): ", gf.emptyFolder)
    model_name = input("Give the model a name (ends with '.pth'): ")
    choice = gf.get_valid_input("Do you want to train the model on valid tiles only(?)(y/n): ", lambda x: gf.yesNo(x) is not None)
    choice = gf.yesNo(choice)
    print()

    info = "\n".join([f"- {key}: {len(value)}" for key, value in geopackages.items()])

    gf.log_info(
        log_file,
        f"""
######################################
Training of a new FarSeg model started\n######################################\n
Input data:\n
Geopackage folder: {geodata_folder}
GeoTIFF folder: {geotiff_folder}
Tile folder: {tile_folder}

Geopackage data:
{info}

Number of GeoTIFFs for training: {len(tif_files)}\n
Hyper parameters:
- Batches: {batches}
- Epochs: {epochs}
- Workers: {num_workers}
- Learning rate: {learning_rate}
- Classes: {num_classes}
- Patience: {patience}
- Minimum improvement: {min_improvement}
- Train-validation-split: {val_split}\n
The trained model will be saved as:
Folder: {model_path}
File: {model_name}
The model will train on valid tiles only: {choice}
"""
    )

    counter = 1

    # Loops through each GeoTIFF file:
    for tif in tqdm(tif_files, 'GeoTIFF files'):
        gf.log_info(log_file, f"\nTraining on GeoTIFF #{counter}\n")
        # Step 1: Generate tile for the current GeoTIFF
        preProcessing.generate_tiles(tif)
        valid_tiles = tileContainer.validate(tile_folder, choice)
        if len(valid_tiles) == 0:
            continue
        # Step 2: Split tiles into training and validation sets
        train_files, val_files = preProcessing.split_data(liste=valid_tiles)
        if train_files == None or val_files == None:
            continue
        if len(train_files) == 0 or len(val_files) == 0:
            continue

        gf.log_info(
            log_file,
            f"""
Training files: {len(train_files)}
Validation files: {len(val_files)}
"""
        )

        # Step 3: Prepare datasets and dataloaders for current tiles
        train_dataset = MapSegmentationDataset(train_files, geopackages)
        val_dataset = MapSegmentationDataset(val_files, geopackages)
        train_loader = DataLoader(train_dataset, batch_size=batches, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batches, shuffle=False, num_workers=num_workers)
        # Step 4: Train the model on this batch of tiles
        loss = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=epochs,
            patience=patience,
            min_delta=min_improvement,
            save_path=os.path.join(model_path, model_name),
            output=True,
            log_file=log_file
        )
        geotiffCounter(loss)
        # Step 5: Clear tiles in the folder to prepare for next GeoTIFF
        del train_dataset, val_dataset, train_loader, val_loader, loss
        gf.emptyFolder(tile_folder)
        torch.cuda.empty_cache()
        torch.save(model.state_dict(), os.path.join(model_path, model_name))
        # Early stop check
        if geotiffCounter.early_stop:
            gf.log_info(
                log_file,
                f"Stopped at GeoTIFF {counter} with loss {geotiffCounter.best_score}"
            )
            break
        counter += 1
    # Removes the tile_folder after training:
    if os.path.exists(tile_folder):
        shutil.rmtree(tile_folder)

    gf.log_info(
        log_file,
        f"Training finished.\nModel saved at {os.path.join(model_path, model_name)}"
    )
