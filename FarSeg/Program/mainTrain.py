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
from Functionality.prepareData import fetchCategorizedTiles
from Functionality.preProcessing import MapSegmentationDataset, preProcessor, geotiff_to_geopackage
from Functionality.farSegModel import initialize_model
from Functionality.train import train

# Function:

def mainTrain():
    """
    Performs the main part of training a new FarSeg model.
    """
    torch.cuda.empty_cache()
    print()
    choice = gf.get_valid_input("Will you upload a trained model(?)(y/n): ", lambda x: gf.yesNo(x) is not None)
    choice = gf.yesNo(choice)
    if choice:
        model_path = gf.get_valid_input("Where are your trained model stored(?): ", gf.doesPathExists)
    print()
    log_file = gf.get_valid_input("Where will you log the process (.log file): ", gf.resetFile)
    # Folder with geographic data (buildings and roads):
    geodata_folder = gf.get_valid_input("Where are the geographic data stored (the solution)(?): ", gf.doesPathExists)
    # Folder with a lot of GeoTIFFs:
    geotiff_folder = gf.get_valid_input("Where are the GeoTIFFs stored(?): ", gf.doesPathExists)
    # New folder to save all the tiles to be generated:
    tile_folder = gf.get_valid_input("Where should the tiles be saved(?): ", gf.emptyFolder)
    # Folder where all the validation files are stored:
    validation_folder = gf.get_valid_input("Where are the folder to the GeoTIFF(s) used for validation stored(?): ", gf.doesPathExists)
    # Define tpe of object to analyze:
    object_type = gf.get_valid_input("What kind of object type are you analyzing(?) (buildings/roads): ", gf.correctObjectType)
    # Loads the GeoPackages:
    geodata_gpkg = [f for f in os.listdir(geodata_folder) if f.endswith('.gpkg')]
    geodata_tif = [f for f in os.listdir(geodata_folder) if f.endswith('.tif') and f.replace('.tif', '.gpkg') not in geodata_gpkg]
    # If some of the training data is stored as GeoTIFF format:
    if len(geodata_tif) > 0:
        for file in tqdm(geodata_tif, desc="GeoTIFFs to GeoPackage", colour="yellow"):
            file = os.path.join(geodata_folder, file)
            geotiff_to_geopackage(
                file,
                file.replace(".tif", ".gpkg"),
                file.split('.')[0].split('/')[-1],
                log_file
            )
    geopackages = gf.load_geopackages(geodata_folder) # {"Object_type": [...]}
    # All GeoTIFF files in the training folder:
    tif_files = glob.glob(geotiff_folder + '/*.tif')
    # Hyper-parameters to use in the training:
    batches = int(gf.get_valid_input("Number of batches (default: 2): ", gf.positiveNumber, default=2))
    epochs = int(gf.get_valid_input("Number of epochs (default: 30): ", gf.positiveNumber, default=30))
    num_workers = int(gf.get_valid_input("Number of workers to use (default: 8): ", gf.positiveNumber, default=8))
    learning_rate = float(gf.get_valid_input("Float number to use as learning rate (default: 0.0001): ", gf.positiveNumber, default=0.0001))
    num_classes = 2 # Object type + Background
    patience = int(gf.get_valid_input("Number of epochs to wait as patience (default: 5): ", gf.positiveNumber, default=5))
    min_improvement = float(gf.get_valid_input("Float number to use as minimum improvement (default: 0.001): ", gf.positiveNumber, default=0.001))
    # Initialize model, loss function and optimizer:
    model, criterion, optimizer = initialize_model(num_classes, lr=learning_rate)
    if choice:
        if model_path:
            model.load_state_dict(torch.load(model_path, weights_only=True))
    # Initializes the pre-processing element:
    gf.emptyFolder(validation_folder + "/Tiles")
    preProcessing = preProcessor(validation_folder + "/Tiles")
    # Give a name for the trained model:
    model_path = gf.get_valid_input("Where will you save the model(?): ", gf.emptyFolder)
    model_name = input("Give the model a name (ends with '.pth'): ")
    print()

    info = "\n".join([f"- {key}: {len(value)}" for key, value in geopackages.items()])

    gf.log_info(
        log_file,
        f"""
######################################
Training of a new FarSeg model for {object_type} started\n######################################\n
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

The trained model will be saved as:
Folder: {model_path}
File: {model_name}
"""
    )

    train_loaders = []
    
    for i, geotiff in enumerate(tif_files): # For every GeoTIFF in the training set
        # Step 1: Generate categorized tiles for the GeoTIFF
        name = os.path.basename(geotiff).split('.')[0] # Fetches the name of the area - corresponding between GeoPackage and GeoTIFF
        if '_' in name:
            name = name.split('_')[0]
        
        gpkg = [file for file in geodata_gpkg if file.split('.')[0] == name][0]
        
        train_files = fetchCategorizedTiles(
            os.path.join(geodata_folder, gpkg),
            geotiff,
            f"{tile_folder}/mask_{i + 1}.tif",
            tile_folder,
            i + 1,
            object_type
        )
        if train_files == None or len(train_files) == 0:
            return
        
        gf.log_info(
            log_file,
f"""
Training files for {name}: {len(train_files)}
"""
            )

        # Step 3: Prepare datasets and dataloaders for training for the current tiles
        train_dataset = MapSegmentationDataset(train_files, {name: geopackages[name]})
        train_loader = DataLoader(train_dataset, batch_size=batches, shuffle=True, num_workers=num_workers)
        train_loaders.append(train_loader)
    # Step 4: Fetch validation data
    for i, geotiff in enumerate(glob.glob(validation_folder + '/*.tif')):
        preProcessing.generate_tiles_no_overlap(geotiff, remove=False, count=i+1)
    val_files = glob.glob(validation_folder + "/Tiles" + "/*.tif")
    if val_files == None or len(val_files) == 0:
            return
    name = os.path.basename(glob.glob(validation_folder + '/*.tif')[0].split('.')[0])
    gf.log_info(
        log_file,
f"""
Validation files: {len(val_files)}
"""
    )
    val_dataset = MapSegmentationDataset(val_files, {name: geopackages[name]})
    val_loader = DataLoader(val_dataset, batch_size=batches, shuffle=False, num_workers=num_workers)
    torch.cuda.empty_cache()
    # Step 5: Train the model on the generated tiles based on the categorization
    loss = train(
        model=model,
        train_loaders=train_loaders,
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
    # Step 6: Clear tiles in the folder to prepare for next GeoTIFF
    del train_dataset, val_dataset, train_loader, val_loader, loss
    torch.cuda.empty_cache()
    # Removes the tile_folder after training:
    if os.path.exists(tile_folder):
        shutil.rmtree(tile_folder)

    gf.log_info(
        log_file,
        f"Training finished.\nModel saved at {os.path.join(model_path, model_name)}"
    )
