# Training/main_train.py

if __name__ == '__main__':

    # Imports libraries:

    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from Data.pre_processing import MapSegmentationDataset, load_geopackages, split_data_list, generate_tiles
    from Data.validateTile import tileValidation
    from Model.farseg_model import initialize_model
    from Training.train_model import train
    
    from torch.utils.data import DataLoader
    import torch
    import glob
    from tqdm import tqdm

    # Program:

    # Paths to data
    # Folder with geopackage files: buildings and roads
    geopackage_folder = 'C:/Users/jshjelse/Documents/Prosjektoppgave/Geopackage/Ver2'
    # Folder with hundreds of different GeoTIFFs
    geotiff_folder = 'C:/images_mj' # 'C:/Users/jshjelse/Documents/Prosjektoppgave/GeoTIFF_Train'
    
    # Where the different tiles will be saved
    tile_folder = './FarSeg/Training/Tiles'
    # Create the tile folder if it doesn't exist
    os.makedirs(tile_folder, exist_ok=True)

    # Load the data
    geopackages = load_geopackages(geopackage_folder) # [Buildings, Roads]
    print("Geopackage loaded")

    tileController = tileValidation(glob.glob(geopackage_folder + '/*.gpkg'))

    # Initialize model, loss function, and optimizer
    num_classes = 3
    model, criterion, optimizer = initialize_model(num_classes, lr=1e-4)
    print("Model initialized")

    # All tif files in the training folder
    tif_files = glob.glob(geotiff_folder + '/*.tif')

    if len(tif_files) > 100:
        print(f"Number of GeoTIFF files {len(tif_files)}.")
        print("Takes the first 9/10 only.")
        tif_files = tif_files[:int(0.9 * len(tif_files))]
        print(f"Number of GeoTIFF files {len(tif_files)}.")

    # Loop through each GeoTIFF file
    for tif in tqdm(tif_files, 'TIFF files'):
        
        # Step 1: Generate tiles for the current GeoTIFF
        generate_tiles(tif, tile_folder)

        valid_tiles = tileController.validate(tile_folder)

        if len(valid_tiles) == 0:
            continue
        
        # Step 2: Split tiles into training and validation sets
        train_files, val_files = split_data_list(valid_tiles)
        print(f"""GeoTIFF {tif} split into training and validation tiles.
              Valid tiles: {len(valid_tiles)}
                Train tiles: {len(train_files)}
                Validation tiles: {len(val_files)}""")

        if train_files == None or val_files == None:
            continue
        if len(train_files) == 0 or len(val_files) == 0:
            continue

        batches = 5
        epochs = 100
        num_workers = 8
        
        # Step 3: Prepare datasets and dataloaders for current tiles
        train_dataset = MapSegmentationDataset(train_files, geopackages)
        val_dataset = MapSegmentationDataset(val_files, geopackages)
        train_loader = DataLoader(train_dataset, batch_size=batches, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batches, shuffle=False, num_workers=num_workers, pin_memory=True)

        # Step 4: Train the model on this batch of tiles
        train(model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs)
        print(f"Model trained on {tif}")

        # Step 5: Clear tiles in the folder to prepare for next GeoTIFF
        for tile_file in glob.glob(tile_folder + '/*'):
            os.remove(tile_file)
        
        torch.cuda.empty_cache()

    # Save the trained model after training
    model_path = './Model/'
    os.makedirs(model_path, exist_ok=True)    
    model_name = './Model/trained_farseg_model_ByggVei_3.pth'
    torch.save(model.state_dict(), model_name)
    print("Model saved")
