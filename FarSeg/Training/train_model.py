# Training/train_model.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from tqdm import tqdm
from Model.farseg_model import EarlyStopping

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """Training loop for FarSeg."""

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using {device} device")
    model = model.to(device)
    criterion = criterion.to(device)

    # Extra
    scaler = torch.amp.GradScaler()

    early_stopping = EarlyStopping(patience=5, min_delta=0.01)
    
    for epoch in tqdm(range(num_epochs), desc='Epochs'):

        if early_stopping.early_stop:
            print("Early stop triggered!")
            break

        print("Started " + str(epoch + 1) + "/" + str(num_epochs))
        
        epoch_loss = 0
        model.train()
        
        for batch_idx, (images, masks) in enumerate(train_loader, 1):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            # Ensure masks are 3D (batch_size, height, width)
            if masks.dim() == 4:
                masks = masks.squeeze(1)  # Remove the channel dimension if it exists
            
            # Convert masks to LongTensor for the loss function
            masks = masks.long()

            # Extra
            with torch.amp.autocast(device_type='cuda:0', dtype=torch.float16):
                # Forward pass
                outputs = model(images)
            
                # outputs shape: (batch_size, num_classes, height, width)
                # masks shape: (batch_size, height, width), with class indices (0: background, 1: roads, etc.)

                # Calculate loss
                loss = criterion(outputs, masks)
            
            # Extra
            # Scale the loss
            scaled_loss = scaler.scale(loss)
            # Backward pass with scaled loss
            scaled_loss.backward()
            scaler.step(optimizer)
            scaler.update()
            #loss.backward()
            #optimizer.step()
            
            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}")
            
            # Extra
            # Clear CUDA cache to prevent memory buildup
            torch.cuda.empty_cache()
        
        # Print average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}')
        
        # Validate at the end of each epoch (or at desired frequency)
        validate(model, val_loader, criterion, device, early_stopping)

        # Extra
        # Clear CUDA cache after validation as well
        torch.cuda.empty_cache()

def validate(model, val_loader, criterion, device, early_stopp):
    """Validation loop for FarSeg."""
    
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            # Ensure masks are 3D (batch_size, height, width)
            if masks.dim() == 4:
                masks = masks.squeeze(1)  # Remove the channel dimension if it exists

            # Convert masks to LongTensor for the loss function
            masks = masks.long()

            # Extra
            with torch.amp.autocast(device_type='cuda'):
                # Forward pass
                outputs = model(images)
            
                # Calculate loss
                loss = criterion(outputs, masks)
            val_loss += loss.item()

            # Clear CUDA cache
            # torch.cuda.empty_cache()
    
    early_stopp(val_loss)

    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
