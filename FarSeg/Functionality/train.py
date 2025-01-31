# FarSeg/Functionality/train.py

# Libraries:

import torch
from torchmetrics import JaccardIndex
from tqdm import tqdm

from farSegModel import EarlyStopping

# Functions:

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, min_delta, save_path=None, output=False):
    """
    Training loop for FarSeg segmentation model.

    Args:
        model (torch.models.FarSeg): Model to be trained
        train_loader (DataLoader): The data used to train the model
        val_loader (DataLoader): The data used to validate the model
        criterion (torch.nn.CrossEntropyLoss): The loss function used for training, suitable for multi-class classification tasks
        optimizer (torch.optim.Adam): The optimizer for updating the model parameters during training, configured with the specified learning rate
        num_epochs (int): Number of epochs to consider
        patience (int): Integer initializing the patience of the earlyStop instance, default 5
        min_delta (float): Float initializing the minimum improvement of the earlyStop instance, default 0.01
        save_path (string): Path to save the best model weights, default None
        output (bool): Wether or not the function should return a value, default False
    
    Returns:
        if output:
            int: Integer describing the best validation score of the model before stopping training
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    scaler = torch.amp.GradScaler()
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, save_path=save_path, model=model)
    for _ in tqdm(range(num_epochs), desc='Epochs'):
        epoch_loss = 0
        model.train()
        for batch_idx, (images, masks) in enumerate(train_loader, 1):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            # Ensures masks are 3D (batch_size, height, width)
            if masks.dim() == 4:
                masks = masks.squeeze(1) # Remove the channel dimension if it exists
            # Convert masks to LongTensor for the loss function
            masks = masks.long()
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(images) # Forward pass
                # output shapes: (batch_size, num_classes, height, width)
                # mask shapes: (batch_size, height, width), with class indices: (0: background, 1: buildings, etc.)
                # Calculates loss:
                loss = criterion(outputs, masks)
            # Scale the loss:
            scaled_loss = scaler.scale(loss)
            # Backward pass with scaled loss:
            scaled_loss.backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            # Clear CUDA cache to prevent memory buildup:
            del images, masks, outputs, loss
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        # Validate at the end of each epoch
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss, avg_iou = validate(model, val_loader, criterion, device)
        early_stopping(avg_val_loss, avg_train_loss)
        torch.cuda.empty_cache()
        if avg_iou > 0.85:
            early_stopping.early_stop = True
        if early_stopping.early_stop:
            break
    if output:
        return early_stopping.best_score

def validate(model, val_loader, criterion, device):
    """
    Validation loop during training of FarSeg segmentation model.

    Args:
        model (torch.models.FarSeg): Model to be trained
        val_loader (DataLoader): The data used to validate the model
        criterion (torch.nn.CrossEntropyLoss): The loss function used for training, suitable for multi-class classification tasks
        device (torch.device): Device (GPU / CPU) that performs the calculations during training and validation
    """
    model.eval()
    val_loss = 0
    iou_metric = JaccardIndex(task="multiclass", num_classes=3).to(device=device)
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            # Ensures masks are 3D (batch_size, height, width):
            if masks.dim() == 4:
                masks = masks.squeeze(1) # Remove the channel dimension if it exists
            # Convert masks to LongTensor for the loss function:
            masks = masks.long()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images) # Forward pass
                # Calculate loss:
                loss = criterion(outputs, masks)
            val_loss += loss.item()
            # IoU-preditions:
            preds = torch.argmax(outputs, dim=1)
            iou_metric.update(preds, masks)
            # Clear CUDA cache to prevent memory buildup:
            del images, masks, outputs, loss
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    avg_val_loss = val_loss / len(val_loader)
    avg_iou = iou_metric.compute()
    iou_metric.reset()
    return avg_val_loss, avg_iou.item()
