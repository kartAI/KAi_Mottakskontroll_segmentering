# FarSeg/Functionality/farSegModel.py

# Libraries:

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchgeo.models import FarSeg

# Classes:

class EarlyStopping():
    """
    Instance to consider good learning rate of the model - stops the training loop when teh model do not learn more

    Attributes:
        patience (int): Number of epochs to wait before stopping early
        min_delta (float): Minimum improvement in the monitored metric required to continue
        save_path (string): Path to save the best model weights
        model (FarSeg model): The model to save the best state_dict from
        monitor_state_dict (bool): If True, stops if train_loss decreases but val_loss increases
        counter (int): Number of epochs waited
        best_score (float): The best achieved score by the model so far, starts like None
        early_stop (bool): Wether or not to stop early
    """

    def __init__(self, patience, min_delta, save_path=None, model=None):
        """
        Creates a new instance of EarlyStopping.

        Args:
            patience (int): Number of epochs to wait before stopping early
            min_delta (flaot): Minimum improvement in the monitored metric required to continue
            save_path (string): Path to save the best model weights
            model (torchgeo.models.FarSeg): The model to save the best state_dict from
        """
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.model = model
        
        self.monitor_train_loss = True # Kan endre og gjøre dette til input istedenfor

        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss, train_loss):
        """
        Checks if the learning is going forward.
        Counts the number of epochs until the patience is passed and
        change early_stop, that will stop the training loop.

        Args:
            val_loss (float): Loss values of the last validation
        """
        if np.isnan(val_loss) or np.isinf(val_loss):
            self.early_stop = True
            return
        elif self.best_score is None:
            self.best_score = val_loss
            if self.save_path is not None:
                self.save_checkpoint()
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
            self.save_checkpoint()
        # If train loss, checks for overfitting:
        if self.monitor_train_loss and train_loss is not None:
            if train_loss < self.best_score and val_loss > self.best_score:
                self.early_stop = True
    
    def save_checkpoint(self):
        """
        Saves the best model weights
        """
        if self.model is not None:
            torch.save(self.model.state_dict(), self.save_path)

class geotiffStopping():
    """
    Instance stopping the training if no improvement has occured during the last GeoTIFFs.

    Attributes:
        patience (int): Number of epochs to wait before stopping early
        min_delta (float): Minimum improvement in the monitored metric required to continue
        counter (int): Number of epochs waited
        best_score (float): The best achieved score by the model so far, starts like None
        early_stop (bool): Wether or not to stop early
    """
    def __init__(self, patience, min_delta):
        """
        Creates a new instance of geotiffStopping.

        Args:
            patience (int): Number of epochs to wait before stopping early
            min_delta (flaot): Minimum improvement in the monitored metric required to continue
        """
        self.patience = patience
        self.min_delta = min_delta

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, loss):
        """
        Checks if the learning is going forward.
        Counts the number of GeoTIFFs until the patience is passed and
        change early_stop, that will stop the training loop.

        Args:
            val_loss (float): Loss values of the last validation
        """
        if np.isnan(loss) or np.isinf(loss):
            self.early_stop = True
            return
        elif self.best_score is None:
            self.best_score = loss
        elif loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = loss
            self.counter = 0

# Function:

def initialize_model(num_classes, lr=1e-4):
    """
    Initialize the FarSeg model with optimizer and loss function.

    Args:
        num_classes (int): Number of classes to segment
        lr (float): Learning rate of the model, default = 1e-4
    
    Returns:
        model (torchgeo.models.FarSeg): The FarSeg segmentation model initialized with the specified number of classes
        criterion (torch.nn.CrossEntropyLoss): The loss function used for training, suitable for multi-class classification tasks
        optimizer (torch.optim.Adam): The optimizer for updating model parameters during training, configured with the specified learning rate
    """
    model = FarSeg(backbone="resnet50", classes=num_classes, backbone_pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer
