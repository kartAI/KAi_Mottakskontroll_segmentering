# FarSeg/Functionality/farSegModel.py

# Libraries:

import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchgeo.models import FarSeg

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import generalFunctions as gf

# Classes:

class EarlyStopping():
    """
    Instance to consider good learning rate of the model - stops the training loop when teh model do not learn more

    Attributes:
        patience (int): Number of epochs to wait before stopping early
        min_delta (float): Minimum improvement in the monitored metric required to continue
        save_path (string): Path to save the best model weights
        model (FarSeg model): The model to save the best state_dict from
        counter (int): Number of epochs waited
        best_score (float): The best achieved score by the model so far, starts like None
        early_stop (bool): Wether or not to stop early
    """

    def __init__(self, patience, min_delta, save_path=None, model=None):
        """
        Creates a new instance of EarlyStopping.

        Arguments:
            patience (int): Number of epochs to wait before stopping early
            min_delta (flaot): Minimum improvement in the monitored metric required to continue
            save_path (string): Path to save the best model weights
            model (torchgeo.models.FarSeg): The model to save the best state_dict from
        """
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.model = model

        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss, log_file=None):
        """
        Checks if the learning is going forward.
        Counts the number of epochs until the patience is passed and
        change early_stop, that will stop the training loop.

        Arguments:
            val_loss (float): Loss value of the last validation
            train_loss (float): Loss value of the last training session
            log_file (string): Path to the log file to log proress, default None
        """
        if np.isnan(val_loss) or np.isinf(val_loss):
            self.early_stop = True
            if log_file != None:
                gf.log_info(log_file, f"Invalid loss: {val_loss}")
            return
        elif self.best_score == None:
            self.best_score = val_loss
            if self.save_path != None:
                self.save_checkpoint(log_file)
        elif val_loss > self.best_score - self.min_delta:
            if val_loss < self.best_score:
                self.best_score = val_loss
                self.save_checkpoint(log_file)
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if log_file != None:
                    gf.log_info(log_file, "The training during this epoch for this GeoTIFF did not make the model better.")
        else:
            self.best_score = val_loss
            self.save_checkpoint(log_file)
            self.counter = 0

    def save_checkpoint(self, log_file=None):
        """
        Saves the best model weights

        Argument:
            log_file (string): Path to the log file to log proress, default None
        """
        if self.model != None and self.save_path != None:
            torch.save(self.model.state_dict(), self.save_path)
            if log_file != None:
                gf.log_info(log_file, f"Model saved. {self.counter} - {self.best_score}")

# Function:

def initialize_model(num_classes, lr=1e-4):
    """
    Initialize the FarSeg model with optimizer and loss function.

    Arguments:
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
