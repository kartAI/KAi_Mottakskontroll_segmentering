# FarSeg_dev/Functionality/farSegModel.py

# Libraries:

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchgeo.models import FarSeg

# Class:

class EarlyStopping():
    """
    Instance to consider good learning rate of the model - stops the training loop when teh model do not learn more

    Attributes:
        patience (int): Number of epochs to wait before stopping early
        min_delta (flaot): Minimum improvement in the monitored metric required to continue
        counter (int): Number of epochs waited
        best_score (float): The best achieved score by the model so far, starts like None
        early_stop (bool): Wether or not to stop early
    """

    def __init__(self, patience=5, min_delta=0):
        """
        Creates a new instance of EarlyStopping.

        Args:
            patience (int): Number of epochs to wait before stopping early
            min_delta (flaot): Minimum improvement in the monitored metric required to continue
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        """
        Checks if the learning is going forward.
        Counts the number of epochs until the patience is passed and
        change early_stop, that will stop the training loop.

        Args:
            val_loss (float): Loss values of the last validation
        """
        if np.isnan(val_loss):
            self.early_stop = True
        elif self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
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
