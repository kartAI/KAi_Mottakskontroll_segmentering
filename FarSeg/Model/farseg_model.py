# Model/farseg_model.py

from torchgeo.models import FarSeg
import torch.nn as nn
import torch.optim as optim
import numpy as np

def initialize_model(num_classes, lr=1e-4):
    """Initialize the FarSeg model with optimizer and loss function."""
    model = FarSeg(backbone="resnet50", classes=num_classes, backbone_pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Args:
            patience (int): Number of epochs to wait before stopping early
            min_delta (float): Minimum improvement in the monitored metric required to continue
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss):
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

