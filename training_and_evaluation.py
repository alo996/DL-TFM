from tracNet import TracNet
from data_preparation import matFiles_to_npArray, reshape

import copy
import datetime
import gc
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from scipy.io import loadmat, savemat
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary


def initialize_weights(module):
    """Sample inital weights for the convolutional layers from a normal distribution."""
    if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
        torch.nn.init.normal_(module.weight, std=0.01)


def run_epoch(model, loss_fn, dataloader, device, optimizer, train):
    # Set model to training mode
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    epoch_rmse = 0.0

    # Iterate over data
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)

        # zero the parameters
        if train:
            optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(train):
            pred = model(xb)
            loss = loss_fn(pred, yb)

            # backward + optimize if in training phase
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                optimizer.step()

        # statistics
        epoch_loss += loss.item()

    epoch_loss /= len(dataloader.dataset)
    epoch_rmse = np.sqrt(2 * epoch_loss)
    return epoch_loss, epoch_rmse


def fit(model, loss_fn, scheduler, dataloaders, optimizer, device, max_epochs, patience):
    best_val_rmse = np.inf
    best_epoch = -1
    best_model_weights = {}

    for epoch in range(1, max_epochs + 1):
        train_loss, train_rmse = run_epoch(model, loss_fn, dataloaders['train'], device, optimizer, train=True)
        scheduler.step()
        val_loss, val_rmse = run_epoch(model, loss_fn, dataloaders['val'], device, optimizer=None, train=False)
        print(
            f"Epoch {epoch}/{max_epochs}, train_loss: {train_loss:.3f}, train_rmse: {train_rmse:.3f}, val_loss: {val_loss:.3f}, val_rmse: {val_rmse:.3f}")

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_rmse', train_rmse, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_rmse', val_rmse, epoch)

        # Save best weights
        if val_rmse < best_val_rmse:
            best_epoch = epoch
            best_val_rmse = val_rmse
            best_model_weights = copy.deepcopy(model.state_dict())

        # Early stopping
        print(
            f"best val_rmse: {best_val_rmse:.3f}, epoch: {epoch}, best_epoch: {best_epoch}, current_patience: {patience - (epoch - best_epoch)}")
        if epoch - best_epoch >= patience:
            break

    torch.save(best_model_weights, f'/home/alexrichard/LRZ Sync+Share/ML in Physics/{NAME}.pth')


