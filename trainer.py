import gc
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn
import sys
import os

from losses import bce_weighted_dice_loss

dtype = torch.cuda.FloatTensor

def train(model, model_name, save_dir, train_dataloader, val_dataloader, criterion, optimizer, epochs=100):
    gc.collect()
    torch.cuda.empty_cache()

    best_weights = None
    best_loss = sys.maxsize

    train_loss = []
    val_loss = []

    # Model name format:
    # model_name, patch_size, dataset, lr, epoch
    model_name = f'{model_name}_{str(epochs)}'

    pbar = tqdm(total=epochs, desc='Training')
    for epoch in range(epochs):
        # Train
        model.train()
        running_loss = 0
        
        for inputs, labels in train_dataloader:
            gc.collect()
            torch.cuda.empty_cache()
            # Move data to GPU
            inputs = torch.unsqueeze(torch.tensor(inputs.type(dtype)), axis=1)
            labels = torch.unsqueeze(torch.tensor(labels.type(dtype)), axis=1)
            optimizer.zero_grad()
            # Run model
            outputs = model.forward(inputs)

            loss = criterion(outputs, labels).cuda()
            
            loss.backward()

            running_loss += loss.item()
            optimizer.step()
            
        train_epoch_loss = running_loss / len(train_dataloader)
        train_loss.append(train_epoch_loss)

        # Val
        model.eval()
        running_loss = 0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                gc.collect()
                torch.cuda.empty_cache()
                # Move data to GPU
                inputs = torch.unsqueeze(torch.tensor(inputs.type(dtype)), axis=1)
                labels = torch.unsqueeze(torch.tensor(labels.type(dtype)), axis=1)
                # Run model
                outputs = model.forward(inputs)

                loss = criterion(outputs, labels).cuda()
                running_loss += loss.item()
        val_epoch_loss = running_loss / len(val_dataloader)
        val_loss.append(val_epoch_loss)
        
        # Keep track of best weights
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            best_weights = model.state_dict()
            
        pbar.set_postfix({'TL': train_epoch_loss, 'VL': val_epoch_loss, 'BV': best_loss})
        pbar.update(1)
    # Save weights
    torch.save(best_weights, os.path.join(save_dir, f'{model_name}_best_loss.npz'))
    torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}_last_epoch.npz'))
    return model, train_loss, val_loss