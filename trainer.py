import gc
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn

from losses import bce_weighted_dice_loss

dtype = torch.cuda.FloatTensor

def train(model, train_dataloader, val_dataloader, label_mean, epochs=100):
    gc.collect()
    torch.cuda.empty_cache()
    
    best_weights = None
    best_loss = 10000000
    
    train_loss = []
    val_loss = []
    

    criterion = bce_weighted_dice_loss
    optimizer = optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-6)
    pbar = tqdm(total=epochs, desc='Training')
    for epoch in range(epochs):
        # Train
        model.train()
        running_loss = 0
        
        for inputs, labels in train_dataloader:
            gc.collect()
            torch.cuda.empty_cache()
            # Move data to GPU
            inputs = inputs.type(dtype)
            labels = labels.type(dtype)
            # Run model
            outputs = model.forward(inputs)

            loss = criterion(outputs, labels, [1-label_mean, label_mean]).cuda()
            optimizer.zero_grad()
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
                inputs = inputs.type(dtype)
                labels = labels.type(dtype)
                # Run model
                outputs = model.forward(inputs)

                loss = criterion(outputs, labels, [1-label_mean, label_mean]).cuda()
                running_loss += loss.item()
        val_epoch_loss = running_loss / len(val_dataloader)
        val_loss.append(val_epoch_loss)
        
        # Save model every epochs
        torch.save(model.state_dict(), f'./models/{model_name}_{loss_name}_t{str(num_train_samples)}_{str(epoch+1)}.npz')
            
        # Keep track of best weights
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            best_weights = model.state_dict()
            
        pbar.set_postfix({'Train Loss': train_epoch_loss, 'Val Loss': val_epoch_loss, 'Best Val': best_loss})
        pbar.update(1)
    # Save weights with best loss
    torch.save(best_weights, f'./models/{model_name}_{loss_name}_t{str(num_train_samples)}_best_loss.npz')
    return model, train_loss, val_loss