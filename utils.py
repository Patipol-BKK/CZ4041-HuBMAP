import numpy as np
import rasterio
from tqdm.auto import tqdm
import torch

# Reads and convert image array to grayscale if needed
def read_tiff(image_path):
    with rasterio.open(image_path) as image:
        image_data = image.read().astype(np.float32)
        
        if image_data.shape[0] == 1:
            image_data /= 255.0
        else:
            image_data = np.average(
                image_data, 
                axis=0, 
                weights=[0.299, 0.587, 0.144]
            )
            
            image_data = np.expand_dims(image_data, axis=0)
            
        # # Histogram Equalization (takes too long and too much memory to do)
        # hist, bins = np.histogram(image_data.flatten(), bins=256, range=[0, 1])
        # cdf = hist.cumsum(dtype=np.float32)
        # cdf_normalized = cdf / cdf[-1]
        # equalized_image = np.interp(image_data.flatten(), bins[:-1], cdf_normalized)
        # equalized_image = equalized_image.reshape(image_data.shape)
            
        return image_data.astype(np.float32)

# loss function with adjusted scaling based on label_mean
def adjusted_mse(output, target, label_mean):
    loss = torch.mean(((output - target) * torch.where(target < 0.5, 1/label_mean, 1/(1-label_mean)))**2)
    return loss

# Calculate average of mask labels, use to weight loss accordingly
def dataset_label_mean(dataset):
    label_sum = 0
    for idx in tqdm(range(len(dataset)), desc='Dataset Mean'):
        label_sum += np.sum(dataset[idx][1])/(dataset[idx][1].shape[1]*dataset[idx][1].shape[2])
    label_mean = label_sum / len(dataset)
    return label_mean