import torch

# loss function with adjusted scaling based on label_mean
def adjusted_mse(output, target, label_mean):
    loss = torch.mean(((output - target) * torch.where(target < 0.5, 1/label_mean, 1/(1-label_mean)))**2)
    return loss

# loss based on label overlaps
def dice_loss(predicted, target, smooth=0.00001):
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target)
    dice = 1 - (2 * intersection + smooth) / (union + smooth)
    return dice

# weighted bce with dice 
def bce_weighted_dice_loss(predicted, target, weights, smooth=0.00001):
    # Clamp values to avoid numerical instability
    predicted = torch.clamp(predicted,min=1e-7,max=1-1e-7)
    bce = - weights[1] * target * torch.log(predicted) - (1 - target) * weights[0] * torch.log(1 - predicted)
    bce = torch.mean(bce) + dice_loss(predicted, target, smooth)
    return bce

def bce_dice_loss(predicted, target, smooth=0.00001):
    # Clamp values to avoid numerical instability
    predicted = torch.clamp(predicted,min=1e-7,max=1-1e-7)
    bce = - target * torch.log(predicted) - (1 - target)  * torch.log(1 - predicted)
    bce = torch.mean(bce) + dice_loss(predicted, target, smooth)
    return bce