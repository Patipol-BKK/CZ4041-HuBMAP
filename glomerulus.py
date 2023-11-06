import numpy as np
import cv2
import json
from shapely.geometry import Polygon
from skimage import draw
import copy
from tqdm.auto import tqdm
import torch.nn as nn
from torch.utils.data import Dataset
import random

class Glomerulus():
    def __init__(self, coordinates):
        self.coordinates = np.array(coordinates)
        centroid = self._compute_centroid(coordinates)
        self.centroid_y, self.centroid_x = centroid.y, centroid.x
        
    def _compute_centroid(self, coordinates):
        polygon = Polygon(coordinates)
        return polygon.centroid
    
    def __repr__(self):
        return f'Glomerulus(centroid_x={self.centroid_x}, centroid_y={self.centroid_y})'
    
    
def get_glomeruli(json_path, target_label):
    label_json = json.load(open(json_path))
    glomeruli = []
    for element in label_json:
        if element['type'] == 'Feature' and element['id'] == 'PathAnnotationObject':
            label = element['geometry']
            if label['type'] == 'Polygon' and element['properties']['classification']['name'] == target_label:
                glomeruli.append(Glomerulus(label['coordinates'][0]))
    return glomeruli

# Stores only patch location and rotation to save memory
class Patch():
    def __init__(self, center_x, center_y, theta, patch_size, glomeruli, image):
        self.center_x = center_x
        self.center_y = center_y
        
        self.theta = theta
        self.patch_size = patch_size
        
        self.image = image
        self.glomeruli = glomeruli
    
    def _rotate(self, image, angle):
        # Get center of sliced image and rotation matrix
        center = (image.shape[2] // 2, image.shape[1] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated_image = np.zeros_like(image)
        for idx in range(image.shape[0]):
            rotated_image[idx] = cv2.warpAffine(
                image[idx], 
                rotation_matrix, 
                (image.shape[2], image.shape[1])
            )
        return rotated_image
    
    # TODO: Optimize by only rendering glomeruli appeared in the mask
    # Renders image binary mask using given image array and patch's coordinate and rotation
    def render_mask(self):
        cell_center_x = int(round(self.center_x))
        cell_center_y = int(round(self.center_y))
        
        # Calculate slicing size that ensures patch will fit with any rotation
        half_slicing_size = int(np.ceil(self.patch_size * np.sqrt(2))) + 1
        
        slicing_size = half_slicing_size * 2
        
        half_patch_size = self.patch_size // 2
        
        padded_base = np.zeros((1, slicing_size, slicing_size), dtype=bool)
        
        for glomerulus in self.glomeruli:
            distance_x = np.abs(glomerulus.centroid_x - self.center_x)
            distance_y = np.abs(glomerulus.centroid_y - self.center_y)
            distance = np.sqrt(distance_x**2 + distance_y**2)
            
            if distance < slicing_size:
                coordinates = copy.deepcopy(glomerulus.coordinates)
                for idx in range(len(coordinates)):
                    coordinates[idx] = np.array([
                        coordinates[idx][0] - cell_center_x + half_slicing_size, 
                        coordinates[idx][1] - cell_center_y + half_slicing_size
                    ], dtype=int)

                binary_mask = np.array(draw.polygon2mask(
                    (slicing_size, slicing_size),
                    np.flip(coordinates, axis=None)
                ), dtype=bool)

                binary_mask = np.expand_dims(binary_mask, axis=0)
                padded_base |= binary_mask
            
        rotated_binary_mask = self._rotate(padded_base.astype(float), self.theta)
        
        # Crop rotated patch to correct patch size
        half_patch_size = self.patch_size // 2
        
        center = (rotated_binary_mask.shape[2] // 2, rotated_binary_mask.shape[1] // 2)
        
        x_tl = center[0] - half_patch_size
        y_tl = center[1] - half_patch_size
        x_br = center[0] + half_patch_size
        y_br = center[1] + half_patch_size
        
        return rotated_binary_mask[:, y_tl:y_br, x_tl:x_br]
    
    # Renders image array using given image array and patch's coordinate and rotation
    def render_image(self):
        # Get center coordinates snapped to cell coordinates
        cell_center_x = int(round(self.center_x))
        cell_center_y = int(round(self.center_y))
        
        # Calculate slicing size that ensures patch will fit with any rotation
        half_slicing_size = int(np.ceil(self.patch_size * np.sqrt(2))) + 1
        
        # X/Y Top-left
        x_tl = cell_center_x - half_slicing_size
        y_tl = cell_center_y - half_slicing_size
        
        # Calculate needed top-left paddings
        x_pad_tl = - min(0, x_tl)
        y_pad_tl = - min(0, y_tl)
        
        # Limit top-left coordinates to within image indices
        x_tl = max(0, x_tl)
        y_tl = max(0, y_tl)
        
        # X/Y Bottom-right
        x_br = cell_center_x + half_slicing_size
        y_br = cell_center_y + half_slicing_size
        
        # Calculate needed bottom-right paddings
        x_pad_br = max(0, x_br - (self.image.shape[2] - 1))
        y_pad_br = max(0, y_br - (self.image.shape[1] - 1))
        
        # Limit bottom-right coordinates to within image indices
        x_br = min(self.image.shape[2] - 1, x_br)
        y_br = min(self.image.shape[1] - 1, y_br)
                   
        slicing_size = half_slicing_size * 2
        
        # Get sliced image without padding
        sliced_image = self.image[:, y_tl:y_br, x_tl:x_br]
        
        # Get 2d array with correct size and add sliced image to it at the appropriate location
        padded_sliced_image = np.zeros((self.image.shape[0], slicing_size, slicing_size))
        padded_sliced_image[:, y_pad_tl:slicing_size - y_pad_br, x_pad_tl:slicing_size - x_pad_br] = sliced_image
        
        rotated_image = self._rotate(padded_sliced_image, self.theta)
            
        # Crop rotated patch to correct patch size
        half_patch_size = self.patch_size // 2
        
        center = (rotated_image.shape[2] // 2, rotated_image.shape[1] // 2)
        
        x_tl = center[0] - half_patch_size
        y_tl = center[1] - half_patch_size
        x_br = center[0] + half_patch_size
        y_br = center[1] + half_patch_size
        
        return rotated_image[:, y_tl:y_br, x_tl:x_br]

# Generates random patches centered around random glomerulus + random transformation (x, y, and rotation)
def generate_glomerulus_patches(patch_size, num_patches, glomeruli, image):
    patches = []
    for _ in tqdm(range(num_patches)):
        glomeruli_num = len(glomeruli)
        glomerulus = glomeruli[random.randrange(0, glomeruli_num)]
        
        half_patch_size = patch_size / 2
        center_x = random.randrange(
            int(glomerulus.centroid_x - half_patch_size), 
            int(glomerulus.centroid_x + half_patch_size)
        )
        
        center_y = random.randrange(
            int(glomerulus.centroid_y - half_patch_size),
            int(glomerulus.centroid_y + half_patch_size)
        )
        
        theta = random.random()*360
        
        patches.append(Patch(
            center_x = center_x,
            center_y = center_y,
            theta = theta,
            patch_size = patch_size,
            glomeruli = glomeruli,
            image = image
        ))
    return patches

# Multi-image version of generate_glomerulus_patches
def generate_glomerulus_patches_multi(patch_size, num_patches, glomeruli_list, image_list):
    patches = []
    for _ in tqdm(range(num_patches)):
        image_idx = random.randrange(0, len(image_list))

        glomeruli_num = len(glomeruli_list[image_idx])
        glomerulus = glomeruli_list[image_idx][random.randrange(0, glomeruli_num)]
        
        half_patch_size = patch_size / 2 + 50
        center_x = random.randrange(
            int(glomerulus.centroid_x - half_patch_size), 
            int(glomerulus.centroid_x + half_patch_size)
        )
        
        center_y = random.randrange(
            int(glomerulus.centroid_y - half_patch_size),
            int(glomerulus.centroid_y + half_patch_size)
        )
        
        theta = random.random()*360
        
        patches.append(Patch(
            center_x = center_x,
            center_y = center_y,
            theta = theta,
            patch_size = patch_size,
            glomeruli = glomeruli_list[image_idx],
            image = image_list[image_idx]
        ))
    return patches

# Generates random patches from the image at random position and rotation
def generate_random_patches(patch_size, num_patches, glomeruli, image):
    patches = []
    for _ in tqdm(range(num_patches)):
        glomeruli_num = len(glomeruli)
        glomerulus = glomeruli[random.randrange(0, glomeruli_num)]
        
        center_x = random.randrange(
            patch_size // 2,
            image.shape[2] - patch_size // 2
        )
        
        center_y = random.randrange(
            patch_size // 2,
            image.shape[1] - patch_size // 2
        )
        
        theta = random.random()*360
        
        patches.append(Patch(
            center_x = center_x,
            center_y = center_y,
            theta = theta,
            patch_size = patch_size,
            glomeruli = glomeruli,
            image = image
        ))
    return patches

# Multi-image version of generate_random_patches
def generate_random_patches_multi(patch_size, num_patches, glomeruli_list, image_list):
    patches = []
    for _ in tqdm(range(num_patches)):
        image_idx = random.randrange(0, len(image_list))

        glomeruli_num = len(glomeruli_list[image_idx])
        glomerulus = glomeruli_list[image_idx][random.randrange(0, glomeruli_num)]
        
        center_x = random.randrange(
            patch_size // 2,
            image_list[image_idx].shape[2] - patch_size // 2
        )
        
        center_y = random.randrange(
            patch_size // 2,
            image_list[image_idx].shape[1] - patch_size // 2
        )
        
        theta = random.random()*360
        
        patches.append(Patch(
            center_x = center_x,
            center_y = center_y,
            theta = theta,
            patch_size = patch_size,
            glomeruli = glomeruli_list[image_idx],
            image = image_list[image_idx]
        ))
    return patches

class KidneySampleDataset():
    def __init__(self, patches):
        self.patches = patches
    def __getitem__(self, idx):
        # Handle slicing, repeatedly call index version of the function
        if isinstance(idx, slice):
            return [self[ii] for ii in iter(range(*idx.indices(len(self))))]

        # Handle index
        elif isinstance(idx, int):
            return self.patches[idx].render_image(), self.patches[idx].render_mask()
        
    def __len__(self):
        return len(self.patches)