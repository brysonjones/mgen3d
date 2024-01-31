import torch
import numpy as np
import json
from PIL import Image
import os

# write a function to ensure an image is a float format and between 0 and 1
def convert_image_to_float(image:np.ndarray):
    """This function converts the input image to float format and ensures that
       the values are between 0 and 1.

    Args:
        image (np.ndarray): input image

    Returns:
        np.ndarray: image in float format between 0 and 1
    """    
    # check if image is float
    if image.dtype != np.float32:
        image = image.astype(np.float32)
        
    # check if image is between 0 and 1
    if np.max(image) > 1.0:
        image = image / 255.0
    return image
    

def read_image(file_path:str, white_background:bool=False):
    image = np.asarray(Image.open(file_path))
    if white_background:
        image = convert_background_to_white(image)
    if image.shape[-1] == 4:
        image = convert_background_to_black(image) # remaining alpha channel converted to black
    image = convert_image_to_float(image)
    return image

def convert_background_to_white(image:np.ndarray):
    """This function converts the background color of the image to the specified color
       provided by the user. This requires the image to be in RGBA format.

    Args:
        image (np.ndarray): input image in RGBA format

    Returns:
        np.ndarray: update image in RGB format
    """    
    image_pil = Image.fromarray(image)
    new_image = Image.new("RGBA", image_pil.size, "WHITE") # Create a white rgba background
    new_image.paste(image_pil, (0, 0), image_pil)              # Paste the image on the background. Go to the links given below for details.
    new_image.convert('RGB')
    new_image = np.asarray(new_image)
    
    return new_image[:, :, :3]

def convert_background_to_black(image:np.ndarray):
    """This function converts the background color of the image to the specified color
       provided by the user. This requires the image to be in RGBA format.

    Args:
        image (np.ndarray): input image in RGBA format

    Returns:
        np.ndarray: update image in RGB format
    """    
    image_pil = Image.fromarray(image)
    new_image = Image.new("RGBA", image_pil.size, "BLACK") 
    new_image.paste(image_pil, (0, 0), image_pil)
    new_image.convert('RGB')
    new_image = np.asarray(new_image)
    
    return new_image[:, :, :3]
    