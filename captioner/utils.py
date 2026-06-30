import numpy as np
from PIL import Image
from typing import Union

def load_image(image: Union[np.ndarray, Image.Image, str], return_type='numpy'):
    """
    Load image from path or PIL.Image or numpy.ndarray to required format.
    """

    # Check if image is already in return_type
    if isinstance(image, Image.Image) and return_type == 'pil' or \
            isinstance(image, np.ndarray) and return_type == 'numpy':
        return image

    # PIL.Image as intermediate format
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if image.mode == "RGBA":
        image = image.convert("RGB")
        
    if return_type == 'pil':
        return image
    elif return_type == 'numpy':
        return np.asarray(image)
    else:
        raise NotImplementedError()
