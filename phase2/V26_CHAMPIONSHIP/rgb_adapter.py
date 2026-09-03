import cv2
import numpy as np

def load_rgb_as_grayscale(image_path: str):
    """
    Experiment F: RGB Adapter
    Lightweight CPU adapter to convert 3-channel RGB optical imagery 
    to robust grayscale without CNNs or external libraries.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None
        
    if len(img_bgr.shape) == 2 or img_bgr.shape[2] == 1:
        return img_bgr # Already grayscale
        
    # Standard luminosity
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # In some optical modalities, specific channels (like Green) 
    # hold better edge contrast than raw luminance.
    # We return the combined consensus gray for existing pipeline.
    return gray
