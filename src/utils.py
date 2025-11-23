# src/utils.py
from PIL import Image
import os

def process_image(image_file):
    """
    Validates and prepares the uploaded image.
    Ensures it is in RGB format (removing Alpha channels if PNG).
    """
    try:
        image = Image.open(image_file)
        
        # Convert to RGB (BiomedCLIP expects 3 channels, not 4 or 1)
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def get_reference_images(folder_path="data/reference_images"):
    """
    Generator that yields images from the reference folder
    so the RAG Engine can index them.
    """
    valid_exts = [".jpg", ".jpeg", ".png"]
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} not found.")
        return []

    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in valid_exts):
            path = os.path.join(folder_path, filename)
            yield path, filename