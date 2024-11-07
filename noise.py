import os
import numpy as np
from PIL import Image
import albumentations as A
from tqdm import tqdm

# Set up input and output directories
input_folder = 'input_screenshots'
output_folder = 'outputs'

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)


augmentations = A.Compose([
    A.GaussNoise(var_limit=(100.0, 1000.0), p=0.8),
    A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
    A.ImageCompression(quality_lower=30, quality_upper=100, p=0.4)
])



def is_blank(image):
    return np.std(image) < 5


image_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]

# Loop through each file in the input directory with a progress bar
for filename in tqdm(image_files, desc="Processing images"):
    # Load image with Pillow
    image_path = os.path.join(input_folder, filename)
    try:
        with Image.open(image_path) as img:
            image = np.array(img)  # Convert Pillow image to NumPy array

        # Check if the image is blank
        if is_blank(image):
            os.remove(image_path)  # Delete blank image without printing
            continue

        # Apply augmentations with independent probabilities
        augmented = augmentations(image=image)
        augmented_image = augmented["image"]

        # Save augmented image using Pillow
        output_image = Image.fromarray(augmented_image)
        mod_filename = f"distorted_{filename}"
        output_path = os.path.join(output_folder, mod_filename)
        output_image.save(output_path)

    except Exception as e:
        # Handle errors silently if needed
        continue

print("Image processing completed.")
