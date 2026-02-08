import os
import cv2
import imgaug.augmenters as iaa
from tqdm import tqdm

# Input and output folders
input_folder = ""  # Change this to your input folder
output_folder = ""  # Change this to your output folder

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define augmentation pipeline
augmentations = iaa.Sequential([
    iaa.Fliplr(0.5),  # Horizontal flip with 50% probability
    iaa.Flipud(0.2),  # Vertical flip with 20% probability
    iaa.Affine(rotate=(-25, 25)),  # Rotate images between -25 and 25 degrees
    iaa.Multiply((0.8, 1.2)),  # Change brightness (80%-120%)
    iaa.GaussianBlur(sigma=(0, 3.0)),  # Apply Gaussian blur
    iaa.AdditiveGaussianNoise(scale=(10, 30)),  # Add Gaussian noise
])

# Process all images in the input folder
print("Augmenting images...")
for filename in tqdm(os.listdir(input_folder)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Read the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

        # Apply augmentations
        augmented_images = augmentations(images=[image])

        # Save augmented images
        for i, augmented_image in enumerate(augmented_images):
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_aug_{i}.jpg")
            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
            cv2.imwrite(output_path, augmented_image)

print("Data augmentation complete. Augmented images are saved in the output folder.")
