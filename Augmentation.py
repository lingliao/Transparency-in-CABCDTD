######################
# Image augmentation #
######################

# Load modules
import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os


# Define the directory containing the original images
original_dir = '/content/all_598'

# Define the directory to save the augmented images
augmented_dir = '/content/all_598_augmented'

# Ensure the augmented directory exists, create if not
if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)

# ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# List all original image files
original_files = os.listdir(original_dir)

# Function to save augmented images
def save_augmented_images(img, prefix, idx):
    filename = os.path.join(augmented_dir, f"{prefix}_{idx}.png")
    cv2.imwrite(filename, img)

# Iterate through each original image
for filename in original_files:
    # Load the original image
    img = cv2.imread(os.path.join(original_dir, filename), cv2.IMREAD_UNCHANGED)

    # Reshape to add a channel dimension if it's missing
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    # Rescale to [0, 1]
    img = img.astype(np.float32) / 65535.0

    # Reshape to 4D array (batch_size, rows, columns, channels) for augmentation
    img = np.expand_dims(img, axis=0)

    # Generate augmented images
    i = 0
    for batch in datagen.flow(img, batch_size=1):
        augmented_img = (batch[0] * 65535).astype(np.uint16)  # Rescale back to 16-bit
        save_augmented_images(augmented_img,os.path.splitext(filename)[0], i + 1)
        print(f"Augmented image {i + 1} saved to : {filename}")
        i += 1
        if i >= 5:  # Generate 5 augmented images for each original image
            break
print("Data augmentation completed.")
