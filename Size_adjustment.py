###################
# Size adjustment #
###################
# Identify wether the size equals to 598 by 598 or not.
# If not, resize it to the desired size while remaining the center of abnormal area at the geometric center

# Load modules
import os
from PIL import Image

# Define the function
def adjust_and_save_image(image_path, target_size=(598, 598), save_dir='same_images', adjusted_dir='adjusted'):
    # Create the destination folder
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(adjusted_dir, exist_ok=True)

    # Open the image
    image = Image.open(image_path)

    # Get the width and height
    width, height = image.size

    # If the size doesn't match witht he target size
    if width != target_size[0] or height != target_size[1]:
        # Calculate the size we need to add
        left = (target_size[0] - width) // 2
        top = (target_size[1] - height) // 2
        right = target_size[0] - width - left
        bottom = target_size[1] - height - top

        # Create a new image and paste the 'image' onto it
        new_image = Image.new(image.mode, target_size, 0)  # fill in with black color
        new_image.paste(image, (left, top))

        # Save the adjusted image to the corresponding directory
        adjusted_path = os.path.join(adjusted_dir, os.path.basename(image_path))
        new_image.save(adjusted_path)
    else:
        # Directly copy images of equal dimensions to a new folder
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        image.save(save_path)


# Define the source and output folders
image_folder = '/content/train_598' #input
adjusted_folder = '/content/train_598_updated' #output 1
same_folder = '/content/train_598_same' #output 2

# Iterate all images in the input folder
for filename in os.listdir(image_folder):
    if filename.endswith('.png'):
        image_path = os.path.join(image_folder, filename)
        adjust_and_save_image(image_path, save_dir=same_folder, adjusted_dir=adjusted_folder)

Print("Done!")
