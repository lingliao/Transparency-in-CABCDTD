######################
# Remove white edges #
######################
# We manually reviewed all cropped images and filtered out the 18 images which contained unwanted white edges
# It shall be conveninent/quick if we choose to view all the images under Gallery view
# We then applied below code to remove the white edges

# Load modules
import os
from PIL import Image

# Define the source and destinaation folder
source_folder = '/content/white_edge'
destination_folder = '/content/adjusted/'

# Create the destination folder if not existed
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Iterate all images inside a folder
for filename in os.listdir(source_folder):
    if filename.endswith('.png'):
        file_path = os.path.join(source_folder, filename)

        img = Image.open(file_path)

        width, height = img.size

        for x in range(width):
            for y in range(height):
                pixel_value = img.getpixel((x, y))

                if pixel_value == 65535:  #we only removed the pure white part, the pixel value = 65535
                    img.putpixel((x, y), 0)

        destination_filename = filename.replace('_content_', 'adjusted_')

        destination_file_path = os.path.join(destination_folder, destination_filename)

        img.save(destination_file_path)

print("All images processed and saved in the destination folder.")
