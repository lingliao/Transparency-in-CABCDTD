##############################
# Crop 598 by 598 pixel area #
##############################

# Load modules
import os
from PIL import Image

#define the function to find the geometric center
def find_nonzero_center(img):
    # get the width and height from the image size
    width, height = img.size

    # Initialize the total pixel count and cumulative values of two coordinates
    total_pixels = 0
    sum_x = 0
    sum_y = 0

    # iterate and calculate the values
    for x in range(width):
        for y in range(height):
            pixel_value = img.getpixel((x, y))
            if pixel_value != 0:
                total_pixels += 1
                sum_x += x
                sum_y += y

    # Calculate the geometric center coordinates of non-zero pixels
    center_x = sum_x // total_pixels if total_pixels > 0 else width // 2
    center_y = sum_y // total_pixels if total_pixels > 0 else height // 2

    return (center_x, center_y)

#define the function for cropping
def extract_center_for_folder(input_folder, output_folder, output_size):
    # create the output_folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # iterate all images
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            # Construct input and output file paths
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # open the image
            img = Image.open(input_path)

            # Find the geometric center coordinates of non-zero pixels
            center_x, center_y = find_nonzero_center(img)

            # Calculate the coordinates of the top-left and bottom-right corners of the cropping area
            left = max(0, center_x - output_size // 2)
            top = max(0, center_y - output_size // 2)
            right = min(center_x + output_size // 2, img.width)
            bottom = min(center_y + output_size // 2, img.height)

            # crop the image
            cropped_img = img.crop((left, top, right, bottom))

            # save the cropped image
            cropped_img.save(output_path)
            print(f"Processed {filename}")

# Call the function for cropping
# Adjust the input directory and the destination directory to suit your need
extract_center_for_folder("/content/train_same_size", "/content/train_598", 598)
