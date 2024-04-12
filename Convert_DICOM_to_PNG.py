##################################################
# Convert DICOM format to PNG format             #
# Our downloaded data is saved in google drive   #
# Please adjust the root folder as your need     #
##################################################

# Make sure to replace any placeholders or paths with the actual paths or values specific to your environment
# We decided to use Subject ID(patientID, left or right breast and image view) to name the converted images and track the corresponding pathology information.
# One full image might have multiple ROIs. It was important to add different suffixes, prefixes, etc., while keeping the Subject ID the same while saving the converted images.

# Install pydicom if needed
!pip install pydicom
# Load modules
import os
import cv2
import shutil
import torch
from torchvision import transforms
from google.colab import drive
import pydicom
from PIL import Image
import pandas as pd

# Load the images
# Connect with google drive, feel free to delete this part if you didn't use google drive to save your downloaded images
# Connect with Google Drive
# Assuming you have already mounted Google Drive in your environment
drive.mount('/content/drive')

# Unzip the downloaded images
# Replace this with the correct path to your zip file
!unzip /content/drive/MyDrive/CBIS-DDSM/roi_crop_train.zip

# Read in the description csv files
train = pd.read_csv('/content/drive/MyDrive/CBIS-DDSM/mass_case_description_train_set.csv')
test = pd.read_csv('/content/drive/MyDrive/CBIS-DDSM/mass_case_description_test_set.csv')

# Extract the columns we are interested in
train_need = train[['patient_id', 'pathology', 'image file path', 'cropped image file path', 'ROI mask file path']]
test_need = test[['patient_id', 'pathology', 'image file path', 'cropped image file path','ROI mask file path']]

# Merge them together and reindex the output dataframe
merged_df = pd.concat([train_need, test_need], axis=0)
merged_df.reset_index(drop=True, inplace=True)

# Extract the name information we need: patientID, left or right breast, and so on.
merged_df['crop_-1'] = merged_df['cropped image file path'].apply(lambda x: x.split('/')[0])
merged_df['ROI_-1'] = merged_df['ROI mask file path'].apply(lambda x: x.split('/')[0])
merged_df['full_-1'] = merged_df['image file path'].apply(lambda x: x.split('/')[0])

###########################################################################
# Confirm whether "SubjectID" was a suitable choice for naming the images #
###########################################################################

## Round 1
# Check if items in column1 and column2 of the same row are the same
merged_df['same_items'] = merged_df['crop_-1'] == merged_df['ROI_-1']
merged_df['full_same_items'] = merged_df['full_-1'].isin(merged_df['crop_-1'])

#Check if items in column A share the same pathology info or not when the items are same.
merged_df['pathology_crop'] = merged_df['pathology'][merged_df['crop_-1'].duplicated(keep=False)].duplicated(keep=False)
merged_df['pathology_roi'] = merged_df['pathology'][merged_df['ROI_-1'].duplicated(keep=False)].duplicated(keep=False)
merged_df['pathology_full'] = merged_df['pathology'][merged_df['full_-1'].duplicated(keep=False)].duplicated(keep=False)

# Count the number of False values
count_false = (merged_df['same_items'] == False).sum()
full_false = (merged_df['full_same_items'] == False).sum()
pathology_crop = (merged_df['pathology_crop'] == False).sum()
pathology_roi = (merged_df['pathology_roi'] == False).sum()
pathology_full = (merged_df['pathology_full'] == False).sum()

# Display the count
print("Number of False values:", count_false)
print("Number of full_False values:", count_false)
print("Number of pathology crop:", pathology_crop)
print("Number of pathylogy roi:", pathology_roi)
print("Number of pathylogy full:", pathology_full)

## Round 2
# We tested for "image file path" only, feel free to try other columns if interested in.

# Identify duplicate 'image file path' values
merged_df['name'] = merged_df['image file path'].str.split('/').str[0]
is_duplicate = merged_df.duplicated(subset=['image file path'], keep='first')

# Filter the DataFrame to keep only rows with unique 'image file path' values
unique_df = merged_df[~is_duplicate]

filtered_df = pd.DataFrame(columns=unique_df.columns)

# Grouping by 'patient_id' and checking if 'pathology' values are the same within each group
for name, group in unique_df.groupby('name'):
    if len(group['pathology'].unique()) > 1:
        print("Inconsistent pathology for name:", name)
        print(group)
        unique_df = pd.concat([unique_df, group])

# Resetting index for the filtered DataFrame
filtered_df.reset_index(drop=True, inplace=True)
print(len(filtered_df))

##############################################################
# Convert DICOM to PNG and save output to destination folder #
##############################################################

# Fonction for the conversion
# Make sure you choose the right pixel value, 16 bit for full images, 8 bits for the remaining
def convert_dcm_to_png(source_folder, destination_root):

    # Initialize a counter for failed conversions
    failed_count = 0

    # Initialize lists to store data for DataFrame
    source_paths = []
    png_paths = []
    source_parts = []
    png_parts = []

    # Traverse through each item in the source folder
    for item in os.listdir(source_folder):
        source_item_path = os.path.join(source_folder, item)
        # Check if it's a directory
        if os.path.isdir(source_item_path):
            # Recursively convert .dcm files in subdirectories
            sub_failed_count, sub_df = convert_dcm_to_png(source_item_path, destination_root)
            failed_count += sub_failed_count
            source_paths.extend(sub_df['source_path'])
            png_paths.extend(sub_df['png_path'])
            source_parts.extend(sub_df['source_part'])
            png_parts.extend(sub_df['png_part'])

        elif item.endswith('.dcm'):
            try:
                # Read DICOM file
                ds = pydicom.dcmread(source_item_path, force=True)

                # Convert pixel data to unit 16 or uint8
                # pixel_array = ds.pixel_array.astype('uint16') #for full image
                pixel_array = ds.pixel_array.astype('uint8') # for ROI and croped image

                # Create an 8-bit image from the array
                image = Image.fromarray(pixel_array)

                # Extract the filename without extension
                name = os.path.splitext(os.path.basename(source_item_path))[0]

                # Extract the last 4 parts of the path
                source_part = source_item_path.split('/')[-4]

                # Construct the destination folder path
                destination_folder = os.path.join(destination_root, source_part)

                # Create destination folder if it doesn't exist
                os.makedirs(destination_folder, exist_ok=True)

                # Construct the destination PNG file path
                png_path = os.path.join(destination_folder, f"_{name}.png")

                # Check if the PNG file already exists
                while os.path.exists(png_path):
                    # Add a prefix "_" before the name
                    name = f"_{name}"
                    png_path = os.path.join(destination_folder, f"_{name}.png")

                # Save as PNG directly in the destination folder
                image.save(png_path, format='PNG')

                # Append paths and parts to lists
                source_paths.append(source_item_path)
                png_paths.append(png_path)
                source_parts.append(source_part)
                png_parts.append(png_path.split('/')[-2])

            except Exception as e:
                print(f"Failed to convert {source_item_path}: {e}")
                failed_count += 1

    # Return the total number of failed conversions and DataFrame
    return failed_count, pd.DataFrame({
        'source_path': source_paths,
        'png_path': png_paths,
        'source_part': source_parts,
        'png_part': png_parts
    })

# Specify the source folder and destination folder, make sure they exist
source_folder = "/content/mass_train_ROI/manifest-LyDgOQGl3853937313152078328/CBIS-DDSM"
destination_root = "/content/roi_train"

# Call the functon to convert DICOM files to PNG and get the total number of failed conversions and DataFrame
# The function will return the number of failed conversions and the paths of the images before and after conversion as well
full_failed, full_df = convert_dcm_to_png(source_folder, destination_root) 

############################################
# Move the converted images to one folder  #
############################################

# Function to move images
def move_images(root_dir, dest_dir):
    # Walk through the directory structure
    for foldername, _, filenames in os.walk(root_dir):
        # Iterate over each file in the current directory
        for filename in filenames:
            # Construct the source and destination paths
            source_path = os.path.join(foldername, filename)
            dest_path = os.path.join(dest_dir, foldername.replace('/', '_') + '_' + filename)

            # Create destination directory if it doesn't exist
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # Move the file
            shutil.move(source_path, dest_path)
            print(f"Moved: {source_path} -> {dest_path}")

# Specify the root directory containing images and the destination directory
root_dir = "/content/roi_train"
dest_dir = "/content/roi_train_needed"

# Call the function to move images
move_images(root_dir, dest_dir)
