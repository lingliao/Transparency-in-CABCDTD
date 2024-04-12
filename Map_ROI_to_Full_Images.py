#################################################################################################
# Check the size between ROI and full image to determine whether there are any mismatch or not  #
# If YES, resize the ROI size to the corresponding full image size                              #
# Map ROI to the full image                                                                     #
#################################################################################################

# Load modules
import os
import re
import pandas as pd
from PIL import Image

##################################################
# Extract the image names of ROI and full images #
##################################################

# Define the function for the extraction
def extract_names_from_folder(folder_path):
    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Define a regular expression pattern to match the desired parts of the filename
    pattern = re.compile(r"Mass-(.*?)_(MLO|CC)(.*?)\.png")
    # Create a list to store the extracted data
    extracted_data = []

    # Iterate over the files and extract the data
    for filename in files:
        match = pattern.search(filename)
        if match:
            # extract the content
            name = f"Mass-{match.group(1)}_{match.group(2)}"
            last_number = re.findall(r'\d+', match.group(3))[-1]
            extracted_data.append({
                'name': name,
                '1or2': last_number,
                'path': os.path.join(folder_path, filename)
            })

    return extracted_data

# Specify the paths of the two folders to compare, feel free to edit the directory
folder1_path = "/content/roi_train_needed"
folder2_path = "/content/full_train_needed"

# Extract data from both folders and convert them to dataframe
data_in_folder1 = extract_names_from_folder(folder1_path)
data_in_folder2 = extract_names_from_folder(folder2_path)
df_folder1 = pd.DataFrame(data_in_folder1)
df_folder2 = pd.DataFrame(data_in_folder2)

# Merge the DataFrames on the 'name' column to find common names
common_data_df = pd.merge(df_folder1, df_folder2, on='name', suffixes=('_ROI', '_Full'), how='left')

##########################################################################
# Check if there is any size mismatch between ROI and full images or not #
##########################################################################

# List to store information about mismated images
mismated_info = []

for index, row in common_data_df.iterrows():
    roi_path = row['path_ROI']
    full_path = row['path_Full']
    name = row['path_ROI'].split('/')[-1]
    name = name.replace("_content_mass_test_failed_", "")

    # Open ROI and full images
    roi_image = Image.open(roi_path)
    full_image = Image.open(full_path)

    # Resize the roi image to meet the full image size
    if roi_image.size != full_image.size:
        mismated_info.append((name, full_image.size, roi_image.size))  # Save mismated image info
        print(f"Found one difference in sizes for {name}.")

print("All images successfully processed.")

# Save mismated images information to CSV
if mismated_info:
    df_mismated = pd.DataFrame(mismated_info, columns=['Image Name', 'Full Image Size', 'ROI Size'])
    df_mismated.to_csv('mismated_image_info.csv', index=False)
    print("Mismated images information saved to mismated_images_info.csv.")
else:
    print("No mismated images found.")


##############################
# Map ROI to the full images #
##############################

for index, row in common_data_df.iterrows():
    roi_path = row['path_ROI']
    full_path = row['path_Full']
    name = row['path_ROI'].split('/')[-1]
    name = name.replace("_content_mass_test_failed_", "")

    # Open ROI and full images
    roi_image = Image.open(roi_path)
    full_image = Image.open(full_path)

    # Convert ROI image to RGBA
    roi_image = roi_image.convert("RGBA")
    datas = roi_image.getdata()
    new_data = []

    for i in range(len(datas)):
        item = datas[i]
        if item[0] == 255:
            new_data.append((255, 255, 255, 0))  # Set white pixels to transparent
        else:
            new_data.append(item)

    roi_image.putdata(new_data)

    # Resize the roi image to meet the full image size
    if roi_image.size != full_image.size:
        mask_size = full_image.size
        roi_image = roi_image.resize(mask_size)
        print(f"found one difference sizes {name} ")
    else:
        mask_size = full_image.size

    # Set the mask position
    mask_position = (0, 0)  # Top-left corner

    # Apply the ROI image as a mask to the full image.
    full_image.paste(roi_image, mask_position, roi_image)

    # Prepare the save path with the corresponding number of underscores
    save_path = f"/content/train_same_size/{name}"

    # Save the processed full image
    full_image.save(save_path)
    # print(f"Mapped image {name} saved.")

    # Print the number of images generated for each iteration
    # print(f"Image {name} saved.")
print("All images successfully processed")
