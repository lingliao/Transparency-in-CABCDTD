###################################
# Save the pathology information
###################################

# Load module
import pandas as pd
import re

# Read in the description data
train = pd.read_csv('/content/drive/MyDrive/CBIS-DDSM/mass_case_description_train_set.csv')
test = pd.read_csv('/content/drive/MyDrive/CBIS-DDSM/mass_case_description_test_set.csv')

# Extract columns we need
train_need = train[['patient_id', 'pathology', 'image file path', 'cropped image file path', 'ROI mask file path']]
test_need = test[['patient_id', 'pathology', 'image file path', 'cropped image file path','ROI mask file path']]

# Merge and reindex the dataframe
merged_df = pd.concat([train_need, test_need], axis=0)
merged_df.reset_index(drop=True, inplace=True)

# Extract the names
merged_df['crop_name'] = merged_df['cropped image file path'].apply(lambda x: x.split('/')[0])
merged_df['ROI_name'] = merged_df['ROI mask file path'].apply(lambda x: x.split('/')[0])
merged_df['full_name'] = merged_df['image file path'].apply(lambda x: x.split('/')[0])

# Save the output
merged_df.to_csv('all_mass_pathology.csv', index=False)
