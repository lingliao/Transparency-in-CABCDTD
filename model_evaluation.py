####################################
# Model evaluation #
####################################

################################
# Image load and visualization #
################################

# Install packages/hubs/ others needed
pip install pydicom

# Load modules
import os
import pydicom
from PIL import Image
import shutil
import cv2
import torch
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import matplotlib.image as mpimg
import random
from keras.utils import to_categorical
from concurrent.futures import ThreadPoolExecutor
from glob import glob

# Read in the pathology info
dicom_data = pd.read_csv('all_mass_pathology.csv')

# Read in the image path info
# Specify the root path where the .png files are located
jpg_root_path = 'all_598_augmented'

# Function to get all .png file paths in a directory
def get_jpg_file_paths(directory):
    jpg_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                jpg_paths.append(os.path.join(root, file))
    return jpg_paths

# Get all .png file paths under the specified directory
jpg_paths = get_jpg_file_paths(jpg_root_path)

# Create a DataFrame with the file paths
df = pd.DataFrame({'File_Paths': jpg_paths})

# Extracting the information from the file paths and matching with dicom_data
df['ID1'] = df['File_Paths'].apply(lambda x: x.split('/')[-1] if len(x.split('/')) > 1 else '')
dicom_data['ID1'] = dicom_data['image file path'].apply(lambda x: x.split('/')[-4] if len(x.split('/')) > 1 else '')

# Assuming unique_df and jpg_paths_df are your DataFrames
for index, row in dicom_data.iterrows():
    # Check if 'image_file_path_first_part' is in 'image_file_path' of jpg_paths_df
    mask = df['ID1'].str.contains(row['ID1'])
    # If there is a match, update 'pathology' in jpg_paths_df
    df.loc[mask, 'pathology'] = row['pathology']
    # df.loc[mask, 'full_image_name'] = row['image_file_path_first_part']

#remove the missing rows
df.dropna(subset=['pathology'], inplace=True)

# Save the df to a CSV file as all.csv
df.to_csv('all.csv', index=False)

###################################
# Label the pathology information #
###################################

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Map labels to binary values
label_mapping = {'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0, 'MALIGNANT': 1}
df['label'] = df['pathology'].map(label_mapping)

# Get the images before augmentation
def check_filename(filename):
    basename = os.path.basename(filename)
    return basename.endswith('1-1.png') or basename.endswith('1-2.png')

# Apply the function to filter rows
original_df = df[df['File_Paths'].apply(check_filename)]

###################
# Build the model #
###################

# Install tensorflow-hub and timm
pip install tensorflow-hub
pip install timm

# Model structure
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # Define the additional convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1),  # Change input channels to 1
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1),
            nn.Conv2d(3, 3, kernel_size=1),  # Change to 3 output channels to match Xception input
            nn.LeakyReLU(0.1)
        )
        
#         # Load the pretrained Xception model
#         self.feature_extractor = timm.create_model('legacy_xception', pretrained=True)
        
#         # Remove the last two layers of the Xception model
#         self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-2])


        # load resnet pretrained model - feel free to adjust based on your needs
        self.feature_extractor = models.resnet50(weights=ResNet50_Weights.DEFAULT)

          # Remove the last fully connected layer of ResNet-50
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])

#         self.feature_extractor = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        
#         # Remove the last fully connected layer of DenseNet-121
#         self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
        
        # Add the desired layers
        self.global_pooling = nn.AdaptiveAvgPool2d(1)  # Global average pooling

        self.fc1 = nn.Linear(2048, 512) # For resnet50 and Xception

        self.relu = nn.ReLU()  # Changed LeakyReLU to ReLU
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        # Forward pass through the convolutional layers
        x = self.conv_layers(x)
        # print("After conv_layers shape:", x.shape)
        
        # Forward pass through the feature extractor
        x = self.feature_extractor(x)
        
        # Forward pass through the added layers
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)  # Moved dropout after ReLU in fc2
        x = self.fc3(x)
        x = torch.sigmoid(x)
        
        return x

# Create an instance of the model
model = CustomModel()

# Print model summary
# print(model)

##########################
# Image data processing  #
##########################

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]  # Assuming the image paths are in the first column
        image = Image.open(img_name)  # Open image without converting to 'L'
        image = np.array(image)  # Convert image to numpy array
        image = image.astype(np.float32)  # Convert to float32
        
        label = self.data.iloc[idx, 3]  # Assuming the labels are in the fourth column
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
  
# Define the transformation pipeline - adjust based on which data you plan to use
transform_ = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Lambda(lambda x: x / 65535.0),  # Normalize tensor by dividing by 65535.0
    # transforms.CenterCrop(299),
    # transforms.Resize((299, 299))  # Ensure the final size is 448x448
    # transforms.CenterCrop(448),  # Center crop the image to 448x448
    # transforms.Resize((448, 448))  # Ensure the final size is 448x448
    transforms.CenterCrop(224),
    transforms.Resize((224, 224))  # Ensure the final size is 448x448

])

# Training, validation, and test spliting

from sklearn.model_selection import train_test_split

train_data, temp_data = train_test_split(original_df, test_size=0.2, random_state=42)

# Function to get the augmented ones for training dataset.
def replicate_rows(df, n_replicates):
    # Create an empty list to collect new rows
    new_rows = []
    
    # Iterate over each row in the original DataFrame
    for _, row in df.iterrows():
        # Add the original row once
        new_rows.append(row)
        
        # Add replicated rows with modified File_Paths
        for i in range(1, n_replicates + 1):
            new_row = row.copy()
            new_row['File_Paths'] = new_row['File_Paths'].replace('.png', f'_{i}.png')
            new_rows.append(new_row)
    
    # Convert list of rows to DataFrame
    return pd.DataFrame(new_rows)

# Get all train data
train_data_augmented = replicate_rows(train_data, 5)

val_data, test_data = train_test_split(temp_data, test_size=1/2, random_state=42)

# Create datasets and dataloaders for train, validation, and test sets
train_dataset_ = CustomDataset(train_data_augmented, transform=transform_)
train_loader_ = DataLoader(train_dataset_, batch_size=32, shuffle=True)

val_dataset = CustomDataset(val_data, transform=transform_)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # No need to shuffle validation data

test_dataset = CustomDataset(test_data, transform=transform_)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # No need to shuffle test data

####################
# Model evaluation #
####################

######################
#1. Calculate the accuracy, precision, recall, AUROC, and F1 score

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

# # Assuming you have defined your device as 'cuda' if available, else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the best model state dictionary based on the validation accuracy
model.load_state_dict(torch.load('model_epoch_32.pth')) # You will want to update this based on your training checkpoints' performances

# Set the model to evaluation mode
model.eval()

# Lists to store true labels and predicted probabilities
true_labels = []
predicted_probs = []

# Loop through the test dataset
with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_images, test_labels = test_images.to(device), test_labels.to(device)

        # Forward pass
        outputs = model(test_images)

        # Calculate predicted probabilities
        predicted_probs.extend(outputs.cpu().numpy())

        # Convert labels to numpy array and append to true_labels list
        true_labels.extend(test_labels.cpu().numpy())

# Convert true_labels and predicted_probs to numpy arrays
true_labels = np.array(true_labels)
predicted_probs = np.array(predicted_probs)

# Calculate predicted labels
predicted_labels = (predicted_probs > 0.5).astype(int)

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
roc_auc = roc_auc_score(true_labels, predicted_probs)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")

######################
#2. calculate 95 CI%

# Define the number of iterations for bootstrap
n_iterations = 1000  # Can be adjusted as needed

# Define lists to store evaluation metric values after bootstrap sampling
accuracy_bootstrap = []
precision_bootstrap = []
recall_bootstrap = []
f1_bootstrap = []
roc_auc_bootstrap = []

# Perform bootstrap resampling and calculate evaluation metric values
for _ in range(n_iterations):
    # Generate bootstrap samples
    bootstrap_indices = np.random.choice(len(true_labels), size=len(true_labels), replace=True)
    bootstrap_true_labels = true_labels[bootstrap_indices]
    bootstrap_predicted_labels = predicted_labels[bootstrap_indices]
    bootstrap_predicted_probs = predicted_probs[bootstrap_indices]

    # Calculate evaluation metric values
    bootstrap_accuracy = accuracy_score(bootstrap_true_labels, bootstrap_predicted_labels)
    bootstrap_precision = precision_score(bootstrap_true_labels, bootstrap_predicted_labels)
    bootstrap_recall = recall_score(bootstrap_true_labels, bootstrap_predicted_labels)
    bootstrap_f1 = f1_score(bootstrap_true_labels, bootstrap_predicted_labels)
    bootstrap_roc_auc = roc_auc_score(bootstrap_true_labels, bootstrap_predicted_probs)

    # Append evaluation metric values to the corresponding lists
    accuracy_bootstrap.append(bootstrap_accuracy)
    precision_bootstrap.append(bootstrap_precision)
    recall_bootstrap.append(bootstrap_recall)
    f1_bootstrap.append(bootstrap_f1)
    roc_auc_bootstrap.append(bootstrap_roc_auc)

# Calculate the upper and lower bounds of the 95% CI
accuracy_ci_bootstrap = np.percentile(accuracy_bootstrap, [2.5, 97.5])
precision_ci_bootstrap = np.percentile(precision_bootstrap, [2.5, 97.5])
recall_ci_bootstrap = np.percentile(recall_bootstrap, [2.5, 97.5])
f1_ci_bootstrap = np.percentile(f1_bootstrap, [2.5, 97.5])
roc_auc_ci_bootstrap = np.percentile(roc_auc_bootstrap, [2.5, 97.5])

# Print the 95% CI results calculated using the bootstrap method
print(f"95% CI for Accuracy (Bootstrap): {accuracy_ci_bootstrap}")
print(f"95% CI for Precision (Bootstrap): {precision_ci_bootstrap}")
print(f"95% CI for Recall (Bootstrap): {recall_ci_bootstrap}")
print(f"95% CI for F1 Score (Bootstrap): {f1_ci_bootstrap}")
print(f"95% CI for ROC AUC (Bootstrap): {roc_auc_ci_bootstrap}")

######################
#3. plot confusion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

conf_matrix # Print the confusion matrix in number as well.

# Calculate row-wise sums
row_sums = conf_matrix.sum(axis=1, keepdims=True)

# Calculate percentages based on row sums
conf_matrix_percent_row = conf_matrix / row_sums * 100

# Create annotations with both count and percentage values
annotations = [f"{conf_matrix[i, j]:d}\n({conf_matrix_percent_row[i, j]:.2f}%)" for i in range(conf_matrix.shape[0]) for j in range(conf_matrix.shape[1])]
annotations = np.array(annotations).reshape(conf_matrix.shape[0], conf_matrix.shape[1])

# Define custom color palette with starting color #6da9ed and ending color #eb6a4d
#colors = sns.diverging_palette(100, 5, s=80, l=60, as_cmap=True)
colors = sns.diverging_palette(80, 5, s=70, l=80, as_cmap=True)


# Plot confusion matrix with annotations and custom colors
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=annotations, fmt="", cmap=colors, cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

# Add custom labels
plt.xticks(ticks=[0.5, 1.5], labels=["Benign", "Malignant"])
plt.yticks(ticks=[0.5, 1.5], labels=["Benign", "Malignant"])

plt.savefig('confusion_matrix.pdf', dpi=1000)
plt.show()

######################
#4. plot ROC curve

from sklearn.metrics import roc_curve, auc

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

# Set Seaborn style and color palette
sns.set_style("white")
sns.set_palette(["#e090b5", "#e6cd73"])

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Save the plot as PDF with 1000dpi
plt.savefig('roc_curve.pdf', dpi=1000)
plt.show()

