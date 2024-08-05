# An Open Codebase to Empower Transparency and Reproducibility in Machine Learning-Based Breast Cancer Diagnosis Using the CBIS-DDSM Dataset

## Overview

we provided a pilot codebase covering the entire  process from image preprocessing to model development and evaluation pipeline, utilizing the publicly available Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM) mass subset, including both full images and regions of interests (ROIs). We identified that increasing the input size improves correctly detection of malignant cases, while results for benign cases remains relatively stable across sizes within each set of models.


Below conatins the overview of the application of the CBIS-DDSM mass subset for breast cancer diagnosis:

<div style="text-align: center;">
  <img width="970" alt="image" src="https://github.com/lingliao/Transparency-in-CABCDTD/assets/91222367/93f7aa76-4a39-4534-be60-ba14a795155f">
</div>


## Dataset Availability
The data utilized in this study is downloaded from [here]( https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22516629#2251662935562334b1e043a3a0512554ef512cad).

Full image size and the cropped area per 598 by 598 pixels are plotted as below:

<div style="text-align: center;">
  <img width="900" alt="image" src="https://github.com/lingliao/Transparency-in-CABCDTD/assets/91222367/b191aeb1-d923-43bb-965c-58ac206a0d2c">
</div>

## Methods
In general, our methods include: 1) converting DICOM to PNG format without altering bit depth, 2) mapping ROIs to corresponding full images to identify and crop abnormal areas while ensuring size congruence, 3) confirming sufficient crop size coverage for most abnormal regions, 4) appending cropped images to the preliminary target **598 × 598** pixels with centered abnormal areas and removal of unwanted backgrounds, 5) performing data augmentation for enhanced diversity, 6) processing and splitting images into training, validation, and testing sets for model development, 7) optimizing computational efficient Xception network for model development, and 8) assessing effectiveness using multiple matrices and visualizations.

Steps to run the code we provided for model development:

1. Convert_DICOM_to_PNG.py
2. Map_ROI_to_Full_Images.py
3. Cropping.py
4. Remove_white_edges.py
5. Size_adjustment.py
6. Augmentation.py
7. pathology.py
8. model_development_and_evaluation.py (You will want to replace the "model_epoch_37.pth" with your best trained result)
9. OR, you could directly evaluate our model's performence with model_evaluation.py and our best saved check point "model_epoch_37.pth"

Steps to run the code for Figure 1B and 1C:
1. make sure you run the first 5 steps of the above lists
2. full_image_size&cropped_area.py
3. R_visualization.R

## Results
The processed images, ready for the model, can be found [here](https://drive.google.com/file/d/1-l-IX4asVuwokRDvzOCYH5Hyj2_--Lx0/view?usp=sharing).

The model's performance evaluation is based on the checkpoint with the highest validation accuracy.

The best performed checkpoint can be downloaded from [here](https://drive.google.com/drive/folders/18RxhTm9Oxak1dA0d2xihnHWekmEVug6h?usp=sharing)(Please note, this one performs a little better than the results presented in our publication in general, detailed in our folder example_output, but we decided to respect our published version and didn't make further update in the manuscript.).

Accuracy, precision, recall, F1 score, and a confusion matrix are outlined in below:


<div style="text-align: center;">
  <img width="900" alt="image" src="https://github.com/user-attachments/assets/87756d60-334e-4f53-9e50-9c8ba8ebb928">
</div>


## Example output
Our example outputs, including data processing, data visualization and model development and evaluation, are saved in folder example_output.

## Identified images with unwanted white edges
Identified cropped images with unwanted white backgrounds are stored in folder image_with_white_edge.

## Others
Other outputs that the authors think might be helpful can be found under folder Others, including the calculated full image size, percentage per 598 by 598 areas for corpped images, pathology info, identified mismatch info in sizewize between ROI and full images.

(598_percentage_all.csv,
all_mass_pathology.csv,
heaght_width_FULL.csv,
mismated_test_image_info(original).csv,
and
mismated_train_image_info(original).csv.)

## We appreciate you attention.

**AI for IMPROVEMENT and for EVERYONE.**
