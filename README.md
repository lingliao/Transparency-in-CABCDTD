# Transparency in Computer-Aided Breast Cancer Diagnosis Tool Development: A CBIS-DDSM Case Study

Ling Liao, 

M.S., Precision Medicine and Healthcare, Tsinghua University

PHD student, Computational and Systems Biology, Washington University in St. Louis

Founder, Biomedical Deep Learning LLC

Our methodology outlines in this case study demonstrates robust performance in classifying benign and malignant breast mammography images, achieving an AUROC of 86.3% [95% CI: 0.842, 0.885] using the CBIS-DDSM mass dataset. Additionally, we provide all the code we employed for the entire process, spanning from data preprocessing to model evaluation. As the old saying goes, the devil is in the details. A complete and annotated codebase is essential for guaranteeing the reproducibility, transparency, effectiveness, and overall reliability of our model, thereby further advancing research communication and improvement in this field.

Here is the overview of the application of the CBIS-DDSM mass subset for breast cancer diagnosis:

<div style="text-align: center;">
  <img width="970" alt="image" src="https://github.com/lingliao/Transparency-in-CABCDTD/assets/91222367/93f7aa76-4a39-4534-be60-ba14a795155f">
</div>


## Dataset Availability
The data utilized in this study is downloaded from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22516629#2251662935562334b1e043a3a0512554ef512cad

Full image size and the cropped area per 598 by 598 pixels are plotted as below:

<div style="text-align: center;">
  <img width="900" alt="image" src="https://github.com/lingliao/Transparency-in-CABCDTD/assets/91222367/b191aeb1-d923-43bb-965c-58ac206a0d2c">
</div>

## Methods
In general, our methods include: 1) converting DICOM to PNG format without altering bit depth, 2) mapping ROIs to corresponding full images to identify and crop abnormal areas while ensuring size congruence, 3) confirming sufficient crop size coverage for most abnormal regions, 4) appending cropped images to the desired 598 × 598 pixels with centered abnormal areas and removal of unwanted backgrounds, 5) performing data augmentation for enhanced diversity, 6) processing and splitting images into training, validation, and testing sets for model development, 7) optimizing computational efficient Xception network for model development, and 8) assessing effectiveness using multiple matrices and visualizations.

Steps to run the code we provided for model development:

1. Convert_DICOM_to_PNG.py
2. Map_ROI_to_Full_Images.py
3. Cropping.py
4. Remove_white_edges.py
5. Size_adjustment.py
6. Augmentation.py
7. pathology.py
8. model_development_and_evaluation.py

Steps to run the code for Figure 1B and 1C:
1. make sure you run the first 5 steps of the above lists
2. full_image_size&cropped_area.py
3. R_visualization

## Results
The model's performance evaluation is based on the checkpoint with the highest validation accuracy.

Accuracy, precision, recall, F1 score, ROC curve, and a confusion matrix are outlined in below:



<div style="text-align: center;">
  <img width="970" alt="image" src="https://github.com/lingliao/Transparency-in-CABCDTD/assets/91222367/447270ad-168a-4e0f-8a6a-eb9ad52fc4ba">
</div>


