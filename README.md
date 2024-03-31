# Transparency in Computer-Aided Breast Cancer Diagnosis Tool Development: A CBIS-DDSM Case Study

Ling Liao

M.S., Precision Medicine and Healthcare, Tsinghua University

PHD student, Computational and Systems Biology, Washington University in St. Louis

Founder, Biomedical Deep Learning LLC

<div style="text-align: center;">
  <img width="970" alt="image" src="https://github.com/lingliao/Transparency-in-CABCDTD/assets/91222367/93f7aa76-4a39-4534-be60-ba14a795155f">
</div>

## Main
In this study, a case study was conducted utilizing the mass subset from CBIS-DDSM, a high-volume publicly available mammography dataset. This subset comprises 1,696 abnormal ROIs and 1,592 corresponding full images obtained from 892 patients. An overview of the application of the CBIS-DDSM mass subset for breast cancer diagnosis is presented as above.

## Dataset Availability
The data utilized in this study is downloaded from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22516629#2251662935562334b1e043a3a0512554ef512cad

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

<div style="text-align: center;">
  <img width="970" alt="image" src="https://github.com/lingliao/Transparency-in-CABCDTD/assets/91222367/447270ad-168a-4e0f-8a6a-eb9ad52fc4ba">
</div>


