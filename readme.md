
# Deep Learning Models for Brain Tumour Classification

## Project Overview
This project focuses on classifying brain MRI scans to detect the presence and type of brain tumours using deep learning. It leverages a ResNet50-based Convolutional Neural Network (CNN) for image classification, trained and fine-tuned on a dataset of brain MRI images.

## Dataset
The dataset used for this project is not directly included in this repository due to its large size. It consists of various brain MRI images categorized by tumour type (e.g., glioma, meningioma, pituitary) and 'no tumour' cases. The dataset can be found at: **[https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?resource=download
]**

## Models
This repository includes two trained Keras models:
- `resnet50_best.h5`: A checkpoint of the ResNet50 model saved during training, representing the best performing model based on validation accuracy.
- `resnet50_finetuned_mri.h5`: The final fine-tuned ResNet50 model after the complete training process.

These models are managed using Git Large File Storage (LFS) to handle their size efficiently.

## How to Use
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sahilgetit/Deep-Learning-Models-for-Brain-Tumour-Classification.git
    ```
2.  **Open the `capstone.ipynb` notebook** in Google Colab or any Jupyter environment.
3.  **Ensure Git LFS is installed and configured** in your environment if you wish to work with the model files locally.
4.  **Download the dataset** from the link provided above and place it in the appropriate directory (as expected by the notebook, typically `/content/extracted`).
5.  **Run the cells sequentially** to preprocess data, train the model, and evaluate its performance.

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- scikit-learn
- matplotlib
- numpy
- Git LFS (for large model files)

