# AI Skin Disease Detection with Visual Highlight

An AI-powered desktop application developed using **Python, OpenCV, Scikit-learn, NumPy, and Tkinter** that detects skin diseases from images and visually highlights the affected region. The application uses image processing and machine learning techniques to assist in the early identification of common skin diseases.

## Project Overview

The AI Skin Disease Detection system is designed to analyze skin images and classify them into different disease categories using Machine Learning. The application preprocesses uploaded images, extracts features using HOG (Histogram of Oriented Gradients), predicts the disease using a trained model, and highlights the infected area to improve visualization.

## Features

### Image Upload

- Upload skin images from the local system
- Supports common image formats (JPG, PNG, JPEG)
- Easy-to-use graphical interface

### Image Preprocessing

- Image resizing
- Noise reduction
- Color conversion
- Image normalization

### Feature Extraction

- Histogram of Oriented Gradients (HOG)
- Extracts important texture features
- Improves disease classification accuracy

### Disease Prediction

- Predicts skin disease using a trained Machine Learning model
- Displays prediction result with confidence
- Fast and accurate classification

### Visual Highlight

- Highlights the affected skin region
- Generates processed output image
- Saves the highlighted image in the outputs folder

### Model Training

- Train the model using custom datasets
- Save trained model using Joblib
- Reuse trained model for prediction

## Technology Stack

- Python
- OpenCV
- NumPy
- Scikit-learn
- Scikit-image

## Project Modules

### Dataset Module

- Training dataset
- Testing dataset
- Disease image collection

### Training Module

- Image preprocessing
- HOG feature extraction
- Model training
- Model saving

### Prediction Module

- Load trained model
- Upload image
- Disease prediction
- Display result

### Output Module

- Highlight infected region
- Save processed image
- Display prediction result

## Project Workflow

```text
           Upload Skin Image
                   │
                   ▼
        Image Preprocessing
                   │
                   ▼
      Feature Extraction (HOG)
                   │
                   ▼
      Load Trained ML Model
                   │
                   ▼
        Disease Prediction
                   │
                   ▼
     Highlight Affected Region
                   │
                   ▼
      Display & Save Result
```

## Future Enhancements

- Deep Learning (CNN) based prediction
- Support for additional skin diseases
- Mobile application
- Cloud-based disease detection
- Real-time camera detection
- Disease severity analysis
- Doctor recommendation system
- Patient report generation

## Author

**G O Chandana**

Department of Computer Science and Engineering

SJB Institute of Technology, Bengaluru
