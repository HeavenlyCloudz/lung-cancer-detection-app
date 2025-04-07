# ONCO AI, a lung cancer detection Streamlit app

This is a Streamlit application for detecting lung cancer using a convolutional neural network (CNN), ONCO AI. The app allows users to upload lung images and receive predictions on whether the images are cancerous or non-cancerous.

## Table of Contents
- Introduction
- Features
  - Image Upload
  - Real-time Predictions
  - Model Training
  - Visualization
  - Testing
- Requierements

## Introduction

This application leverages a convolutional neural network (CNN) to classify lung images as either cancerous or non-cancerous. 
It is built using Streamlit, making it user-friendly and accessible for users without a technical background. It also implements Grad CAM, a fascinating feature that displays the model's output.

## Features

- **Image Upload**: Users can upload lung cancer CT scans (JPG, PNG) for analysis.
- **Real-time Predictions**: The app provides predictions based on the uploaded images. These predictions are binary; cancerous or non-cancerous, as well as additional classes of malignant, adenocarcinoma, squamous cell carcinoma, and large cell carcinoma (cancerous), while on the opposite side of the spectrum is normal and benign (non-cancerous) 
- **Model Training**: Users can train the model using the dataset provided, and see the results with an accompanying graph.
- **Visualization**: The app includes Grad-CAM to visualize model predictions.
- **Testing**: Users can test the trained model on a separate dataset that the model has not been exposed to, and see the results of the model with a graph.


## Requierements

- Python 3.7 or higher
- pip
- Streamlit
