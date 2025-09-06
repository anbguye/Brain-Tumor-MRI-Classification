# Brain Tumor MRI Classification

A comprehensive Jupyter notebook project for classifying brain tumor MRI scans into four categories: Glioma, Meningioma, Pituitary, or No Tumor.

## Overview

This project demonstrates end-to-end deep learning for medical image classification, featuring:
- Data exploration and preprocessing
- Two neural network architectures (transfer learning and custom CNN)
- Model training, evaluation, and visualization
- Interactive web application with AI-powered explanations
- Deployment capabilities

## Features

- **Data Analysis**: Exploratory data analysis with class distribution and sample visualization
- **Model Training**: Train and compare Xception (transfer learning) and custom CNN models
- **Evaluation**: Comprehensive metrics, confusion matrices, and classification reports
- **Web App**: Streamlit interface for real-time classification with saliency maps
- **AI Explanations**: Google Gemini integration for model decision explanations
- **Deployment**: Ngrok tunneling for public access

## Dataset

The project uses the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle, containing:
- Glioma tumors
- Meningioma tumors
- Pituitary tumors
- No tumor (healthy scans)

## Requirements

- Python 3.7+
- Jupyter Notebook or Google Colab
- Required packages:
  - tensorflow
  - keras
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - opencv-python
  - pillow
  - streamlit
  - google-generativeai
  - python-dotenv
  - pyngrok
  - plotly

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/anbguye/Brain-Tumor-MRI-Classification.git
   cd Brain-Tumor-MRI-Classification
   ```

2. **Install dependencies**:
   ```bash
   pip install tensorflow pandas numpy matplotlib seaborn scikit-learn opencv-python pillow streamlit google-generativeai python-dotenv pyngrok plotly
   ```

3. **Set up API keys**:
   - Create a `.env` file in the project root
   - Add your Google Gemini API key: `GOOGLE_API_KEY=your_api_key_here`
   - Add your Ngrok token: `NGROK_AUTH_TOKEN=your_token_here`

## How to Run

### Option 1: Google Colab (Recommended)

1. Upload `Brain_Tumor_Classification.ipynb` to Google Colab
2. The notebook includes automatic dataset download via Kaggle API
3. Run all cells sequentially
4. The Streamlit app will launch with a public URL via Ngrok

### Option 2: Local Environment

1. Download the dataset from Kaggle and extract to `Training/` and `Testing/` folders
2. Open `Brain_Tumor_Classification.ipynb` in Jupyter Notebook
3. Update file paths in the notebook if necessary
4. Run the cells to train models and launch the web app

## Project Structure

The notebook is divided into two main parts:

### Part 1: Model Development
- Dataset loading and preprocessing
- Data augmentation and generators
- Xception model training and evaluation
- Custom CNN model training and evaluation
- Model comparison and visualization

### Part 2: Web Application
- Streamlit app code for interactive classification
- Saliency map generation
- Google Gemini integration for explanations
- Ngrok deployment

## Model Architectures

### Xception (Transfer Learning)
- **Input**: 299x299 RGB images
- **Base Model**: Xception pretrained on ImageNet
- **Top Layers**: Flatten → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.25) → Dense(4, Softmax)
- **Training**: Adamax optimizer, 0.001 learning rate, 5 epochs

### Custom CNN
- **Input**: 224x224 RGB images
- **Architecture**:
  - Conv2D(512) → MaxPool → Conv2D(256) → MaxPool → Dropout(0.25)
  - Conv2D(128) → MaxPool → Dropout(0.25) → Conv2D(64) → MaxPool
  - Flatten → Dense(256, L2 reg) → Dropout(0.35) → Dense(256, L2 reg) → Dropout(0.35) → Dense(4, Softmax)
- **Training**: Adamax optimizer, 0.001 learning rate, 5 epochs

## Results

Both models achieve high accuracy on the test set with comprehensive evaluation metrics including precision, recall, and confusion matrices.

## Web Application Features

- Upload MRI images for classification
- Real-time prediction with confidence scores
- Saliency maps highlighting model focus areas
- AI-generated explanations using Google Gemini
- Interactive visualizations with Plotly

## License

This project uses the CC0-1.0 license.
