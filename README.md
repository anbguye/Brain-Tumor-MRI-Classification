# Brain Tumor MRI Classification

A deep learning project that classifies brain tumor MRI scans into four categories: Glioma, Meningioma, Pituitary, or No Tumor.

## Features

- **Multiple Model Options**:
  - Transfer Learning with Xception
  - Custom CNN Architecture

- **Interactive Web Interface**:
  - Built with Streamlit
  - Real-time image upload and classification
  - Visualization of model predictions

- **Advanced Visualization**:
  - Saliency maps showing model focus areas
  - Probability distribution for all classes
  - AI-powered explanations of model decisions using Google's Gemini

## Dataset

The project uses the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle, which includes MRI scans of:
- Glioma tumors
- Meningioma tumors
- Pituitary tumors
- No tumor (healthy)

## Technical Stack

- **Deep Learning**: TensorFlow, Keras
- **Image Processing**: OpenCV, Pillow
- **Web App**: Streamlit
- **Visualization**: Plotly, Matplotlib, Seaborn
- **AI Explanations**: Google Gemini API
- **Deployment**: Ngrok for tunneling

## Setup and Installation

1. Install required packages:
pip install tensorflow opencv-python streamlit pillow google-generativeai python-dotenv


2. Set up environment variables:
- Create a `.env` file
- Add your Google API key for Gemini
- Add your Ngrok authentication token

3. Run the Streamlit app:
   streamlit run app.py


## Model Architecture

### Transfer Learning Model
- Base: Xception (pretrained on ImageNet)
- Additional layers:
  - Flatten
  - Dropout (0.3)
  - Dense (128, ReLU)
  - Dropout (0.25)
  - Dense (4, Softmax)

### Training
- Optimizer: Adamax (learning rate: 0.001)
- Loss: Categorical Crossentropy
- Metrics: Accuracy, Precision, Recall

## Features in Development
- Model performance metrics
- Batch prediction capabilities
- Enhanced visualization options
- Export of analysis reports

## License
This project uses the CC0-1.0 license.
