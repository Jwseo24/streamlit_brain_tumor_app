# Brain Tumor MRI Classification App 🧠

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)
[![EfficientNet](https://img.shields.io/badge/EfficientNet-B0-green.svg)](https://keras.io/api/applications/efficientnet/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Latest-blue.svg)](https://opencv.org/)

A Streamlit web application that uses deep learning models to classify brain tumor MRI scans into four different categories: glioma, meningioma, no tumor, and pituitary tumor.

## Project Overview 🔍

This project implements two different deep learning models:
1. Custom CNN Architecture
2. EfficientNetB0 (Transfer Learning)

Both models are trained on the Brain Tumor MRI Dataset to classify brain MRI scans into four categories.

## Models 🤖

### 1. Custom CNN Model
- Custom-designed CNN architecture with multiple convolutional blocks
- Uses data augmentation for better generalization
- Implements dropout layers to prevent overfitting
- Trained with categorical cross-entropy loss

### 2. EfficientNetB0 Model
- Pretrained on ImageNet dataset
- Implemented with transfer learning approach
- Uses focal loss to handle class imbalance
- Two-phase training:
  - Phase 1: Train only the classification head
  - Phase 2: Fine-tune the deeper layers

## Dataset 📊

The models are trained on the Brain Tumor MRI Dataset from Kaggle, which includes:
- Four classes: glioma, meningioma, no tumor, and pituitary
- Training and testing splits
- MRI scans in various orientations

## Requirements 📝

```
streamlit
tensorflow
numpy
pandas
Pillow
```

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Files in the Project 📁

1. `app.py` - Main Streamlit application file
   - Handles image upload and preprocessing
   - Loads both models
   - Displays predictions and confidence scores
   - Provides side-by-side model comparison

2. `Brain_tumor.ipynb` - Jupyter notebook containing:
   - Model development and training code
   - Data preprocessing pipeline
   - Model evaluation and comparison
   - Visualization of results

3. `final_brain_tumor_cnn.keras` - Trained Custom CNN model

4. `best_effnet.keras` - Trained EfficientNetB0 model

## How to Run 🚀

1. Clone the repository:
```bash
git clone https://github.com/Mohit10133/streamlit_brain_tumor_app.git
cd streamlit_brain_tumor_app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Using the App 💻

1. Open the app in your browser (typically at `http://localhost:8501`)
2. Upload a brain MRI scan image
3. Both models will analyze the image
4. View predictions and confidence scores for each model
5. Compare the results side by side

## Model Performance 📈

The application provides:
- Real-time predictions
- Confidence scores for each class
- Side-by-side model comparison
- Visualization of prediction probabilities

## Technical Details 🔧

### Image Preprocessing
- Images are resized to 224x224 pixels
- Proper channel handling (RGB/Grayscale)
- Pixel value normalization
- Model-specific preprocessing pipelines

### Model Architecture
- Custom CNN: Multiple convolutional blocks with batch normalization
- EfficientNetB0: Pretrained ImageNet weights with custom classification head

## Contributors 👥

- [Mohit10133](https://github.com/Mohit10133)

## License 📄

This project is open source and available under the [MIT License](LICENSE).
