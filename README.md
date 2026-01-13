Here is a comprehensive README file for your GitHub repository, based on the architecture, training pipeline, and inference logic found in your code.

---

# Webcam-Based Upright Posture Detection System

This repository provides an end-to-end solution for detecting human sitting posture in real-time using a standard webcam. It utilizes **MediaPipe Pose** for landmark extraction and a hybrid **CNN-LSTM** deep learning model for high-accuracy classification.

## Overview

The project is designed to help users maintain a healthy posture while working at a computer. It identifies five distinct posture classes:

* **Upright** (Ideal)
* **Leaning Forward**
* **Leaning Backward**
* **Leaning Left**
* **Leaning Right**

## Model Architecture

The system employs a sophisticated neural network designed for high-precision (0.99+ target) classification of spatial-temporal pose data:

1. **Input Layer**: Accepts pose landmarks (X, Y, Z, and visibility).
2. **1D Convolutional Layers**: Extract spatial features and local patterns from landmark coordinates.
3. **LSTM Layers**: Capture temporal dependencies and the "flow" of movement.
4. **Dense Layers**: Deep feature integration with Batch Normalization and Dropout to prevent overfitting.
5. **Output Layer**: Softmax activation for 5-class classification.

## Project Structure

* `Working_Transfer_Learning_Training_27_09.ipynb`: The training pipeline including data augmentation, balanced dataset creation, and model checkpointing.
* `Working_Inference 4.ipynb`: The real-time application using OpenCV and MediaPipe for live posture feedback.
* `pose_landmarker_lite.task`: MediaPipe model asset for landmark detection.
* `high_accuracy_posture_model.h5`: Trained Keras model weights.
* `high_accuracy_scaler.pkl`: StandardScaler used for feature normalization.
* `high_accuracy_encoder.pkl`: LabelEncoder for posture categories.

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/posture-detection-system.git
cd posture-detection-system

```


2. **Install dependencies**:
```bash
pip install opencv-python mediapipe tensorflow pandas numpy scikit-learn joblib matplotlib tqdm

```



## Training Pipeline

The training notebook supports **Transfer Learning**, allowing you to fine-tune existing weights with new data.

* **Data Augmentation**: Enhances the dataset using Gaussian noise, scaling, and spatial shifting.
* **Class Balancing**: Automatically upsamples minority classes to create a high-quality 10,000-sample balanced dataset.
* **Optimizer**: Uses `AMSGrad` for better convergence over 200 epochs.
* **Callbacks**: Includes `EarlyStopping` and `ReduceLROnPlateau` for optimized training.

## Usage (Inference)

Run the inference notebook to start the live detection app.

1. **Initialize**: The app loads the model, scaler, and encoder.
2. **Calibration**: Sit in your ideal "upright" position and trigger calibration to set your baseline.
3. **Monitoring**: The system provides a live video feed with overlaid skeletal landmarks and real-time posture classification.
4. **Temporal Smoothing**: Uses a prediction history buffer to prevent flickering results.

```python
# Example snippet to run the detector
app = PostureDetectionApp(
    model_path='high_accuracy_posture_model_XXXXXXXXXX.h5',
    scaler_path='high_accuracy_scaler_XXXXXXXXXX.pkl',
    label_encoder_path='high_accuracy_encoder_XXXXXXXXXX.pkl'
)
app.run()

```

Performance Analysis

The training process generates comprehensive evaluation metrics:

Classification Reports: Precision, recall, and F1-score for every posture.
Confidence Analysis: Mean and max prediction confidence levels.
Learning Curves: Visualization of accuracy/loss over epochs.

License

This project is licensed under the MIT License - see the LICENSE file for details.



* **MediaPipe Pose Implementation**: Cites the use of landmark indices [0, 2, 5, 7, 8, 9, 10, 11, 12, etc.].
* **Deep Learning Details**: Reflects the `Conv1D`, `LSTM`, and `Dense` architecture found in your inference and training code.
* **Training Enhancements**: Highlights the use of `AMSGrad`, `EarlyStopping`, and 200-epoch targets specified in the training logs.
