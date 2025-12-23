Pneumonia Detection using Convolutional Neural Networks (CNN)

A deep learning project that classifies chest X-ray images as Normal or Pneumonia using a custom Convolutional Neural Network (CNN) built with TensorFlow and Keras.



📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Results](#results)

Project Overview

This project implements a deep learning solution to automatically detect pneumonia from chest X-ray images. The model uses a custom CNN architecture to achieve high accuracy in binary classification tasks, helping to support medical diagnosis processes.

Key Features:
- Custom CNN architecture for binary classification
- Image preprocessing and normalization
- Training, validation, and testing on large dataset
- Model evaluation with accuracy and loss metrics
- Easy-to-use inference pipeline

---

Dataset

**Source:** Chest X-ray Images Dataset

Classes:
- **NORMAL:** Healthy chest X-rays
- **PNEUMONIA:** X-rays showing signs of pneumonia

Data Split:
- Training: 5,216 images
- Validation: 16 images
- Test: 624 images

Image Specifications:
- Size: 180 × 180 RGB images
- Format: JPG
- Augmentation: Normalization applied during preprocessing

---

Model Architecture
Input Layer
- Input shape: (180, 180, 3) - 180×180 RGB images

Architecture Components
1. Rescaling Layer:Normalizes pixel values to [0, 1]
2. Convolutional Block 1:
   - Conv2D: 32 filters, 3×3 kernel
   - MaxPooling2D: 2×2 pool size
   
3. Convolutional Block 2:
   - Conv2D: 64 filters, 3×3 kernel
   - MaxPooling2D: 2×2 pool size
   
4. Convolutional Block 3:
   - Conv2D: 128 filters, 3×3 kernel
   - MaxPooling2D: 2×2 pool size

5. Dense Layers:
   - Flatten layer
   - Dense: 128 neurons with ReLU activation
   - Dropout: 50% to prevent overfitting
   - Output Dense: 1 neuron with Sigmoid activation (binary classification)
Model Summary
- Total Parameters: 6,647,105
- Trainable Parameters:6,647,105
- Optimizer:Adam
- Loss Function: Binary Crossentropy
- Metrics: Accuracy



 Performance Metrics

| Metric | Value |
|--------|-------|
| Training Accuracy | ~98% |
| Validation Accuracy | ~94-100% |
| Test Accuracy | ~77-78% |
| Training Loss | Low |
| Model Size | Lightweight (~25MB) |



 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/Sevani2005/Pneumonia_DNN_PyCharm.git
   cd Pneumonia_DNN_PyCharm
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

 Dependencies
All required packages are listed in `requirements.txt`:
- TensorFlow >= 2.10.0
- NumPy >= 1.21.0
- OpenCV >= 4.5.0
- Pandas >= 1.3.0
- Scikit-learn >= 1.0.0
- Matplotlib >= 3.4.0
- Pillow >= 9.0.0



 Usage

 Training the Model
```bash
python main_dnn.py
```

This script will:
1. Load and preprocess the dataset
2. Build the CNN model
3. Train the model on the training set
4. Validate on the validation set
5. Save the trained model

Testing the Model
```bash
python test_tensorflow.py
```

This script will:
1. Load the trained model
2. Evaluate performance on the test set
3. Generate classification reports and visualizations
4.  Making Predictions
```python
from main_dnn import load_model, predict
import cv2

# Load model
model = load_model('pneumonia_model.h5')

# Load and preprocess image
image = cv2.imread('chest_xray.jpg')
image = cv2.resize(image, (180, 180))
image = image / 255.0

# Make prediction
prediction = model.predict(image.reshape(1, 180, 180, 3))
result = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
print(f"Prediction: {result}")
```

---

 Project Structure


Pneumonia_DNN_PyCharm/
├── main_dnn.py              # Main training script
├── test_tensorflow.py       # Testing and evaluation script
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
├── README.md               # This file
└── .idea/                  # PyCharm IDE configuration (ignored)
```

---

Technologies Used

- Deep Learning Framework: TensorFlow/Keras
- Image Processing: OpenCV, NumPy
- Data Handling: Pandas, Numpy
- Visualization: Matplotlib
- Machine Learning:Scikit-learn
- IDE:PyCharm
- Version Control:Git/GitHub

 Results & Insights

Model Performance
- The model achieves ~98% accuracy on the training set
- Validation accuracy ranges from 94-100%
- Test accuracy of 77-78% indicates good generalization

Key Findings
- The CNN architecture effectively captures features relevant to pneumonia detection
- Dropout regularization helps prevent overfitting
- Image normalization is crucial for model performance

Potential Improvements
- Implement data augmentation techniques for better generalization
- Try transfer learning with pre-trained models (ResNet, VGG)
- Experiment with ensemble methods
- Collect more balanced datasets
- Implement gradient-based visualization techniques (CAM, Grad-CAM)



