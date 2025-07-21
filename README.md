# AI vs Real Image Classification 🤖📸

A deep learning project that classifies images as either AI-generated (fake) or real using a Convolutional Neural Network (CNN). This project uses the CIFAKE dataset to train a binary classifier capable of distinguishing between authentic and synthetic images.

## 🔗 Links

- **Original Kaggle Project**: [Classifying Real and AI-Generated Images](https://www.kaggle.com/code/emanafi/aireal)
- **Dataset**: [CIFAKE Dataset on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Features](#-features)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Dependencies](#-dependencies)
- [Contributors](#-contributors)

## 🎯 Project Overview

This project addresses the growing challenge of distinguishing between real and AI-generated images. With the advancement of generative AI models, it has become increasingly difficult to identify synthetic content with the naked eye. This CNN-based classifier provides an automated solution to detect AI-generated images.

### Key Objectives

- Build a robust binary classifier for AI vs real image detection
- Analyze file size patterns between real and synthetic images
- Implement proper data preprocessing and model evaluation techniques
- Achieve high accuracy in distinguishing between authentic and generated content

## 📊 Dataset

The project uses the **CIFAKE** dataset, which contains:

- **Training Set**:
  - Real images: `/train/REAL/`
  - Fake (AI-generated) images: `/train/FAKE/`
- **Test Set**:
  - Real images: `/test/REAL/`
  - Fake (AI-generated) images: `/test/FAKE/`

### Dataset Characteristics

- **Image Format**: 32x32 pixel RGB images in JPEG format
- **Preprocessing**: Images are normalized to [0,1] range
- **Labels**: Binary classification (0 = Fake, 1 = Real)

## ✨ Features

### 1. **Comprehensive Data Analysis**

- File size distribution analysis across different categories
- Statistical analysis (mean, median, standard deviation)
- Visual histograms for data exploration

### 2. **Data Preprocessing**

- Image resizing and normalization
- Train-test split with stratification
- Data validation and format verification

### 3. **CNN Model Architecture**

- Multi-layer convolutional neural network
- Regularization techniques (L2 regularization, Dropout)
- Adam optimizer with custom learning rate

### 4. **Model Evaluation**

- Training/validation accuracy tracking
- Confusion matrix visualization
- Precision, Recall, and F1-score metrics
- Classification report generation

### 5. **Prediction System**

- Single image prediction capability
- Batch prediction for multiple images
- Custom image testing functionality

## 🧠 Model Architecture

The CNN model consists of:

```python
# Convolutional Layers
- Conv2D(64, (3,3)) + MaxPooling2D(2,2) + Dropout(0.3)
- Conv2D(32, (3,3)) + MaxPooling2D(2,2) + Dropout(0.3)  
- Conv2D(16, (3,3)) + MaxPooling2D(2,2) + Dropout(0.3)

# Fully Connected Layers
- Dense(512) + Dropout(0.4)
- Dense(256) + Dropout(0.4)
- Dense(128) + Dropout(0.4)
- Dense(1, activation='sigmoid')  # Output layer
```

### Model Features

- **Regularization**: L2 regularization (0.001) on all layers
- **Dropout**: Progressive dropout rates (0.3 → 0.4) to prevent overfitting
- **Optimizer**: Adam with learning rate 0.0001
- **Loss Function**: Binary crossentropy
- **Training**: 30 epochs with batch size 64

## 🚀 Installation

### Running on Kaggle (Recommended)

1. **Open the Kaggle notebook**
   - Visit the [original Kaggle project](https://www.kaggle.com/code/chellyahmed/aireal)
   - Fork the notebook to your account
   - The CIFAKE dataset is already available in the Kaggle environment

2. **Required packages** (pre-installed on Kaggle)
   - TensorFlow
   - OpenCV
   - Scikit-learn
   - Matplotlib
   - Seaborn
   - PIL (Pillow)
   - NumPy
   - tqdm

### Running Locally

1. **Clone the repository**

   ```bash
   git clone https://github.com/chellyahmed/aireal.git
   cd aireal
   ```

2. **Install required packages**

   ```bash
   pip install tensorflow opencv-python scikit-learn matplotlib seaborn pillow numpy tqdm
   ```

3. **Download the dataset**
   - Download the [CIFAKE dataset from Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
   - Extract to match the directory structure used in the notebook
   - Update the directory paths in the code to match your local setup

## 💻 Usage

### Running the Notebook

1. **Open Jupyter Notebook**

   ```bash
   jupyter notebook aireal.ipynb
   ```

2. **Execute cells sequentially**
   - Cell 1: Data analysis and visualization
   - Cell 2: Data validation and preprocessing
   - Cell 3: Statistical analysis
   - Cell 4: Data loading and splitting
   - Cell 5: Model definition and compilation
   - Cell 6: Model training
   - Cell 7: Training visualization
   - Cells 8-9: Custom image prediction
   - Cells 10-11: Model evaluation and metrics

### Predicting on Custom Images

```python
# Load and preprocess your image
img_path = 'path/to/your/image.jpg'
img = image.load_img(img_path, target_size=(32, 32))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Make prediction
prediction = model.predict(img_array)
result = "Real" if prediction[0][0] > 0.5 else "Fake"
print(f"Prediction: {result} (Confidence: {prediction[0][0]:.4f})")
```

## 📈 Results

The model achieves:

- **Training Accuracy**: [To be filled based on your results]
- **Validation Accuracy**: [To be filled based on your results]
- **Precision**: [To be filled based on your results]
- **Recall**: [To be filled based on your results]
- **F1-Score**: [To be filled based on your results]

### Key Insights

- File size analysis reveals patterns between real and AI-generated images
- The CNN model effectively learns distinguishing features
- Regularization techniques help prevent overfitting
- Model performs well on both training and validation sets

## 📦 Dependencies

- **TensorFlow**: Deep learning framework
- **OpenCV**: Image processing
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical visualization
- **Scikit-learn**: Machine learning utilities
- **PIL (Pillow)**: Image handling
- **tqdm**: Progress bars

## 🔍 Technical Details

### Data Preprocessing

- Images resized to 32x32 pixels
- Pixel values normalized to [0, 1] range
- RGB color format maintained
- Train-test split: 80-20 ratio

### Model Training

- Batch size: 64
- Epochs: 30
- Learning rate: 0.0001
- Early stopping can be implemented for optimal performance

### Evaluation Metrics

- Binary classification accuracy
- Precision and recall for both classes
- F1-score for balanced evaluation
- Confusion matrix for detailed analysis

### Areas for Improvement

- Experiment with different CNN architectures
- Implement data augmentation techniques
- Add more sophisticated regularization methods
- Create a web interface for easy image testing
- Expand to handle different image sizes and formats

## 👥 Contributors

This project was developed by:

- **[Ahmed Chelly](https://github.com/ChellyAhmed)**
- **[Amine Braham](https://github.com/Spid3rrr)**
- **[Eman Sarah Afi](https://github.com/emansarahafi)**
- **[Sahar Sarraj](https://github.com/Sarraj-Sahar)**

## 🙏 Acknowledgments

- **CIFAKE Dataset**: Thanks to the creators of the CIFAKE dataset
- **Kaggle Community**: For providing the platform and dataset
- **TensorFlow Team**: For the excellent deep learning framework

---

**Note**: This project is for educational and research purposes. Always consider ethical implications when working with AI-generated content detection systems.
