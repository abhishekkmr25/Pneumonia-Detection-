# Pneumonia Classification

This project detects pneumonia from chest X-ray images using both classical machine learning and deep learning approaches.

The project compares traditional feature-based models with transfer learning CNNs and evaluates their performance using standard classification metrics.

## Features

* Chest X-ray image preprocessing and exploratory analysis
* Chest-region cropping and CLAHE-based image enhancement
* Texture and intensity feature extraction
* PCA-based dimensionality reduction
* Patient-level train-test splitting
* Classical ML model training and hyperparameter tuning
* Transfer learning using pretrained CNN architectures
* Model evaluation using Accuracy, Precision, Recall, F1-Score and ROC-AUC
* Comparison of classical ML and deep learning approaches

## Methodology

1. Loaded and explored chest X-ray images from the dataset.
2. Preprocessed images using grayscale conversion and resizing for classical ML.
3. Extracted intensity, statistical and GLCM texture features from X-ray images.
4. Applied feature scaling and PCA within ML pipelines to reduce dimensionality and prevent data leakage.
5. Trained and tuned Logistic Regression, Random Forest and SVM models using patient-aware cross-validation.
6. Applied chest-region cropping, CLAHE enhancement and image augmentation for deep learning.
7. Trained pretrained VGG16 and ResNet50-based models using transfer learning.
8. Evaluated the models using Accuracy, Precision, Recall, F1-Score and ROC-AUC.
9. Compared classical ML and deep learning models to understand their strengths and trade-offs.

## Models Used

### Classical Machine Learning

* Logistic Regression
* Random Forest
* SVM

### Deep Learning

* VGG16
* ResNet50

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

## Technologies Used

* Python
* NumPy
* Pandas
* OpenCV
* Scikit-learn
* Scikit-image
* TensorFlow / Keras
* Albumentations
* Matplotlib
* Seaborn
