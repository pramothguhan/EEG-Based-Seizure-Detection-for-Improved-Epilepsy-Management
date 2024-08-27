## EEG Classification Model :brain:

## Overview :memo:

This project aims to develop and evaluate machine learning models for classifying EEG data, specifically for detecting seizure activity. The models were trained on data from the CHB-MIT Scalp EEG Database, which consists of EEG recordings from pediatric subjects diagnosed with epilepsy. The project explores two primary models: a Convolutional Neural Network (CNN) and a Decision Tree, both optimized for binary classification tasks.

## Abstract :notebook_with_decorative_cover:

Electroencephalogram (EEG) data is critical in the diagnosis and monitoring of epilepsy. This project develops and assesses models for classifying EEG data into seizure and non-seizure states. Leveraging data preprocessing, feature extraction, and machine learning techniques, the project evaluates the effectiveness of CNN and Decision Tree models in detecting seizure activity. The results indicate that while both models have potential, further optimization and development are required for clinical application.

## Project Goals :dart:

1. **Data Preprocessing:** Prepare EEG data for classification through normalization, filtering, and feature extraction.
2. **Model Development:** Develop and train CNN and Decision Tree models for binary classification of EEG data.
3. **Performance Evaluation:** Assess the models' performance using accuracy, F1 score, and confusion matrices.
4. **Comparison and Analysis:** Compare the CNN and Decision Tree models to determine their strengths and weaknesses.
5. **Future Work:** Identify potential areas for further research and model refinement, including real-time monitoring and clinical application.

## Technologies Used :computer:

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## Methodology :gear:

### Data Preprocessing and Feature Extraction:
1. **Normalization and Statistical Measures:** The EEG signal is normalized, and statistical measures (mean, standard deviation, skewness, kurtosis) are calculated to capture essential characteristics.
2. **Entropy Measures:** Sample and fuzzy entropy are computed to provide insights into the complexity of the EEG signal.
3. **Advanced Feature Extraction:** Techniques like Short-Time Fourier Transform (STFT) and Power Spectral Density (PSD) are used to extract frequency domain information.
4. **Label Extraction:** Seizure start and end times are extracted, and each time window is labeled as seizure (1) or non-seizure (0).

### Model Architecture and Training:
#### Decision Tree Model:
1. **SMOTE Oversampling:** Synthetic Minority Over-sampling Technique (SMOTE) is used to handle class imbalance.
2. **Training:** A Decision Tree classifier is trained using the resampled data.
3. **Evaluation:** The model's accuracy and F1 score are calculated, and the results are visualized using confusion matrices.

#### CNN Model:
1. **Model Architecture:** The CNN consists of Conv1D layers, MaxPooling1D layers, and Dense layers, using ReLU activation and sigmoid output for binary classification.
2. **Training:** The CNN is trained with oversampled data, using the Adam optimizer and binary cross-entropy loss.
3. **Evaluation:** The CNN model's performance is assessed using accuracy and F1 score metrics.

## Results :chart_with_upwards_trend:

- **Decision Tree Model:**
  - **Accuracy:** 88.9%
  - **F1 Score:** 0.89
  - **Summary:** The Decision Tree model, optimized with SMOTE, performed effectively in classifying seizures, demonstrating strong accuracy and F1 score.

- **CNN Model:**
  - **Accuracy:** 63.1%
  - **F1 Score:** 0.64
  - **Summary:** The CNN model showed potential but requires further refinement through hyperparameter tuning and architectural adjustments.

## Project Folder Structure :file_folder:
```
📦 EEG_Classification_Model
├── data
│ ├── EEG_data.csv
│ ├── Seizure_annotations.csv
├── notebooks
│ └── Project_03_Group_22.ipynb
├── src
│ ├── data_preprocessing.py
│ ├── feature_extraction.py
│ ├── model_training.py
│ └── evaluation.py
├── results
│ └── confusion_matrix.png
└── README.md
```


## How to Run Locally :house:

1. **Clone the repository:**
   ```bash
   git clone <repository-link>

## Conclusion and Future Work :mag:
The CNN and Decision Tree models demonstrated the potential for seizure detection in EEG data. While the Decision Tree model showed higher accuracy, the CNN model's architecture needs further refinement. Future work will focus on exploring ensemble methods, real-time monitoring, transfer learning, and improving model interpretability and clinical validation.
