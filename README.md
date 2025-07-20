# 🌿 Plant Disease Classification Project

## 📌 Project Overview

This project focuses on classifying plant leaf images into four categories: healthy, **multiple_diseases, **rust, and **scab using both Traditional Machine Learning and Convolutional Neural Networks (CNNs).

The goal is to accurately identify diseases from leaf images and compare different model performances, followed by building a deep learning solution that significantly improves accuracy.


## 📁 Dataset

The dataset used in this project is sourced from Kaggle’s Plant Pathology 2020 competition.

- train.csv: Contains image_id and one-hot encoded class labels.
- images/: Folder containing training images named as Train_0.jpg, Train_1.jpg, etc.
- test_images/: Folder containing test images named as Test_0.jpg, Test_1.jpg, etc.


## 📚 Technologies Used

- Python 3.11
- NumPy, Pandas, Matplotlib
- Scikit-learn
- OpenCV
- TensorFlow / Keras
- ImageDataGenerator (for data augmentation)


## ⚙ Project Structure

*CELEBAL PROJECT/*
  - .venv/  # Virtual environment
  - images/ # Folder with training images
  - test images # Folder with test images/
  - model/  # Directory for saved models
  - utils/  # Utility files/functions
  - output_snippets/
  - cnn_preprocessing.ipynb  # Main CNN training and preprocessing code
  - traditionl_method_proj.ipynb # Code for traditional ML models
  - streamlit_app.py   # Streamlit web app for predictions
  - train.csv  # Training label data
  - test.csv  # Test image ids
  - cnn_submission.csv
  - submission_final.csv
  - requirements.txt  # Python dependencies
  - report.txt
  - README.md

## 🧠 Models Used

### ✅ Traditional ML Models:
- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)
- Voting Ensemble

Best Accuracy (Traditional): ~60.5%

### ✅ Deep Learning (CNN):
- Built a CNN from scratch
- Applied Data Augmentation and Class Weights
- Used LabelEncoder and to_categorical for preprocessing

Best Accuracy (CNN): 85%

## 📊 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- Special focus was given to handling the imbalanced multiple_diseases class using:
  - Class weights
  - Data augmentation

## 🧪 How to Run the Project

1.  **Download and Extract**
    Download and extract the submitted project folder to your local machine.

2.  **Navigate to Project Directory**
    Open your terminal or command prompt and change to the project's root directory.
    ```bash
    cd path/to/the/project_folder
    ```

3.  **Create and Activate Virtual Environment (Optional but Recommended)**
    This keeps your project dependencies isolated.

    * **On Windows:**
        ```powershell
        # Create the environment
        python -m venv .venv

        # Activate the environment
        .venv\Scripts\activate
        ```

    * **On macOS/Linux:**
        ```bash
        # Create the environment
        python3 -m venv .venv

        # Activate the environment
        source .venv/bin/activate
        ```

4.  **Install Dependencies**
    Install all the required Python packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    
5. Run notebooks
plant_disease_cnn_model_final/cnn_preprocessing.ipynb → For CNN training and evaluation
traditional_model_pre-processing.ipynb → For SVM, RF, Logistic Regression, etc.

## 📤 Submission File

The final prediction files, **`submission_final.csv`** (from traditional methods) and **`cnn_submission.csv`** (from the CNN), are formatted as follows:

```csv
image_id,label
Test_0,rust
Test_1,rust
```

## 💾 Model Saving & Loading

# Save the trained CNN model
model.save("plant_disease_cnn_model_final.h5")

# Load the model later
from tensorflow.keras.models import load_model
model = load_model("plant_disease_cnn_model_final.h5")

# Streamlit UI
Run the web app:
      streamlit run streamlit_app.py
Upload a test image to get prediction and probability of each class.

## 📚 Learnings & Achievements
1. Applied both traditional ML and CNN to a real-world image classification problem.
2. Handled class imbalance using class weights and data augmentation.
3. Improved model performance from ~60% to 84% using CNN and augmentation.
4. Built a complete ML pipeline: preprocessing, training, evaluation, and prediction.

## 🧑‍💻 Author
Nandini Pachpandiya  
🎓 B.Tech in Artificial Intelligence & Data Science  
💼 Internship: Celebal Technologies  
📅 Project Duration: 02.06.2025 - 03.08.2025


## 📩 Feedback & Suggestions
Feel free to raise issues or suggestions for improvement. This project helped me deeply understand how machine learning and deep learning models are built and evaluated on real-world datasets.
