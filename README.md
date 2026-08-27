# 🌾 Crop Yield Prediction System

An AI/ML-based Crop Yield Prediction System that predicts agricultural yield using historical crop data and machine learning algorithms.

## 📌 Overview

This project uses historical crop data to preprocess agricultural information, train multiple machine learning regression models, compare their performance, and select the best-performing model for crop yield prediction.

The system also provides an interactive Streamlit dashboard for making crop yield predictions.

## 🚀 Features

- Crop dataset preprocessing
- Missing value handling
- Duplicate removal
- Categorical feature encoding
- Feature scaling
- Yield-per-hectare feature engineering
- Train-test split
- Multiple ML model training
- Model performance comparison
- Best model selection
- Interactive crop yield prediction dashboard
- Model performance evaluation

## 🤖 Machine Learning Models

The project currently evaluates:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Models are evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Explained Variance Score

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

## 📂 Project Files

```text
Crop-Yield-Prediction/
│
├── Crops_data.csv
├── cleaned_crop_data.csv
│
├── Data Cleaning and Preprocessing.py
├── Train-Test Split and Model Training.py
├── Complete Crop Yield Prediction Pipeline.py
├── Accuracy Check.py
│
├── dashboard.py
│
├── best_crop_yield_model.pkl
├── scaler.pkl
├── label_encoders.pkl
│
├── X_train.csv
├── X_test.csv
├── y_train.csv
├── y_test.csv
│
├── model_performance_summary.csv
├── best_model_predictions.csv
│
├── requirements.txt
├── .gitignore
└── README.md
