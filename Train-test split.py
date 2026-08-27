# Train-Test Split and Model Training Script for Crop Yield Dataset

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load the cleaned dataset
print("Loading cleaned dataset...")
df = pd.read_csv('cleaned_crop_data.csv')
print(f"Dataset loaded successfully. Shape: {df.shape}")

# Display basic info about the dataset
print("\n" + "="*60)
print("DATASET INFORMATION")
print("="*60)
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")
print(f"\nFirst 5 rows:")
print(df.head())

# Define target variable and features
print("\n" + "="*60)
print("DEFINING TARGET AND FEATURES")
print("="*60)

# Check if target column exists
if 'yield_per_hectare' not in df.columns:
    raise ValueError("'yield_per_hectare' column not found in the dataset!")

# Define target variable (y)
y = df['yield_per_hectare']

# Define features (X) - all columns except the target
X = df.drop('yield_per_hectare', axis=1)

print(f"Target variable shape: {y.shape}")
print(f"Features shape: {X.shape}")
print(f"\nFeatures columns: {X.columns.tolist()}")

# Perform Train-Test Split (80% training, 20% testing)
print("\n" + "="*60)
print("TRAIN-TEST SPLIT")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

# Scale numerical features using StandardScaler
print("\n" + "="*60)
print("FEATURE SCALING")
print("="*60)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit scaler on training data and transform both train and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training data scaled successfully")
print(f"Testing data scaled successfully")
print(f"Scaler mean: {scaler.mean_[:5]}...")  # Show first 5 means
print(f"Scaler scale: {scaler.scale_[:5]}...")  # Show first 5 scales

# Train Linear Regression model
print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)

# Initialize Linear Regression model
lr_model = LinearRegression()

# Train the model on scaled training data
lr_model.fit(X_train_scaled, y_train)

print("Linear Regression model trained successfully")
print(f"Model coefficients: {lr_model.coef_[:5]}...")  # Show first 5 coefficients
print(f"Model intercept: {lr_model.intercept_}")

# Make predictions on test data
print("\n" + "="*60)
print("MAKING PREDICTIONS")
print("="*60)

y_pred = lr_model.predict(X_test_scaled)

print(f"Predictions made on {len(y_pred)} test samples")
print(f"\nFirst 10 actual vs predicted values:")
for i in range(min(10, len(y_test))):
    print(f"Actual: {y_test.iloc[i]:.4f} | Predicted: {y_pred[i]:.4f} | Difference: {abs(y_test.iloc[i] - y_pred[i]):.4f}")

# Evaluate model performance
print("\n" + "="*60)
print("MODEL EVALUATION METRICS")
print("="*60)

# Calculate R2 Score
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2:.4f}")

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Calculate additional metrics for better insight
print("\n" + "="*60)
print("ADDITIONAL INSIGHTS")
print("="*60)

# Mean of actual values
mean_actual = y_test.mean()
print(f"Mean of actual values: {mean_actual:.4f}")

# Mean of predicted values
mean_predicted = y_pred.mean()
print(f"Mean of predicted values: {mean_predicted:.4f}")

# Calculate Mean Absolute Percentage Error (MAPE)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Calculate Explained Variance Score
from sklearn.metrics import explained_variance_score
explained_var = explained_variance_score(y_test, y_pred)
print(f"Explained Variance Score: {explained_var:.4f}")

# Summary of model performance
print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)
print(f"Model: Linear Regression")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")

# Determine if the model is performing well
print("\n" + "="*60)
print("MODEL PERFORMANCE ASSESSMENT")
print("="*60)

if r2 > 0.7:
    print(f"✓ Good R2 Score ({r2:.4f}) - Model explains {r2*100:.1f}% of variance")
elif r2 > 0.5:
    print(f"● Moderate R2 Score ({r2:.4f}) - Model explains {r2*100:.1f}% of variance")
else:
    print(f"✗ Low R2 Score ({r2:.4f}) - Model may need improvement")

if mape < 10:
    print(f"✓ Excellent MAPE ({mape:.2f}%) - Predictions are very accurate")
elif mape < 20:
    print(f"● Good MAPE ({mape:.2f}%) - Predictions are reasonably accurate")
elif mape < 30:
    print(f"○ Moderate MAPE ({mape:.2f}%) - Predictions have acceptable accuracy")
else:
    print(f"✗ High MAPE ({mape:.2f}%) - Predictions may need improvement")

print("\n" + "="*60)
print("TRAIN-TEST SPLIT AND MODEL TRAINING COMPLETED")
print("="*60)
