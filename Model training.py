# Complete Crop Yield Prediction Pipeline
# This script performs: Data Loading -> Preprocessing -> Train-Test Split -> Model Training

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CROP YIELD PREDICTION PIPELINE")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND PREPROCESS THE DATASET
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING AND PREPROCESSING DATASET")
print("="*80)

# Load the original dataset
try:
    df = pd.read_csv('Crops_data.csv')
    print(f"✓ Dataset loaded successfully from 'Crops_data.csv'")
    print(f"  Dataset shape: {df.shape}")
except FileNotFoundError:
    try:
        df = pd.read_csv('cleaned_crop_data.csv')
        print(f"✓ Dataset loaded successfully from 'cleaned_crop_data.csv'")
        print(f"  Dataset shape: {df.shape}")
    except FileNotFoundError:
        print("Error: Neither 'Crops_data.csv' nor 'cleaned_crop_data.csv' found!")
        exit(1)

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nDataset info:")
print(df.info())

# Handle missing values
print("\n" + "-"*60)
print("HANDLING MISSING VALUES")
print("-"*60)

# Drop rows with completely empty values
initial_shape = df.shape
df = df.dropna(how='all')
print(f"Rows dropped with all null values: {initial_shape[0] - df.shape[0]}")

# Fill numerical missing values with mean
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)

# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if not df[col].empty:
        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)

print(f"Missing values after imputation: {df.isnull().sum().sum()}")

# Remove duplicate rows
initial_shape = df.shape
df = df.drop_duplicates()
print(f"Duplicate rows removed: {initial_shape[0] - df.shape[0]}")

# Standardize column names
df.columns = df.columns.str.lower().str.replace(' ', '_')
print(f"\nStandardized column names: {df.columns.tolist()}")

# Convert year column to integer type if exists
if 'year' in df.columns:
    df['year'] = df['year'].astype(int)
    print("✓ Year column converted to integer type")

# Encode categorical columns
print("\n" + "-"*60)
print("ENCODING CATEGORICAL COLUMNS")
print("-"*60)

categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns found: {categorical_columns}")

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  ✓ Encoded column: {col}")

# Create yield_per_hectare if not present
print("\n" + "-"*60)
print("FEATURE ENGINEERING")
print("-"*60)

if 'yield_per_hectare' not in df.columns:
    production_cols = [col for col in df.columns if 'production' in col.lower()]
    area_cols = [col for col in df.columns if 'area' in col.lower()]
    
    if len(production_cols) > 0 and len(area_cols) > 0:
        main_prod_col = production_cols[0]
        main_area_col = area_cols[0]
        
        df['yield_per_hectare'] = np.where(
            df[main_area_col] > 0,
            df[main_prod_col] / df[main_area_col],
            0
        )
        print(f"✓ Created 'yield_per_hectare' using {main_prod_col} / {main_area_col}")
    else:
        print("Warning: Could not create yield_per_hectare - required columns not found")
else:
    print("✓ 'yield_per_hectare' column already exists")

# ============================================================================
# STEP 2: TRAIN-TEST SPLIT
# ============================================================================

print("\n" + "="*80)
print("STEP 2: TRAIN-TEST SPLIT")
print("="*80)

# Define target and features
if 'yield_per_hectare' not in df.columns:
    print("Error: 'yield_per_hectare' column not found!")
    exit(1)

y = df['yield_per_hectare']
X = df.drop('yield_per_hectare', axis=1)

print(f"Target variable shape: {y.shape}")
print(f"Features shape: {X.shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Save the split datasets for future use
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print(f"\n✓ Split datasets saved:")
print(f"  - X_train.csv")
print(f"  - X_test.csv")
print(f"  - y_train.csv")
print(f"  - y_test.csv")

# Scale the features
print("\n" + "-"*60)
print("FEATURE SCALING")
print("-"*60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Features scaled using StandardScaler")

# ============================================================================
# STEP 3: TRAIN MULTIPLE MODELS
# ============================================================================

print("\n" + "="*80)
print("STEP 3: TRAINING MULTIPLE MODELS")
print("="*80)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42)
}

print(f"Models to train: {list(models.keys())}\n")

# Dictionary to store results
results = {}

# Train each model
for model_name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    print(f"✓ Model training completed")
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    print(f"✓ Predictions made on test set")
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Store results
    results[model_name] = {
        'model': model,
        'r2_score': r2,
        'mae': mae,
        'rmse': rmse,
        'predictions': y_pred
    }
    
    # Print metrics
    print(f"\nPerformance Metrics:")
    print(f"  R² Score: {r2:.6f}")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.6f}")

# ============================================================================
# STEP 4: MODEL COMPARISON AND SELECTION
# ============================================================================

print("\n" + "="*80)
print("STEP 4: MODEL COMPARISON AND SELECTION")
print("="*80)

# Create comparison dataframe
comparison_data = []
for model_name, metrics in results.items():
    comparison_data.append({
        'Model': model_name,
        'R² Score': metrics['r2_score'],
        'MAE': metrics['mae'],
        'RMSE': metrics['rmse']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('R² Score', ascending=False)

print("\nModels ranked by R² Score (highest to lowest):")
print(comparison_df.to_string(index=False))

# Identify the best model
best_model_name = max(results.keys(), key=lambda x: results[x]['r2_score'])
best_model = results[best_model_name]['model']
best_r2 = results[best_model_name]['r2_score']

print(f"\n{'='*60}")
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"{'='*60}")
print(f"  R² Score: {best_r2:.6f}")
print(f"  MAE: {results[best_model_name]['mae']:.6f}")
print(f"  RMSE: {results[best_model_name]['rmse']:.6f}")

# ============================================================================
# STEP 5: SAVE MODEL AND RESULTS
# ============================================================================

print("\n" + "="*80)
print("STEP 5: SAVING MODEL AND RESULTS")
print("="*80)

# Save the best trained model
model_filename = 'best_crop_yield_model.pkl'
joblib.dump(best_model, model_filename)
print(f"✓ Best model saved as: {model_filename}")

# Save the scaler for future use
scaler_filename = 'scaler.pkl'
joblib.dump(scaler, scaler_filename)
print(f"✓ Scaler saved as: {scaler_filename}")

# Save label encoders
joblib.dump(label_encoders, 'label_encoders.pkl')
print(f"✓ Label encoders saved as: label_encoders.pkl")

# Save model performance summary
comparison_df.to_csv('model_performance_summary.csv', index=False)
print(f"✓ Model performance summary saved as: model_performance_summary.csv")

# Save best model predictions
best_predictions = results[best_model_name]['predictions']
relative_errors = np.abs((y_test - best_predictions) / y_test) * 100

predictions_df = pd.DataFrame({
    'Actual': y_test.values if hasattr(y_test, 'values') else y_test,
    'Predicted': best_predictions,
    'Absolute_Error': np.abs(y_test - best_predictions),
    'Percentage_Error': relative_errors
})
predictions_df.to_csv('best_model_predictions.csv', index=False)
print(f"✓ Best model predictions saved as: best_model_predictions.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("="*80)

print("\n📁 Files generated:")
print("  • X_train.csv, X_test.csv, y_train.csv, y_test.csv - Split datasets")
print("  • best_crop_yield_model.pkl - Trained best model")
print("  • scaler.pkl - Feature scaler for preprocessing")
print("  • label_encoders.pkl - Label encoders for categorical variables")
print("  • model_performance_summary.csv - Performance metrics comparison")
print("  • best_model_predictions.csv - Predictions from best model")

print(f"\n📊 Best Model: {best_model_name}")
print(f"   R² Score: {best_r2:.4f}")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. Use 'best_crop_yield_model.pkl' for making predictions on new data")
print("2. Load the model using: model = joblib.load('best_crop_yield_model.pkl')")
print("3. Don't forget to apply the same preprocessing (scaling, encoding) to new data")
print("="*80)
