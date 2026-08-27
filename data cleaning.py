# Data Cleaning and Preprocessing Script for Crop Dataset

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("Loading dataset...")
df = pd.read_csv('Crops_data.csv')

# Display basic dataset info
print("\n" + "="*50)
print("BASIC DATASET INFORMATION")
print("="*50)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nDataset info:")
print(df.info())

print("\nMissing values count:")
print(df.isnull().sum())

# Handle missing values
print("\n" + "="*50)
print("HANDLING MISSING VALUES")
print("="*50)

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

# Standardize column names (lowercase, replace spaces with underscores)
df.columns = df.columns.str.lower().str.replace(' ', '_')
print("\nStandardized column names:")
print(df.columns.tolist())

# Convert year column to integer type (assuming 'year' column exists)
if 'year' in df.columns:
    df['year'] = df['year'].astype(int)
    print("\nYear column converted to integer type")

# Identify categorical columns for encoding
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns to encode: {categorical_columns}")

# Convert categorical columns using Label Encoding
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded column: {col}")

# Feature Engineering: Create yield_per_hectare if not present
# Check for relevant columns (production and area columns)
production_cols = [col for col in df.columns if 'production' in col.lower()]
area_cols = [col for col in df.columns if 'area' in col.lower()]

# Create yield_per_hectare for the main crop if not exists
if 'yield_per_hectare' not in df.columns and len(production_cols) > 0 and len(area_cols) > 0:
    # Assuming first production and area columns are for the main crop
    main_prod_col = production_cols[0]
    main_area_col = area_cols[0]
    
    # Avoid division by zero
    df['yield_per_hectare'] = np.where(
        df[main_area_col] > 0,
        df[main_prod_col] / df[main_area_col],
        0
    )
    print(f"\nCreated 'yield_per_hectare' using {main_prod_col} / {main_area_col}")
else:
    print("\n'yield_per_hectare' column already exists or required columns not found")

# Scale numerical features
print("\n" + "="*50)
print("FEATURE SCALING")
print("="*50)

# Select numerical features for scaling (exclude target if it exists)
if 'yield_per_hectare' in df.columns:
    numerical_features = df.select_dtypes(include=[np.number]).columns.drop('yield_per_hectare')
else:
    numerical_features = df.select_dtypes(include=[np.number]).columns

# Apply StandardScaler
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
print(f"Scaled {len(numerical_features)} numerical features")

# Split features (X) and target (y)
print("\n" + "="*50)
print("SPLITTING DATA")
print("="*50)

if 'yield_per_hectare' in df.columns:
    X = df.drop('yield_per_hectare', axis=1)
    y = df['yield_per_hectare']
    
    # Split into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
else:
    print("Warning: 'yield_per_hectare' column not found. Cannot split data.")
    X_train, X_test, y_train, y_test = None, None, None, None

# Print final processed dataset shape
print("\n" + "="*50)
print("FINAL PROCESSED DATASET")
print("="*50)
print(f"Final dataset shape: {df.shape}")
print(f"Final dataset columns: {df.columns.tolist()}")

# Save cleaned dataset to CSV
output_filename = 'cleaned_crop_data.csv'
df.to_csv(output_filename, index=False)
print(f"\nCleaned dataset saved to: {output_filename}")

# Optional: Basic visualization
print("\n" + "="*50)
print("BASIC VISUALIZATION")
print("="*50)

if 'yield_per_hectare' in df.columns:
    plt.figure(figsize=(10, 6))
    plt.hist(df['yield_per_hectare'].dropna(), bins=50, alpha=0.7, color='blue')
    plt.title('Distribution of Yield per Hectare')
    plt.xlabel('Yield per Hectare')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('yield_distribution.png')
    plt.show()
    print("Yield distribution histogram saved as 'yield_distribution.png'")

print("\n" + "="*50)
print("PREPROCESSING COMPLETE")
print("="*50)
