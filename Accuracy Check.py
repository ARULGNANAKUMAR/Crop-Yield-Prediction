# Crop Yield Regression Model Evaluation Script

# Import required libraries
import pandas as pd
import numpy as np
import joblib
import math
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CROP YIELD REGRESSION MODEL EVALUATION")
print("="*80)

# ============================================================================
# STEP 1: LOAD THE TRAINED MODEL AND TEST DATA
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOADING MODEL AND TEST DATA")
print("="*80)

# Load the trained model
try:
    model = joblib.load('best_crop_yield_model.pkl')
    print("✓ Trained model loaded successfully from 'best_crop_yield_model.pkl'")
except FileNotFoundError:
    print("Error: 'best_crop_yield_model.pkl' not found!")
    print("Please ensure the model file exists in the current directory.")
    exit(1)

# Load the test data
try:
    X_test = pd.read_csv('X_test.csv')
    print("✓ X_test.csv loaded successfully")
    print(f"  X_test shape: {X_test.shape}")
except FileNotFoundError:
    print("Error: 'X_test.csv' not found!")
    exit(1)

try:
    y_test = pd.read_csv('y_test.csv')
    print("✓ y_test.csv loaded successfully")
    print(f"  y_test shape: {y_test.shape}")
except FileNotFoundError:
    print("Error: 'y_test.csv' not found!")
    exit(1)

# Convert y_test to 1D array if necessary
print("\n" + "-"*60)
print("DATA PREPROCESSING")
print("-"*60)

if isinstance(y_test, pd.DataFrame):
    if y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]
        print("✓ Converted y_test from DataFrame to 1D Series")
    else:
        print(f"Warning: y_test has {y_test.shape[1]} columns. Using first column.")
        y_test = y_test.iloc[:, 0]
elif isinstance(y_test, pd.Series):
    print("✓ y_test is already a Series")
else:
    print(f"✓ y_test is a {type(y_test)}")

print(f"  Final y_test shape: {y_test.shape}")

# ============================================================================
# STEP 2: MAKE PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: MAKING PREDICTIONS")
print("="*80)

# Predict yield values on X_test
y_pred = model.predict(X_test)

print(f"✓ Predictions made on {len(y_pred)} test samples")
print(f"  Prediction array shape: {y_pred.shape}")

# ============================================================================
# STEP 3: SAMPLE PREDICTED VS ACTUAL VALUES
# ============================================================================

print("\n" + "="*80)
print("STEP 3: SAMPLE PREDICTED VS ACTUAL VALUES (First 10 rows)")
print("="*80)

# Create a comparison DataFrame
comparison_df = pd.DataFrame({
    'Actual': y_test[:10].values if hasattr(y_test, 'values') else y_test[:10],
    'Predicted': y_pred[:10],
    'Difference': (y_test[:10].values if hasattr(y_test, 'values') else y_test[:10]) - y_pred[:10],
    'Absolute_Error': np.abs((y_test[:10].values if hasattr(y_test, 'values') else y_test[:10]) - y_pred[:10])
})

print("\nSample predictions (first 10 rows):")
print(comparison_df.to_string(index=True))

# Calculate average absolute error for sample
sample_avg_error = comparison_df['Absolute_Error'].mean()
print(f"\nAverage absolute error for first 10 samples: {sample_avg_error:.4f}")

# ============================================================================
# STEP 4: CALCULATE PERFORMANCE METRICS
# ============================================================================

print("\n" + "="*80)
print("STEP 4: CALCULATING PERFORMANCE METRICS")
print("="*80)

# Calculate R2 Score
r2 = r2_score(y_test, y_pred)
print(f"\n✓ R2 Score calculated: {r2:.6f}")

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"✓ Mean Absolute Error (MAE) calculated: {mae:.6f}")

# Calculate Root Mean Squared Error (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
print(f"✓ Root Mean Squared Error (RMSE) calculated: {rmse:.6f}")

# Calculate additional metrics for better insight
print("\n" + "-"*60)
print("ADDITIONAL METRICS")
print("-"*60)

# Mean of actual values
mean_actual = y_test.mean()
print(f"Mean of actual values: {mean_actual:.6f}")

# Mean of predicted values
mean_predicted = y_pred.mean()
print(f"Mean of predicted values: {mean_predicted:.6f}")

# Calculate Mean Absolute Percentage Error (MAPE)
# Avoid division by zero by adding a small epsilon
epsilon = 1e-10
mape = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Calculate Explained Variance Score
from sklearn.metrics import explained_variance_score
explained_var = explained_variance_score(y_test, y_pred)
print(f"Explained Variance Score: {explained_var:.6f}")

# ============================================================================
# STEP 5: PRINT COMPREHENSIVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("STEP 5: COMPREHENSIVE EVALUATION RESULTS")
print("="*80)

print("\n" + "-"*60)
print("MODEL PERFORMANCE SUMMARY")
print("-"*60)

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL EVALUATION METRICS                     │
├─────────────────────────────────────────────────────────────────┤
│  R² Score (Coefficient of Determination):  {r2:.6f}            │
│  Mean Absolute Error (MAE):                {mae:.6f}            │
│  Root Mean Squared Error (RMSE):           {rmse:.6f}            │
│  Mean Absolute Percentage Error (MAPE):    {mape:.2f}%           │
│  Explained Variance Score:                 {explained_var:.6f}   │
└─────────────────────────────────────────────────────────────────┘
""")

# ============================================================================
# STEP 6: MODEL PERFORMANCE INTERPRETATION
# ============================================================================

print("\n" + "="*80)
print("STEP 6: MODEL PERFORMANCE INTERPRETATION")
print("="*80)

# Interpret R2 Score
print("\n" + "-"*60)
print("R² SCORE INTERPRETATION")
print("-"*60)

if r2 > 0.8:
    print(f"✓ R² Score = {r2:.4f} → Excellent Model Performance")
    print("  The model explains more than 80% of the variance in the target variable.")
    print("  Predictions are highly accurate and reliable.")
elif r2 > 0.6:
    print(f"● R² Score = {r2:.4f} → Good Model Performance")
    print("  The model explains between 60-80% of the variance in the target variable.")
    print("  Predictions are reasonably accurate with acceptable error margins.")
else:
    print(f"✗ R² Score = {r2:.4f} → Model Needs Improvement")
    print("  The model explains less than 60% of the variance in the target variable.")
    print("  Consider feature engineering, collecting more data, or trying different algorithms.")

# Interpret MAE
print("\n" + "-"*60)
print("MAE INTERPRETATION")
print("-"*60)

mae_percentage = (mae / mean_actual) * 100 if mean_actual != 0 else 0
print(f"MAE as percentage of mean actual value: {mae_percentage:.2f}%")

if mae_percentage < 10:
    print("✓ Excellent: Predictions are very close to actual values")
elif mae_percentage < 20:
    print("● Good: Predictions have reasonable accuracy")
elif mae_percentage < 30:
    print("○ Moderate: Predictions have acceptable accuracy")
else:
    print("✗ Poor: Predictions deviate significantly from actual values")

# Interpret RMSE
print("\n" + "-"*60)
print("RMSE INTERPRETATION")
print("-"*60)

rmse_percentage = (rmse / mean_actual) * 100 if mean_actual != 0 else 0
print(f"RMSE as percentage of mean actual value: {rmse_percentage:.2f}%")

if rmse_percentage < 15:
    print("✓ Excellent: Model has low prediction error")
elif rmse_percentage < 25:
    print("● Good: Model has moderate prediction error")
elif rmse_percentage < 35:
    print("○ Moderate: Model has acceptable prediction error")
else:
    print("✗ High: Model has significant prediction error")

# ============================================================================
# STEP 7: FINAL RECOMMENDATION
# ============================================================================

print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

print("\nBased on the evaluation metrics:")

if r2 > 0.8 and mae_percentage < 15:
    print("""
    🎯 RECOMMENDATION: DEPLOY THE MODEL
    • The model shows excellent performance metrics
    • It can be confidently used for crop yield prediction
    • Consider monitoring performance over time and retraining periodically
    """)
elif r2 > 0.6 and mae_percentage < 25:
    print("""
    📊 RECOMMENDATION: MODEL IS USABLE WITH CAUTION
    • The model shows good but not excellent performance
    • It can be used for initial predictions
    • Consider collecting more data or feature engineering for improvement
    """)
else:
    print("""
    ⚠️ RECOMMENDATION: MODEL NEEDS IMPROVEMENT
    • The model performance is below acceptable thresholds
    • Consider the following improvements:
      1. Collect more training data
      2. Perform additional feature engineering
      3. Try different algorithms (Gradient Boosting, XGBoost, etc.)
      4. Perform hyperparameter tuning
      5. Check for data quality issues
    """)

# ============================================================================
# SAVE EVALUATION RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING EVALUATION RESULTS")
print("="*80)

# Create evaluation summary
evaluation_summary = pd.DataFrame({
    'Metric': ['R² Score', 'MAE', 'RMSE', 'MAPE (%)', 'Explained Variance'],
    'Value': [r2, mae, rmse, mape, explained_var]
})

evaluation_summary.to_csv('model_evaluation_results.csv', index=False)
print("✓ Evaluation results saved to 'model_evaluation_results.csv'")

# Save predictions with actual values
predictions_df = pd.DataFrame({
    'Actual': y_test.values if hasattr(y_test, 'values') else y_test,
    'Predicted': y_pred,
    'Absolute_Error': np.abs(y_test - y_pred),
    'Percentage_Error': np.abs((y_test - y_pred) / (y_test + epsilon)) * 100
})
predictions_df.to_csv('model_predictions_with_actual.csv', index=False)
print("✓ Predictions with actual values saved to 'model_predictions_with_actual.csv'")

print("\n" + "="*80)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("="*80)

print("\n📁 Files generated:")
print("  • model_evaluation_results.csv - Summary of evaluation metrics")
print("  • model_predictions_with_actual.csv - All predictions with actual values")

print("\n" + "="*80)
