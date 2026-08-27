# Streamlit Application for Crop Yield Prediction
# File: app.py

# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Crop Yield Prediction System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 3rem;
        border-top: 1px solid #ddd;
        color: #666;
    }
    .info-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🌾 Crop Yield Prediction System</h1>
    <p>Machine Learning-based Prediction for Agricultural Crop Yields</p>
</div>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    """Load the trained crop yield prediction model"""
    try:
        model = joblib.load('best_crop_yield_model.pkl')
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'best_crop_yield_model.pkl' not found!")
        st.info("Please ensure the model file is in the same directory as this application.")
        return None

# Load the scaler if available
@st.cache_resource
def load_scaler():
    """Load the scaler for feature scaling"""
    try:
        scaler = joblib.load('scaler.pkl')
        return scaler
    except:
        return None

# Load feature names from training data
@st.cache_resource
def get_feature_names():
    """Get the feature names expected by the model"""
    try:
        # Try to load feature names from a saved file
        feature_names = joblib.load('feature_names.pkl')
        return feature_names
    except:
        # Return None if not available
        return None

# Load model and preprocessing objects
model = load_model()
scaler = load_scaler()
expected_features = get_feature_names()

if model is None:
    st.stop()

# Sidebar for user inputs
st.sidebar.markdown("## 📋 Input Parameters")
st.sidebar.markdown("Enter the details below to predict crop yield:")

# Create tabs for different input categories in sidebar
st.sidebar.markdown("### 🌱 Basic Information")

# Basic inputs
state = st.sidebar.text_input("State Name", value="Punjab", help="Enter the state name")
district = st.sidebar.text_input("District Name", value="Ludhiana", help="Enter the district name")
year = st.sidebar.number_input("Year", min_value=2010, max_value=2025, value=2023, step=1)

st.sidebar.markdown("### 🌾 Crop Selection")

# Crop selection dropdown
crop_options = [
    "RICE", "WHEAT", "MAIZE", "SORGHUM", "PEARL MILLET", 
    "BARLEY", "CHICKPEA", "PIGEONPEA", "GROUNDNUT", 
    "SUGARCANE", "COTTON", "SOYABEAN"
]
selected_crop = st.sidebar.selectbox("Select Crop", crop_options)

st.sidebar.markdown("### 📊 Crop Area and Production")

# Area input (in 1000 hectares) - matches dataset format
area = st.sidebar.number_input(
    f"{selected_crop} AREA (1000 ha)",
    min_value=0.0,
    max_value=10000.0,
    value=100.0,
    step=10.0,
    help=f"Area under {selected_crop} cultivation in thousand hectares"
)

# Production input (in 1000 tons)
production = st.sidebar.number_input(
    f"{selected_crop} PRODUCTION (1000 tons)",
    min_value=0.0,
    max_value=50000.0,
    value=500.0,
    step=50.0,
    help=f"Expected {selected_crop} production in thousand tons"
)

# Function to create all features expected by the model
def create_all_features(state, district, year, selected_crop, area, production):
    """Create a DataFrame with all features expected by the model"""
    
    # Initialize all features with zeros
    features = {}
    
    # Add basic features
    features['dist_code'] = 0
    features['year'] = year
    features['state_code'] = 0
    
    # Add state and district (will be encoded)
    features['state_name'] = state
    features['dist_name'] = district
    
    # Initialize all crop-related features to zero
    crop_columns = [
        'rice_area_(1000_ha)', 'rice_production_(1000_tons)', 'rice_yield_(kg_per_ha)',
        'wheat_area_(1000_ha)', 'wheat_production_(1000_tons)', 'wheat_yield_(kg_per_ha)',
        'kharif_sorghum_area_(1000_ha)', 'kharif_sorghum_production_(1000_tons)', 'kharif_sorghum_yield_(kg_per_ha)',
        'rabi_sorghum_area_(1000_ha)', 'rabi_sorghum_production_(1000_tons)', 'rabi_sorghum_yield_(kg_per_ha)',
        'sorghum_area_(1000_ha)', 'sorghum_production_(1000_tons)', 'sorghum_yield_(kg_per_ha)',
        'pearl_millet_area_(1000_ha)', 'pearl_millet_production_(1000_tons)', 'pearl_millet_yield_(kg_per_ha)',
        'maize_area_(1000_ha)', 'maize_production_(1000_tons)', 'maize_yield_(kg_per_ha)',
        'finger_millet_area_(1000_ha)', 'finger_millet_production_(1000_tons)', 'finger_millet_yield_(kg_per_ha)',
        'barley_area_(1000_ha)', 'barley_production_(1000_tons)', 'barley_yield_(kg_per_ha)',
        'chickpea_area_(1000_ha)', 'chickpea_production_(1000_tons)', 'chickpea_yield_(kg_per_ha)',
        'pigeonpea_area_(1000_ha)', 'pigeonpea_production_(1000_tons)', 'pigeonpea_yield_(kg_per_ha)',
        'minor_pulses_area_(1000_ha)', 'minor_pulses_production_(1000_tons)', 'minor_pulses_yield_(kg_per_ha)',
        'groundnut_area_(1000_ha)', 'groundnut_production_(1000_tons)', 'groundnut_yield_(kg_per_ha)',
        'sesamum_area_(1000_ha)', 'sesamum_production_(1000_tons)', 'sesamum_yield_(kg_per_ha)',
        'rapeseed_and_mustard_area_(1000_ha)', 'rapeseed_and_mustard_production_(1000_tons)', 'rapeseed_and_mustard_yield_(kg_per_ha)',
        'safflower_area_(1000_ha)', 'safflower_production_(1000_tons)', 'safflower_yield_(kg_per_ha)',
        'castor_area_(1000_ha)', 'castor_production_(1000_tons)', 'castor_yield_(kg_per_ha)',
        'linseed_area_(1000_ha)', 'linseed_production_(1000_tons)', 'linseed_yield_(kg_per_ha)',
        'sunflower_area_(1000_ha)', 'sunflower_production_(1000_tons)', 'sunflower_yield_(kg_per_ha)',
        'soyabean_area_(1000_ha)', 'soyabean_production_(1000_tons)', 'soyabean_yield_(kg_per_ha)',
        'oilseeds_area_(1000_ha)', 'oilseeds_production_(1000_tons)', 'oilseeds_yield_(kg_per_ha)',
        'sugarcane_area_(1000_ha)', 'sugarcane_production_(1000_tons)', 'sugarcane_yield_(kg_per_ha)',
        'cotton_area_(1000_ha)', 'cotton_production_(1000_tons)', 'cotton_yield_(kg_per_ha)',
        'fruits_area_(1000_ha)', 'vegetables_area_(1000_ha)', 'fruits_and_vegetables_area_(1000_ha)',
        'potatoes_area_(1000_ha)', 'onion_area_(1000_ha)', 'fodder_area_(1000_ha)'
    ]
    
    # Initialize all crop columns to 0
    for col in crop_columns:
        features[col] = 0.0
    
    # Set values for the selected crop
    crop_lower = selected_crop.lower()
    
    # Map crop to column prefixes
    crop_mapping = {
        'rice': 'rice',
        'wheat': 'wheat',
        'maize': 'maize',
        'sorghum': 'sorghum',
        'pearl millet': 'pearl_millet',
        'barley': 'barley',
        'chickpea': 'chickpea',
        'pigeonpea': 'pigeonpea',
        'groundnut': 'groundnut',
        'sugarcane': 'sugarcane',
        'cotton': 'cotton',
        'soyabean': 'soyabean'
    }
    
    if crop_lower in crop_mapping:
        prefix = crop_mapping[crop_lower]
        features[f'{prefix}_area_(1000_ha)'] = area
        features[f'{prefix}_production_(1000_tons)'] = production
        # Yield will be calculated if area > 0
        if area > 0:
            features[f'{prefix}_yield_(kg_per_ha)'] = (production / area) * 1000
        else:
            features[f'{prefix}_yield_(kg_per_ha)'] = 0
    
    # Create DataFrame
    input_df = pd.DataFrame([features])
    
    return input_df

# Load label encoders if available
@st.cache_resource
def load_label_encoders():
    """Load label encoders for categorical variables"""
    try:
        encoders = joblib.load('label_encoders.pkl')
        return encoders
    except:
        return None

label_encoders = load_label_encoders()

# Function to encode categorical variables
def encode_categorical_features(df, encoders):
    """Apply label encoding to categorical features"""
    if encoders:
        for col, encoder in encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except ValueError:
                    # Use first class if value not seen
                    df[col] = 0
    return df

# Main content area - Display input summary
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📍 Location", f"{district}, {state}")
with col2:
    st.metric("🌾 Crop", selected_crop)
with col3:
    st.metric("📅 Year", year)

# Display area and production
col1, col2 = st.columns(2)
with col1:
    st.metric("📊 Cultivated Area", f"{area:,.2f} thousand hectares")
with col2:
    st.metric("📈 Expected Production", f"{production:,.2f} thousand tons")

# Information box
st.markdown("""
<div class="info-box">
    💡 <strong>Note:</strong> This system predicts crop yield based on historical agricultural data 
    from various Indian states. The model considers multiple crops and their cultivation patterns.
</div>
""", unsafe_allow_html=True)

# Prediction button
predict_button = st.button("🌾 Predict Crop Yield", type="primary", use_container_width=True)

# Make prediction when button is clicked
if predict_button:
    with st.spinner("🌾 Analyzing crop data and predicting yield..."):
        try:
            # Create all features
            input_features = create_all_features(state, district, year, selected_crop, area, production)
            
            # Encode categorical features
            if label_encoders:
                input_features = encode_categorical_features(input_features, label_encoders)
            
            # Ensure all expected features are present
            if expected_features:
                # Add missing columns with default values
                for col in expected_features:
                    if col not in input_features.columns:
                        input_features[col] = 0
                # Reorder columns to match training data
                input_features = input_features[expected_features]
            
            # Apply scaling if available
            if scaler:
                numeric_cols = input_features.select_dtypes(include=[np.number]).columns
                input_features[numeric_cols] = scaler.transform(input_features[numeric_cols])
            
            # Make prediction
            prediction = model.predict(input_features)[0]
            
            # Display prediction result
            st.markdown("---")
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f"""
            <h2>📈 Predicted {selected_crop} Yield</h2>
            <h1 style="font-size: 3rem; margin: 0;">{prediction:,.2f}</h1>
            <h3 style="margin-top: 0;">tons per hectare</h3>
            """, unsafe_allow_html=True)
            
            # Calculate total predicted production
            total_predicted_production = prediction * area
            st.markdown(f"""
            <p style="font-size: 1.2rem; margin-top: 1rem;">
            📊 Total Predicted Production: <strong>{total_predicted_production:,.2f} thousand tons</strong>
            </p>
            """, unsafe_allow_html=True)
            
            # Yield interpretation
            if prediction < 1:
                st.warning("⚠️ Low yield prediction. Consider reviewing agricultural practices.")
            elif prediction < 2.5:
                st.info("📊 Moderate yield prediction. Good potential for improvement.")
            elif prediction < 4:
                st.success("✅ Good yield prediction! Efficient farming practices detected.")
            else:
                st.success("🌟 Excellent yield prediction! Optimal conditions identified.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display input summary
            st.markdown("### 📋 Prediction Summary")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Input Parameters:**")
                st.write(f"- Location: {district}, {state}")
                st.write(f"- Crop: {selected_crop}")
                st.write(f"- Year: {year}")
            with col_b:
                st.write("**Agricultural Inputs:**")
                st.write(f"- Area: {area:,.2f} thousand hectares")
                st.write(f"- Expected Production: {production:,.2f} thousand tons")
                st.write(f"- Predicted Yield: {prediction:.2f} tons/hectare")
            
            st.markdown("""
            <div class="info-box">
                ℹ️ <strong>Note:</strong> This prediction is based on historical agricultural data and machine learning models. 
                Actual yields may vary due to weather conditions, pest attacks, soil quality, and other environmental factors.
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.info("Please check if all input values are valid and try again.")

# Footer
st.markdown("""
<div class="footer">
    <p>🌾 Crop Yield Prediction System | Powered by Machine Learning | Data-driven Agricultural Insights</p>
    <p style="font-size: 0.8rem;">Based on Indian Agricultural Data (2010-2017)</p>
</div>
""", unsafe_allow_html=True)

# Run the app with: streamlit run app.py
