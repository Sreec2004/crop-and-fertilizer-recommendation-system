import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from streamlit_lottie import st_lottie
import requests

# Function to load Lottie animations from URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load the models
crop_stacking_clf = joblib.load('crop_stacking_classifier.joblib')
fertilizer_stacking_clf = joblib.load('fertilizer_stacking_classifier.joblib')
link_stacking_clf = joblib.load('link_stacking_classifier.joblib')

# Load the scaler, feature names, crop names, fertilizer names, and links mapping
scaler = joblib.load('scaler.joblib')
feature_names = joblib.load('feature_names.joblib')
crop_names = joblib.load('crop_names.joblib')
fertilizer_names = joblib.load('fertilizer_names.joblib')
links = joblib.load('links.joblib')

# Define the numerical columns
numeric_cols = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Rainfall', 'Temperature']

# Define the district names
district_names = ['Kolhapur', 'Solapur', 'Satara', 'Sangli', 'Pune']

# Function to map predicted crop, fertilizer, and link codes to names and URLs
def get_crop_name(crop_code):
    return crop_names[crop_code]

def get_fertilizer_name(fertilizer_code):
    return fertilizer_names[fertilizer_code]

def get_link(link_code):
    return links[link_code]

# Load Lottie animations
lottie_url = "https://assets3.lottiefiles.com/packages/lf20_lk80fpsm.json"
lottie_success_url = "https://assets7.lottiefiles.com/packages/lf20_vd7ybbsl.json"
lottie_animation = load_lottieurl(lottie_url)
lottie_success = load_lottieurl(lottie_success_url)

# Streamlit app
st.set_page_config(page_title="Crop and Fertilizer Recommendation System", page_icon="🌾")
st.title('🌾 Crop and Fertilizer Recommendation System')

# Display Lottie animation
if lottie_animation:
    st_lottie(lottie_animation, height=200)

# Input fields
district_name = st.selectbox('Select District Name', district_names)
nitrogen = st.number_input('Enter Nitrogen Level', value=0.0)
phosphorus = st.number_input('Enter Phosphorus Level', value=0.0)
potassium = st.number_input('Enter Potassium Level', value=0.0)
ph = st.number_input('Enter pH Level', value=0.0)
rainfall = st.number_input('Enter Rainfall', value=0.0)
temperature = st.number_input('Enter Temperature', value=0.0)

# Predict button
if st.button('Predict'):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'Nitrogen': [nitrogen],
        'Phosphorus': [phosphorus],
        'Potassium': [potassium],
        'pH': [ph],
        'Rainfall': [rainfall],
        'Temperature': [temperature],
    })

    # Add one-hot encoded district name columns
    for district in district_names:
        input_data[f'District_Name_{district}'] = [1 if district_name == district else 0]

    # Ensure all required columns are present
    for col in feature_names:
        if col not in input_data.columns:
            input_data[col] = 0

    # Scale the input data
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # Reorder the columns to match the training data
    input_data = input_data[feature_names]

    # Predict crop name using the crop stacking classifier
    predicted_crop_code = crop_stacking_clf.predict(input_data)[0]
    predicted_crop = get_crop_name(predicted_crop_code)

    # Predict fertilizer using the fertilizer stacking classifier
    predicted_fertilizer_code = fertilizer_stacking_clf.predict(input_data)[0]
    predicted_fertilizer = get_fertilizer_name(predicted_fertilizer_code)

    # Predict link using the link stacking classifier
    predicted_link_code = link_stacking_clf.predict(input_data)[0]
    fertilizer_link = get_link(predicted_link_code)

    # Display the results with animations and enhanced style
    st.success(f'Recommended Crop: **{predicted_crop}**')
    st.info(f'Recommended Fertilizer: **{predicted_fertilizer}**')
    st.write(f"[Click here for more information on {predicted_fertilizer}]({fertilizer_link})")

    # Display Lottie success animation
    if lottie_success:
        st_lottie(lottie_success, height=200)
