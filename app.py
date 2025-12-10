import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
import os

# --- Page Configuration ---
st.set_page_config(page_title="DermaVision AI", page_icon="🩺", layout="wide")

# --- UI Styling (High Contrast Fix) ---
st.markdown("""
<style>
    /* Force Light Background */
    .stApp { background-color: #f4f6f9; }
    
    /* Force Dark Text for Readability */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #2c3e50 !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Button Styling */
    div.stButton > button:first-child { 
        background-color: #3498db; 
        color: white !important; 
        border-radius: 8px; 
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #2980b9;
    }
    
    /* Card/Metric Box Styling */
    div[data-testid="stMetricValue"] {
        color: #2c3e50 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Load Model with Debugging ---
@st.cache_resource
def load_derma_model():
    model_path = 'models/best_skin_model.keras'
    
    # Debug Check: Does the file exist?
    if not os.path.exists(model_path):
        st.error(f"❌ Error: Model file not found at `{model_path}`")
        st.warning("👉 Please create a folder named 'models' and put 'best_skin_model.keras' inside it.")
        st.stop()
        
    model = tf.keras.models.load_model(model_path)
    return model

try:
    model = load_derma_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 2. Class Labels ---
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
class_full_names = {
    'akiec': 'Actinic Keratoses',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi (Mole)',
    'vasc': 'Vascular Lesions'
}

# --- 3. Sidebar (Patient Metadata) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80)
    st.title("Patient Data")
    st.info("ℹ️ This model fuses Image + Clinical Data for higher accuracy.")
    
    age = st.slider("Patient Age", 0, 100, 45)
    sex = st.selectbox("Sex", ["Male", "Female", "Unknown"])
    loc = st.selectbox("Localization", [
        "Abdomen", "Back", "Chest", "Ear", "Face", "Foot", "Genital", 
        "Hand", "Lower Extremity", "Neck", "Scalp", "Trunk", "Upper Extremity", "Unknown"
    ])

# --- 4. Helper: Metadata Vector ---
def build_meta_vector(age, sex, loc):
    # 1. Sex
    sex_v = [0, 0, 0]
    if sex == 'Female': sex_v[0] = 1
    elif sex == 'Male': sex_v[1] = 1
    else: sex_v[2] = 1
    
    # 2. Localization
    locs = ["abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital", 
            "hand", "lower extremity", "neck", "scalp", "trunk", "upper extremity", "unknown"]
    
    loc_v = [0] * len(locs)
    user_loc = loc.lower()
    if user_loc in locs:
        idx = locs.index(user_loc)
        loc_v[idx] = 1
        
    # 3. Age
    age_norm = [age / 100.0]
    
    return np.array(sex_v + loc_v + age_norm).reshape(1, -1)

# --- 5. Main Interface ---
st.title("🩺 DermaVision AI")
st.write("Upload a dermatoscopic image for AI-assisted diagnosis.")

col1, col2 = st.columns([1, 1.5])

with col1:
    uploaded_file = st.file_uploader("Upload Lesion Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Clinical View", use_column_width=True)

with col2:
    if uploaded_file:
        if st.button("Analyze Lesion"):
            with st.spinner("Processing image & metadata..."):
                # A. Image Preprocessing
                img = image.resize((224, 224))
                img_array = np.array(img)
                img_preprocessed = preprocess_input(img_array)
                img_batch = np.expand_dims(img_preprocessed, axis=0)
                
                # B. Metadata Preprocessing
                meta_batch = build_meta_vector(age, sex, loc)
                
                # C. Prediction
                # Try-Catch for Shape Mismatch errors
                try:
                    predictions = model.predict({'image_input': img_batch, 'meta_input': meta_batch})
                except ValueError as e:
                    st.error(f"⚠️ Shape Error: {e}")
                    st.warning("This usually means the Metadata Vector doesn't match the training size. Check the 'locs' list in the code.")
                    st.stop()
                
                # D. Results
                pred_idx = np.argmax(predictions)
                pred_label = classes[pred_idx]
                confidence = np.max(predictions)

            # Display Result
            st.success("Analysis Complete")
            
            st.markdown(f"""
            <div style="padding: 20px; background-color: white; border-radius: 10px; border-left: 6px solid #e74c3c; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="margin:0; color: #555;">Diagnosis</h3>
                <h1 style="color: #e74c3c; margin: 10px 0;">{class_full_names[pred_label]}</h1>
                <p style="color: #555;">Confidence: <b>{confidence*100:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("### Probability Distribution")
            chart_data = pd.DataFrame({
                "Condition": [class_full_names[c] for c in classes],
                "Probability": predictions[0]
            })
            st.bar_chart(chart_data.set_index("Condition"))

    elif not uploaded_file:
        st.info("👈 Please upload an image to start.")
