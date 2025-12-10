import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# --- Page Configuration ---
st.set_page_config(page_title="DermaVision AI", page_icon="🩺", layout="wide")

# --- Custom Styling ---
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    div.stButton > button:first-child { background-color: #3498db; color: white; border-radius: 10px; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

# --- 1. Load Model ---
@st.cache_resource
def load_derma_model():
    # Helper to load custom objects if needed
    model = tf.keras.models.load_model('models/best_skin_model.keras')
    return model

try:
    model = load_derma_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 2. Class Labels (From your Notebook) ---
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

# --- 3. Sidebar (Patient Metadata Input) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80)
    st.title("Patient Metadata")
    st.info("Your model is **Multimodal**: It needs clinical data + image for best accuracy.")
    
    # Inputs matching your notebook's One-Hot Encoding
    age = st.slider("Patient Age", 0, 100, 45)
    sex = st.selectbox("Sex", ["Male", "Female", "Unknown"])
    loc = st.selectbox("Localization", [
        "Abdomen", "Back", "Chest", "Ear", "Face", "Foot", "Genital", 
        "Hand", "Lower Extremity", "Neck", "Scalp", "Trunk", "Upper Extremity", "Unknown"
    ])

# --- 4. Helper: Construct Metadata Vector ---
def build_meta_vector(age, sex, loc):
    # This must match the exact order of pd.get_dummies from training
    # Standard alphabetical order for pandas dummies
    
    # 1. Sex (3 columns: Female, Male, Unknown)
    sex_v = [0, 0, 0]
    if sex == 'Female': sex_v[0] = 1
    elif sex == 'Male': sex_v[1] = 1
    else: sex_v[2] = 1
    
    # 2. Localization (14 columns alphabetical)
    locs = ["abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital", 
            "hand", "lower extremity", "neck", "scalp", "trunk", "upper extremity", "unknown"]
    
    # Note: Your notebook used 'localization' prefix. We map closest user input.
    loc_v = [0] * len(locs)
    user_loc = loc.lower()
    if user_loc in locs:
        idx = locs.index(user_loc)
        loc_v[idx] = 1
        
    # 3. Age (Normalized)
    age_norm = [age / 100.0]
    
    # Combine: Sex + Localization + Age
    # Check your notebook for exact column count (likely ~18 features)
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
    if uploaded_file and st.button("Analyze Lesion"):
        with st.spinner("Processing image & metadata..."):
            # A. Image Preprocessing (ResNet50 Standard)
            img = image.resize((224, 224)) # ResNet expects 224
            img_array = np.array(img)
            
            # CRITICAL: Use ResNet preprocess_input (Zero-centers data)
            # Do NOT use / 255.0
            img_preprocessed = preprocess_input(img_array)
            img_batch = np.expand_dims(img_preprocessed, axis=0)
            
            # B. Metadata Preprocessing
            # We construct the vector based on sidebar inputs
            meta_batch = build_meta_vector(age, sex, loc)
            
            # Verify shapes (optional debug)
            # st.write(f"Img shape: {img_batch.shape}, Meta shape: {meta_batch.shape}")

            # C. Prediction (Dual Input)
            # We pass a DICTIONARY matching input layer names from your notebook
            # Names found in notebook: 'image_input', 'meta_input'
            predictions = model.predict({'image_input': img_batch, 'meta_input': meta_batch})
            
            # D. Result Parsing
            pred_idx = np.argmax(predictions)
            pred_label = classes[pred_idx]
            confidence = np.max(predictions)

        # Display
        st.success("Analysis Complete")
        
        # Metric Card
        st.markdown(f"""
        <div style="padding: 20px; background-color: white; border-radius: 10px; border-left: 6px solid #e74c3c;">
            <h3 style="margin:0; color: #555;">Diagnosis</h3>
            <h1 style="color: #e74c3c; margin: 10px 0;">{class_full_names[pred_label]}</h1>
            <p>Confidence: <b>{confidence*100:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed Chart
        st.write("### Probability Distribution")
        chart_data = pd.DataFrame({
            "Condition": [class_full_names[c] for c in classes],
            "Probability": predictions[0]
        })
        st.bar_chart(chart_data.set_index("Condition"))

    elif not uploaded_file:
        st.info("👈 Waiting for image upload...")
