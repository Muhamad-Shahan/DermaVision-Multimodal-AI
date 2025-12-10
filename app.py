import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tensorflow.keras.applications.resnet50 import preprocess_input
import os

# --- 1. PAGE CONFIGURATION (HCI Principle: Aesthetic Integrity) ---
st.set_page_config(
    page_title="DermaVision Pro",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM MEDICAL CSS (HCI Principle: Clarity) ---
st.markdown("""
<style>
    /* Main Background - Clean Medical Grey */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Sidebar - Professional Dark Blue */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e1e4e8;
    }
    
    /* Headers - High Readability Fonts */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
    }
    
    /* Result Cards - Shadow Depth */
    .result-card {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* Status Indicators */
    .status-danger { color: #e74c3c; font-weight: bold; }
    .status-safe { color: #27ae60; font-weight: bold; }
    .status-warning { color: #f39c12; font-weight: bold; }

    /* Custom Button Styling */
    div.stButton > button {
        width: 100%;
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_derma_model():
    model_path = 'models/best_skin_model.keras'
    if not os.path.exists(model_path):
        st.error(f"❌ System Error: Model file missing at {model_path}")
        st.stop()
    return tf.keras.models.load_model(model_path)

try:
    model = load_derma_model()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# --- 4. DATA STANDARDS (HAM10000 Exact Match) ---
# Official classes from HAM10000 paper
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
class_details = {
    'akiec': ('Actinic Keratoses', 'Pre-cancerous'),
    'bcc': ('Basal Cell Carcinoma', 'Cancerous'),
    'bkl': ('Benign Keratosis', 'Benign'),
    'df': ('Dermatofibroma', 'Benign'),
    'mel': ('Melanoma', 'Cancerous (High Risk)'),
    'nv': ('Melanocytic Nevi', 'Benign (Mole)'),
    'vasc': ('Vascular Lesion', 'Benign')
}

# --- 5. SIDEBAR (Patient Intake) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=60)
    st.markdown("### Patient Intake")
    st.info("ℹ️ Enter clinical metadata to improve diagnostic accuracy.")
    
    # Medical Inputs
    age = st.number_input("Age", min_value=0, max_value=120, value=45, step=1)
    sex = st.selectbox("Biological Sex", ["Male", "Female", "Unknown"])
    
    # UPDATED LOCALIZATION (Exact HAM10000 Values)
    loc = st.selectbox("Lesion Location", [
        "Back", "Lower Extremity", "Trunk", "Upper Extremity", "Abdomen", 
        "Face", "Chest", "Foot", "Neck", "Scalp", "Hand", "Ear", 
        "Genital", "Acral", "Unknown"
    ])
    
    st.markdown("---")
    st.caption("DermaVision Pro v1.2 | Powered by TensorFlow")

# --- 6. HELPER FUNCTIONS ---
def build_meta_vector(age, sex, loc):
    # One-Hot Encoding Logic matching training
    sex_v = [0, 0, 0]
    if sex == 'Female': sex_v[0] = 1
    elif sex == 'Male': sex_v[1] = 1
    else: sex_v[2] = 1
    
    # Official HAM10000 Localization Columns (Alphabetical)
    locs = ["abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital", 
            "hand", "lower extremity", "neck", "scalp", "trunk", "upper extremity", "unknown"]
    
    loc_v = [0] * len(locs)
    user_loc = loc.lower()
    if user_loc in locs:
        loc_v[locs.index(user_loc)] = 1
        
    return np.array(sex_v + loc_v + [age / 100.0]).reshape(1, -1)

# --- 7. MAIN DASHBOARD ---
# Header
col_header, col_logo = st.columns([4, 1])
with col_header:
    st.title("DermaVision Pro")
    st.markdown("##### AI-Assisted Dermatoscopy Analysis System")

# Layout: Left (Image) | Right (Diagnostics)
col_img, col_diag = st.columns([1, 1.3], gap="large")

with col_img:
    st.markdown("### 1. Image Acquisition")
    uploaded_file = st.file_uploader("Upload Dermoscopic Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        # Consistent Display Size
        st.image(image, caption="Current Specimen", use_column_width=True)
        
        # Quality Check (HCI: Error Prevention)
        if image.size[0] < 224 or image.size[1] < 224:
            st.warning("⚠️ Low resolution image detected. Diagnosis may be less accurate.")

with col_diag:
    st.markdown("### 2. Diagnostic Analysis")
    
    if uploaded_file:
        analyze_btn = st.button("RUN ANALYSIS", type="primary")
        
        if analyze_btn:
            with st.spinner("Processing ResNet50 Feature Extraction..."):
                # A. Preprocessing
                img_resized = image.resize((224, 224))
                img_array = preprocess_input(np.array(img_resized))
                img_batch = np.expand_dims(img_array, axis=0)
                meta_batch = build_meta_vector(age, sex, loc)
                
                # B. Prediction
                try:
                    preds = model.predict({'image_input': img_batch, 'meta_input': meta_batch})
                except ValueError:
                    st.error("Model input mismatch. Check metadata vector size.")
                    st.stop()
                
                # C. Interpretation
                top_idx = np.argmax(preds)
                top_label = classes[top_idx]
                top_name, condition_type = class_details[top_label]
                confidence = np.max(preds)
                
                # D. Display Results (The "Beautiful" Part)
                st.markdown(f"""
                <div class="result-card">
                    <h4 style="margin:0; color:#7f8c8d;">Primary Diagnosis</h4>
                    <h1 style="margin:5px 0; color:#2c3e50;">{top_name}</h1>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:18px; font-weight:bold; 
                            color:{'#e74c3c' if 'Cancer' in condition_type else '#27ae60'}">
                            {condition_type}
                        </span>
                        <span style="font-size:18px; font-weight:bold; color:#3498db;">
                            {confidence*100:.1f}% Confidence
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # E. Detailed Breakdown
                st.markdown("#### Differential Diagnosis")
                chart_df = pd.DataFrame({
                    "Condition": [class_details[c][0] for c in classes],
                    "Probability": preds[0]
                }).sort_values(by="Probability", ascending=False)
                
                st.dataframe(
                    chart_df.style.background_gradient(cmap="Blues"),
                    use_container_width=True,
                    hide_index=True
                )
                
                if 'Cancer' in condition_type and confidence > 0.6:
                    st.error("🚨 **High Risk Alert:** Immediate biopsy recommended.")
                elif confidence < 0.5:
                    st.warning("⚠️ **Low Confidence:** Consider repeating image capture.")

    else:
        # Empty State (HCI: Guidance)
        st.info("👈 Upload an image to activate the diagnostic engine.")
        st.markdown("""
        **System Capabilities:**
        * Detects Melanoma & Carcinoma
        * Differentiates benign moles (Nevi)
        * Multimodal analysis (Image + Age/Site)
        """)
