import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DermaVision Pro",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS STYLING (Simplified because config.toml handles the heavy lifting) ---
st.markdown("""
<style>
    /* 1. INPUT BOXES - Force White Background & Dark Text */
    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput div[data-baseweb="input"] > div,
    .stTextInput div[data-baseweb="input"] > div {
        background-color: white !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1;
    }
    
    /* 2. DROPDOWN MENU TEXT - Force Black */
    div[data-baseweb="popover"] li div {
        color: #1e293b !important;
    }
    
    /* 3. RESULT CARD */
    .diagnosis-card {
        background-color: white;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    
    /* 4. BUTTONS */
    div.stButton > button {
        background: linear-gradient(135deg, #0284c7, #0369a1);
        color: white !important;
        border: none;
        padding: 14px;
        font-size: 16px;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
    
    /* 5. EXPANDERS */
    .streamlit-expanderHeader {
        background-color: white !important;
        color: #1e293b !important;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_derma_model():
    model_path = 'models/best_skin_model.keras'
    if not os.path.exists(model_path):
        st.error(f"❌ System Error: Model missing at `{model_path}`")
        st.stop()
    return tf.keras.models.load_model(model_path)

try:
    model = load_derma_model()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# --- 4. DATA DEFINITIONS ---
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

# --- 5. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=70)
    st.title("Clinical Intake")
    st.markdown("---")
    
    age = st.number_input("Patient Age", min_value=0, max_value=120, value=45)
    sex = st.selectbox("Biological Sex", ["Male", "Female", "Unknown"])
    
    # RENAMED to Localization
    loc = st.selectbox("Localization", [
        "Back", "Lower Extremity", "Trunk", "Upper Extremity", "Abdomen", 
        "Face", "Chest", "Foot", "Neck", "Scalp", "Hand", "Ear", 
        "Genital", "Acral", "Unknown"
    ])
    
    st.markdown("---")
    st.caption("Session ID: " + str(hash(datetime.datetime.now()))[:8])

# --- 6. HELPER FUNCTIONS ---
def build_meta_vector(age, sex, loc):
    sex_v = [0, 0, 0]
    if sex == 'Female': sex_v[0] = 1
    elif sex == 'Male': sex_v[1] = 1
    else: sex_v[2] = 1
    
    locs = ["abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital", 
            "hand", "lower extremity", "neck", "scalp", "trunk", "upper extremity", "unknown"]
    
    loc_v = [0] * len(locs)
    if loc.lower() in locs:
        loc_v[locs.index(loc.lower())] = 1
        
    return np.array(sex_v + loc_v + [age / 100.0]).reshape(1, -1)

# --- 7. MAIN INTERFACE ---
col_logo, col_title = st.columns([1, 5])
with col_title:
    st.title("DermaVision Pro")
    st.markdown("**AI-Assisted Dermatoscopy Analysis System**")

# USER GUIDE
with st.expander("📖 New User Guide: How to use this tool", expanded=True):
    st.markdown("""
    1. **Enter Patient Data:** Set Age, Sex, and Localization in the sidebar.
    2. **Upload Image:** Upload a clear photo of the lesion.
    3. **Analyze:** Click 'Run Diagnostic Analysis'.
    """)

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("1. Specimen Acquisition")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Clinical Specimen", use_column_width=True)
        if image.size[0] < 224:
            st.warning("⚠️ Low Resolution Warning")

with col2:
    st.subheader("2. Diagnostic Engine")
    
    if uploaded_file:
        if st.button("RUN DIAGNOSTIC ANALYSIS"):
            with st.spinner("Processing..."):
                # Preprocessing
                img_resized = image.resize((224, 224))
                img_array = preprocess_input(np.array(img_resized))
                img_batch = np.expand_dims(img_array, axis=0)
                meta_batch = build_meta_vector(age, sex, loc)
                
                try:
                    preds = model.predict({'image_input': img_batch, 'meta_input': meta_batch})
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    st.stop()
                
                # Results
                top_idx = np.argmax(preds)
                label = classes[top_idx]
                full_name, status = class_details[label]
                conf = np.max(preds)
                
                # Styling
                if "Cancer" in status:
                    border_color = "#e74c3c"
                    bg_color = "#fdf2f2" 
                    icon = "🚨"
                else:
                    border_color = "#27ae60"
                    bg_color = "#f0fdf4" 
                    icon = "✅"
                
                # Result Card
                st.markdown(f"""
                <div class="diagnosis-card" style="border-top: 6px solid {border_color}; background-color: {bg_color};">
                    <h4 style="margin:0; color:#555 !important;">AI Prediction</h4>
                    <h1 style="margin:10px 0; color:#1e293b !important;">{icon} {full_name}</h1>
                    <div style="display:flex; justify-content:center; gap:20px; margin-top:15px;">
                        <span style="background:white; padding:5px 15px; border-radius:15px; border:1px solid {border_color}; color:{border_color}; font-weight:bold;">
                            {status}
                        </span>
                        <span style="background:white; padding:5px 15px; border-radius:15px; border:1px solid #3498db; color:#3498db; font-weight:bold;">
                            {conf*100:.1f}% Confidence
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                st.markdown("#### Differential Probabilities")
                
                res_df = pd.DataFrame({
                    "Condition": [class_details[c][0] for c in classes],
                    "Risk Level": [class_details[c][1] for c in classes],
                    "Probability": [f"{p*100:.2f}%" for p in preds[0]]
                }).sort_values(by="Probability", ascending=False)
                
                st.table(res_df)
    else:
        st.info("👈 Waiting for image upload...")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 12px;">
    <strong>Medical Disclaimer:</strong> This tool is for educational purposes only. 
    AI predictions should never replace professional medical advice.
</div>
""", unsafe_allow_html=True)
