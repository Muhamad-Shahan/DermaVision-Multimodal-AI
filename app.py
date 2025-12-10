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

# --- 2. ADVANCED CSS (Fixing Popups & Expanders) ---
st.markdown("""
<style>
    /* 1. Main App Background */
    .stApp {
        background-color: #f4f6f9;
    }

    /* 2. GLOBAL TEXT COLOR - Default to Dark Blue-Grey */
    h1, h2, h3, h4, h5, h6, p, li, label, .stMarkdown, .stText {
        color: #1e293b !important;
    }

    /* 3. EXPANDER STYLING (User Guide) */
    /* Force the header to be Dark Grey with White Text */
    .streamlit-expanderHeader {
        background-color: #475569 !important; /* Dark Slate */
        color: white !important;
        border-radius: 8px;
    }
    /* Force the content inside to be readable */
    .streamlit-expanderContent {
        background-color: white !important;
        color: #1e293b !important;
        border: 1px solid #e2e8f0;
        border-radius: 0 0 8px 8px;
    }
    /* Fix the arrow icon color */
    .streamlit-expanderHeader svg {
        fill: white !important;
    }

    /* 4. DROPDOWN MENU STYLING (The List that pops up) */
    /* This targets the actual popup container in Streamlit */
    div[data-baseweb="popover"], div[data-baseweb="menu"] {
        background-color: #262730 !important; /* Dark Grey Background */
    }
    /* This targets the text options inside the dropdown */
    div[data-baseweb="option"] {
        color: white !important; /* White Text */
    }
    /* Highlight color when hovering over an option */
    div[data-baseweb="option"]:hover {
        background-color: #3498db !important;
        color: white !important;
    }

    /* 5. INPUT BOX STYLING (The box itself before clicking) */
    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput div[data-baseweb="input"] > div,
    .stTextInput div[data-baseweb="input"] > div {
        background-color: white !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1;
    }
    
    /* 6. SIDEBAR & CARDS */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 2px solid #e2e8f0;
    }
    .diagnosis-card {
        background-color: white;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    
    /* 7. BUTTONS */
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
    
    age = st.number_input("Patient Age", min_value=0, max_value=120, value=45, help="Age is a key factor in skin cancer risk.")
    sex = st.selectbox("Biological Sex", ["Male", "Female", "Unknown"])
    
    # RENAMED to Localization
    loc = st.selectbox("Localization", [
        "Back", "Lower Extremity", "Trunk", "Upper Extremity", "Abdomen", 
        "Face", "Chest", "Foot", "Neck", "Scalp", "Hand", "Ear", 
        "Genital", "Acral", "Unknown"
    ], help="Select the exact body part where the lesion is located.")
    
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

# USER GUIDE (Styled with Dark Header + White Text)
with st.expander("📖 New User Guide: How to use this tool", expanded=True):
    st.markdown("""
    1. **Enter Patient Data:** Use the sidebar to set Age, Sex, and Localization.
    2. **Upload Image:** Take a clear, close-up photo of the skin lesion and upload it.
    3. **Analyze:** Click the 'Run Diagnostic Analysis' button.
    4. **Review:** Check the AI prediction and risk confidence.
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
                    st.error(f"Shape Error: {e}")
                    st.stop()
                
                # Results
                top_idx = np.argmax(preds)
                label = classes[top_idx]
                full_name, status = class_details[label]
                conf = np.max(preds)
                
                # Styling Logic
                if "Cancer" in status:
                    border_color = "#e74c3c" # Red
                    bg_color = "#fdf2f2" 
                    icon = "🚨"
                else:
                    border_color = "#27ae60" # Green
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
