# 🩺 DermaVision: Multimodal Skin Lesion Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://skin-lesion-analyzer-macrkhfljjpy7ahxeqy7fv.streamlit.app/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://www.tensorflow.org/)
[![Dataset](https://img.shields.io/badge/Data-HAM10000-green.svg)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
[![Model](https://img.shields.io/badge/Model-ResNet50_Fusion-blue.svg)](https://arxiv.org/abs/1512.03385)

## 📄 Abstract
Melanoma is the deadliest form of skin cancer, but survival rates exceed 95% if detected early. While many AI models analyze images alone, real-world diagnosis relies on patient context (age, sex, anatomical site).

**DermaVision** is a **Multimodal Deep Learning System** that mimics this clinical process. It fuses **Dermatoscopic Imaging (CNN)** with **Clinical Metadata** to classify skin lesions into 7 diagnostic categories. The system is deployed as a high-contrast, accessibility-focused web application for real-time analysis.

> **[🔴 Launch Live Diagnostic Tool](https://skin-lesion-analyzer-macrkhfljjpy7ahxeqy7fv.streamlit.app/)**

## 📊 Dataset & Research
The model was trained on the **HAM10000 ("Human Against Machine with 10000 training images")** dataset, the gold standard for dermatoscopic research.

* **Dataset Source:** [Kaggle: Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
* **Data Composition:** 10,015 dermatoscopic images across 7 diagnostic categories.
* **Preprocessing:** Images were resized to `224x224`, normalized using ResNet50 standards, and augmented to handle class imbalance.

## 🧠 Diagnostic Capabilities
The system predicts the following 7 conditions:

| Class | Diagnosis | Clinical Significance |
|:-----:|-----------|-----------------------|
| **mel** | **Melanoma** | 🚨 **High Risk:** Malignant skin cancer. |
| **bcc** | **Basal Cell Carcinoma** | 🚨 **High Risk:** Common malignant growth. |
| **akiec** | **Actinic Keratoses** | ⚠️ **Risk:** Pre-cancerous / intraepithelial carcinoma. |
| **nv** | Melanocytic Nevi | ✅ Benign: Common mole. |
| **bkl** | Benign Keratosis | ✅ Benign: Seborrheic keratosis. |
| **df** | Dermatofibroma | ✅ Benign: Skin nodule. |
| **vasc** | Vascular Lesions | ✅ Benign: Cherry angiomas. |

## 🛠️ Technical Architecture
This project implements a **Dual-Stream Neural Network**:

1.  **Visual Stream (ResNet50):**
    * Extracts spatial features from skin images using Transfer Learning (ImageNet weights).
    * *Technique:* Global Average Pooling + Batch Normalization.
2.  **Metadata Stream (Dense Network):**
    * Processes clinical inputs (Age, Sex, Anatomical Site).
    * *Technique:* One-Hot Encoding matching the HAM10000 feature space.
3.  **Feature Fusion:**
    * Concatenates visual features (2048-dim) with clinical features (18-dim) before the final Softmax classification layer.
  
## 📂 Project Structure
* `models/best_skin_model.keras`: The trained Multimodal Keras model.
* `app.py`: The production Streamlit application.
* `.streamlit/config.toml`: Configuration for high-contrast UI rendering.
* `skin_cancer_multimodal.ipynb`: Research notebook.

---

## ⚠️ Medical Disclaimer
**This tool is for educational and research evaluation only.**
It is NOT a certified medical device. Predictions should never replace a professional diagnosis by a certified dermatologist. The model's accuracy is contingent on image quality and is intended to assist, not replace, clinical judgment.

## 📦 Installation
**Prerequisites:** Python 3.9+, TensorFlow 2.10+

```bash
# 1. Clone the repository
git clone [https://github.com/Muhammad-Shahan/DermaVision-Multimodal-AI.git](https://github.com/Muhammad-Shahan/DermaVision-Multimodal-AI.git)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Application
streamlit run app.py

