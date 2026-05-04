from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="DermaVision API")

# ✅ Load model
model = tf.keras.models.load_model("models/best_skin_model.keras")

# ✅ Class labels
class_names = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

# ✅ Full names
label_map = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions"
}

# ✅ Preprocessing
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ✅ Home route
@app.get("/")
def home():
    return {"message": "DermaVision API is running 🚀"}

# ✅ Prediction route (with correct dummy metadata)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        processed = preprocess_image(image)

        # 🔥 FINAL FIX (19 features)
        dummy_metadata = np.zeros((1, 19))

        # 🔥 Multimodal prediction
        prediction = model.predict([processed, dummy_metadata])

        predicted_index = np.argmax(prediction)
        predicted_label = class_names[predicted_index]
        confidence = float(np.max(prediction))

        return {
            "prediction_code": predicted_label,
            "prediction": label_map[predicted_label],
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}
