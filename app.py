"""
Flask Application for Flower Classification
"""

from flask import Flask, request, jsonify, render_template
import mlflow.keras
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf
import gdown  # ✅ to download from Google Drive

app = Flask(__name__)

# Configuration
IMG_SIZE = 180
MODEL_NAME = "flowers_classifier"
MODEL_STAGE = "Production"

# DagsHub Credentials - UPDATE THESE!
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

# Google Drive model download link
# Replace FILE_ID below with your actual Drive file ID
# (for example, if your shareable link is
#  https://drive.google.com/file/d/1ABCxyz123/view?usp=sharing
#  then FILE_ID = "1ABCxyz123")
#https://drive.google.com/file/d/1DL-yT9VpRe8A8tlTdEgUMluekjoUCL5Z/view?usp=sharing
GOOGLE_DRIVE_FILE_ID = "1DL-yT9VpRe8A8tlTdEgUMluekjoUCL5Z"
MODEL_LOCAL_PATH = "flower_classifier_model.keras"

# Class names
CLASS_NAMES = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']

model = None


def download_model_from_drive():
    """Download model from Google Drive if not present locally"""
    if not os.path.exists(MODEL_LOCAL_PATH):
        print("📥 Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_LOCAL_PATH, quiet=False)
        print("✅ Model downloaded successfully!")


def load_model_from_registry():
    """Load model from MLflow registry or Google Drive"""
    global model
    try:
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
        mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")

        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.keras.load_model(model_uri)
        print("✅ Model loaded from MLflow Registry!")
    except Exception as e:
        print(f"⚠️  Registry load failed: {e}")
        try:
            download_model_from_drive()
            model = tf.keras.models.load_model(MODEL_LOCAL_PATH)
            print("✅ Local model loaded from Google Drive!")
        except Exception as err:
            print(f"❌ Model loading failed: {err}")
            model = None


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image)
    return np.expand_dims(image_array, axis=0)


@app.route('/')
def home():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)

        predictions = model.predict(processed_image, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[0][predicted_idx])

        all_predictions = {
            CLASS_NAMES[i]: float(predictions[0][i])
            for i in range(len(CLASS_NAMES))
        }
        all_predictions = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))

        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'confidence_percentage': f"{confidence * 100:.2f}%",
            'all_predictions': all_predictions
        })
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    print("Starting Flask App...")
    load_model_from_registry()
    print("App running at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
