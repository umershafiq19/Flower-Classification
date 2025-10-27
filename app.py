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

app = Flask(__name__)

# Configuration
IMG_SIZE = 180
MODEL_NAME = "flowers_classifier"
MODEL_STAGE = "Production"

# DagsHub Credentials - UPDATE THESE!
DAGSHUB_USERNAME = "<YOUR_DAGSHUB_USERNAME>"
DAGSHUB_REPO = "<YOUR_DAGS_HUB_REPO>"
DAGSHUB_TOKEN = "<DAGSHUB_TOKEN>"

# Class names
CLASS_NAMES = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']

model = None


def load_model_from_registry():
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
            model = tf.keras.models.load_model('flower_classifier_model.keras')
            print("✅ Local model loaded!")
        except:
            print("❌ No model loaded!")
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
