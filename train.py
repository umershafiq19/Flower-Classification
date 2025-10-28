"""
MLflow + DagsHub Flower Classification - Training Script
"""

import os
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
IMG_SIZE = 180
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# DagsHub Credentials - UPDATE THESE!
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

# Set MLflow tracking
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("DAGSHUB_USERNAME")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("DAGSHUB_TOKEN")
mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")


def load_and_prepare_data():
    print("Loading TensorFlow Flowers dataset...")
    import tensorflow_datasets as tfds
    
    (train_ds, val_ds), info = tfds.load(
        'tf_flowers',
        split=['train[:80%]', 'train[80%:]'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    
    class_names = info.features['label'].names
    print(f"Found {len(class_names)} classes: {class_names}")
    
    def prepare(ds):
        ds = ds.map(lambda x, y: (tf.image.resize(x, [IMG_SIZE, IMG_SIZE]), y))
        ds = ds.batch(BATCH_SIZE)
        ds = ds.cache().prefetch(tf.data.AUTOTUNE)
        return ds
    
    return prepare(train_ds.shuffle(1000)), prepare(val_ds), class_names


def create_model(num_classes):
    return keras.Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])


def train_and_log_model():
    with mlflow.start_run(run_name=f"flowers_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("batch_size", BATCH_SIZE)
        
        # Load data
        train_ds, val_ds, class_names = load_and_prepare_data()
        
        # Create model
        model = create_model(len(class_names))
        model.compile(
            optimizer=keras.optimizers.Adam(LEARNING_RATE),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        print("\nTraining model...")
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
        
        # Log metrics
        for epoch in range(EPOCHS):
            mlflow.log_metric("train_acc", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_acc", history.history['val_accuracy'][epoch], step=epoch)
        
        # Save model
        model.save("flower_classifier_model.keras")
        
        # Log model
        mlflow.keras.log_model(model, "model", registered_model_name="flowers_classifier")
        
        print(f"\n✅ Training complete!")
        print(f"✅ Model registered as 'flowers_classifier'")


if __name__ == "__main__":
    if DAGSHUB_USERNAME == "your_username":
        print("⚠️  Update credentials in train.py first!")
        response = input("Continue with local MLflow? (y/n): ")
        if response.lower() != 'y':
            exit(1)
        mlflow.set_tracking_uri("file:./mlruns")
    
    train_and_log_model()
