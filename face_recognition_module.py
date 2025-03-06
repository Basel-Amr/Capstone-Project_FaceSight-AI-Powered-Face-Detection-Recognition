# =============================== #
# 📌 Standard Library Imports
# =============================== #
import os
import sys
import glob
import shutil
import time
import datetime
import random
import warnings

# =============================== #
# 📌 Numerical & Data Handling
# =============================== #
import numpy as np
import pandas as pd

# =============================== #
# 📌 Image Processing Libraries
# =============================== #
import cv2
from PIL import Image

# =============================== #
# 📌 Deep Learning & Model Handling
# =============================== #
import tensorflow as tf
import tensorflow.lite as tflite
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model


# =============================== #
# 📌 Data Science & Visualization
# =============================== #
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# =============================== #
# 📌 Interactive & Console Utilities
# =============================== #
from IPython.display import display
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import print as rprint
from tqdm import tqdm

from FaceDetector import FaceDetector

# =============================== #
# 📌 Load Pre-trained Models
# =============================== #

# Load Face Detection Model
prototxt_path = "models/deploy.prototxt"
model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
face_detector = FaceDetector(prototxt_path, model_path)



# =============================== #
# ⚠️ Suppress Warnings
# =============================== #
warnings.filterwarnings("ignore")

def load_image(image_path):
    """Loads an image from a given path and converts it to RGB."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is None:
        raise FileNotFoundError(f"Error: Image not found at {image_path}")
    return image

def preprocess_face(image):
    """Preprocesses the face image to fit InceptionV3 input requirements."""
    faces = face_detector.detect_faces(image)
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")
    processed_face = face_detector.preprocess_image(faces[0])
    processed_face = np.expand_dims(processed_face, axis=0)
    img = preprocess_input(processed_face)
    return img


def get_embedding(model, face_image):
    """Extracts a 2048-d feature vector from the face image using InceptionV3."""
    processed_face = preprocess_face(face_image)
    embedding = model.predict(processed_face)[0]  # Get feature vector
    return embedding

from sklearn.preprocessing import normalize

def compare_faces(embedding1, embedding2, threshold=0.8):
    """Compares two face embeddings using cosine similarity and determines if they match."""
    embedding1 = normalize([embedding1])[0]
    embedding2 = normalize([embedding2])[0]
    similarity = 1 - cosine(embedding1, embedding2)
    is_match = similarity >= threshold
    return is_match, similarity


def recognize_face(known_faces, known_names, test_face, model, threshold=0.8):
    """Compares a test face against known faces and returns the recognized name or 'Non-Defined'."""
    test_embedding = get_embedding(model, test_face)
    best_match = ('Non-Defined', 0)  # Default: No match
    
    for name, known_embedding in zip(known_names, known_faces):
        is_match, similarity = compare_faces(known_embedding, test_embedding, threshold)

        if is_match and similarity > best_match[1]:
            best_match = (name, similarity)
    
    if best_match[1] < threshold + 0.1:  # Adding margin to avoid false positives
        return 'Non-Defined'
    
    return best_match[0]

