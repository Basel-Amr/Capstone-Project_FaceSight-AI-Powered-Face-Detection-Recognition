# =============================== #
# 📌 Image Processing Libraries
# =============================== #
import cv2
from PIL import Image

# =============================== #
# 📌 Data Science & Visualization
# =============================== #
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import numpy as np
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


from tensorflow.keras.models import load_model

class FaceDetector:
    """ 🚀 Face Detection & Preprocessing Pipeline using OpenCV's DNN module. """

    def __init__(self, prototxt_path, model_path, confidence_threshold=0.5):
        try:
            self.model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            self.confidence_threshold = confidence_threshold
        except Exception as e:
            print(f"Error loading model: {e}")

    def detect_faces(self, image):
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                     mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
        self.model.setInput(blob)
        detections = self.model.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x_max, y_max) = box.astype("int")
                x, y = max(0, x), max(0, y)
                x_max, y_max = min(w, x_max), min(h, y_max)

                if x_max > x and y_max > y:
                    face = image[y:y_max, x:x_max]
                    faces.append(face)
        return faces

    def preprocess_image(self, image):
        """ Convert image to RGB and resize """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = cv2.resize(image, (299, 299))
        image = image.astype("float32") / 255.0  # Normalize
        return image

    def process_and_save_faces(self, image_path, save_path, show=False):
        """ Detect faces, preprocess, augment, and save them """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert once for visualization

        faces = self.detect_faces(image)

        if not faces:
            print("⚠️ No faces detected in image.")
            return

        for i, face in enumerate(faces):
            processed_face = self.preprocess_image(face)

            if show:
                print(f"Displaying face {i+1} in RGB format:")
                plt.imshow(processed_face)  # Should display the image in RGB
                plt.axis('off')
                plt.show()

            # Save the processed face
            save_name = f"{save_path}/processed_face_{i}.png"
            processed_rgb = cv2.cvtColor((processed_face * 255).astype("uint8"), cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
            cv2.imwrite(save_name, processed_rgb)
            

# Define Pathes
# ✅ Load Face Detection Model
# 📝 Model structure file
prototxt_path  = r"models\deploy.prototxt"
# 🏋️‍♂️ Pre-trained model file
model_path  = r"models\res10_300x300_ssd_iter_140000.caffemodel"

# Get the models
chechpoint_path = "models\InceptionV3_checkpoint.weights.h5"
h5_model_path = r"models\InceptionV3_model.h5"
keras_model_path = r"models\InceptionV3_model.keras"

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Verify successful loading
if net.empty():
    print("❌ Error: Failed to load the face detection model.")
else:
    print("✅ Face detection model loaded successfully!")

face_detector = FaceDetector(prototxt_path, model_path)