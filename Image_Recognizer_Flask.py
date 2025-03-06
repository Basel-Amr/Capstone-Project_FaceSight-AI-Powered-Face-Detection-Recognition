from flask import Flask, render_template, request, jsonify
import sqlite3
import os
import numpy as np
from werkzeug.utils import secure_filename
import cv2
from tensorflow.keras.models import load_model
from face_recognition_module import load_image, get_embedding, compare_faces
from FaceDetector import FaceDetector

app = Flask(__name__)

# Database path
DB_PATH = 'celebrity_embeddings.db'

# Load Face Detection Model
prototxt_path = "models/deploy.prototxt"
model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
face_detector = FaceDetector(prototxt_path, model_path)

# Load Keras Model
keras_model_path = "models/InceptionV3_model.keras"
model = load_model(keras_model_path)

UPLOAD_FOLDER = './static/uploads'  # Updated to static folder for serving images
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Helper function to check allowed extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load all celebrity embeddings from the database
def load_embeddings():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, embedding FROM celebrity_embeddings")
    rows = cursor.fetchall()
    known_faces = []
    known_names = []
    
    for row in rows:
        known_names.append(row[0])
        known_faces.append(np.frombuffer(row[1], dtype=np.float32))  # Convert binary to numpy array
    
    conn.close()
    return known_faces, known_names

known_faces, known_names = load_embeddings()

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for uploading image and recognizing
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process uploaded image
        uploaded_image = load_image(file_path)
        uploaded_embedding = get_embedding(model, uploaded_image)

        # Compare with known faces
        best_match = ('Non-Defined', 0)  # Default: No match
        
        for name, known_embedding in zip(known_names, known_faces):
            is_match, similarity = compare_faces(known_embedding, uploaded_embedding)
            if is_match and similarity > best_match[1]:
                best_match = (name, similarity)
        
        if best_match[1] < 0.6:
            return jsonify({'result': 'Non-Defined', 'image_path': '/static/uploads/' + filename})
        
        return jsonify({'result': best_match[0], 'image_path': '/static/uploads/' + filename})

    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    app.run(debug=False)
