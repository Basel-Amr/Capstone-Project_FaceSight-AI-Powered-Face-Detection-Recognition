import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import onnxruntime as ort
import time
from FaceDetector import FaceDetector

# =============================== #
# 📌 Load Pre-trained Models
# =============================== #

# Load Face Detection Model
prototxt_path = "models/deploy.prototxt"
model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
face_detector = FaceDetector(prototxt_path, model_path)

# Load Keras Model 
keras_model_path = "models/InceptionV3_model.keras"
loaded_model_keras = load_model(keras_model_path)

Classes = {
    'pins_Adriana Lima': 0, 'pins_Alex Lawther': 1, 'pins_Alexandra Daddario': 2, 'pins_Alvaro Morte': 3, 'pins_Amanda Crew': 4,
    'pins_Andy Samberg': 5, 'pins_Anne Hathaway': 6, 'pins_Anthony Mackie': 7, 'pins_Avril Lavigne': 8, 'pins_Ben Affleck': 9,
    'pins_Bill Gates': 10, 'pins_Bobby Morley': 11, 'pins_Brenton Thwaites': 12, 'pins_Brian J. Smith': 13, 'pins_Brie Larson': 14,
    'pins_Chris Evans': 15, 'pins_Chris Hemsworth': 16, 'pins_Chris Pratt': 17, 'pins_Christian Bale': 18, 'pins_Cristiano Ronaldo': 19,
    'pins_Danielle Panabaker': 20, 'pins_Dominic Purcell': 21, 'pins_Dwayne Johnson': 22, 'pins_Eliza Taylor': 23, 'pins_Elizabeth Lail': 24,
    'pins_Emilia Clarke': 25, 'pins_Emma Stone': 26, 'pins_Emma Watson': 27, 'pins_Gwyneth Paltrow': 28, 'pins_Henry Cavil': 29,
    'pins_Hugh Jackman': 30, 'pins_Inbar Lavi': 31, 'pins_Irina Shayk': 32, 'pins_Jake Mcdorman': 33, 'pins_Jason Momoa': 34,
    'pins_Jennifer Lawrence': 35, 'pins_Jeremy Renner': 36, 'pins_Jessica Barden': 37, 'pins_Jimmy Fallon': 38, 'pins_Johnny Depp': 39,
    'pins_Josh Radnor': 40, 'pins_Katharine Mcphee': 41, 'pins_Katherine Langford': 42, 'pins_Keanu Reeves': 43, 'pins_Krysten Ritter': 44,
    'pins_Leonardo DiCaprio': 45, 'pins_Lili Reinhart': 46, 'pins_Lindsey Morgan': 47, 'pins_Lionel Messi': 48, 'pins_Logan Lerman': 49,
    'pins_Madelaine Petsch': 50, 'pins_Maisie Williams': 51, 'pins_Maria Pedraza': 52, 'pins_Marie Avgeropoulos': 53, 'pins_Mark Ruffalo': 54,
    'pins_Mark Zuckerberg': 55, 'pins_Megan Fox': 56, 'pins_Miley Cyrus': 57, 'pins_Millie Bobby Brown': 58, 'pins_Morena Baccarin': 59,
    'pins_Morgan Freeman': 60, 'pins_Nadia Hilker': 61, 'pins_Natalie Dormer': 62, 'pins_Natalie Portman': 63, 'pins_Neil Patrick Harris': 64,
    'pins_Pedro Alonso': 65, 'pins_Penn Badgley': 66, 'pins_Rami Malek': 67, 'pins_Rebecca Ferguson': 68, 'pins_Richard Harmon': 69,
    'pins_Rihanna': 70, 'pins_Robert De Niro': 71, 'pins_Robert Downey Jr': 72, 'pins_Sarah Wayne Callies': 73, 'pins_Selena Gomez': 74,
    'pins_Shakira Isabel Mebarak': 75, 'pins_Sophie Turner': 76, 'pins_Stephen Amell': 77, 'pins_Taylor Swift': 78, 'pins_Tom Cruise': 79,
    'pins_Tom Hardy': 80, 'pins_Tom Hiddleston': 81, 'pins_Tom Holland': 82, 'pins_Tuppence Middleton': 83, 'pins_Ursula Corbero': 84,
    'pins_Wentworth Miller': 85, 'pins_Zac Efron': 86, 'pins_Zendaya': 87, 'pins_Zoe Saldana': 88, 'pins_alycia dabnem carey': 89,
    'pins_amber heard': 90, 'pins_barack obama': 91, 'pins_barbara palvin': 92, 'pins_camila mendes': 93, 'pins_elizabeth olsen': 94,
    'pins_ellen page': 95, 'pins_elon musk': 96, 'pins_gal gadot': 97, 'pins_grant gustin': 98, 'pins_jeff bezos': 99, 'pins_kiernen shipka': 100,
    'pins_margot robbie': 101, 'pins_melissa fumero': 102, 'pins_scarlett johansson': 103, 'pins_tom ellis': 104
}


# =============================== #
# 📌 Face Detection and Prediction Function
# =============================== #

def predict_face(image):
    # Detect faces
    faces = face_detector.detect_faces(image)
    
    if not faces:
        return None, None  # No faces detected

    # Process the first face
    face = faces[0]
    processed_face = face_detector.preprocess_image(face)

    # Predict using the loaded model
    face_array = np.expand_dims(processed_face, axis=0)  # Add batch dimension
    prediction = loaded_model_keras.predict(face_array)
    predicted_class_index = np.argmax(prediction)  # Get the index of the highest probability
    predicted_class_name = [key for key, value in Classes.items() if value == predicted_class_index][0]
    confidence = np.max(prediction)  # Get confidence level
    
    return predicted_class_name, confidence

# =============================== #
# 📌 Streamlit UI Components
# =============================== #

# Title
st.title("Face Detection and Classification")

# Image Upload Section
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read and process image
    image = Image.open(uploaded_image)
    image = np.array(image)

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict and display result
    class_name, confidence = predict_face(image)

    if class_name:
        st.success(f"Prediction: {class_name}")
        st.write(f"Confidence: {confidence:.2f}")
    else:
        st.warning("No faces detected in the image.")

# =============================== #
# 📌 ONNX Model Inference (Optional)
# =============================== #
# I have no GPU
# def predict_with_onnx(image):
#     input_name = onnx_model.get_inputs()[0].name
#     output_name = onnx_model.get_outputs()[0].name
#     face_array = np.expand_dims(image, axis=0)
#     prediction = onnx_model.run([output_name], {input_name: face_array.astype(np.float32)})[0]
#     predicted_class_index = np.argmax(prediction)
#     return predicted_class_index, np.max(prediction)

