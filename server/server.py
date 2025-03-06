import cv2
import numpy as np
import tensorflow as tf
import json
import logging
import ssl
import os
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaRecorder
import asyncio
from aiohttp import web
from av import VideoFrame
import concurrent.futures
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
        bounding_boxes = []
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
                    bounding_boxes.append((x, y, x_max, y_max))
        return faces, bounding_boxes

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Face Detection Model
prototxt_path = "models/deploy.prototxt"
model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
face_detector = FaceDetector(prototxt_path, model_path)

# Load Keras Model (replace with your model later if needed)
keras_model_path = "models/InceptionV3_model.keras"
loaded_model_keras = load_model(keras_model_path)

# Create thread pool for CPU-intensive operations
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
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


# Prediction function
def predict_face(image):
    """
    Function to detect faces and predict their class using the pre-trained model.

    Args:
    - image (np.array): Input image in numpy array format.

    Returns:
    - predicted_class_name (str): Predicted class name of the face.
    - confidence (float): Confidence of the prediction.
    """
    # Detect faces
    faces, bounding_boxes = face_detector.detect_faces(image)
    
    if not faces:
        return None, None  # No faces detected

    # Process the first face
    face = faces[0]
    processed_face = face_detector.preprocess_image(face)

    # Predict using the loaded model
    face_array = np.expand_dims(processed_face, axis=0)  # Add batch dimension
    prediction = loaded_model_keras.predict(face_array,verbose=0)
    predicted_class_index = np.argmax(prediction)  # Get the index of the highest probability
    predicted_class_name = [key for key, value in Classes.items() if value == predicted_class_index][0]
    confidence = np.max(prediction)  # Get confidence level
    
    return predicted_class_name, confidence




class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.frame_count = 0
        self.last_boxes = []
        self.last_scores = []
        
        # Pre-allocate buffers
        self.img_buffer = None
        self.draw_buffer = None

    async def recv(self):
        """
        Receive and process frames from the video stream. Performs face detection
        and classification, then draws results on the frame.

        Returns:
        - new_frame (VideoFrame): Processed video frame with detections.
        """
        frame = await self.track.recv()
        self.frame_count += 1

        try:
            # Convert frame to numpy array efficiently
            img = frame.to_ndarray(format="bgr24")
            
            # Initialize buffers if needed
            if self.img_buffer is None:
                self.img_buffer = np.zeros_like(img)
                self.draw_buffer = np.zeros_like(img)

            # Copy frame to buffer
            np.copyto(self.img_buffer, img)
            
            # Run face detection and prediction in thread pool to avoid blocking
            predicted_class_name, confidence = await asyncio.get_event_loop().run_in_executor(
                thread_pool, predict_face, self.img_buffer
            )
            
            if predicted_class_name:
                # Draw the bounding box and the prediction result on the image
                height, width = img.shape[:2]
                
                # Here, use face_detector.detect_faces() to get the bounding box
                faces, bounding_boxes = face_detector.detect_faces(self.img_buffer)
                
                for (x, y, x_max, y_max) in bounding_boxes:
                    # Draw the rectangle around the face
                    cv2.rectangle(img, (x, y), (x_max, y_max), (0, 255, 0), 2)
                
                # Draw the prediction result text on the image
                cv2.putText(img, f"{predicted_class_name}: {confidence:.2f}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Create new frame with the drawn results
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame

async def index(request):
    content = open(os.path.join(os.path.dirname(__file__), "templates/index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            video_transform = VideoTransformTrack(track)
            pc.addTrack(video_transform)

            @track.on("ended")
            async def on_ended():
                logger.info("Track ended")

    # Process offer
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )


# Setup CORS middleware
async def cors_middleware(app, handler):
    async def middleware(request):
        if request.method == 'OPTIONS':
            response = web.Response()
        else:
            response = await handler(request)
        
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    return middleware


# Application setup
async def init_app():
    app = web.Application(middlewares=[cors_middleware])
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.router.add_options("/offer", lambda r: web.Response())  # Handle CORS preflight
    
    return app


# SSL configuration
def init_ssl():
    if os.path.exists('cert.pem') and os.path.exists('key.pem'):
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain('cert.pem', 'key.pem')
        logger.info("SSL context created successfully")
        return ssl_context
    logger.warning("SSL certificates not found, running without HTTPS")
    return None


# Cleanup
async def cleanup(app):
    # Close thread pool
    thread_pool.shutdown(wait=True)
    logger.info("Thread pool shut down")


if __name__ == "__main__":
    try:
        ssl_context = init_ssl()
        app = asyncio.get_event_loop().run_until_complete(init_app())
        app.on_cleanup.append(cleanup)
        
        web.run_app(
            app,
            host="0.0.0.0",
            port=8080,
            ssl_context=ssl_context,
            access_log=logger
        )
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        thread_pool.shutdown(wait=False)
