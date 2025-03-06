# 🌟 Face Detection and Recognition System  

![Logo](static/sprints_logo.png)  

### 🚀 AI-driven tool for real-time face detection and recognition!  

## 📌 Table of Contents  
- [📜 About the Project](#-about-the-project)  
- [📂 Folder Structure](#-folder-structure)  
- [⚡ Features](#-features)  
- [💻 Installation](#-installation)  
- [🚀 Running the Project](#-running-the-project)  
- [📢 Team Members](#-team-members)  
- [📜 License](#-license)  

---  

## 📜 About the Project  
The **Face Detection and Recognition System** is a web-based application built using **OpenCV**, **InceptionV3**, and **Streamlit** for real-time face detection and recognition. The system can:  

✅ Detect faces from images and live camera feed 📷  
✅ Identify individuals using a trained model 🧠  
✅ Provide a user-friendly interface with **Streamlit** 🎨  
✅ Optimize inference time using **TFLite** and **ONNX** models ⚡  

---  

## 📂 Folder Structure  
```
📂 face_recognition_system/
│── 📁 data/
│   │── raw/                     # Raw dataset before preprocessing
│   │── processed/               # Processed images after face detection
│   │── train_val_test/          # Split dataset for training and validation
│── 📁 models/                   # Saved models (H5, ONNX, TFLite)
│── 📁 notebooks/                # Jupyter notebooks for testing and training
│   │── face_detection_test.ipynb  # Face detection testing
│   │── fine_tune_inceptionv3.ipynb # Model training notebook
│── 📁 server/
│   │── server.py                # Real-time face recognition server
│── FaceDetector.py              # Face detection and preprocessing module
│── app.py                       # Main Streamlit app
│── requirements.txt              # Required Python libraries
│── static/
│   │── sprints_logo.png         # Project logo
```  

---  

## ⚡ Features  

✨ **Real-Time Face Detection:** Uses OpenCV’s DNN module for face detection.  
🎭 **Face Recognition:** Identifies faces using **InceptionV3** model.  
📡 **Live Camera Support:** Detects faces in real-time from a webcam.  
🚀 **Optimized Performance:** Models converted to **TFLite** and **ONNX** for faster inference.  
📊 **Performance Metrics:** Displays accuracy, precision, recall, and F1 score.  
🖼️ **User-Friendly UI:** Built with **Streamlit** for easy interaction.  

---  

## 💻 Installation  

### 🔹 Step 1: Clone the Repository  
```bash
git clone https://github.com/Basel-Amr/Capstone-Project_FaceSight-AI-Powered-Face-Detection-Recognition.git
```  

### 🔹 Step 2: Create & Activate a Virtual Environment (Optional but Recommended)  
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```  

### 🔹 Step 3: Install Dependencies  
```bash
pip install -r requirements.txt
```  

---  

## 🚀 Running the Project  

### 🖥️ **Run the Streamlit Web App**  
```bash
streamlit run app.py --server.port 8502
```  
Open your browser and go to `http://localhost:8502` to access the app.  

### 📷 **Run the Live Camera Detection**  
```bash
python server.py
```  
This will start the live camera feed and detect faces in real-time.  

---  

## 📢 Team Members  
📧 **Menna Mohamed**  
📧 **Dina Fakhry**  
📧 **Abanoub Younan**  
📧 **Karen Emad**  
📧 **Basel Amr**  
📧 **Ahmed Hesham**  
📧 **Omar Tarek**  

---  

## 📜 License  
This project is licensed under the **Sprints License**. Feel free to use and modify it! 🚀  

## 🛠️ Face Detection Pipeline

### 1. **Preprocessing:**
   - The **FaceDetector** class uses OpenCV's **DNN module** for face detection.
   - Detected faces are cropped, resized to 299x299, and normalized.
   - Data augmentation (rotation, shifting, and flipping) is applied for training.

### 2. **Model Training:**
   - **InceptionV3** model is fine-tuned for face recognition.
   - Training results:
     - **Accuracy**: 76.35%
     - **Loss**: 1.1564
     - **Validation Accuracy**: 71.33%
     - **Validation Loss**: 1.4291
   - Testing results:
     - **Accuracy**: 69.68%
     - **Precision**: 0.6999
     - **Recall**: 0.6968
     - **F1 Score**: 0.6850

---

## 📊 Model Comparison

| Model Type     | Size      | Inference Time  |
|---------------|----------|----------------|
| **H5 Model**   | 112.98 MB | 1.58 seconds  |
| **Keras Model**| 113.06 MB | 1.57 seconds  |
| **ONNX Model** | 87.79 MB  | 0.30 seconds  |
| **TFLite Model**| 87.78 MB | 0.29 seconds  |

---

## 📣 Acknowledgments
- [OpenCV DNN Module](https://opencv.org/)
- [InceptionV3](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3)
- [Streamlit](https://www.streamlit.io/)
- [Keras](https://keras.io/)
---
## 📱 How to Contribute

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.
---