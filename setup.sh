#!/bin/bash
echo "Setting up virtual environment..."
python3 -m venv face_recognition_env
source face_recognition_env/bin/activate
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Setup complete."
