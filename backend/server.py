from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import sys
import cv2
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

# Add parent directory to path to import app.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.app import load_emotion_model, detect_faces, predict_emotion

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Create temp directory for uploaded images
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once when server starts
MODEL_PATH = "./emotion_model.h5"  # Update with your model path
model = load_emotion_model(MODEL_PATH)

@app.route('/api/detect-emotion', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Save uploaded image with unique filename
    file_ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(image_path)
    
    try:
        # Detect faces
        img, faces, face_coords = detect_faces(image_path)
        if img is None:
            return jsonify({"error": "Could not process image"}), 500
        
        if len(faces) == 0:
            return jsonify({"emotions": [], "message": "No faces detected in the image"}), 200
        
        # Process each face and collect results
        emotions_data = []
        
        # Draw rectangles and labels on the image
        img_with_emotions = img.copy()
        for i, ((x, y, w, h), face_img) in enumerate(zip(face_coords, faces)):
            # Predict emotion
            emotion, confidence = predict_emotion(model, face_img)
            
            # Add to results
            emotions_data.append({
                "emotion": emotion,
                "confidence": float(confidence)  # Convert numpy float to Python float for JSON
            })
            
            # Draw rectangle
            cv2.rectangle(img_with_emotions, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add emotion label
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(img_with_emotions, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Convert the image to base64 for sending to frontend
        _, buffer = cv2.imencode('.jpg', img_with_emotions)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Clean up - remove temp file
        os.remove(image_path)
        
        return jsonify({
            "emotions": emotions_data,
            "image_with_emotions": f"data:image/jpeg;base64,{img_base64}",
            "message": f"Detected {len(emotions_data)} face(s) with emotions"
        })
        
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(image_path):
            os.remove(image_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)