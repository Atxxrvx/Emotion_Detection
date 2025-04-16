from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import cv2
import matplotlib.pyplot as plt
import base64
import numpy as np
import tensorflow as tf
import dlib  # Add dlib import

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Create temp directory for uploaded images
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Functions that were previously imported from src.app
def load_emotion_model(model_path):
    """Load the emotion recognition model"""
    model = tf.keras.models.load_model(model_path)
    return model

def detect_faces(image_path):
    """Detect faces in an image using Dlib instead of Haar cascades"""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        return None, [], []
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Initialize dlib's face detector
    detector = dlib.get_frontal_face_detector()
    
    # Detect faces
    dlib_rects = detector(gray, 1)
    
    faces = []
    face_coords = []
    
    for rect in dlib_rects:
        # Convert dlib rectangle to OpenCV format (x, y, w, h)
        x = rect.left()
        y = rect.top()
        w = rect.right() - rect.left()
        h = rect.bottom() - rect.top()
        
        # Ensure coordinates are within image boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)
        
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize for emotion recognition (assuming model expects 48x48)
        face_roi_resized = cv2.resize(face_roi, (48, 48))
        
        faces.append(face_roi_resized)
        face_coords.append((x, y, w, h))
    
    return img, faces, face_coords

def predict_emotion(model, face_img):
    """Predict emotion from face image"""
    # Ensure face image is grayscale and properly sized
    if len(face_img.shape) > 2:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Resize to match model input size if needed
    face_img = cv2.resize(face_img, (48, 48))
    
    # Normalize pixel values
    face_img = face_img / 255.0
    
    # Reshape for model input
    face_tensor = np.expand_dims(face_img, axis=0)
    face_tensor = np.expand_dims(face_tensor, axis=-1)
    
    # Predict
    predictions = model.predict(face_tensor)
    
    # Map predictions to emotions
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotion_idx = np.argmax(predictions[0])
    emotion = emotions[emotion_idx]
    confidence = predictions[0][emotion_idx]
    
    return emotion, confidence

# Load model once when server starts
MODEL_PATH = "./emotion_model.keras"  # Updated to use .keras model
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
            
            # Add face number to the results
            emotions_data.append({
                "face_number": i + 1,
                "emotion": emotion,
                "confidence": float(confidence)  # Convert numpy float to Python float for JSON
            })
            
            # Draw rectangle
            cv2.rectangle(img_with_emotions, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Only add the face number (no emotion text or confidence)
            face_num = f"{i+1}"
            cv2.putText(img_with_emotions, face_num, (x+5, y+25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
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