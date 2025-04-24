from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import cv2
import matplotlib.pyplot as plt
import base64
import numpy as np
import tensorflow as tf
# Removed dlib import

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Create temp directory for uploaded images
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Haar Cascade Setup ---
# Ensure you have the Haar cascade XML file.
# If cv2 is installed correctly, this path should work.
# Otherwise, provide the full path to 'haarcascade_frontalface_default.xml'.
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(HAAR_CASCADE_PATH):
    raise FileNotFoundError(f"Haar cascade file not found at {HAAR_CASCADE_PATH}. "
                          "Please ensure OpenCV is installed correctly or provide the full path.")
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
if face_cascade.empty():
    raise IOError(f"Failed to load Haar cascade classifier from {HAAR_CASCADE_PATH}")
# --- End Haar Cascade Setup ---

mp = {
    'Surprise' : 'Sad',
    'Sad' : 'Neutral',
    'Neutral' : 'Surprise',
    'Angry' : 'Angry',
    'Disgust' : 'Disgust',
    'Fear' : 'Fear',
    'Happy' : 'Happy'
}

# Functions that were previously imported from src.app
def load_emotion_model(model_path):
    """Load the emotion recognition model"""
    model = tf.keras.models.load_model(model_path)
    return model

def detect_faces(image_path):
    """Detect faces in an image using Haar cascades"""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None, [], []

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar cascade
    # Parameters: image, scaleFactor, minNeighbors, minSize
    # Adjust scaleFactor and minNeighbors for performance/accuracy trade-off
    haar_rects = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1, # How much the image size is reduced at each image scale.
        minNeighbors=5,  # How many neighbors each candidate rectangle should have to retain it.
        minSize=(30, 30) # Minimum possible object size. Objects smaller than this are ignored.
    )

    faces = []
    face_coords = []

    # Iterate over detected faces (already in x, y, w, h format)
    for (x, y, w, h) in haar_rects:
        # Ensure coordinates are within image boundaries (though detectMultiScale usually handles this)
        x = max(0, x)
        y = max(0, y)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)

        # Extract face region from the grayscale image
        face_roi = gray[y:y+h, x:x+w]

        # Check if the extracted ROI is valid
        if face_roi.size == 0:
            print(f"Warning: Empty face ROI detected for coordinates (x={x}, y={y}, w={w}, h={h}). Skipping.")
            continue

        # Resize for emotion recognition (assuming model expects 48x48)
        try:
            face_roi_resized = cv2.resize(face_roi, (48, 48))
        except cv2.error as e:
            print(f"Error resizing face ROI: {e}. ROI shape: {face_roi.shape}. Skipping.")
            continue

        faces.append(face_roi_resized)
        face_coords.append((x, y, w, h))

    return img, faces, face_coords

def predict_emotion(model, face_img):
    """Predict emotion from face image"""
    # Ensure face image is grayscale and properly sized (already done in detect_faces, but good failsafe)
    if len(face_img.shape) > 2:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    # Ensure the input size matches the model's expected input
    # This resize might be redundant if detect_faces always provides 48x48
    if face_img.shape != (48, 48):
         face_img = cv2.resize(face_img, (48, 48))

    # Normalize pixel values
    face_img = face_img / 255.0

    # Reshape for model input (assuming a channel dimension is needed)
    face_tensor = np.expand_dims(face_img, axis=0) # Add batch dimension
    face_tensor = np.expand_dims(face_tensor, axis=-1) # Add channel dimension

    # Predict
    predictions = model.predict(face_tensor)

    # Map predictions to emotions
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotion_idx = np.argmax(predictions[0])
    emotion = mp[emotions[emotion_idx]]
    
    confidence = predictions[0][emotion_idx]

    return emotion, confidence

# Load model once when server starts
MODEL_PATH = "./Emotion_CNN_KDEF_RAFDB_best.keras"  # Updated to use .keras model
model = load_emotion_model(MODEL_PATH)

@app.route('/api/detect-emotion', methods=['POST'])
def detect_emotion_route(): # Renamed function to avoid conflict with the local variable
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    # Save uploaded image with unique filename
    file_ext = os.path.splitext(file.filename)[1].lower()
    # Basic check for common image extensions
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    if file_ext not in allowed_extensions:
        return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(allowed_extensions)}'}), 400

    unique_filename = f"{uuid.uuid4()}{file_ext}"
    image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(image_path)

    try:
        # Detect faces using the updated function
        img, faces, face_coords = detect_faces(image_path)
        if img is None:
            os.remove(image_path) # Clean up failed image
            return jsonify({"error": "Could not read or process image"}), 500

        if len(faces) == 0:
             os.remove(image_path) # Clean up image with no faces
             return jsonify({"emotions": [], "image_with_emotions": None, "message": "No faces detected in the image"}), 200

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
            # Adjust text placement and size for visibility
            cv2.putText(img_with_emotions, face_num, (x + 5, y - 10 if y > 20 else y + h + 20), # Place above or below rect
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA) # Use Anti-aliasing

        # Convert the image to base64 for sending to frontend
        _, buffer = cv2.imencode('.jpg', img_with_emotions)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "emotions": emotions_data,
            "image_with_emotions": f"data:image/jpeg;base64,{img_base64}",
            "message": f"Detected {len(emotions_data)} face(s) with emotions"
        })

    except Exception as e:
        print(f"An error occurred: {e}") # Log the error server-side
        # Clean up in case of error
        if os.path.exists(image_path):
            os.remove(image_path)
        return jsonify({'error': 'An internal server error occurred.'}), 500
    finally:
        # Ensure cleanup happens even if processing is successful but before returning
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError as e:
                print(f"Error removing temporary file {image_path}: {e}")


if __name__ == '__main__':
    app.run(debug=True, port=5000)