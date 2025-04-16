import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Function to load the emotion detection model
def load_emotion_model(model_path):
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to detect faces using Haar Cascade
def detect_faces(image_path):
    # Load the face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None, []
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    # Extract face regions
    face_images = []
    face_coords = []
    
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_images.append(face_img)
        face_coords.append((x, y, w, h))
    
    return img, face_images, face_coords

# Function to preprocess a face for the emotion model
def preprocess_face(face_img, target_size=(48, 48)):
    # Convert to grayscale if needed (most emotion models are trained on grayscale)
    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_img
    
    # Resize to target size
    face_resized = cv2.resize(face_gray, target_size)
    
    # Normalize pixel values
    face_normalized = face_resized / 255.0
    
    # Expand dimensions to match model input shape
    face_input = np.expand_dims(np.expand_dims(face_normalized, -1), 0)
    
    return face_input

# Function to predict emotion
def predict_emotion(model, face_img):
    # Common emotion labels (adjust based on your model's classes)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # Preprocess the face
    face_input = preprocess_face(face_img)
    
    # Make prediction
    prediction = model.predict(face_input)
    
    # Get the emotion label with highest probability
    emotion_idx = np.argmax(prediction[0])
    emotion = emotion_labels[emotion_idx]
    confidence = prediction[0][emotion_idx]
    
    return emotion, confidence

# Main function to process an image
def process_image(model_path, image_path):
    # Load the model
    model = load_emotion_model(model_path)
    if model is None:
        return
    
    # Detect faces
    img, faces, face_coords = detect_faces(image_path)
    if img is None:
        return
    
    if len(faces) == 0:
        print("No faces detected in the image.")
        return
    
    # Set up the plot
    fig = plt.figure(figsize=(12, 8))
    num_faces = len(faces)
    rows = (num_faces + 2) // 3  # Calculate rows needed (at most 3 faces per row)
    
    # Display original image with face boxes
    ax = fig.add_subplot(rows+1, 1, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.set_title("Original Image with Detected Faces")
    ax.axis('off')
    
    # Draw rectangles around faces
    for (x, y, w, h) in face_coords:
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
    ax.imshow(img_rgb)
    
    # Process each face
    for i, (face_img, (x, y, w, h)) in enumerate(zip(faces, face_coords)):
        # Predict emotion
        emotion, confidence = predict_emotion(model, face_img)
        
        # Display each face with emotion label
        ax = fig.add_subplot(rows+1, 3, i+4)  # Start from position 4 (after the original image)
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        ax.imshow(face_rgb)
        ax.set_title(f"Face {i+1}: {emotion} ({confidence:.2f})")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    model_path = "/home/atharva/Desktop/Container/ML_LAB/project/emotion_model.h5"  # Update with your model path
    image_path = "/home/atharva/Desktop/Container/ML_LAB/project/image3.jpg"        # Update with your image path
    
    process_image(model_path, image_path)