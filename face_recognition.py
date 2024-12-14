import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import logging
from PIL import Image
import io

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleFaceRecognition:
    def __init__(self, faces_dir='Faces'):
        self.faces_dir = faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        
    def load_image(self, image_path):
        """Load image handling different formats including HEIF"""
        try:
            # Try loading with PIL first
            image = Image.open(image_path)
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Convert to numpy array
            return np.array(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
        
    def load_faces(self):
        """Load face encodings from the database"""
        logger.info("Loading faces...")
        
        for person_name in os.listdir(self.faces_dir):
            person_dir = os.path.join(self.faces_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            logger.info(f"Loading images for {person_name}")
            person_encodings = []
            
            # Load all images for this person
            for image_file in os.listdir(person_dir):
                if not any(image_file.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.heic']):
                    continue
                    
                image_path = os.path.join(person_dir, image_file)
                try:
                    # Load image using our custom function
                    image = self.load_image(image_path)
                    if image is None:
                        continue
                        
                    # Find face locations first
                    face_locations = face_recognition.face_locations(image, model="hog")
                    
                    if face_locations:
                        # Get encodings for each face found
                        encodings = face_recognition.face_encodings(image, face_locations)
                        if encodings:
                            person_encodings.extend(encodings)
                            logger.info(f"Successfully loaded {image_file}")
                        else:
                            logger.warning(f"Could not encode face in {image_file}")
                    else:
                        logger.warning(f"No face found in {image_file}")
                        
                except Exception as e:
                    logger.error(f"Error processing {image_file}: {str(e)}")
                    
            # If we found any faces for this person, add them
            if person_encodings:
                # Add each encoding separately
                for encoding in person_encodings:
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(person_name)
                    
        logger.info(f"Loaded {len(self.known_face_encodings)} face encodings")
        
    def run_recognition(self):
        """Run real-time face recognition"""
        if not self.known_face_encodings:
            logger.error("No faces loaded! Please load faces first.")
            return
            
        logger.info("Starting face recognition...")
        
        # Initialize webcam
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            logger.error("Could not open webcam")
            return
            
        # Reduce recognition frequency for better performance
        process_this_frame = True
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
                
            # Only process every other frame for better performance
            if process_this_frame:
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                
                # Convert from BGR to RGB
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                try:
                    # Find faces in current frame
                    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                    if face_locations:
                        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                        
                        # Process each face found
                        for face_encoding, face_location in zip(face_encodings, face_locations):
                            # Get distances to all known faces
                            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                            
                            if len(face_distances) > 0:
                                best_match_idx = np.argmin(face_distances)
                                min_distance = face_distances[best_match_idx]
                                
                                # Use a stricter threshold
                                if min_distance < 0.5:
                                    name = self.known_face_names[best_match_idx]
                                    confidence = f"{(1 - min_distance) * 100:.1f}%"
                                else:
                                    name = "Unknown"
                                    confidence = "Low"
                                    
                                # Scale back face location
                                top, right, bottom, left = [coord * 4 for coord in face_location]
                                
                                # Draw box and label
                                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                                cv2.putText(frame, f"{name} ({confidence})", 
                                          (left + 6, bottom - 6),
                                          cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                                
                                logger.info(f"Detected: {name} with confidence {confidence}")
                except Exception as e:
                    logger.error(f"Error processing frame: {str(e)}")
                        
            process_this_frame = not process_this_frame
            
            # Display the frame
            cv2.imshow('Face Recognition', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        video_capture.release()
        cv2.destroyAllWindows()

def main():
    recognizer = SimpleFaceRecognition()
    recognizer.load_faces()
    recognizer.run_recognition()

if __name__ == "__main__":
    main()
