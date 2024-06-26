import cv2
import face_recognition
import datetime
import os
import logging

# Set up logging configuration
logging.basicConfig(filename='face_capture.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Set up camera
cap = cv2.VideoCapture(0)

# Directory and file paths
known_faces_dir = "known_faces/"
known_faces_file = "known_faces.txt"

# Ensure known_faces directory exists
os.makedirs(known_faces_dir, exist_ok=True)

def add_new_face():
    """
    Captures an image from the camera, detects a face, and saves it as a new face in the known_faces directory.
    Updates the known_faces.txt file with the path to the new face image.
    """
    # Capture image from camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from camera.")
        return

    # Detect faces in image
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    if len(face_encodings) != 1:
        print("Please ensure only one face is visible and clear in the frame.")
        return

    # Assume only one face in the image
    new_face_encoding = face_encodings[0]

    # Save image with timestamp
    image_filename = f"face_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    image_path = os.path.join(known_faces_dir, image_filename)
    cv2.imwrite(image_path, frame)
    print(f"New face saved as {image_path}")

    # Append to known_faces.txt
    with open(known_faces_file, "a") as f:
        f.write("known_faces/" + image_filename + "\n")

    print("New face added to known faces.")
    logging.info(f"New face added: {image_path}")

# Main loop to capture and display frames
while True:
    # Capture image from camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from camera.")
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Wait for user input to add a new face (press 'a')
    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):
        add_new_face()
        break  # Exit after adding a face

    # Break the loop on 'q' key press
    elif key == ord('q'):
        break

# Release camera resources
cap.release()
cv2.destroyAllWindows()
