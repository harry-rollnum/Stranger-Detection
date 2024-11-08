import cv2
import face_recognition
import smtplib
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from dotenv import load_dotenv
import os
import logging

# Load environment variables from .env file
load_dotenv()

# Set up logging configuration
logging.basicConfig(filename='face_detection.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Set up camera
cap = cv2.VideoCapture(0)

# Load known faces and their encodings from known_faces.txt
known_faces = []
known_face_encodings = []

with open("known_faces.txt", "r") as f:
    for line in f:
        image = face_recognition.load_image_file(line.strip())
        known_faces.append(image)
        known_face_encodings.append(face_recognition.face_encodings(image)[0])

# Email credentials
FROM_EMAIL = os.getenv('EMAIL_USER')
PASSWORD = os.getenv('EMAIL_PASS')

def send_notification(subject, message, image_filename):
    """
    Sends an email notification with a subject, message body, and attached image file.

    Args:
    - subject (str): Subject of the email.
    - message (str): Body text of the email.
    - image_filename (str): File path of the image to attach.
    """
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = FROM_EMAIL
    msg['To'] = FROM_EMAIL

    msg.attach(MIMEText(message, 'plain'))

    with open(image_filename, 'rb') as f:
        img = MIMEImage(f.read())
        img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_filename))
        msg.attach(img)

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(FROM_EMAIL, PASSWORD)
            server.sendmail(FROM_EMAIL, FROM_EMAIL, msg.as_string())
            logging.info(f"Notification email sent successfully: {subject}")
    except Exception as e:
        logging.error(f"Error sending notification email: {e}")

while True:
    # Capture image from camera
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the captured frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    detected_unknown = False
    detected_known = False

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare face encodings with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        if any(matches):
            detected_known = True
            logging.info("Known face detected!")
            print("Known Face detected")

        if not any(matches):
            detected_unknown = True

            # Save image of unknown face with timestamp
            filename = f"unknown_faces/unknown_face_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            logging.info(f"Unknown face saved as {filename}")
            print(f"Unknown face saved as {filename}")

            # Send notification email with attached image
            subject = "Unknown face detected!"
            message = "An unknown face was detected at your home. Please check the camera feed."
            send_notification(subject, message, filename)

    # Display the resulting frame with detected faces
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera resources
cap.release()
cv2.destroyAllWindows()
