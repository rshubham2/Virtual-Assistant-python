# assistant/face_auth.py

import cv2
import numpy as np
import sqlite3
import os
import logging
from typing import Optional, List, Tuple
import pickle
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import face_recognition
except ImportError:
    logger.warning("face_recognition library not found. Using fallback authentication method.")
    face_recognition = None

class FaceAuthenticator:
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.create_tables()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            face_encoding BLOB
        )
        ''')
        self.conn.commit()

    def register_user(self, username: str, password: str) -> bool:
        face_encoding = self.capture_face_encoding()
        if face_encoding is None:
            logger.warning("Failed to capture face. Using fallback registration.")
            return self.fallback_register(username, password)

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password, face_encoding) VALUES (?, ?, ?)",
                (username, password, pickle.dumps(face_encoding))
            )
            self.conn.commit()
            logger.info(f"User {username} registered successfully with face encoding.")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Username {username} already exists.")
            return False

    def fallback_register(self, username: str, password: str) -> bool:
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, password)
            )
            self.conn.commit()
            logger.info(f"User {username} registered successfully without face encoding.")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Username {username} already exists.")
            return False

    def capture_face_encoding(self) -> Optional[np.ndarray]:
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            logger.error("Failed to open camera.")
            return None

        frame_width = int(video_capture.get(3))
        frame_height = int(video_capture.get(4))
        face_frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
        cv2.rectangle(face_frame, (frame_width//4, frame_height//4),
                      (3*frame_width//4, 3*frame_height//4), (0,255,0), 2)

        loading_chars = ['|', '/', '-', '\\']
        loading_idx = 0
        start_time = time.time()
        face_detected = False

        while True:
            ret, frame = video_capture.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            combined_frame = frame.copy()
            cv2.addWeighted(combined_frame, 1, face_frame, 0.3, 0, combined_frame)

            if len(faces) > 0:
                face_detected = True
                (x, y, w, h) = faces[0]
                cv2.rectangle(combined_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                if face_recognition:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)
                    if face_locations:
                        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                        video_capture.release()
                        cv2.destroyAllWindows()
                        return face_encoding

            # Add loading animation
            loading_text = f"Detecting face {loading_chars[loading_idx]}"
            cv2.putText(combined_frame, loading_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            loading_idx = (loading_idx + 1) % len(loading_chars)

            cv2.imshow('Capture Face', combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Timeout after 20 seconds
            if time.time() - start_time > 20:
                logger.warning("Face capture timed out.")
                break

        video_capture.release()
        cv2.destroyAllWindows()

        if face_detected:
            logger.warning("Face detected but not recognized. Please try again.")
        else:
            logger.warning("No face detected during capture.")
        return None

    def authenticate(self) -> Optional[str]:
        face_encoding = self.capture_face_encoding()
        if face_encoding is None:
            logger.warning("Failed to capture face for authentication. Using fallback method.")
            return self.fallback_authentication()

        cursor = self.conn.cursor()
        cursor.execute("SELECT username, face_encoding FROM users WHERE face_encoding IS NOT NULL")
        for username, stored_encoding in cursor.fetchall():
            stored_encoding = pickle.loads(stored_encoding)
            if face_recognition.compare_faces([stored_encoding], face_encoding)[0]:
                return username

        logger.info("Face not recognized. Using fallback authentication.")
        return self.fallback_authentication()

    def fallback_authentication(self) -> Optional[str]:
        username = input("Enter your username: ")
        password = input("Enter your password: ")  # In a real system, use getpass and proper security measures

        cursor = self.conn.cursor()
        cursor.execute("SELECT username FROM users WHERE username = ? AND password = ?", (username, password))
        result = cursor.fetchone()

        if result:
            return result[0]
        else:
            logger.warning("Invalid username or password.")
            return None

    def __del__(self):
        self.conn.close()