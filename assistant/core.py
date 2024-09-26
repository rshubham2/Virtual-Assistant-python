# assistant/core.py
from .speech_recognition import SpeechRecognizer
from .text_to_speech import TextToSpeech
from .nlp_model import NLPModel
from .face_auth import FaceAuthenticator
from .vision import Vision
import logging
import threading


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VirtualAssistant:
    def __init__(self):
        self.speech_recognizer = SpeechRecognizer()
        self.tts = TextToSpeech()
        self.nlp_model = NLPModel()
        self.face_authenticator = FaceAuthenticator()
        self.vision = Vision()
        self.skills = {}
        self.wake_word = "hey assistant"
        self.authenticated_user = None
        self.vision_active = False

    def authenticate_user(self):
        self.speak("Please look at the camera for face authentication.")
        user = self.face_authenticator.authenticate()
        if user:
            self.authenticated_user = user
            self.speak(f"Welcome, {user}! How can I help you today?")
            return True
        else:
            self.speak("Authentication failed. Access denied.")
            return False

    def listen(self):
        return self.speech_recognizer.listen()

    def process_detections(self, detections):
        objects = [f"{d['label']} (confidence: {d['confidence']:.2f})" for d in detections]
        response = f"I can see: {', '.join(objects)}"
        self.speak(response)

    def toggle_vision(self):
        if not self.vision_active:
            self.vision_active = True
            self.vision.run_detection(self.process_detections)
            self.vision_active = False
            return "Vision deactivated. I'm now listening for commands again."
        else:
            return "Vision is already active."

    def process_command(self, text):
        if "activate vision" in text.lower():
            return self.toggle_vision()

        # Existing skill processing
        for skill_name, skill_func in self.skills.items():
            if skill_name.lower() in text.lower():
                return skill_func(text)

        # If no specific skill matches, use the general NLP model
        return self.nlp_model.generate_response(text)

    def speak(self, text):
        self.tts.speak(text)

    def register_user(self):
        self.speak("Welcome to user registration. Please provide a username and password.")
        username = input("Enter a username: ")
        password = input("Enter a password: ")  # In a real system, use getpass and proper security measures

        self.speak(
            "Now, let's capture your face for authentication. Please look at the camera and position your face within the green rectangle.")
        if self.face_authenticator.register_user(username, password):
            self.speak(f"User {username} registered successfully. You can now authenticate.")
        else:
            self.speak("Registration failed. The username may already exist or face capture was unsuccessful.")

    def authenticate_user(self):
        self.speak(
            "Starting user authentication. Please look at the camera and position your face within the green rectangle.")
        max_attempts = 3
        for attempt in range(max_attempts):
            user = self.face_authenticator.authenticate()
            if user:
                self.authenticated_user = user
                self.speak(f"Welcome, {user}! How can I help you today?")
                return True
            else:
                remaining_attempts = max_attempts - attempt - 1
                if remaining_attempts > 0:
                    self.speak(f"Authentication failed. {remaining_attempts} attempts remaining. Please try again.")
                else:
                    self.speak("Authentication failed. Access denied.")
                    return False

    def run(self):
        self.speak(
            "Hello! I'm your virtual assistant. Say 'Hey Assistant' to wake me up, or 'Register' to create a new account.")
        while True:
            try:
                logger.info("Listening for wake word or registration command...")
                audio = self.listen()
                if audio:
                    text = self.speech_recognizer.recognize_speech(audio)
                    if text:
                        if "register" in text.lower():
                            self.register_user()
                        elif self.wake_word in text.lower():
                            if self.authenticate_user():
                                while True:
                                    logger.info("Listening for command...")
                                    command_audio = self.listen()
                                    if command_audio:
                                        command_text = self.speech_recognizer.recognize_speech(command_audio)
                                        if command_text:
                                            logger.info(f"User said: {command_text}")
                                            if "goodbye" in command_text.lower():
                                                self.speak(
                                                    f"Goodbye, {self.authenticated_user}! Say 'Hey Assistant' when you need me again.")
                                                self.authenticated_user = None
                                                break
                                            response = self.process_command(command_text)
                                            logger.info(f"Assistant response: {response}")
                                            self.speak(response)
                            else:
                                continue
            except Exception as e:
                logger.error(f"An error occurred: {str(e)}")
                self.speak("I'm sorry, but an error occurred. Please try again.")

    def add_skill(self, name, func):
        self.skills[name] = func
