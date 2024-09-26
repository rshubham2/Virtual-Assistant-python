import speech_recognition as sr
from mtranslate import translate


class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def listen(self):
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = self.recognizer.listen(source)
        return audio

    def recognize_speech(self, audio, language='en-IN'):
        try:
            text = self.recognizer.recognize_google(audio, language=language)
            print(f"Recognized: {text}")
            if language.startswith('hi'):
                translated = translate(text, 'en')
                print(f"Translated to English: {translated}")
                return translated
            return text
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
            return None
        except sr.RequestError:
            print("Sorry, there was an error with the speech recognition service.")
            return None