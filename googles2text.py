import sys
import sounddevice as sd
import numpy as np
from PySide6.QtCore import QThread, Signal, QObject
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from google.cloud import speech_v1p1beta1 as speech
import tempfile
import os
from dotenv import load_dotenv
import google.auth

load_dotenv()

class TranscriptionThread(QThread):
    transcription_signal = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        credentials, _ = google.auth.load_credentials_from_file(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        self.client = speech.SpeechClient(credentials=credentials)

    def run(self):
        self.running = True
        samplerate = 16000
        duration = 5

        with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32") as stream:
            print("Listening...")
            while self.running:
                try:
                    audio_chunk = stream.read(int(samplerate * duration))[0].flatten()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                        temp_audio_path = temp_audio_file.name
                        self.save_wav(temp_audio_path, audio_chunk, samplerate)
                        transcription = self.transcribe_audio(temp_audio_path)
                        self.transcription_signal.emit(transcription)
                    os.remove(temp_audio_path)
                except Exception as e:
                    print(f"Error during transcription: {e}")
                    self.transcription_signal.emit("Transcription failed. Check logs for details.")

    def stop(self):
        self.running = False

    def save_wav(self, file_path, audio_data, samplerate):
        import scipy.io.wavfile as wav
        wav.write(file_path, samplerate, audio_data)

    def transcribe_audio(self, audio_file_path):
        try:
            with open(audio_file_path, "rb") as audio_file:
                content = audio_file.read()
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US",
            )
            response = self.client.recognize(config=config, audio=audio)
            transcription = ""
            for result in response.results:
                transcription += result.alternatives[0].transcript
            return transcription
        except Exception as e:
            print(f"Error during Google Speech-to-Text job: {e}")
            return "Transcription failed. Check logs for details."

if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()
    engine.load("transcription_app.qml")
    if not engine.rootObjects():
        print("Error: Failed to load QML file.")
        sys.exit(-1)
    root = engine.rootObjects()[0]
    if root is None:
        print("Error: Root object not found in QML file.")
        sys.exit(-1)
    transcription_display = root.findChild(QObject, "transcriptionDisplay")
    if transcription_display is None:
        print("Error: 'transcriptionDisplay' element not found in QML.")
        sys.exit(-1)
    transcription_thread = TranscriptionThread()
    transcription_thread.transcription_signal.connect(
        lambda text: transcription_display.setProperty("text", text)
    )
    root.findChild(QObject, "startButton").clicked.connect(transcription_thread.start)
    root.findChild(QObject, "stopButton").clicked.connect(transcription_thread.stop)
    sys.exit(app.exec())