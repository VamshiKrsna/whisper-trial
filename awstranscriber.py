import sys
import sounddevice as sd
import numpy as np
import boto3
from PySide6.QtCore import QThread, Signal, QObject
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
import tempfile
import os
import requests
import time
import uuid


class TranscriptionThread(QThread):
    """
    Thread to handle microphone input and transcription using AWS Transcribe.
    """
    transcription_signal = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.client = boto3.client("transcribe", region_name="eu-north-1")  
        self.s3_client = boto3.client("s3", region_name="eu-north-1")  
        self.bucket_name = "text2speechqt" 

    def run(self):
        """
        Continuously captures audio from the microphone and transcribes it using AWS Transcribe.
        """
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

                        s3_key = os.path.basename(temp_audio_path)
                        self.s3_client.upload_file(temp_audio_path, self.bucket_name, s3_key)

                        transcription = self.transcribe_audio(s3_key)
                        self.transcription_signal.emit(transcription)

                    os.remove(temp_audio_path)
                except Exception as e:
                    print(f"Error during transcription: {e}")
                    self.transcription_signal.emit("Transcription failed. Check logs for details.")

    def stop(self):
        """
        Stop the transcription thread.
        """
        self.running = False

    def save_wav(self, file_path, audio_data, samplerate):
        """
        Save audio data to a WAV file.
        """
        import scipy.io.wavfile as wav
        wav.write(file_path, samplerate, audio_data)

    def transcribe_audio(self, s3_key):
        """
        Transcribe audio using AWS Transcribe.
        """
        try:
            job_name = f"transcription_job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            file_uri = f"s3://{self.bucket_name}/{s3_key}"

        
            self.client.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={"MediaFileUri": file_uri},
                MediaFormat="wav",
                LanguageCode="en-US",  
            )

            while True:
                response = self.client.get_transcription_job(TranscriptionJobName=job_name)
                status = response["TranscriptionJob"]["TranscriptionJobStatus"]
                if status in ["COMPLETED", "FAILED"]:
                    break

            if status == "COMPLETED":
                transcript_uri = response["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
                transcript_response = requests.get(transcript_uri)
                transcript_text = transcript_response.json()["results"]["transcripts"][0]["transcript"]
                return transcript_text
            else:
                return "Transcription failed."
        except Exception as e:
            print(f"Error during AWS Transcribe job: {e}")
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

    transcription_thread = TranscriptionThread() #AWS Transcribe thread

    # Connecting transcription thread signal to UI
    transcription_thread.transcription_signal.connect(
        lambda text: transcription_display.setProperty("text", text)
    )

    # Start and stop transcription via buttons
    root.findChild(QObject, "startButton").clicked.connect(transcription_thread.start)
    root.findChild(QObject, "stopButton").clicked.connect(transcription_thread.stop)

    sys.exit(app.exec())