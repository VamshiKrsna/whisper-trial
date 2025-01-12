import sys
import whisper
import sounddevice as sd
import numpy as np
from PySide6.QtCore import QThread, Signal, QObject  # Import QObject
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine


class TranscriptionThread(QThread):
    """
    Thread to handle microphone input and transcription.
    """
    transcription_signal = Signal(str)  # Signal to send transcription to the UI

    def __init__(self, model_name="tiny", parent=None):
        super().__init__(parent)
        self.running = False
        self.model = whisper.load_model(model_name)

    def run(self):
        """
        Continuously captures audio from the microphone and transcribes it.
        """
        self.running = True
        samplerate = 16000  # Whisper's expected sample rate
        duration = 5  # Process audio in chunks of 5 seconds

        with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32") as stream:
            print("Listening...")
            while self.running:
                # Record audio chunk
                audio_chunk = stream.read(int(samplerate * duration))[0].flatten()

                # Whisper expects audio as numpy array with dtype float32
                audio = np.array(audio_chunk, dtype=np.float32)

                # Transcribe audio
                result = self.model.transcribe(audio, fp16=False)
                transcription = result.get("text", "")

                # Emit the transcription to the UI
                self.transcription_signal.emit(transcription)

    def stop(self):
        """
        Stop the transcription thread.
        """
        self.running = False


if __name__ == "__main__":
    # Qt application setup
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # Load the QML file for the GUI
    engine.load("transcription_app.qml")
    if not engine.rootObjects():
        print("Error: Failed to load QML file.")
        sys.exit(-1)

    # Retrieve the root object
    root = engine.rootObjects()[0]
    if root is None:
        print("Error: Root object not found in QML file.")
        sys.exit(-1)

    # Find the transcription display element
    transcription_display = root.findChild(QObject, "transcriptionDisplay")
    if transcription_display is None:
        print("Error: 'transcriptionDisplay' element not found in QML.")
        sys.exit(-1)

    # Load the Whisper transcription thread
    transcription_thread = TranscriptionThread()

    # Connect transcription thread signal to UI
    transcription_thread.transcription_signal.connect(
        lambda text: transcription_display.setProperty("text", text)
    )

    # Start and stop transcription via buttons
    root.findChild(QObject, "startButton").clicked.connect(transcription_thread.start)
    root.findChild(QObject, "stopButton").clicked.connect(transcription_thread.stop)

    sys.exit(app.exec())