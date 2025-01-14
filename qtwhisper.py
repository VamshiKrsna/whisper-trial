import sys
import whisper
import sounddevice as sd
import numpy as np
from PySide6.QtCore import QThread, Signal, QObject 
from PySide6.QtCore import QThread, Signal, QObject  
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine


class TranscriptionThread(QThread):
    transcription_signal = Signal(str)  
    """
    Thread to handle microphone input and transcription.
    """
    transcription_signal = Signal(str) 

    def __init__(self, model_name="tiny", parent=None):
        super().__init__(parent)
        self.running = False
        self.model = whisper.load_model(model_name)

    def run(self):
        self.running = True
        samplerate = 16000  
        duration = 5  

        with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32") as stream:
            print("Listening...")
            while self.running:
                audio_chunk = stream.read(int(samplerate * duration))[0].flatten()

                audio = np.array(audio_chunk, dtype=np.float32)


                result = self.model.transcribe(audio, fp16=False, language = "en")
                result = self.model.transcribe(audio, fp16=False)
                transcription = result.get("text", "")

                self.transcription_signal.emit(transcription)

    def stop(self):
        self.running = False


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
