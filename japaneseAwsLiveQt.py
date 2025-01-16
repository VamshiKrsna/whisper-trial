import sys
import asyncio
import sounddevice as sd
from PySide6.QtCore import QThread, Signal, QObject
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent


class TranscriptionThread(QThread):
    transcription_signal = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False

    async def mic_stream(self):
        loop = asyncio.get_event_loop()
        input_queue = asyncio.Queue()

        def callback(indata, frame_count, time_info, status):
            loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

        stream = sd.RawInputStream(
            channels=1,
            samplerate=16000,
            callback=callback,
            blocksize=1024 * 2,
            dtype="int16",
        )

        with stream:
            while self.running:
                indata, status = await input_queue.get()
                yield indata, status

    async def write_chunks(self, stream):
        async for chunk, status in self.mic_stream():
            await stream.input_stream.send_audio_event(audio_chunk=chunk)
        await stream.input_stream.end_stream()

    async def basic_transcribe(self):
        client = TranscribeStreamingClient(region="us-east-1")

        stream = await client.start_stream_transcription(
            language_code="ja-JP", 
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )

        handler = MyEventHandler(stream.output_stream, self.transcription_signal)
        await asyncio.gather(self.write_chunks(stream), handler.handle_events())

    def run(self):
        self.running = True
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.basic_transcribe())

    def stop(self):
        self.running = False


class MyEventHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream, update_signal):
        super().__init__(output_stream)
        self.update_signal = update_signal

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            for alt in result.alternatives:
                self.update_signal.emit(alt.transcript)


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