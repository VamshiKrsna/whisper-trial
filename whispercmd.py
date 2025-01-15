import sys
import whisper
import sounddevice as sd
import numpy as np
import queue
import threading

# Global variables
audio_queue = queue.Queue()
stop_event = threading.Event()

def audio_callback(indata, frames, time, status):
    """
    Callback function to capture audio from the microphone.
    """
    if status:
        print(f"Audio stream error: {status}")
    audio_queue.put(indata.copy())

def transcribe_audio(model, samplerate):
    """
    Continuously transcribe audio from the microphone.
    """
    while not stop_event.is_set():
        try:
            # Get audio data from the queue
            audio_data = audio_queue.get(timeout=1.0)
            audio_data = audio_data.flatten().astype(np.float32)

            # Transcribe the audio chunk
            result = model.transcribe(audio_data, fp16=False, language="en")
            transcription = result.get("text", "").strip()

            # Print the transcription immediately
            if transcription:
                print("\rTranscription:", transcription, end="", flush=True)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"\nError during transcription: {e}")

def main():
    # Load the Whisper model
    model_name = "base"  # Use "base" for better real-time performance
    model = whisper.load_model(model_name)

    # Set up audio stream parameters
    samplerate = 16000  # Whisper's expected sample rate
    blocksize = int(samplerate * 1)  # Process audio in 1-second chunks for real-time feel
    channels = 1  # Mono audio

    # Start the audio stream
    print("Starting live transcription... Press Ctrl+C to stop.")
    with sd.InputStream(
        samplerate=samplerate,
        blocksize=blocksize,
        channels=channels,
        dtype="float32",
        callback=audio_callback
    ):
        # Start the transcription thread
        transcription_thread = threading.Thread(
            target=transcribe_audio,
            args=(model, samplerate)
        )
        transcription_thread.start()

        # Wait for the user to stop the program
        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("\nStopping transcription...")
            stop_event.set()
            transcription_thread.join()

if __name__ == "__main__":
    main()