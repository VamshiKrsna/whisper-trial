import streamlit as st
import whisper
import torch
from io import BytesIO

st.title("Audio Transcription with Whisper (Tiny Model)")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg"])

# Loading the Whisper tiny model (70MB)
@st.cache_resource
def load_model():
    model = whisper.load_model("tiny")
    return model

model = load_model()

def transcribe_audio(audio_file):
    audio = whisper.load_audio(audio_file)
    
    result = model.transcribe(audio)
    
    return result["text"]

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    if st.button("Transcribe"):
        with st.spinner("Transcribing..."):
            with open("temp_audio", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            transcription = transcribe_audio("temp_audio")
            
            st.success("Transcription Complete!")
            st.text_area("Transcription", transcription, height=200)