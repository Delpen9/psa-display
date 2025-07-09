import streamlit as st
import numpy as np
from st_audiorec import st_audiorec
import av
import whisper
import io
import soundfile as sf
import librosa

# page config
st.set_page_config(layout="wide")

# cache Whisper model
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

model = load_whisper()

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Front Image")
    front_file = st.file_uploader("Upload front image", type=["png","jpg","jpeg"], key="front")
    if front_file:
        st.image(front_file, caption="Front", use_column_width=True)

with col2:
    st.subheader("Back Image")
    back_file = st.file_uploader("Upload back image", type=["png","jpg","jpeg"], key="back")
    if back_file:
        st.image(back_file, caption="Back", use_column_width=True)

with col3:
    st.subheader("Details (Voice → Text)")

    # 1. Record
    wav_bytes = st_audiorec()  
    if wav_bytes is not None:
        st.audio(wav_bytes, format="audio/wav")

    # 2. Transcribe
    if st.button("📝 Transcribe"):
        if wav_bytes:
            # Load the WAV bytes into numpy
            audio_buffer = io.BytesIO(wav_bytes)
            data, samplerate = sf.read(audio_buffer)  

            # If stereo, convert to mono
            if data.ndim > 1:
                data = data.mean(axis=1)

            # Whisper expects float32 in [–1, +1]
            audio = data.astype("float32")

            # Resample to 16 kHz if needed
            if samplerate != 16000:
                audio = librosa.resample(audio, orig_sr=samplerate, target_sr=16000)

            # Transcribe
            result = model.transcribe(audio, fp16=False)
            st.session_state.transcript = result["text"]
        else:
            st.warning("No audio to transcribe—please make a recording first!")

    # 3. Show & save
    note = st.text_area("Transcription", st.session_state.get("transcript",""), height=180)
    if st.button("💾 Save Note"):
        with open("saved_note.txt","w") as f:
            f.write(note)
        st.success("Your note has been saved!")