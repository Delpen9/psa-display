import streamlit as st
import numpy as np
import av
import whisper
import io
import soundfile as sf
import librosa
from audiorecorder import audiorecorder

# page config
st.set_page_config(layout="wide")

# cache Whisper model
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

model = load_whisper()

# Initialize card list in session state
def init_cards():
    if 'cards' not in st.session_state:
        st.session_state.cards = [0]

# Add a new card after index idx
def add_card(idx):
    new_id = max(st.session_state.cards) + 1 if st.session_state.cards else 0
    st.session_state.cards.insert(idx + 1, new_id)

# Render a single card given its unique ID
def render_card(card_id):
    with st.expander(f"Show Card Details", expanded=True):
        col1, col2, col3 = st.columns(3)

        # Front Image
        with col1:
            st.subheader("Front Image")
            front_file = st.file_uploader(
                "Upload front image", type=["png", "jpg", "jpeg"], key=f"front_{card_id}"
            )
            if front_file:
                st.image(front_file, caption="Front", use_column_width=True)

        # Back Image
        with col2:
            st.subheader("Back Image")
            back_file = st.file_uploader(
                "Upload back image", type=["png", "jpg", "jpeg"], key=f"back_{card_id}"
            )
            if back_file:
                st.image(back_file, caption="Back", use_column_width=True)

        # Details (Voice → Text)
        with col3:
            st.subheader("Details (Voice → Text)")

            audio = audiorecorder(
                start_prompt="Start recording",
                stop_prompt="Stop recording",
                pause_prompt="",
                custom_style={'color': 'white'},
                start_style={},
                pause_style={},
                stop_style={},
                show_visualizer=True,
                key=f"audio_{card_id}"
            )

            if len(audio) > 0:
                # To play audio in frontend:
                st.audio(audio.export().read())

            if st.button("📝 Transcribe", key=f"transcribe_{card_id}"):
                if audio:
                    audio_buffer = io.BytesIO(audio.export().read())
                    data, samplerate = sf.read(audio_buffer)

                    if data.ndim > 1:
                        data = data.mean(axis=1)

                    audio = data.astype("float32")

                    if samplerate != 16000:
                        audio = librosa.resample(audio, orig_sr=samplerate, target_sr=16000)

                    result = model.transcribe(audio, fp16=False)
                    st.session_state[f"transcript_{card_id}"] = result["text"]
                else:
                    st.warning("No audio to transcribe—please record first!")

            note = st.text_area(
                "Transcription",
                st.session_state.get(f"transcript_{card_id}", ""),
                height=180,
                key=f"note_{card_id}",
            )
            if st.button("💾 Save Note", key=f"save_{card_id}"):
                with open(f"saved_note_{card_id}.txt", "w") as f:
                    f.write(note)
                st.success("Your note has been saved!")

    st.markdown("</div>", unsafe_allow_html=True)

# Main function to render all cards
if __name__ == "__main__":
    init_cards()

    for card_id in st.session_state.cards:
        render_card(card_id)

        # Button to add a new card below this one
        if st.button("Add Card Below", key=f"add_{card_id}"):
            idx = st.session_state.cards.index(card_id)
            add_card(idx)
