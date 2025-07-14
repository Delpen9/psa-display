import streamlit as st
import io
import soundfile as sf
import librosa
from audiorecorder import audiorecorder

# Page Config & Styles
st.set_page_config(page_title="Audio Card Notes", layout="wide", initial_sidebar_state="expanded")

# Hide Streamlit default menu and footer & add basic card styling
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stApp {padding: 2rem;}
        .card-container {background: #f9f9f9; border-radius: 1rem; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1);}    
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    import whisper
    return whisper.load_model("base")

model = load_model()

# Initialise cards state
if "cards" not in st.session_state:
    st.session_state.cards = [0]

# Helper to insert a card id after current index

def add_card(idx: int):
    next_id = max(st.session_state.cards, default=-1) + 1
    st.session_state.cards.insert(idx + 1, next_id)

# Card renderer

def render_card(idx: int, card_id: int):
    with st.container():
        # Provide a default name
        default_name = f"Details for Card"

        # Allow user to edit the name
        card_name = st.text_input(f"Card Name", value=default_name, key=f"card_name_{idx}")

        # Use the edited name in the expander
        with st.expander(card_name, expanded=True):
            c1, c2, c3 = st.columns([1, 1, 1], gap="small", border=True)

            # Image Uploaders + Camera Capture
            for col, label in zip((c1, c2), ("Front", "Back")):
                with col:
                    st.markdown(f"**Upload {label} Image**")
                    # for each card and each label (“Front”, “Back”)
                    tabs = st.tabs(["Upload", "Camera"])
                    with tabs[0]:
                        upload = st.file_uploader(
                            "",
                            type=["png", "jpg", "jpeg"],
                            key=f"upload_{label.lower()}_{card_id}",
                        )
                    with tabs[1]:
                        camera = st.camera_input(
                            f"Snap {label} Photo",
                            key=f"camera_{label.lower()}_{card_id}",
                        )

                    # pick whichever the user provided
                    image = upload if upload is not None else camera
                    if image is not None:
                        st.image(image, caption=label, use_column_width=True)

            # Audio recorder & transcription
            with c3:
                st.markdown("**Record & Transcribe**")

                audio_data = audiorecorder(
                    start_prompt="",
                    stop_prompt="",
                    pause_prompt="",
                    show_visualizer=True,
                    key=f"audio_{card_id}",
                )

                if audio_data:
                    st.audio(audio_data.export().read(), format="audio/wav")

                if st.button("📝 Transcribe", key=f"trans_{card_id}"):
                    if audio_data and len(audio_data) > 0:
                        buf = io.BytesIO(audio_data.export().read())
                        data, sr = sf.read(buf)
                        if data.ndim > 1:
                            data = data.mean(axis=1)
                        audio = data.astype("float32")
                        if sr != 16000:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                        result = model.transcribe(audio, fp16=False)
                        st.session_state[f"transcript_{card_id}"] = result["text"]
                    else:
                        st.warning("No audio recorded!")

                transcript = st.session_state.get(f"transcript_{card_id}", "")
                note = st.text_area("Transcription", transcript, height=150, key=f"note_{card_id}")

        # Divider & add-below button
        if st.button("Add Card Below", key=f"add_{card_id}"):
            add_card(idx)

# Render all cards
for i, cid in enumerate(st.session_state.cards):
    st.markdown("---")
    render_card(i, cid)