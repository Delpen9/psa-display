import streamlit as st
import io
import soundfile as sf
import librosa
from audiorecorder import audiorecorder

@st.cache_resource
def load_model():
    import whisper
    return whisper.load_model("base")

# Helper to insert a Item id after current index
def add_Item(idx: int):
    next_id = max(st.session_state.Items, default=-1) + 1
    st.session_state.Items.insert(idx + 1, next_id)

@st.dialog("Confirm delete", width="small")   # "large" is ~750 px
def confirm_delete(idx, cid):
    st.write(f"Delete item **#{cid}**?")
    yes, no = st.columns(2)
    with yes:
        if st.button("Yes, delete"):
            # remove card
            st.session_state.Items.pop(idx)
            # optional: clean up any related keys
            for k in list(st.session_state.keys()):
                if k.endswith(f"_{cid}"):
                    st.session_state.pop(k)
            st.rerun()          # closes the dialog + refreshes page
    with no:
        if st.button("Cancel"):
            st.rerun()          # just close the dialog
    
# Item renderer
def render_Item(idx: int, Item_id: int, allow_delete: bool):
    with st.container():
        # Provide a default name
        default_name = f"Default Item Name"

        # Allow user to edit the name
        Item_name = st.text_input(f"", value=default_name, key=f"Item_name_{idx}")

        with st.expander("Details", expanded=True):
            c1, c2, c3 = st.columns([1,1,1], gap="small")

            # Image Uploaders + Camera Capture
            for col, label in zip((c1, c2), ("Front", "Back")):
                with col:
                    st.markdown(f"**Upload {label} Image**")
                    tabs = st.tabs(["Upload", "Camera"])
                    with tabs[0]:
                        upload = st.file_uploader(
                            "", type=["png","jpg","jpeg"],
                            key=f"upload_{label.lower()}_{Item_id}"
                        )
                    with tabs[1]:
                        camera = st.camera_input(
                            f"Snap {label} Photo",
                            key=f"camera_{label.lower()}_{Item_id}"
                        )

                    image = upload or camera
                    if image:
                        st.image(image, caption=label, use_container_width=True)

                    # capture for return
                    if label == "Front":
                        front_image = image
                        st.session_state[f"front_{Item_id}"] = image
                    else:
                        back_image = image
                        st.session_state[f"back_{Item_id}"] = image

            # Audio recorder & transcription
            with c3:
                st.markdown("**Record & Transcribe**")

                audio_data = audiorecorder(
                    start_prompt="",
                    stop_prompt="",
                    pause_prompt="",
                    show_visualizer=True,
                    key=f"audio_{Item_id}",
                )

                if audio_data:
                    st.audio(audio_data.export().read(), format="audio/wav")

                if st.button("📝 Transcribe", key=f"trans_{Item_id}"):
                    if audio_data and len(audio_data) > 0:
                        buf = io.BytesIO(audio_data.export().read())

                        data, sr = sf.read(buf)

                        if data.ndim > 1:
                            data = data.mean(axis=1)

                        audio = data.astype("float32")

                        if sr != 16000:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

                        result = model.transcribe(audio, fp16=False)

                        st.session_state[f"transcript_{Item_id}"] = result["text"]
                    else:
                        st.warning("No audio recorded!")

                transcript = st.session_state.get(f"transcript_{Item_id}", "")
                note = st.text_area("Transcription", transcript, height=150, key=f"note_{Item_id}")

        if st.button("➕ Add Item Below", key=f"add_{Item_id}"):
            add_Item(idx)

        if allow_delete:
            if st.button("🗑️ Delete Item", key=f"del_{Item_id}"):
                confirm_delete(idx, Item_id)   # ← opens modal

    return front_image, back_image

if __name__ == "__main__":
    # Page config
    st.set_page_config(
        page_title="Collectible Documenter", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )

    model = load_model()

    # Initialise Items state
    if "Items" not in st.session_state:
        st.session_state.Items = [0]

    # Base CSS (hide menu/footer, base Item styling)
    st.markdown(
        """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stApp {padding: 2rem;}

            .banner-container {
                padding: 1rem 2rem;
                border-radius: 0.5rem;
                margin-bottom: 1.5rem;
                text-align: center;
                font-family: 'Segoe UI', sans-serif;
                border: 1px solid;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
                transition: background-color 0.3s, color 0.3s, border-color 0.3s;
            }

            .Item-container {
                background: #f9f9f9;
                border-radius: 1rem;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                transition: background 0.3s, color 0.3s, box-shadow 0.3s;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Banner (media-query fallback for Auto mode)
    st.markdown(
        """
        <style>
        @media (prefers-color-scheme: light) {
            .banner-container {
                background-color: #f5f5f5;
                color: #333;
                border-color: #e0e0e0;
            }
        }
        @media (prefers-color-scheme: dark) {
            .banner-container {
                background-color: #1e1e1e;
                color: #f5f5f5;
                border-color: #444;
                box-shadow: 0 2px 8px rgba(255, 255, 255, 0.05);
            }
        }
        </style>

        <div class="banner-container">
            <h1 style="margin: 0; font-size: 2.2rem;">Collectible Documenter</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        .banner-container {
            background-color: #f5f5f5 !important;
            color: #333 !important;
            border-color: #e0e0e0 !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
        }
        .Item-container {
            background: #f9f9f9 !important;
            color: #000 !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    allow_delete = len(st.session_state.Items) > 1
    # Render items
    for i, cid in enumerate(st.session_state.Items):
        st.markdown("---")

        front_image = st.session_state.get(f"front_{cid}")
        back_image  = st.session_state.get(f"back_{cid}")

        if front_image or back_image:
            c1, c2 = st.columns([4, 1], gap="small")
            with c1:
                render_Item(i, cid, allow_delete=allow_delete)

            with c2:
                if front_image:
                    st.image(front_image, caption="Front", use_container_width=True)

                if back_image:
                    st.image(back_image,  caption="Back",  use_container_width=True)
        else:
            render_Item(i, cid, allow_delete=allow_delete)
