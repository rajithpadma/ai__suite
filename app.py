
"""
AI Suite - Streamlit App
Features:
1. Text Summarization
2. Web Scraping
3. Speech to Text
4. Neural Style Transfer

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, Optional, Tuple

import requests
import streamlit as st
from PIL import Image
from bs4 import BeautifulSoup

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Suite",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Suite")
st.caption(
    "Text Summarizer • Web Scraping • Speech to Text • Neural Style Transfer"
)

# =========================================================
# DIRECTORIES
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"

OUTPUTS_DIR.mkdir(exist_ok=True)

# =========================================================
# WEB SCRAPING
# =========================================================
def scrape_url(url: str, timeout: int = 20) -> Tuple[str, str]:

    url = (url or "").strip()

    if not url:
        raise ValueError("URL is empty.")

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=timeout
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    title_tag = soup.find("title")

    title = (
        title_tag.get_text(strip=True)
        if title_tag
        else "(No title)"
    )

    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.find_all("p")
    ]

    text = " ".join(paragraphs)

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if len(text) < 100:
        raise ValueError(
            "Could not extract enough article text."
        )

    return title, text


# =========================================================
# SUMMARIZATION
# =========================================================
@st.cache_resource(show_spinner="Loading summarization model...")
def get_summarizer_pipeline():

    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1

    # FIXED MODEL
    summarizer = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        device=device,
    )

    return summarizer


def split_text(
    text: str,
    chunk_words: int = 400
):

    words = text.split()

    return [
        " ".join(words[i:i + chunk_words])
        for i in range(0, len(words), chunk_words)
    ]


def summarize_text(
    text: str,
    max_length: int = 180,
    min_length: int = 60,
    chunk_words: int = 400,
    progress: Optional[Callable[[str], None]] = None,
):

    text = (text or "").strip()

    if not text:
        raise ValueError("No text provided.")

    summarizer = get_summarizer_pipeline()

    chunks = split_text(
        text,
        chunk_words
    )

    summaries = []

    for idx, chunk in enumerate(chunks):

        if progress:
            progress(
                f"Summarizing chunk {idx + 1}/{len(chunks)}..."
            )

        prompt = f"Summarize this article:\n\n{chunk}"

        output = summarizer(
            prompt,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )

        generated = output[0].get(
            "generated_text",
            ""
        )

        summaries.append(
            generated.strip()
        )

    return " ".join(summaries)


# =========================================================
# SPEECH TO TEXT
# =========================================================
def transcribe_wav_bytes(
    wav_bytes: bytes,
    language: str = "en-US"
):

    import speech_recognition as sr

    recognizer = sr.Recognizer()

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as tmp:

        tmp.write(wav_bytes)

        tmp_path = tmp.name

    try:

        with sr.AudioFile(tmp_path) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(
            audio,
            language=language
        )

        return text

    except sr.UnknownValueError:
        raise RuntimeError(
            "Could not understand the audio."
        )

    except sr.RequestError as e:
        raise RuntimeError(
            f"Speech recognition error: {e}"
        )

    finally:

        try:
            os.remove(tmp_path)
        except:
            pass


# =========================================================
# NEURAL STYLE TRANSFER
# =========================================================
@st.cache_resource(show_spinner="Loading VGG19...")
def get_vgg():

    import torch
    from torchvision import models

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    try:
        weights = models.VGG19_Weights.DEFAULT

        model = models.vgg19(
            weights=weights
        ).features

    except:
        model = models.vgg19(
            pretrained=True
        ).features

    model = model.to(device).eval()

    for param in model.parameters():
        param.requires_grad = False

    return model, device


def neural_style_transfer(
    content_img: Image.Image,
    style_img: Image.Image,
    steps: int = 50,
    max_size: int = 256,
    content_weight: float = 1e4,
    style_weight: float = 1e2,
    progress=None,
):

    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    vgg, device = get_vgg()

    transform = transforms.Compose([
        transforms.Resize(max_size),
        transforms.ToTensor(),
    ])

    content = transform(
        content_img
    ).unsqueeze(0).to(device)

    style = transform(
        style_img
    ).unsqueeze(0).to(device)

    target = content.clone().requires_grad_(True)

    optimizer = torch.optim.Adam(
        [target],
        lr=0.02
    )

    for step in range(steps):

        optimizer.zero_grad()

        content_loss = F.mse_loss(
            target,
            content
        )

        style_loss = F.mse_loss(
            target,
            style
        )

        total_loss = (
            content_weight * content_loss
            +
            style_weight * style_loss
        )

        total_loss.backward()

        optimizer.step()

        if progress and step % 10 == 0:
            progress(
                f"Step {step}/{steps}"
            )

    out = target.squeeze(0).detach().cpu()

    out_img = transforms.ToPILImage()(out)

    return out_img


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "📝 Summarizer + Web Scraping",
    "🎤 Speech to Text",
    "🎨 Neural Style Transfer",
])

# =========================================================
# TAB 1
# =========================================================
with tab1:

    st.subheader(
        "Text Summarizer & Web Scraper"
    )

    mode = st.radio(
        "Input method",
        [
            "Article URL",
            "Paste text",
            "Upload .txt file"
        ],
        horizontal=True,
    )

    url_input = ""
    pasted_text = ""
    uploaded_file = None

    if mode == "Article URL":

        url_input = st.text_input(
            "Article URL"
        )

    elif mode == "Paste text":

        pasted_text = st.text_area(
            "Paste article text",
            height=200
        )

    else:

        uploaded_file = st.file_uploader(
            "Upload .txt file",
            type=["txt"]
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        max_len = st.slider(
            "Max summary length",
            50,
            300,
            180
        )

    with col2:
        min_len = st.slider(
            "Min summary length",
            20,
            150,
            60
        )

    with col3:
        chunk_size = st.slider(
            "Chunk size",
            200,
            800,
            400
        )

    summarize_btn = st.button(
        "Summarize",
        type="primary"
    )

    if summarize_btn:

        status = st.empty()

        def progress(msg):
            status.info(msg)

        try:

            if mode == "Article URL":

                title, body = scrape_url(
                    url_input
                )

                st.write(
                    f"### {title}"
                )

            elif mode == "Paste text":

                body = pasted_text.strip()

            else:

                body = uploaded_file.read().decode(
                    "utf-8",
                    errors="ignore"
                )

            summary = summarize_text(
                body,
                max_length=max_len,
                min_length=min_len,
                chunk_words=chunk_size,
                progress=progress,
            )

            status.empty()

            st.markdown("## Summary")

            st.write(summary)

            st.download_button(
                "Download Summary",
                data=summary.encode("utf-8"),
                file_name="summary.txt",
                mime="text/plain",
            )

        except Exception as e:
            st.error(str(e))

# =========================================================
# TAB 2
# =========================================================
with tab2:

    st.subheader("Speech to Text")

    audio_file = st.file_uploader(
        "Upload WAV File",
        type=["wav"]
    )

    language = st.selectbox(
        "Language",
        [
            "en-US",
            "en-GB",
            "hi-IN",
            "es-ES",
        ]
    )

    if st.button(
        "Transcribe",
        type="primary"
    ):

        if not audio_file:
            st.error(
                "Please upload WAV file."
            )

        else:

            try:

                with st.spinner(
                    "Transcribing..."
                ):

                    text = transcribe_wav_bytes(
                        audio_file.getvalue(),
                        language=language
                    )

                st.markdown(
                    "## Transcription"
                )

                st.write(text)

            except Exception as e:
                st.error(str(e))

# =========================================================
# TAB 3
# =========================================================
with tab3:

    st.subheader(
        "Neural Style Transfer"
    )

    colA, colB = st.columns(2)

    with colA:

        content_file = st.file_uploader(
            "Content Image",
            type=["png", "jpg", "jpeg"],
            key="content"
        )

    with colB:

        style_file = st.file_uploader(
            "Style Image",
            type=["png", "jpg", "jpeg"],
            key="style"
        )

    s1, s2, s3 = st.columns(3)

    with s1:
        steps = st.slider(
            "Steps",
            10,
            100,
            50
        )

    with s2:
        max_size = st.slider(
            "Max Image Size",
            128,
            384,
            256
        )

    with s3:

        content_weight = st.number_input(
            "Content Weight",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=500
        )

    style_weight = st.number_input(
        "Style Weight",
        min_value=10,
        max_value=5000,
        value=100,
        step=10
    )

    if st.button(
        "Generate Stylized Image",
        type="primary"
    ):

        if not content_file or not style_file:

            st.error(
                "Upload both images."
            )

        else:

            try:

                content_img = Image.open(
                    content_file
                ).convert("RGB")

                style_img = Image.open(
                    style_file
                ).convert("RGB")

                st.image(
                    [content_img, style_img],
                    caption=["Content", "Style"],
                    width=250
                )

                progress_box = st.empty()

                def progress(msg):
                    progress_box.info(msg)

                result = neural_style_transfer(
                    content_img,
                    style_img,
                    steps=steps,
                    max_size=max_size,
                    content_weight=float(content_weight),
                    style_weight=float(style_weight),
                    progress=progress,
                )

                progress_box.empty()

                st.image(
                    result,
                    caption="Stylized Output",
                    use_container_width=True
                )

                buf = io.BytesIO()

                result.save(
                    buf,
                    format="PNG"
                )

                st.download_button(
                    "Download PNG",
                    data=buf.getvalue(),
                    file_name="stylized.png",
                    mime="image/png",
                )

            except Exception as e:
                st.error(str(e))

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## Run App")

st.sidebar.code(
    "pip install -r requirements.txt\n"
    "streamlit run streamlit_app.py",
    language="bash",
)

st.sidebar.markdown(
    "Uses HuggingFace Transformers, Torch, BeautifulSoup, and Streamlit."
)

