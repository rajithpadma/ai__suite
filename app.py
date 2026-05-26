"""
AI Suite - single-file Streamlit app
Run:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, Tuple
import bs4
import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image

# ---------------------------------------------------------------------------
# Paths (models / outputs created next to this file at runtime)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1) WEB SCRAPING
# ---------------------------------------------------------------------------
def scrape_url(url: str, timeout: int = 20) -> Tuple[str, str]:
    url = (url or "").strip()
    if not url:
        raise ValueError("URL is empty.")

    headers = {"User-Agent": "Mozilla/5.0 (compatible; AI-Suite/1.0)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "(no title)"

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = " ".join(p for p in paragraphs if p)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 200:
        article = soup.find("article")
        if article:
            paras = [p.get_text(" ", strip=True) for p in article.find_all("p")]
            alt = " ".join(p for p in paras if p)
            alt = re.sub(r"\s+", " ", alt).strip()
            if len(alt) > len(text):
                text = alt

    if len(text) < 100:
        raise ValueError("Could not extract enough text from this URL.")

    return title, text


def scrape_html_bytes(html_bytes: bytes) -> Tuple[str, str]:
    raw = html_bytes.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "(no title)"
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = " ".join(p for p in paragraphs if p)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 50:
        raise ValueError("Could not extract enough text from the uploaded HTML.")
    return title, text


# ---------------------------------------------------------------------------
# 2) TEXT SUMMARIZATION
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading summarization model (first time may take a minute)...")
def get_summarizer_pipeline():
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model="facebook/bart-large-cnn", device=device)


def split_text(text: str, chunk_words: int = 400) -> list[str]:
    words = text.split()
    size = max(50, chunk_words)
    return [" ".join(words[i : i + size]) for i in range(0, len(words), size)]


def summarize_text(
    text: str,
    max_length: int = 180,
    min_length: int = 60,
    chunk_words: int = 400,
    progress: Optional[Callable[[str], None]] = None,
) -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("No text to summarize.")

    summarizer = get_summarizer_pipeline()
    chunks = split_text(text, chunk_words)
    parts: list[str] = []

    for i, chunk in enumerate(chunks):
        if progress:
            progress(f"Summarizing part {i + 1} of {len(chunks)}...")
        out = summarizer(
            chunk,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )
        parts.append(out[0]["summary_text"].strip())

    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# 3) SPEECH TO TEXT
# ---------------------------------------------------------------------------
def transcribe_wav_bytes(wav_bytes: bytes, language: str = "en-US") -> str:
    if not wav_bytes:
        raise ValueError("Audio file is empty.")

    import speech_recognition as sr

    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name

    try:
        with sr.AudioFile(tmp_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language=language)
        text = (text or "").strip()
        if not text:
            raise RuntimeError("Transcription was empty.")
        return text
    except sr.UnknownValueError:
        raise RuntimeError("Could not understand the audio. Try a clearer WAV recording.")
    except sr.RequestError as e:
        raise RuntimeError(f"Speech service error: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 4) NEURAL STYLE TRANSFER
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading VGG19 for style transfer...")
def get_vgg_features():
    import torch
    from torchvision import models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        weights = models.VGG19_Weights.DEFAULT
        vgg = models.vgg19(weights=weights).features
    except Exception:
        vgg = models.vgg19(pretrained=True).features

    vgg = vgg.to(device).eval()
    for p in vgg.parameters():
        p.requires_grad = False
    return vgg, device


def neural_style_transfer(
    content_img: Image.Image,
    style_img: Image.Image,
    steps: int = 50,
    max_size: int = 256,
    content_weight: float = 1e4,
    style_weight: float = 1e2,
    progress: Optional[Callable[[str], None]] = None,
) -> Image.Image:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    vgg, device = get_vgg_features()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    style_layers = (0, 5, 10, 19, 28)
    content_layer = 21

    def resize(img: Image.Image) -> Image.Image:
        w, h = img.size
        if max(w, h) <= max_size:
            return img
        if w >= h:
            return img.resize((max_size, max(1, int(h * max_size / w))), Image.Resampling.LANCZOS)
        return img.resize((max(1, int(w * max_size / h)), max_size), Image.Resampling.LANCZOS)

    def to_tensor(img: Image.Image) -> torch.Tensor:
        img = resize(img.convert("RGB"))
        t = transforms.ToTensor()(img).unsqueeze(0).to(device)
        return (t - mean) / std

    def gram(feat: torch.Tensor) -> torch.Tensor:
        _, c, h, w = feat.shape
        x = feat.view(c, h * w)
        return torch.mm(x, x.t()) / (c * h * w)

    def extract(x: torch.Tensor, layers: tuple[int, ...]) -> dict[int, torch.Tensor]:
        out: dict[int, torch.Tensor] = {}
        wanted = set(layers)
        for i, layer in enumerate(vgg):
            x = layer(x)
            if i in wanted:
                out[i] = x
        return out

    content = to_tensor(content_img)
    style = to_tensor(style_img)

    content_feat = extract(content, (content_layer,))
    style_feat = extract(style, style_layers)
    style_grams = {i: gram(style_feat[i]) for i in style_layers}

    target = content.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([target], lr=0.02)

    min_norm = (0.0 - mean) / std
    max_norm = (1.0 - mean) / std

    for step in range(steps):
        optimizer.zero_grad()
        with torch.no_grad():
            target.clamp_(min_norm, max_norm)

        feats = extract(target, style_layers + (content_layer,))
        c_loss = F.mse_loss(feats[content_layer], content_feat[content_layer])
        s_loss = sum(F.mse_loss(gram(feats[i]), style_grams[i]) for i in style_layers)
        loss = content_weight * c_loss + style_weight * s_loss
        loss.backward()
        optimizer.step()

        if progress and step % max(1, steps // 5) == 0:
            progress(f"Style transfer step {step + 1}/{steps} (loss={loss.item():.4f})")

    with torch.no_grad():
        target.clamp_(min_norm, max_norm)
        out = target * std + mean

    result = transforms.ToPILImage()(out.squeeze(0).cpu().clamp(0, 1))

    key = hashlib.sha256(
        (str(steps) + str(max_size) + str(content_weight) + str(style_weight)).encode()
    ).hexdigest()[:12]
    save_path = OUTPUTS_DIR / f"nst_{key}.png"
    result.save(save_path)
    return result


# ---------------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Suite", page_icon="AI", layout="wide")

st.title("AI Suite")
st.caption("All-in-one: Text Summarizer + Web Scraping | Speech to Text | Neural Style Transfer")

tab1, tab2, tab3 = st.tabs(
    [
        "Text Summarizer & Web Scraping",
        "Speech to Text",
        "Neural Style Transfer",
    ]
)

# ===================== TAB 1: SUMMARIZER + SCRAPING =====================
with tab1:
    st.subheader("Text Summarizer & Web Scraping")

    mode = st.radio(
        "Input method",
        ["Article URL", "Paste text", "Upload .txt file", "Upload .html file (scrape only)"],
        horizontal=True,
    )

    url_val = ""
    pasted = ""
    txt_upload = None
    html_upload = None

    if mode == "Article URL":
        url_val = st.text_input("Enter article URL")
    elif mode == "Paste text":
        pasted = st.text_area("Paste article text", height=200)
    elif mode == "Upload .txt file":
        txt_upload = st.file_uploader("Upload .txt file", type=["txt"])
    else:
        html_upload = st.file_uploader("Upload .html file", type=["html", "htm"])

    c1, c2, c3 = st.columns(3)
    with c1:
        max_len = st.slider("Max summary length", 50, 300, 180)
    with c2:
        min_len = st.slider("Min summary length", 20, 150, 60)
    with c3:
        chunk_w = st.slider("Chunk size (words)", 200, 800, 400)

    col_a, col_b = st.columns(2)
    with col_a:
        run_scrape = st.button("Scrape only", type="secondary")
    with col_b:
        run_summary = st.button("Summarize", type="primary")

    status = st.empty()

    def show_progress(msg: str):
        status.info(msg)

    if run_scrape or run_summary:
        try:
            title = ""
            body = ""

            if mode == "Article URL":
                if not url_val.strip():
                    st.error("Enter a URL.")
                    st.stop()
                show_progress("Scraping URL...")
                title, body = scrape_url(url_val.strip())

            elif mode == "Paste text":
                body = pasted.strip()
                if not body:
                    st.error("Paste some text.")
                    st.stop()
                title = "(pasted text)"

            elif mode == "Upload .txt file":
                if txt_upload is None:
                    st.error("Upload a .txt file.")
                    st.stop()
                body = txt_upload.read().decode("utf-8", errors="ignore").strip()
                title = txt_upload.name
                if not body:
                    st.error("File is empty.")
                    st.stop()

            else:
                if html_upload is None:
                    st.error("Upload an HTML file.")
                    st.stop()
                show_progress("Parsing HTML...")
                title, body = scrape_html_bytes(html_upload.getvalue())

            st.success("Text extracted successfully.")
            st.markdown(f"**Title:** {title}")
            with st.expander("View extracted text", expanded=not run_summary):
                st.write(body)
            st.download_button(
                "Download extracted text",
                data=body.encode("utf-8"),
                file_name="extracted.txt",
                mime="text/plain",
            )

            if run_summary:
                show_progress("Running summarization model...")
                summary = summarize_text(
                    body,
                    max_length=max_len,
                    min_length=min_len,
                    chunk_words=chunk_w,
                    progress=show_progress,
                )
                status.empty()
                st.markdown("### Summary")
                st.write(summary)
                st.download_button(
                    "Download summary",
                    data=summary.encode("utf-8"),
                    file_name="summary.txt",
                    mime="text/plain",
                )

        except Exception as e:
            status.empty()
            st.error(str(e))

# ===================== TAB 2: SPEECH TO TEXT =====================
with tab2:
    st.subheader("Speech to Text")
    st.write("Upload a **WAV** file. Requires internet (Google Speech API).")

    audio = st.file_uploader("Upload WAV audio", type=["wav"])
    lang = st.selectbox("Language", ["en-US", "en-GB", "hi-IN", "es-ES", "fr-FR", "de-DE"])

    if st.button("Transcribe", type="primary"):
        if audio is None:
            st.error("Upload a WAV file first.")
        else:
            try:
                with st.spinner("Transcribing..."):
                    text = transcribe_wav_bytes(audio.getvalue(), language=lang)
                st.markdown("### Transcription")
                st.write(text)
                st.download_button(
                    "Download transcription",
                    data=text.encode("utf-8"),
                    file_name="transcription.txt",
                    mime="text/plain",
                )
            except Exception as e:
                st.error(str(e))

# ===================== TAB 3: NEURAL STYLE TRANSFER =====================
with tab3:
    st.subheader("Neural Style Transfer")
    st.write("Upload a **content** image and a **style** image.")

    col1, col2 = st.columns(2)
    with col1:
        content_file = st.file_uploader("Content image", type=["png", "jpg", "jpeg"], key="c")
    with col2:
        style_file = st.file_uploader("Style image", type=["png", "jpg", "jpeg"], key="s")

    s1, s2, s3 = st.columns(3)
    with s1:
        nst_steps = st.slider("Steps", 10, 100, 50)
    with s2:
        nst_size = st.slider("Max image size", 128, 384, 256, step=16)
    with s3:
        nst_cw = st.number_input("Content weight", 1000, 50000, 10000, step=500)
    nst_sw = st.number_input("Style weight", 10, 5000, 100, step=10)

    if st.button("Generate stylized image", type="primary"):
        if content_file is None or style_file is None:
            st.error("Upload both content and style images.")
        else:
            try:
                content_img = Image.open(content_file).convert("RGB")
                style_img = Image.open(style_file).convert("RGB")

                st.image([content_img, style_img], caption=["Content", "Style"], width=280)

                prog = st.empty()

                def nst_progress(msg: str):
                    prog.info(msg)

                with st.spinner("Running neural style transfer (may take 1-3 minutes)..."):
                    result = neural_style_transfer(
                        content_img,
                        style_img,
                        steps=nst_steps,
                        max_size=nst_size,
                        content_weight=float(nst_cw),
                        style_weight=float(nst_sw),
                        progress=nst_progress,
                    )

                prog.empty()
                st.image(result, caption="Stylized output", use_container_width=True)

                buf = io.BytesIO()
                result.save(buf, format="PNG")
                st.download_button(
                    "Download PNG",
                    data=buf.getvalue(),
                    file_name="stylized.png",
                    mime="image/png",
                )
            except Exception as e:
                st.error(str(e))

st.sidebar.markdown("### How to run locally")
st.sidebar.code('pip install -r requirements.txt\nstreamlit run app.py', language="bash")
st.sidebar.markdown("**Note:** First summarization run downloads the BART model (~1.6 GB).")
