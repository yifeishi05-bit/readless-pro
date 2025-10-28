# app.py  — ReadLess Pro (CPU-friendly, Py3.11)

import os
import io
import sys
import math
import warnings
from typing import List

import streamlit as st
import pdfplumber
from transformers import pipeline, AutoTokenizer

warnings.filterwarnings("ignore", category=SyntaxWarning)

# ----------------- 页面设置 -----------------
st.set_page_config(page_title="📘 ReadLess Pro – Book Summarizer", page_icon="📘", layout="wide")
st.title("📚 ReadLess Pro – AI Book Summarizer")
st.caption("Upload a long PDF (even full books!) and get automatic chapter summaries powered by AI (T5-small).")

# ----------------- 会员与控制面板 -----------------
REAL_CODE = os.getenv("ACCESS_CODE") or st.secrets.get("ACCESS_CODE", "")
BUY_LINK = "https://readlesspro.lemonsqueezy.com/buy/d0a09dc2-f156-4b4b-8407-12a87943bbb6"

with st.sidebar:
    st.header("🔒 Member Login")
    code = st.text_input("Enter access code (for paid users)", type="password")

    st.divider()
    st.subheader("⚙️ Controls")
    max_sections = st.number_input("Max sections to summarize", 5, 100, 20, 1)
    per_section_max_len = st.slider("Per-section max length", 80, 300, 180, 10)
    per_section_min_len = st.slider("Per-section min length", 30, 200, 60, 10)
    final_max_len = st.slider("Final summary max length", 150, 500, 300, 10)
    final_min_len = st.slider("Final summary min length", 80, 300, 120, 10)

    st.divider()
    # 清缓存按钮（适用于 Streamlit Cloud）
    if st.button("♻️ Reset app cache"):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        try:
            st.cache_resource.clear()
        except Exception:
            pass
        st.success("Cache cleared. Rerun the app.")

    # 显示运行环境版本
    try:
        import torch
        torch_ver = torch.__version__
    except Exception:
        torch_ver = "N/A"
    st.caption(f"Python: {sys.version.split()[0]} • Torch: {torch_ver}")

if code != REAL_CODE:
    st.warning("Please enter a valid access code to continue.")
    st.markdown(f"💳 [Click here to subscribe ReadLess Pro]({BUY_LINK})")
    st.stop()

# ----------------- 模型（懒加载） -----------------
@st.cache_resource(show_spinner=True)
def load_summarizer_and_tokenizer():
    tok = AutoTokenizer.from_pretrained("t5-small", use_fast=True)
    # CPU 友好；如果平台提供 GPU，会自动用 device=0
    device = 0 if os.getenv("CUDA_VISIBLE_DEVICES") not in (None, "", "-1") else -1
    summarizer = pipeline(
        "summarization",
        model="t5-small",
        tokenizer=tok,
        framework="pt",
        device=device,
    )
    return summarizer, tok

# ----------------- Token 级分块 -----------------
def chunk_by_tokens(tokenizer: AutoTokenizer, text: str, max_tokens: int = 480, overlap: int = 40) -> List[str]:
    if not text.strip():
        return []
    paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    buf_tokens = 0

    def tok_len(t: str) -> int:
        return len(tokenizer.encode(t, add_special_tokens=False))

    for p in paras:
        p_tokens = tok_len(p)
        if p_tokens > max_tokens:
            # 按句子再细分
            sentences = []
            tmp = []
            for seg in p.replace("。", "。|").replace("！", "！|").replace("？", "？|").split("|"):
                s = seg.strip()
                if s:
                    tmp.append(s)
                    if s[-1:] in "。！？.!?":
                        sentences.append("".join(tmp))
                        tmp = []
            if tmp:
                sentences.append("".join(tmp))
            for s in sentences:
                s_tokens = tok_len(s)
                if buf_tokens + s_tokens <= max_tokens:
                    buf.append(s)
                    buf_tokens += s_tokens
                else:
                    if buf:
                        piece = " ".join(buf)
                        chunks.append(piece)
                        tail = piece[-overlap * 2 :] if overlap > 0 else ""
                        buf = [tail] if tail else []
                        buf_tokens = tok_len(" ".join(buf)) if buf else 0
                    if s_tokens <= max_tokens:
                        buf.append(s)
                        buf_tokens = tok_len(" ".join(buf))
                    else:
                        ids = tokenizer.encode(s, add_special_tokens=False)
                        for i in range(0, len(ids), max_tokens):
                            chunks.append(tokenizer.decode(ids[i : i + max_tokens]))
                        buf, buf_tokens = [], 0
        else:
            if buf_tokens + p_tokens <= max_tokens:
                buf.append(p)
                buf_tokens += p_tokens
            else:
                piece = " ".join(buf)
                chunks.append(piece)
                tail = piece[-overlap * 2 :] if overlap > 0 else ""
                buf = [tail, p] if tail else [p]
                buf_tokens = tok_len(" ".join(buf))
    if buf:
        chunks.append(" ".join(buf))
    return [c.strip() for c in chunks if c.strip()]

# ----------------- 文件上传 -----------------
uploaded = st.file_uploader("📄 Upload a PDF file (book, report, or notes)", type="pdf")
if not uploaded:
    st.stop()

# ----------------- PDF 解析 -----------------
st.info("✅ File uploaded successfully. Extracting text…")
text_parts: List[str] = []
try:
    # 使用 BytesIO，避免重复读取
    raw = uploaded.read()
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        total_pages = len(pdf.pages)
        st.write(f"Total pages detected: **{total_pages}**")
        progress_pages = st.progress(0.0)
        for i, page in enumerate(pdf.pages, start=1):
            try:
                t = page.extract_text(x_tolerance=2, y_tolerance=2)
            except Exception:
                t = ""
            if t:
                text_parts.append(t)
            if i % 10 == 0 or i == total_pages:
                progress_pages.progress(i / total_pages)
except Exception as e:
    st.error(f"❌ Failed to parse PDF: {e}")
    st.stop()

full_text = "\n".join(text_parts).strip()
if not full_text:
    st.error("❌ No readable text found in PDF. It may be scanned images.")
    st.stop()

# ----------------- 分块与摘要 -----------------
summarizer, tokenizer = load_summarizer_and_tokenizer()
token_chunks = chunk_by_tokens(tokenizer, full_text, max_tokens=480, overlap=40)
st.write(f"🔍 Split into **{len(token_chunks)}** sections for summarization.")

token_chunks = token_chunks[: int(max_sections)]
progress = st.progress(0.0)
chapter_summaries: List[str] = []

for i, chunk in enumerate(token_chunks, start=1):
    inp = "summarize: " + chunk
    result = summarizer(
        inp,
        max_length=int(per_section_max_len),
        min_length=int(per_section_min_len),
        do_sample=False,
        clean_up_tokenization_spaces=True,
    )
    chapter_summary = result[0]["summary_text"].strip()
    chapter_summaries.append(f"### 📖 Chapter {i}\n{chapter_summary}")
    progress.progress(i / len(token_chunks))

# ----------------- 输出章节摘要 -----------------
st.success("✅ Chapter Summaries Generated!")
for ch in chapter_summaries:
    st.markdown(ch)

# ----------------- 全书综合摘要 -----------------
st.divider()
st.subheader("📙 Final Book Summary")
combined = " ".join([s.replace("### 📖 Chapter", "Chapter") for s in chapter_summaries])
final = summarizer(
    "summarize: " + combined[:12000],
    max_length=int(final_max_len),
    min_length=int(final_min_len),
    do_sample=False,
    clean_up_tokenization_spaces=True,
)[0]["summary_text"].strip()
st.write(final)

st.caption("🚀 Powered by T5-small • Token-aware chunking • Optimized for long PDFs")
