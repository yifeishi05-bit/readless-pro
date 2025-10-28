import os
import io
import math
import warnings
from typing import List, Tuple

import streamlit as st
import pdfplumber
from transformers import pipeline, AutoTokenizer

warnings.filterwarnings("ignore", category=SyntaxWarning)

# ----------------- 页面设置 -----------------
st.set_page_config(page_title="📘 ReadLess Pro – Book Summarizer", page_icon="📘", layout="wide")
st.title("📚 ReadLess Pro – AI Book Summarizer")
st.caption("Upload a long PDF (even full books!) and get automatic chapter summaries powered by AI (T5-small model).")

# ----------------- 安全码逻辑 -----------------
REAL_CODE = os.getenv("ACCESS_CODE") or st.secrets.get("ACCESS_CODE", "")
BUY_LINK = "https://readlesspro.lemonsqueezy.com/buy/d0a09dc2-f156-4b4b-8407-12a87943bbb6"

with st.sidebar:
    st.title("🔒 Member Login")
    code = st.text_input("Enter access code (for paid users)", type="password")
    max_sections = st.number_input("Max sections to summarize", 5, 100, 20, 1, help="上限避免超时/超额调用")
    per_section_max_len = st.slider("Per-section max length (tokens)", 80, 300, 180, 10)
    per_section_min_len = st.slider("Per-section min length (tokens)", 30, 200, 60, 10)
    final_max_len = st.slider("Final summary max length (tokens)", 150, 500, 300, 10)
    final_min_len = st.slider("Final summary min length (tokens)", 80, 300, 120, 10)

if code != REAL_CODE:
    st.warning("Please enter a valid access code to continue.")
    st.markdown(f"💳 [Click here to subscribe ReadLess Pro]({BUY_LINK})")
    st.stop()

# ----------------- 模型加载（懒加载） -----------------
@st.cache_resource(show_spinner=True)
def load_summarizer_and_tokenizer():
    tok = AutoTokenizer.from_pretrained("t5-small", use_fast=True)
    summarizer = pipeline(
        "summarization",
        model="t5-small",
        tokenizer=tok,
        framework="pt",
        device=0 if os.getenv("CUDA_VISIBLE_DEVICES") not in (None, "", "-1") else -1,
    )
    return summarizer, tok

# ---------- Token 级分块（保证 T5 输入不溢出） ----------
def chunk_by_tokens(tokenizer: AutoTokenizer, text: str, max_tokens: int = 480, overlap: int = 50) -> List[str]:
    if not text.strip():
        return []
    # 先用段落粗分，再合并到 token 限制
    paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    buf_tokens = 0

    def tok_len(t: str) -> int:
        return len(tokenizer.encode(t, add_special_tokens=False))

    for p in paras:
        p_tokens = tok_len(p)
        if p_tokens > max_tokens:  # 极长段落再细切（按句号/顿号/句末标点）
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
                        chunks.append(" ".join(buf))
                        # overlap
                        if overlap > 0:
                            tail = " ".join(buf)[-overlap * 2 :]
                            buf = [tail]
                            buf_tokens = tok_len(tail)
                        else:
                            buf, buf_tokens = [], 0
                    if s_tokens <= max_tokens:
                        buf.append(s)
                        buf_tokens = tok_len(" ".join(buf))
                    else:
                        # 硬切：超长句，按 token 硬切
                        ids = tokenizer.encode(s, add_special_tokens=False)
                        for i in range(0, len(ids), max_tokens):
                            piece = tokenizer.decode(ids[i : i + max_tokens])
                            chunks.append(piece)
                        buf, buf_tokens = [], 0
        else:
            if buf_tokens + p_tokens <= max_tokens:
                buf.append(p)
                buf_tokens += p_tokens
            else:
                chunks.append(" ".join(buf))
                # overlap
                if overlap > 0:
                    tail = " ".join(buf)[-overlap * 2 :]
                    buf = [tail, p]
                    buf_tokens = tok_len(" ".join(buf))
                else:
                    buf, buf_tokens = [p], p_tokens
    if buf:
        chunks.append(" ".join(buf))
    return [c.strip() for c in chunks if c.strip()]

# ----------------- 上传文件 -----------------
uploaded = st.file_uploader("📄 Upload a PDF file (book, report, or notes)", type="pdf")

if not uploaded:
    st.stop()

# ----------------- PDF 解析 -----------------
st.info("✅ File uploaded successfully. Extracting text...")
text_parts: List[str] = []
try:
    with pdfplumber.open(io.BytesIO(uploaded.read())) as pdf:
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

# T5-small 默认最大输入 512，留一点余量
token_chunks = chunk_by_tokens(tokenizer, full_text, max_tokens=480, overlap=40)
st.write(f"🔍 Split into **{len(token_chunks)}** sections for summarization.")

# 限制节数，避免超时
token_chunks = token_chunks[: int(max_sections)]
progress = st.progress(0.0)
chapter_summaries: List[str] = []

for i, chunk in enumerate(token_chunks, start=1):
    # T5 需要以 "summarize: " 前缀更稳定
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
