# ========== ReadLess Pro (Streamlit) ==========
# 功能：
# 1) 访问码门禁（配合 Lemon Squeezy 销售页）
# 2) 上传 PDF 或粘贴文本，自动分块摘要（Map-Reduce）
# 3) 输出 Executive summary / Key takeaways / Action items / Keywords / TL;DR
# 4) 导出 Markdown
# ----------------------------------------------

import os
import io
import math
import time
from typing import List, Tuple

import streamlit as st

# PDF 解析
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# token 计数 + 分块
try:
    import tiktoken
except Exception:
    tiktoken = None

# OpenAI
from openai import OpenAI

# ----------------- 基础设置 -----------------
st.set_page_config(page_title="ReadLess Pro", page_icon="📘", layout="wide")

# 付款链接（你的 Lemon Squeezy 结账地址）
BUY_LINK = "https://readlesspro.lemonsqueezy.com/buy/d0a09dc2-f156-4b4b-8407-12a87943bbb6"

# Secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
REAL_CODE = os.getenv("ACCESS_CODE") or st.secrets.get("ACCESS_CODE", "")

# ----------------- 门禁 -----------------
st.sidebar.title("🔒 Member Login")
code = st.sidebar.text_input("Enter your access code (paid users)", type="password")

if code != REAL_CODE:
    st.warning("Please enter a valid access code to use ReadLess Pro.")
    st.sidebar.markdown(f"💳 **No code yet?** [Subscribe here]({BUY_LINK})")
    st.stop()

# ----------------- 工具函数 -----------------
def get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not found. Set it in Streamlit Secrets first.")
        st.stop()
    return OpenAI(api_key=OPENAI_API_KEY)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """优先用 pdfplumber；失败再用 pypdf。"""
    text = ""
    if pdfplumber is not None:
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    if t.strip():
                        text += t + "\n\n"
        except Exception:
            text = ""
    if (not text.strip()) and (PdfReader is not None):
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = []
            for p in reader.pages:
                t = p.extract_text() or ""
                if t.strip():
                    pages.append(t)
            text = "\n\n".join(pages)
        except Exception:
            pass
    return text.strip()

def count_tokens(s: str, model: str = "gpt-4o-mini") -> int:
    if tiktoken is None:
        # 粗略计数：四字一 token（够用）
        return max(1, len(s) // 4)
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(s))

def chunk_text(s: str, max_tokens: int = 1200, overlap_tokens: int = 80) -> List[str]:
    """按 token 长度分块，带少量重叠，提升连贯性。"""
    if not s.strip():
        return []
    if tiktoken is None:
        # 无 tiktoken 时，退化为字符切分
        step = max_tokens * 4
        ovlp = overlap_tokens * 4
        chunks = []
        i = 0
        while i < len(s):
            end = min(len(s), i + step)
            chunks.append(s[i:end])
            i = end - ovlp
            if i < 0:
                i = 0
            if i >= len(s):
                break
        return chunks

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(s)
    chunks = []
    i = 0
    while i < len(tokens):
        end = min(len(tokens), i + max_tokens)
        chunk_tokens = tokens[i:end]
        chunks.append(enc.decode(chunk_tokens))
        i = end - overlap_tokens
        if i < 0:
            i = 0
        if i >= len(tokens):
            break
    return chunks

# ----------------- 摘要调用 -----------------
def call_chunk_summary(client: OpenAI, model: str, text: str, lang: str, style: str) -> str:
    """对单个分块进行摘要。"""
    prompt = f"""
You are a professional summarizer. Summarize the following content in {lang}.
Style: {style}. Be concise and informative.
Return a well-structured markdown with:
- Short summary (3-5 sentences)
- 3-7 key bullets
- Important terms or data (if any)

Text:
{text}
"""
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You produce clear, compact, and faithful summaries."},
            {"role": "user", "content": prompt}
        ]
    )
    return resp.choices[0].message.content.strip()

def call_combine_summary(client: OpenAI, model: str, pieces: List[str], lang: str, detail: str) -> str:
    """把多个分块摘要合并为最终报告。"""
    joined = "\n\n---\n\n".join(pieces)
    prompt = f"""
You are ReadLess Pro. Combine the following chunk summaries into ONE final report in {lang}.
Output markdown sections:

# Executive Summary
(4-8 sentences)

## Key Takeaways
- 5–10 bullets

## Action Items (if applicable)
- 3–8 bullets focused on next steps

## Keywords / Terms
- comma-separated keywords

## TL;DR
- one sentence under 30 words

Focus level: {detail}
Chunk summaries:
{joined}
"""
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a precise editor. You merge points, remove duplicates, ensure coherence."},
            {"role": "user", "content": prompt}
        ]
    )
    return resp.choices[0].message.content.strip()

def build_markdown_report(title: str, final_md: str, meta: dict) -> str:
    header = f"# {title}\n\n"
    info = []
    for k, v in meta.items():
        info.append(f"- **{k}**: {v}")
    meta_md = "## Document Info\n" + "\n".join(info) + "\n\n"
    return header + meta_md + final_md + "\n"

# ----------------- 页面 UI -----------------
st.title("📘 ReadLess Pro")
st.caption("Upload a PDF or paste text → AI summarizes it for you.")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    lang = st.selectbox("Report language", ["English", "Chinese"], index=0)
    if lang == "Chinese":
        lang_disp = "Chinese (Simplified)"
    else:
        lang_disp = "English"
    style = st.selectbox("Tone / Style", ["Concise & Neutral", "Consulting Style", "Academic Tone"], index=0)
    detail = st.selectbox("Detail level", ["High", "Medium", "Low"], index=0)
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    max_tokens_per_chunk = st.slider("Max tokens per chunk", 600, 2000, 1200, step=100)

tab1, tab2 = st.tabs(["📄 Upload PDF", "📝 Paste Text"])

input_text = ""
meta_info = {}

with tab1:
    pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if pdf_file is not None:
        file_bytes = pdf_file.read()
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(file_bytes)
        if not text:
            st.error("Could not extract text from this PDF.")
        else:
            input_text = text
            meta_info["Source"] = pdf_file.name
            meta_info["Chars"] = len(text)
            meta_info["Tokens (approx)"] = count_tokens(text)

with tab2:
    pasted = st.text_area("Paste raw text here", height=260, placeholder="Paste article, report, or notes...")
    if pasted and pasted.strip():
        input_text = pasted.strip()
        meta_info["Source"] = "Pasted Text"
        meta_info["Chars"] = len(input_text)
        meta_info["Tokens (approx)"] = count_tokens(input_text)

if not input_text:
    st.info("Upload a PDF or paste text to start.")
    st.stop()

# ----------------- 摘要执行 -----------------
client = get_openai_client()

colA, colB, colC = st.columns([1,1,1])
with colA:
    st.metric("Characters", f"{len(input_text):,}")
with colB:
    st.metric("Approx. tokens", f"{count_tokens(input_text):,}")
with colC:
    st.metric("Chunks (est.)", f"{math.ceil(count_tokens(input_text)/max_tokens_per_chunk):,}")

run = st.button("🚀 Generate Summary", type="primary")
if not run:
    st.stop()

with st.spinner("Splitting into chunks..."):
    chunks = chunk_text(input_text, max_tokens=max_tokens_per_chunk, overlap_tokens=80)
if not chunks:
    st.error("Failed to split the input into chunks.")
    st.stop()

st.success(f"Prepared {len(chunks)} chunks. Summarizing...")

chunk_summaries = []
progress = st.progress(0.0)
status = st.empty()

for i, ch in enumerate(chunks, start=1):
    status.write(f"Summarizing chunk {i}/{len(chunks)} ...")
    try:
        s = call_chunk_summary(client, model=model, text=ch, lang=lang_disp, style=style)
    except Exception as e:
        s = f"*(Chunk {i} failed: {e})*"
    chunk_summaries.append(s)
    progress.progress(i/len(chunks))
    # 轻微节流，避免并发限速
    time.sleep(0.2)

with st.spinner("Combining chunks into final report..."):
    try:
        final_report_md = call_combine_summary(client, model=model, pieces=chunk_summaries, lang=lang_disp, detail=detail)
    except Exception as e:
        st.error(f"Failed to combine summaries: {e}")
        st.stop()

title = meta_info.get("Source", "ReadLess Report")
full_md = build_markdown_report(title, final_report_md, meta_info)

st.divider()
st.subheader("✅ Final Report")
st.markdown(full_md)

st.download_button(
    label="⬇️ Download Markdown",
    data=full_md.encode("utf-8"),
    file_name=f"{os.path.splitext(title)[0]}_ReadLess_Report.md",
    mime="text/markdown",
)
st.caption("Tip: you can paste the Markdown into Notion/Word.")
