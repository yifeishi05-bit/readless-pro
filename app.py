import os
import streamlit as st
import pdfplumber
from transformers import pipeline

# ----------------- 页面设置 -----------------
st.set_page_config(page_title="📘 ReadLess Pro – Book Summarizer", page_icon="📘")

st.title("📚 ReadLess Pro – AI Book Summarizer")
st.caption("Upload a long PDF (even full books!) and get automatic chapter summaries powered by AI (T5-small model).")

# ----------------- 安全码逻辑 -----------------
REAL_CODE = os.getenv("ACCESS_CODE") or st.secrets.get("ACCESS_CODE", "")
BUY_LINK = "https://readlesspro.lemonsqueezy.com/buy/d0a09dc2-f156-4b4b-8407-12a87943bbb6"
st.sidebar.title("🔒 Member Login")
code = st.sidebar.text_input("Enter access code (for paid users)", type="password")

if code != REAL_CODE:
    st.warning("Please enter a valid access code to continue.")
    st.markdown(f"💳 [Click here to subscribe ReadLess Pro]({BUY_LINK})")
    st.stop()

# ----------------- 模型加载（懒加载） -----------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="t5-small", tokenizer="t5-small")

# ----------------- 上传文件 -----------------
uploaded = st.file_uploader("📄 Upload a PDF file (book, report, or notes)", type="pdf")

if uploaded:
    st.info("✅ File uploaded successfully. Extracting text in batches...")
    text = ""
    with pdfplumber.open(uploaded) as pdf:
        total_pages = len(pdf.pages)
        st.write(f"Total pages detected: **{total_pages}**")
        for i, page in enumerate(pdf.pages):
            t = page.extract_text()
            if t:
                text += t + "\n"
            if (i + 1) % 20 == 0:  # 每 20 页分段处理
                st.text(f"Loaded {i+1}/{total_pages} pages...")

    if not text.strip():
        st.error("❌ No readable text found in PDF. It may be scanned images.")
        st.stop()

    # ----------------- 分段生成章节摘要 -----------------
    summarizer = load_summarizer()
    chunks = [text[i:i + 3000] for i in range(0, len(text), 3000)]
    st.write(f"🔍 Splitting text into {len(chunks)} sections for summarization...")

    progress = st.progress(0)
    chapter_summaries = []

    for i, chunk in enumerate(chunks[:10]):  # 限制最多10章（防止超时）
        result = summarizer(chunk, max_length=180, min_length=60, do_sample=False)
        chapter_summary = result[0]["summary_text"]
        chapter_summaries.append(f"### 📖 Chapter {i+1}\n{chapter_summary}")
        progress.progress((i + 1) / min(len(chunks), 10))

    # ----------------- 输出章节摘要 -----------------
    st.success("✅ Chapter Summaries Generated!")
    for ch in chapter_summaries:
        st.markdown(ch)

    # ----------------- 全书综合摘要 -----------------
    st.divider()
    st.subheader("📙 Final Book Summary")
    combined_text = " ".join([s for s in chapter_summaries])
    final_summary = summarizer(combined_text[:6000], max_length=250, min_length=100, do_sample=False)[0]["summary_text"]
    st.write(final_summary)

    st.caption("🚀 Powered by T5-small summarization • Lightweight and optimized for long PDFs")
