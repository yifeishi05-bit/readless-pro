import os
import streamlit as st
import pdfplumber
from transformers import pipeline

# --- 页面配置 ---
st.set_page_config(page_title="ReadLess Pro – Offline Edition", page_icon="📘")

# --- 门禁逻辑 ---
REAL_CODE = os.getenv("ACCESS_CODE") or st.secrets.get("ACCESS_CODE", "")
BUY_LINK = "https://readlesspro.lemonsqueezy.com/buy/d0a09dc2-f156-4b4b-8407-12a87943bbb6"

st.sidebar.title("🔒 Member Login")
code = st.sidebar.text_input("Enter access code (for paid users)", type="password")

if code != REAL_CODE:
    st.warning("Please enter a valid access code to continue.")
    st.markdown(f"💳 [Click here to subscribe ReadLess Pro]({BUY_LINK})")
    st.stop()

# --- 延迟加载 summarizer 模型（轻量版） ---
@st.cache_resource
def load_model():
    try:
        summarizer = pipeline(
            "summarization",
            model="t5-small",           # ✅ 轻量模型，启动快
            tokenizer="t5-small"
        )
        return summarizer
    except Exception as e:
        st.error(f"⚠️ 模型加载失败: {e}")
        return None

# --- 页面主体 ---
st.title("📘 ReadLess Pro – Lightweight AI Summarizer")
st.subheader("Upload a PDF and get instant AI summaries (no API key required)")

uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf")

if uploaded_file:
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if not text.strip():
            st.error("❌ No extractable text found in this PDF.")
            st.stop()

        st.info("✅ PDF uploaded successfully! Loading summarizer model...")

        summarizer = load_model()
        if summarizer is None:
            st.error("⚠️ Summarizer model not available. Please retry later.")
            st.stop()

        # 分段摘要（每2000字符一段）
        chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)]
        summaries = []

        progress = st.progress(0)
        for idx, chunk in enumerate(chunks[:3]):  # 限制最多3段
            summary = summarizer(chunk, max_length=180, min_length=40, do_sample=False)
            summaries.append(summary[0]["summary_text"])
            progress.progress((idx + 1) / min(len(chunks), 3))

        full_summary = "\n\n".join(summaries)
        st.success("🧠 Summary generated successfully!")
        st.write(full_summary)

    except Exception as e:
        st.error(f"⚠️ Error processing PDF: {e}")

st.caption("🚀 Powered by Hugging Face Transformers • Model: t5-small (lightweight, offline-safe)")
