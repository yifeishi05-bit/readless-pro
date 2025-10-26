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

# --- 初始化本地 summarizer ---
@st.cache_resource
def load_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_model()

# --- 页面主体 ---
st.title("📘 ReadLess Pro – 100% Free Offline Version")
st.subheader("Upload PDFs or text to get instant AI summaries (no key, no API)")

uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf")

if uploaded_file:
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    st.info("✅ PDF uploaded successfully! Generating local summary...")

    try:
        chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
        summaries = []
        for chunk in chunks[:3]:  # 限制前三段避免太长
            summary = summarizer(chunk, max_length=180, min_length=40, do_sample=False)
            summaries.append(summary[0]["summary_text"])

        full_summary = "\n\n".join(summaries)
        st.success("🧠 Summary generated:")
        st.write(full_summary)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

st.caption("🚀 Powered by Hugging Face Transformers • Local model: distilBART-CNN")
