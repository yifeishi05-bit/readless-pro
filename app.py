import os
import streamlit as st
import pdfplumber
import requests

st.set_page_config(page_title="ReadLess Pro Free Edition", page_icon="📘")

# --- 门禁 ---
REAL_CODE = os.getenv("ACCESS_CODE") or st.secrets.get("ACCESS_CODE", "")
BUY_LINK = "https://readlesspro.lemonsqueezy.com/buy/d0a09dc2-f156-4b4b-8407-12a87943bbb6"

st.sidebar.title("🔒 Member Login")
code = st.sidebar.text_input("Enter access code (for paid users)", type="password")
if code != REAL_CODE:
    st.warning("Please enter a valid access code to continue.")
    st.markdown(f"💳 [Click here to subscribe ReadLess Pro]({BUY_LINK})")
    st.stop()

# --- 页面 ---
st.title("📘 ReadLess Pro – Completely Free Version (Zero API Keys)")
st.subheader("Upload PDFs and get instant AI summaries without any login or tokens")

uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf")

if uploaded_file:
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"

    st.info("✅ PDF uploaded successfully! Generating summary via public AI API...")

    try:
        # 调用匿名公开 summarizer（T5-base 模型）
        response = requests.post(
            "https://hf.space/embed/pszemraj/document-summarizer/api/predict/",
            json={"inputs": text[:3000]}  # 限制长度避免超时
        )

        if response.status_code == 200:
            data = response.json()
            summary = data["data"][0]
            st.success("🧠 Summary generated:")
            st.write(summary)
        else:
            st.error(f"⚠️ Request failed ({response.status_code}): {response.text}")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

st.caption("🚀 Powered by Hugging Face Spaces – Public T5 Summarizer API (free & no login)")
