import os
import streamlit as st
import pdfplumber
import requests

# --- 页面配置 ---
st.set_page_config(page_title="ReadLess Pro", page_icon="📘")

# --- 门禁逻辑：访问码验证 + 购买引导 ---
REAL_CODE = os.getenv("ACCESS_CODE") or st.secrets.get("ACCESS_CODE", "")
BUY_LINK = "https://readlesspro.lemonsqueezy.com/buy/d0a09dc2-f156-4b4b-8407-12a87943bbb6"

st.sidebar.title("🔒 Member Login")
code = st.sidebar.text_input("Enter access code (for paid users)", type="password")

if code != REAL_CODE:
    st.warning("Please enter a valid access code to continue.")
    st.markdown(f"💳 [Click here to subscribe ReadLess Pro]({BUY_LINK})")
    st.stop()

# --- 页面主体 ---
st.title("📘 ReadLess Pro – 100% Free AI Reading Assistant")
st.subheader("Upload PDFs or text to get instant AI summaries (no API key required)")

uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf")

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    st.info("✅ PDF uploaded successfully! Generating summary...")

    # --- 免费 summarizer API（huggingface 免费接口） ---
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
            headers={"Authorization": "Bearer hf_yOUR_FREE_TOKEN_IF_YOU_HAVE_ONE"},
            json={"inputs": text[:3000]}  # 限制长度避免 Hugging Face 免费接口超时
        )

        if response.status_code == 200:
            summary = response.json()[0]["summary_text"]
            st.success("🧠 Summary generated:")
            st.write(summary)
        else:
            st.error(f"⚠️ Request failed ({response.status_code}): {response.text}")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

st.caption("Powered by Hugging Face • Free model: BART-large-CNN")
