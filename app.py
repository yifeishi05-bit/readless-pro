import os
import streamlit as st
import pdfplumber
import requests

# --- 页面配置 ---
st.set_page_config(page_title="ReadLess Pro – Free Edition", page_icon="📘")

# --- 门禁逻辑 ---
REAL_CODE = os.getenv("ACCESS_CODE") or st.secrets.get("ACCESS_CODE", "")
BUY_LINK = "https://readlesspro.lemonsqueezy.com/buy/d0a09dc2-f156-4b4b-8407-12a87943bbb6"

st.sidebar.title("🔒 Member Login")
code = st.sidebar.text_input("Enter access code (for paid users)", type="password")

if code != REAL_CODE:
    st.warning("Please enter a valid access code to continue.")
    st.markdown(f"💳 [Click here to subscribe ReadLess Pro]({BUY_LINK})")
    st.stop()

# --- 页面主体 ---
st.title("📘 ReadLess Pro – 100% Free Public Version")
st.subheader("Upload PDFs or text to get instant AI summaries (no key, no login)")

uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf")

if uploaded_file:
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"

    st.info("✅ PDF uploaded successfully! Generating summary...")

    try:
        # 使用公开 summarization 模型（T5-base）
        response = requests.post(
            "https://api-inference.huggingface.co/models/t5-small",
            json={"inputs": f"summarize: {text[:3000]}"}
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                summary = result[0]["summary_text"]
                st.success("🧠 Summary generated:")
                st.write(summary)
            else:
                st.error("⚠️ Could not parse model output. Try a shorter file.")
        else:
            st.error(f"⚠️ Request failed ({response.status_code}): {response.text}")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

st.caption("🚀 Powered by Hugging Face Public API (Model: t5-small, no key required)")
