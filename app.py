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

# --- OpenRouter 免费API配置 ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", "")

headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

# --- 页面主体 ---
st.title("📘 ReadLess Pro – AI Reading Assistant (Free OpenRouter Version)")
st.subheader("Upload PDFs or text to get instant AI summaries for free!")

uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf")

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    st.info("✅ PDF uploaded successfully! Generating summary...")

    # --- 发送请求到 OpenRouter ---
    data = {
        "model": "mistralai/mixtral-8x7b-instruct",  # ✅ 免费稳定模型
        "messages": [
            {"role": "system", "content": "You are a professional summarizer. Output in English or Chinese automatically."},
            {"role": "user", "content": f"Summarize this document clearly and concisely:\n\n{text[:10000]}"}  # 限制长度避免超时
        ]
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )

        if response.status_code == 200:
            summary = response.json()["choices"][0]["message"]["content"]
            st.success("🧠 Summary generated:")
            st.write(summary)
        else:
            st.error(f"⚠️ Request failed ({response.status_code}): {response.text}")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

st.caption("Powered by OpenRouter • Free model: Mixtral-8x7B-Instruct")
