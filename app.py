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

# --- 页面内容 ---
st.title("📘 ReadLess Pro – AI Reading Assistant (Free Version)")
st.subheader("Upload PDFs or text to get instant AI summaries – powered by Mixtral 8x7B")

uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf")

if uploaded_file:
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    st.info("✅ PDF uploaded successfully! Generating summary...")

    try:
        headers = {
            "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "mistralai/mixtral-8x7b",
            "messages": [
                {"role": "system", "content": "You are a professional summarizer."},
                {"role": "user", "content": f"Summarize this text:\n\n{text[:10000]}"},
            ],
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

        if response.status_code == 200:
            summary = response.json()["choices"][0]["message"]["content"]
            st.success("🧠 Summary:")
            st.write(summary)
        else:
            st.error(f"⚠️ Request failed ({response.status_code}): {response.text}")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

st.caption("Powered by OpenRouter • Free model: Mixtral-8x7B")
