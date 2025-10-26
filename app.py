import os
import streamlit as st
import pdfplumber
from openai import OpenAI

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

# --- 使用 OpenRouter 免费模型 ---
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"]
)

# --- 页面内容 ---
st.title("📘 ReadLess Pro – AI Reading Assistant (Free OpenRouter Version)")
st.subheader("Upload PDFs or text to get instant AI summaries for free!")

uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf")

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t

    if not text.strip():
        st.error("⚠️ Could not extract text from the PDF.")
        st.stop()

    st.info("✅ PDF uploaded successfully! Generating summary...")

    try:
        response = client.chat.completions.create(
            model="mistralai/mistral-tiny",  # 永久免费模型
            messages=[
                {"role": "system", "content": "You are a professional summarizer. Write clear, concise summaries in English."},
                {"role": "user", "content": f"Summarize this text clearly and concisely:\n\n{text[:8000]}"}  # 限制长度防止超限
            ],
        )

        summary = response.choices[0].message.content
        st.success("🧠 Summary generated:")
        st.write(summary)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

st.caption("Powered by OpenRouter • Model: mistralai/mistral-tiny (Free)")
