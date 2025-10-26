import os
import streamlit as st
import pdfplumber
from openai import OpenAI

st.set_page_config(page_title="ReadLess Pro", page_icon="📘")

# --- 门禁 ---
REAL_CODE = os.getenv("ACCESS_CODE") or st.secrets.get("ACCESS_CODE", "")
BUY_LINK = "https://readlesspro.lemonsqueezy.com/buy/d0a09dc2-f156-4b4b-8407-12a87943bbb6"

st.sidebar.title("🔒 Member Login")
code = st.sidebar.text_input("Enter access code (for paid users)", type="password")

if code != REAL_CODE:
    st.warning("Please enter a valid access code to continue.")
    st.markdown(f"💳 [Click here to subscribe ReadLess Pro]({BUY_LINK})")
    st.stop()

# --- 使用 OpenRouter 免费 API ---
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["OPENROUTER_API_KEY"]
)

st.title("📘 ReadLess Pro – AI Reading Assistant (Free Edition)")
st.subheader("Upload PDFs or text to get instant AI summaries for free!")

uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf")

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        text = "".join(page.extract_text() or "" for page in pdf.pages)

    st.info("✅ PDF uploaded successfully! Generating summary...")

    try:
        response = client.chat.completions.create(
            model="mistralai/mistral-tiny",
            messages=[
                {"role": "system", "content": "You are a professional summarizer."},
                {"role": "user", "content": f"Summarize this text clearly and concisely:\n\n{text}"}
            ],
        )
        summary = response.choices[0].message.content
        st.success("🧠 Summary generated:")
        st.write(summary)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

st.caption("Powered by OpenRouter • Free model: Mistral-Tiny")
