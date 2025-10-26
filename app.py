import os
import streamlit as st
from openai import OpenAI
import PyPDF2

# --- 页面配置 ---
st.set_page_config(page_title="ReadLess Pro", page_icon="📘")

# --- 访问码门禁 ---
REAL_CODE = os.getenv("ACCESS_CODE") or st.secrets.get("ACCESS_CODE", "")
BUY_LINK = "https://readlesspro.lemonsqueezy.com/buy/d0a09dc2-f156-4b4b-8407-12a87943bbb6"

st.sidebar.title("🔒 Member Login")
code = st.sidebar.text_input("Enter access code (for paid users)", type="password")

if code != REAL_CODE:
    st.warning("Please enter a valid access code to continue.")
    st.markdown(f"💳 [Click here to subscribe ReadLess Pro]({BUY_LINK})")
    st.stop()

# --- 初始化 OpenAI ---
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
if not api_key:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY in Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

# --- 主界面 ---
st.title("📘 ReadLess Pro — Your AI Reading Assistant")
st.write("Upload PDFs or paste text below to get instant AI summaries in English and Chinese.")

# --- 上传或输入 ---
upload = st.file_uploader("📤 Upload a PDF file", type=["pdf"])
text_input = st.text_area("📝 Or paste your text here:")

content = ""

if upload is not None:
    pdf_reader = PyPDF2.PdfReader(upload)
    content = "".join(page.extract_text() or "" for page in pdf_reader.pages)
elif text_input.strip():
    content = text_input.strip()

if content:
    if st.button("✨ Summarize"):
        with st.spinner("Generating summary using AI..."):
            prompt = f"""Summarize the following text in English and Chinese.

Text:
{content[:6000]}"""   # 截取前 6000 字符避免 token 超限

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a bilingual academic assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            result = response.choices[0].message.content
            st.success("✅ Summary generated successfully!")
            st.write(result)
else:
    st.info("👆 Upload a PDF or paste text to start.")
