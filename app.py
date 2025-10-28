import os
import streamlit as st
import pdfplumber

st.set_page_config(page_title="ReadLess Pro – Cloud Lite", page_icon="📘")

# --- Access Control ---
REAL_CODE = os.getenv("ACCESS_CODE") or st.secrets.get("ACCESS_CODE", "")
BUY_LINK = "https://readlesspro.lemonsqueezy.com/buy/d0a09dc2-f156-4b4b-8407-12a87943bbb6"

st.sidebar.title("🔒 Member Login")
code = st.sidebar.text_input("Enter access code (for paid users)", type="password")

if code != REAL_CODE:
    st.warning("Please enter a valid access code to continue.")
    st.markdown(f"💳 [Click here to subscribe ReadLess Pro]({BUY_LINK})")
    st.stop()

st.title("📘 ReadLess Pro – Cloud Lite Edition")
st.caption("Upload a PDF and get an AI summary. Model loads only when needed.")

uploaded_file = st.file_uploader("📄 Upload a PDF file", type="pdf")

if uploaded_file:
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"

    if not text.strip():
        st.error("❌ No text found in the PDF.")
        st.stop()

    st.info("✅ File uploaded! Click the button below to generate summary.")
    if st.button("🧠 Generate Summary"):
        with st.spinner("Loading summarizer model (this may take 30–60s)..."):
            from transformers import pipeline
            summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

        chunks = [text[i:i + 2000] for i in range(0, len(text), 2000)]
        summaries = []
        for chunk in chunks[:3]:
            summary = summarizer(chunk, max_length=180, min_length=40, do_sample=False)
            summaries.append(summary[0]["summary_text"])

        st.success("🧠 Summary generated:")
        st.write("\n\n".join(summaries))

st.caption("🚀 Powered by Hugging Face Transformers • Model: t5-small (lightweight, cloud-safe)")
