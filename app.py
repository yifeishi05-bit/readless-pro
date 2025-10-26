import os
import streamlit as st

st.set_page_config(page_title="ReadLess Pro", page_icon="📘")

# --- 统一访问码门禁 ---
REAL_CODE = os.getenv("ACCESS_CODE") or st.secrets.get("ACCESS_CODE", "")
if REAL_CODE:
    code = st.sidebar.text_input("访问码（付费用户获取）", type="password")
    if code != REAL_CODE:
        st.warning("请输入有效访问码后再使用。购买或试用请联系：support@yourdomain.com")
        st.stop()
# --- 门禁结束 ---

st.title("📘 ReadLess Pro is Live!")
st.subheader("Welcome to your first deployed Streamlit app 🎉")

st.write("Congratulations, Alex! Your app is successfully running on Streamlit Cloud.")
st.write("Now you can start adding features, upload PDFs, or connect AI summaries here.")
