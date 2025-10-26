import os
import streamlit as st

st.set_page_config(page_title="ReadLess Pro", page_icon="📘")

# --- 门禁逻辑：访问码验证 + 购买引导 ---
REAL_CODE = os.getenv("ACCESS_CODE") or st.secrets.get("ACCESS_CODE", "")
BUY_LINK = "https://readlesspro.lemonsqueezy.com/buy/d0a09dc2-f156-4b4b-8407-12a87943bbb6"

st.sidebar.title("🔒 会员登录")
code = st.sidebar.text_input("请输入访问码（付费用户获取）", type="password")

if code != REAL_CODE:
    st.warning("请输入正确的访问码后使用 ReadLess Pro 功能。")
    st.markdown(f"💳 [点此购买订阅 ReadLess Pro]({BUY_LINK})")
    st.stop()

# --- 门禁结束，以下是正式功能区 ---
st.title("📘 ReadLess Pro is Live!")
st.subheader("欢迎使用你的 AI 阅读助手 ✨")

st.write("你已成功登录高级版，可以在这里上传文件、总结内容或生成笔记。")
