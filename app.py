import os
import re
import jieba
import streamlit as st
import pdfplumber
from collections import Counter

# -------------------- 页面配置 --------------------
st.set_page_config(page_title="ReadLess Pro – Lite (No-Model)", page_icon="📘")

# -------------------- 门禁逻辑 --------------------
REAL_CODE = os.getenv("ACCESS_CODE") or st.secrets.get("ACCESS_CODE", "")
BUY_LINK = "https://readlesspro.lemonsqueezy.com/buy/d0a09dc2-f156-4b4b-8407-12a87943bbb6"

st.sidebar.title("🔒 Member Login")
code = st.sidebar.text_input("Enter access code (for paid users)", type="password")
if code != REAL_CODE:
    st.warning("Please enter a valid access code to continue.")
    st.markdown(f"💳 [Click here to subscribe ReadLess Pro]({BUY_LINK})")
    st.stop()

# -------------------- 工具函数：语言检测与分句 --------------------
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
EN_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CN_SENT_SPLIT_RE = re.compile(r"[。！？…]+")

EN_STOPWORDS = {
    "the","a","an","is","are","was","were","am","be","to","of","and","in","that",
    "for","on","at","by","with","as","it","its","this","these","those","from",
    "or","not","but","than","then","so","such","if","into","their","there","they",
    "you","your","we","our","i","me","my","he","she","his","her","them","us",
}

CN_STOPWORDS = {
    "的","了","和","与","及","也","并","或","而","被","对","在","是","为","于","及其",
    "一个","一些","我们","你们","他们","它们","以及","同时","因此","但是","如果",
}

def is_chinese(text: str) -> bool:
    if not text:
        return False
    cjk = len(CJK_RE.findall(text))
    return (cjk / max(1, len(text))) > 0.2

def split_sentences(text: str, lang_cn: bool):
    if lang_cn:
        # 保留分隔符作为句末标记
        parts = [s.strip() for s in CN_SENT_SPLIT_RE.split(text) if s.strip()]
        ends  = CN_SENT_SPLIT_RE.findall(text)
        sents = []
        for i, p in enumerate(parts):
            end = ends[i] if i < len(ends) else ""
            sents.append((p + (end if end else "")))
        return sents
    else:
        return [s.strip() for s in EN_SENT_SPLIT_RE.split(text) if s.strip()]

# -------------------- 词频与句子打分 --------------------
def word_tokens(text: str, lang_cn: bool):
    if lang_cn:
        # 中文用结巴分词，去掉标点和停用词
        tokens = [t.strip() for t in jieba.cut(text) if t.strip()]
        tokens = [t for t in tokens if t not in CN_STOPWORDS and not re.fullmatch(r"\W+", t)]
        return tokens
    else:
        tokens = re.findall(r"[A-Za-z]+", text.lower())
        tokens = [t for t in tokens if t not in EN_STOPWORDS and len(t) > 1]
        return tokens

def score_sentences(sentences, lang_cn: bool):
    # 建立全文词频
    all_tokens = []
    for s in sentences:
        all_tokens.extend(word_tokens(s, lang_cn))
    if not all_tokens:
        return [0.0]*len(sentences)

    freq = Counter(all_tokens)
    maxf = max(freq.values())
    # TF（简化）归一化
    for k in freq:
        freq[k] = freq[k]/maxf

    scores = []
    for s in sentences:
        toks = word_tokens(s, lang_cn)
        if not toks:
            scores.append(0.0)
            continue
        # 句子分数 = 词频和 / log(句长+1)（防止偏袒超长句）
        raw = sum(freq.get(t, 0.0) for t in toks)
        denom = max(1.0, (len(toks))**0.6)
        scores.append(raw/denom)
    return scores

def extractive_summary(text: str, max_sentences: int = 8) -> str:
    lang_cn = is_chinese(text)
    sentences = split_sentences(text, lang_cn)
    if not sentences:
        return "（未能从文本中识别到有效句子）"

    # 太长则仅取前若干字符进行摘要以保证速度
    joined = "\n".join(sentences)
    if len(joined) > 20000:
        # 取前面部分，通常已覆盖主要信息
        head = int(len(joined)*0.6)
        text = joined[:head]
        sentences = split_sentences(text, lang_cn)

    scores = score_sentences(sentences, lang_cn)
    # 选取得分最高的 N 个句子，并按原文顺序恢复
    idx_sorted = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    chosen_idx = sorted(idx_sorted[:max_sentences])
    chosen = [sentences[i] for i in chosen_idx]

    # 合并，避免过短摘要
    result = " ".join(chosen).strip()
    if not result:
        result = "（摘要为空，可能原文为图片或不可选文本）"
    return result

# -------------------- 读取 PDF 文本 --------------------
def read_pdf_text(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

# -------------------- 页面主体 --------------------
st.title("📘 ReadLess Pro – Lite Edition (No Model, Cloud-safe)")
st.subheader("上传 PDF 或粘贴文本，生成快速摘要（无需大模型、无需 API Key）")

tab1, tab2 = st.tabs(["📄 上传 PDF", "✍️ 直接粘贴文本"])

with tab1:
    uploaded = st.file_uploader("选择 PDF 文件", type="pdf")
    max_sent = st.slider("摘要句子数", 3, 12, 8)
    if uploaded is not None:
        with st.spinner("正在解析 PDF 文本…"):
            content = read_pdf_text(uploaded)
        if not content.strip():
            st.error("❌ 没有从 PDF 中提取到可用文本（可能是扫描件或图片）。")
        else:
            st.info("✅ 解析完成，开始生成摘要…")
            summary = extractive_summary(content, max_sentences=max_sent)
            st.success("🧠 摘要：")
            st.write(summary)

with tab2:
    raw = st.text_area("把要摘要的内容粘贴到这里", height=220, placeholder="支持中英混合文本")
    max_sent2 = st.slider("摘要句子数（文本模式）", 3, 12, 8, key="txt_slider")
    if st.button("生成摘要", type="primary"):
        if not raw.strip():
            st.warning("请先粘贴文本。")
        else:
            with st.spinner("正在生成摘要…"):
                summary = extractive_summary(raw, max_sentences=max_sent2)
            st.success("🧠 摘要：")
            st.write(summary)

st.caption("⚡ 轻量版：不使用 transformers/torch，安装迅速、云端稳定。")
