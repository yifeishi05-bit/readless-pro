# 🟦 ReadLess Pro — 大PDF稳固版（纯Python抽取式摘要）
import io
import re
import sys
from collections import Counter, defaultdict

import streamlit as st
import pdfplumber

# ============== 页面与侧边栏 ==============
st.set_page_config(page_title="📘 ReadLess Pro – Book Summarizer", page_icon="📘", layout="wide")
st.title("📚 ReadLess Pro – AI Book Summarizer (No-ML, Stable for Large PDFs)")
st.caption("不依赖深度学习推理，按页分段 + 抽取式摘要，稳定处理超长PDF。")

with st.sidebar:
    st.header("⚙️ 摘要设置（适合大文件）")
    mode = st.radio("摘要强度", ["精简", "标准（推荐）", "详细"], index=1)
    preset = {
        "精简": dict(pages_per_chunk=25, top_k_per_chunk=3, final_top_k=6),
        "标准（推荐）": dict(pages_per_chunk=20, top_k_per_chunk=5, final_top_k=10),
        "详细": dict(pages_per_chunk=12, top_k_per_chunk=8, final_top_k=16),
    }[mode]

    # 可选：根据预估页数微调分段
    est_pages = st.number_input("估计总页数（可选）", min_value=1, value=200, step=50,
                                help="用于自动调节每段页数：页数越大，每段页数也相应变大以减少段数。")
    if est_pages:
        preset["pages_per_chunk"] = max(8, min(40, int(est_pages / 10)))

    pages_per_chunk = st.number_input("每段包含页数", min_value=5, max_value=60, value=preset["pages_per_chunk"], step=1)
    top_k_per_chunk = st.slider("每段选取关键句数", 2, 15, preset["top_k_per_chunk"])
    final_top_k = st.slider("全书最终摘要关键句数", 4, 30, preset["final_top_k"])

    st.divider()
    st.caption(f"Python: {sys.version.split()[0]} • 纯Python摘要（无需GPU/模型）")

# ============== 文本与摘要工具（纯Python，无依赖） ==============
CN_PUNCS = "。！？；："
EN_PUNCS = r"\.\!\?\;\:"
SPLIT_REGEX = re.compile(rf"(?<=[{CN_PUNCS}])|(?<=[{EN_PUNCS}])")
WHITES = re.compile(r"\s+")
# 迷你停用集合（中英混合）
STOPWORDS = set("""
的 了 和 与 及 或 而 被 将 把 在 之 其 这 那 本 该 并 对 于 从 中 等 比 更 很 非 常 我们 他们 你们 以及 因此 因而 所以 但是 但是却 然而
the a an and or but so of in on at to for with from by this that these those is are was were be been being it they we you he she as if than then
""".split())

def clean_text(t: str) -> str:
    t = t.replace("\x00", " ").replace("\u200b", " ").replace("\ufeff", " ")
    t = WHITES.sub(" ", t)
    return t.strip()

def split_sentences(text: str):
    # 先按中英标点切，再合并太短的碎片
    raw = [s.strip() for s in SPLIT_REGEX.split(text) if s and s.strip()]
    sents = []
    buf = ""
    for s in raw:
        if len(s) < 8:  # 避免极短碎句
            buf += s
            continue
        if buf:
            s = buf + s
            buf = ""
        sents.append(s.strip())
    if buf:
        sents.append(buf.strip())
    return sents

def tokenize(sent: str):
    # 简单混合：按空白切词 + 对中文进一步按字符滑动
    parts = []
    for w in WHITES.split(sent):
        w = w.strip().lower()
        if not w:
            continue
        # 英文词直接收
        if re.search(r"[a-z]", w):
            parts.append(w)
        else:
            # 中文：按单字（可选：双字）简化
            for ch in w:
                if re.match(r"[\u4e00-\u9fff]", ch):
                    parts.append(ch)
    return [p for p in parts if p and p not in STOPWORDS and not p.isdigit()]

def score_sentences(sentences):
    # 词频加权 + 位置轻微加权
    freq = Counter()
    sent_tokens = []
    for s in sentences:
        toks = tokenize(s)
        sent_tokens.append(toks)
        freq.update(toks)

    if not freq:
        return [0.0] * len(sentences)

    maxf = max(freq.values()) or 1
    weights = {w: v / maxf for w, v in freq.items()}

    scores = []
    n = len(sentences)
    for i, toks in enumerate(sent_tokens):
        if not toks:
            scores.append(0.0)
            continue
        base = sum(weights.get(t, 0.0) for t in toks) / len(toks)
        # 位置加权：段首段尾略高
        pos_boost = 1.0 + 0.15 * (1 - abs((i + 1) - (n / 2)) / (n / 2 + 1e-9))
        scores.append(base * pos_boost)
    return scores

def summarize_chunk(text: str, top_k: int = 5):
    text = clean_text(text)
    if not text:
        return "(空段)"
    sents = split_sentences(text)
    if len(sents) <= top_k:
        return " ".join(sents)
    scores = score_sentences(sents)
    idx = sorted(range(len(sents)), key=lambda i: (-scores[i], i))[:top_k]
    idx.sort()  # 保留原文顺序
    return " ".join(sents[i] for i in idx)

# ============== 主流程（按页分段->分段摘要->全书摘要） ==============
uploaded = st.file_uploader("📄 上传PDF（支持上百/上千页）", type="pdf")
if not uploaded:
    st.stop()

st.info("✅ 文件已上传，开始解析文本…")
page_texts = []
try:
    raw = uploaded.read()
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        total_pages = len(pdf.pages)
        st.write(f"检测到总页数：**{total_pages}**")
        bar = st.progress(0.0)
        for i, page in enumerate(pdf.pages, start=1):
            try:
                t = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            except Exception:
                t = ""
            page_texts.append(clean_text(t))
            if i % 10 == 0 or i == total_pages:
                bar.progress(i / total_pages)
except Exception as e:
    st.error(f"❌ 解析PDF失败：{e}")
    st.stop()

if not any(page_texts):
    st.error("❌ 没有读到可用文本（可能是扫描版或加密PDF）。请先做OCR或换可检索版再试。")
    st.stop()

# 分段（按页数），避免一次性拼超大文本
chunks = []
buf = []
for i, t in enumerate(page_texts, start=1):
    if t:
        buf.append(t)
    if (i % pages_per_chunk == 0) or (i == len(page_texts)):
        chunk_text = "\n".join(buf).strip()
        if chunk_text:
            chunks.append(chunk_text)
        buf = []

st.write(f"🔍 已按每 {pages_per_chunk} 页分成 **{len(chunks)}** 段进行摘要。")
st.divider()

# 分段摘要
section_summaries = []
prog = st.progress(0.0)
for idx, ch in enumerate(chunks, start=1):
    summary = summarize_chunk(ch, top_k=top_k_per_chunk)
    section_summaries.append(summary)
    st.markdown(f"### 📖 第 {idx} 段")
    st.write(summary)
    prog.progress(idx / len(chunks))

# 全书摘要（对分段摘要再做一次抽取）
st.divider()
st.subheader("📙 全书最终摘要")
joined = " ".join(section_summaries)
final_summary = summarize_chunk(joined, top_k=final_top_k)
st.write(final_summary)

st.caption("✅ 纯Python抽取式摘要：不依赖Torch/Transformers，适合大体量PDF；若需神经网络精炼，可后续再接入轻量模型。")
