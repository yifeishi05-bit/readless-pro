# ReadLess Pro — 无模型大 PDF 稳定摘要版（超大文件友好）
import io
import math
import re
from collections import Counter, defaultdict
from typing import List, Tuple

import streamlit as st
import pdfplumber

st.set_page_config(page_title="📘 ReadLess Pro (Model-free)", page_icon="📘", layout="wide")
st.title("📚 ReadLess Pro — 大 PDF 稳定摘要（无模型）")
st.caption("不依赖任何大模型；针对 500~1000 页长文档做提取式摘要。若是扫描件请先做 OCR。")

# ---------------- 工具函数 ----------------
_SENT_SPLIT = re.compile(r"(?<=[。！？!?．.])\s+|(?<=[;；])\s+|(?<=[\n])")
_WORD_SPLIT = re.compile(r"[^\w\u4e00-\u9fff]+")

STOPWORDS = set("""
的 了 和 与 及 而 且 在 为 对 以 并 将 把 被 这 那 其 之 于 从 到 等 等等
是 就 都 又 很 及其 比 较 更 最 各 个 已 已经 如果 因为 所以 但是 然而
我们 你们 他们 她们 它们 本 文 之一 其中 通过 进行 能够 可以
a an the and or but if then else for to of in on with as by from into over under
be is are was were been being this that these those it its their our your
""".split())

def split_sentences(text: str) -> List[str]:
    # 先统一空白
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    parts = _SENT_SPLIT.split(text)
    sents = []
    for s in parts:
        s2 = s.strip()
        if len(s2) >= 2:
            sents.append(s2)
    return sents

def words_in(s: str) -> List[str]:
    # 中英混合分词（极简）：中文按字词混合、英文按 \w
    tokens = []
    for t in _WORD_SPLIT.split(s):
        t = t.strip().lower()
        if not t:
            continue
        # 单个汉字也保留，英文去停用词
        if (len(t) == 1 and not re.match(r"[\u4e00-\u9fff]", t)):
            continue
        if t in STOPWORDS:
            continue
        tokens.append(t)
    return tokens

def summarize_extractive(text: str, max_sent: int = 6) -> str:
    sents = split_sentences(text)
    if not sents:
        return ""
    # 计算词频 & 句子分数（TF * IDF-近似）
    docs = [words_in(s) for s in sents]
    df = Counter()
    for dw in docs:
        for w in set(dw):
            df[w] += 1
    N = len(sents)
    scores = []
    for i, dw in enumerate(docs):
        if not dw:
            scores.append((0.0, i)); continue
        tf = Counter(dw)
        s = 0.0
        for w, c in tf.items():
            idf = math.log(1 + N / (1 + df[w]))
            s += (c / len(dw)) * idf
        # 轻度长度惩罚，避免超长句独霸
        s = s / (1.0 + 0.15 * max(0, len(dw) - 40))
        scores.append((s, i))
    # 选 top-k，按原文顺序还原，增强可读性
    k = max(3, min(max_sent, max(3, int(N * 0.1))))
    top_idx = [i for _, i in sorted(scores, key=lambda x: x[0], reverse=True)[:k]]
    top_idx.sort()
    return " ".join(sents[i] for i in top_idx)

def chunk_pages(pages_text: List[str], pages_per_chunk: int) -> List[Tuple[int, int, str]]:
    chunks = []
    for i in range(0, len(pages_text), pages_per_chunk):
        j = min(len(pages_text), i + pages_per_chunk)
        t = "\n".join(pages_text[i:j]).strip()
        if t:
            chunks.append((i+1, j, t))  # (起始页, 结束页, 文本)
    return chunks

# ---------------- 侧边栏参数 ----------------
with st.sidebar:
    st.header("⚙️ 设置")
    pages_per_chunk = st.slider("每段合并页数", 10, 50, 20, 2, help="按页合并后做分段摘要，提升稳定性与速度")
    summary_sents = st.slider("每段摘要句数（上限）", 4, 12, 6, 1)
    final_sents = st.slider("总摘要句数（上限）", 6, 20, 12, 1)
    hard_cap_chars = st.number_input("单段字符硬上限", min_value=10000, value=25000, step=5000,
                                     help="防止超长段导致内存/时间暴涨；超出即截断")
    st.caption("提示：若 PDF 是扫描件（无可抽取文本），会提示需先 OCR。")

# ---------------- 主流程 ----------------
uploaded = st.file_uploader("📄 上传 PDF（支持超长文档）", type="pdf")
if not uploaded:
    st.stop()

st.info("文件已上传，开始解析页面文本…")
raw = uploaded.read()

pages_text = []
try:
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        total = len(pdf.pages)
        st.write(f"检测到页数：**{total}**")
        prog = st.progress(0.0)
        for i, page in enumerate(pdf.pages, start=1):
            try:
                t = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            except Exception:
                t = ""
            t = t.strip()
            if t:
                # 简单清洗 & 限制单页长度，避免极端页面
                t = re.sub(r"[ \t]+", " ", t)
                if len(t) > 12000:
                    t = t[:12000]
                pages_text.append(t)
            if i % 10 == 0 or i == total:
                prog.progress(i / total)
except Exception as e:
    st.error(f"❌ 解析 PDF 失败：{e}")
    st.stop()

if not pages_text:
    st.error("❌ 未抽取到正文文本。这通常是扫描版/图片版 PDF。请先用 OCR（如 Google Drive OCR、Adobe、ABBYY）转成可复制文本的 PDF。")
    st.stop()

# 分段
chunks = chunk_pages(pages_text, pages_per_chunk)
st.write(f"🔍 按每 {pages_per_chunk} 页合并，共得到 **{len(chunks)}** 段。")
summs = []
prog2 = st.progress(0.0)

for idx, (p_from, p_to, text) in enumerate(chunks, start=1):
    if len(text) > hard_cap_chars:
        text = text[:hard_cap_chars]
    s = summarize_extractive(text, max_sent=summary_sents) or "(本段内容过于稀疏，未生成摘要)"
    summs.append(f"### 📖 第 {idx} 段（页 {p_from}–{p_to}）\n{s}")
    prog2.progress(idx / len(chunks))

st.success("✅ 分段摘要完成！")
for s in summs:
    st.markdown(s)

# 最终总摘要（对所有分段摘要再次做提取式汇总）
st.divider()
st.subheader("📙 全书摘要（提取式）")
joined = " ".join(s.replace("### ", "").replace("\n", " ") for s in summs)
final_sum = summarize_extractive(joined, max_sent=final_sents) or "(总摘要生成失败——原文可能过短)"
st.write(final_sum)

# 导出
st.download_button(
    label="⬇️ 下载摘要 Markdown",
    data=("\n\n".join(summs) + "\n\n---\n\n## 全书摘要\n" + final_sum).encode("utf-8"),
    file_name="summary.md",
    mime="text/markdown"
)

st.caption("🚀 模型自由（无 Torch/Transformers）· 长文稳定 · 进度可视 · 适合教材/讲义/报告")
