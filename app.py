# ReadLess Pro — 无模型大 PDF 稳定摘要 + 诊断版（强制 Py3.11 环境）
import re, io, math
from collections import Counter
from typing import List

import streamlit as st

# —— 诊断区：无论如何先把页面跑起来 —— #
st.set_page_config(page_title="📘 ReadLess Pro (Model-free, Py3.11)", page_icon="📘", layout="wide")
st.title("📚 ReadLess Pro — 大 PDF 稳定摘要（无模型，Py3.11）")
st.caption("纯提取式摘要；不依赖 torch/transformers。首先确认页面正常渲染，然后上传大 PDF。")

# 打印环境信息，帮助你确认 Cloud 已按 runtime.txt 起了 3.11
import sys, platform
st.info(f"Python: **{sys.version}**  •  Platform: **{platform.platform()}**")

# 尝试导入 pdfplumber，并把版本打印出来；失败会在页面显示详细异常（不白屏）
try:
    import pdfplumber
    import pdfminer
    st.success(f"pdfplumber ✅  | pdfplumber={getattr(pdfplumber,'__version__','?')}  pdfminer={getattr(pdfminer,'__version__','?')}")
except Exception as e:
    st.error("❌ 导入 pdfplumber 失败，请截图这段错误给我：")
    st.exception(e)

# ---------------- 文本处理（提取式摘要，无模型） ----------------
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
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    parts = _SENT_SPLIT.split(text)
    return [s.strip() for s in parts if len(s.strip()) >= 2]

def words_in(s: str) -> List[str]:
    tokens = []
    for t in _WORD_SPLIT.split(s):
        t = t.strip().lower()
        if not t:
            continue
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
        score = 0.0
        for w, c in tf.items():
            idf = math.log(1 + N / (1 + df[w]))
            score += (c / len(dw)) * idf
        score = score / (1.0 + 0.15 * max(0, len(dw) - 40))
        scores.append((score, i))
    k = max(3, min(max_sent, max(3, int(N * 0.1))))
    top_idx = [i for _, i in sorted(scores, key=lambda x: x[0], reverse=True)[:k]]
    top_idx.sort()
    return " ".join(sents[i] for i in top_idx)

def chunk_pages(pages_text: List[str], pages_per_chunk: int):
    chunks = []
    for i in range(0, len(pages_text), pages_per_chunk):
        j = min(len(pages_text), i + pages_per_chunk)
        t = "\n".join(pages_text[i:j]).strip()
        if t:
            chunks.append((i+1, j, t))
    return chunks

# ---------------- 侧边栏参数 ----------------
with st.sidebar:
    st.header("⚙️ 设置")
    pages_per_chunk = st.slider("每段合并页数", 10, 50, 20, 2)
    summary_sents = st.slider("每段摘要句数（上限）", 4, 12, 6, 1)
    final_sents = st.slider("总摘要句数（上限）", 6, 20, 12, 1)
    hard_cap_chars = st.number_input("单段字符硬上限", min_value=10000, value=25000, step=5000)

# ---------------- 主流程（保证页面始终渲染，不用 st.stop） ----------------
uploaded = st.file_uploader("📄 上传 PDF（支持 700+ 页）", type="pdf")
if not uploaded:
    st.warning("未上传文件。页面正常即说明部署成功；请上传大 PDF 测试。")
else:
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
                    t = re.sub(r"[ \t]+", " ", t)
                    if len(t) > 12000:
                        t = t[:12000]
                    pages_text.append(t)
                if i % 10 == 0 or i == total:
                    prog.progress(i / total)
    except Exception as e:
        st.error("❌ 解析 PDF 失败（多为扫描版或损坏 PDF）。错误详情：")
        st.exception(e)
        pages_text = []

    if not pages_text:
        st.error("❌ 未抽取到正文文本。若为扫描/图片版，请先 OCR 再上传。")
    else:
        chunks = chunk_pages(pages_text, pages_per_chunk)
        st.write(f"🔍 按每 {pages_per_chunk} 页合并，共 **{len(chunks)}** 段。")
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

        st.divider()
        st.subheader("📙 全书摘要（提取式）")
        joined = " ".join(s.replace("### ", "").replace("\n", " ") for s in summs)
        final_sum = summarize_extractive(joined, max_sent=final_sents) or "(总摘要生成失败——原文可能过短)"
        st.write(final_sum)

        st.download_button(
            label="⬇️ 下载摘要 Markdown",
            data=("\n\n".join(summs) + "\n\n---\n\n## 全书摘要\n" + final_sum).encode("utf-8"),
            file_name="summary.md",
            mime="text/markdown"
        )

st.caption("🚀 模型自由 · 长文稳定 · 进度可视 · 若仍失败，页面会显示详细异常（截图给我）")
