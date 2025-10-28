# 📘 ReadLess Pro — Torch-free bulletproof edition (Streamlit Cloud / Py3.13)
import io
import re
import math
import sys
from collections import Counter, defaultdict
from typing import List, Tuple

import streamlit as st
import pdfplumber

st.set_page_config(page_title="📘 ReadLess Pro – Book Summarizer", page_icon="📘", layout="wide")
st.title("📚 ReadLess Pro – Book Summarizer (No-ML, Torch-free)")
st.caption("超稳：纯 Python 摘要算法（无深度学习依赖），支持超长 PDF。")

# -------------------- 控件 --------------------
with st.sidebar:
    st.header("⚙️ Controls")
    mode = st.radio("摘要强度", ["快速（提要）", "标准（推荐）", "详细（更长）"], index=1)
    presets = {
        "快速（提要）": dict(target_ratio=0.06, chunk_pages=20, final_sentences=6),
        "标准（推荐）": dict(target_ratio=0.09, chunk_pages=16, final_sentences=9),
        "详细（更长）": dict(target_ratio=0.12, chunk_pages=12, final_sentences=12),
    }
    P = presets[mode]
    # 语言：简单选择影响分句与停用词
    lang = st.selectbox("语言", ["english", "chinese"], index=0)
    custom_pages = st.number_input("分块页数（越小越稳）", 8, 40, P["chunk_pages"], 1,
                                   help="按页切块后分别摘要，避免一次性处理太大文本导致卡顿。")
    target_ratio = st.slider("章节摘要比例（句子数/原句子数）", 0.03, 0.2, P["target_ratio"], 0.01)
    final_sentences = st.slider("最终总摘要句子数", 3, 30, P["final_sentences"], 1)
    st.caption(f"Python: {sys.version.split()[0]} • 无 PyTorch/Transformers")

# -------------------- 文本工具 --------------------
EN_STOPS = set("""
a about above after again against all am an and any are as at be because been before being below between both but by
could did do does doing down during each few for from further had has have having he her here hers herself him himself his
how i if in into is it its itself let me more most my myself nor of on once only or other our ours ourselves out over own
same she should so some such than that the their theirs them themselves then there these they this those through to too
under until up very was we were what when where which while who whom why with you your yours yourself yourselves
""".split())

def sentence_split(text: str, language: str) -> List[str]:
    text = re.sub(r"\s+", " ", text)
    if language == "chinese":
        # 依据中文标点切句
        parts = re.split(r"(?<=[。！？；])", text)
    else:
        # 英文分句
        parts = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in parts if s and len(s.strip()) > 2]
    return sents

def tokenize(text: str, language: str) -> List[str]:
    if language == "chinese":
        # 粗粒度：按字母数字与汉字分词
        tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text)
        return [t.lower() for t in tokens]
    else:
        tokens = re.findall(r"[A-Za-z']+", text.lower())
        return [t for t in tokens if t not in EN_STOPS and len(t) > 1]

def build_idf(all_docs_tokens: List[List[str]]) -> defaultdict:
    N = len(all_docs_tokens)
    df = Counter()
    for toks in all_docs_tokens:
        df.update(set(toks))
    idf = defaultdict(float)
    for w, d in df.items():
        idf[w] = math.log((1 + N) / (1 + d)) + 1.0
    return idf

def summarize_chunk(text: str, language: str, target_ratio: float) -> Tuple[str, List[str]]:
    sents = sentence_split(text, language)
    if not sents:
        return "", []
    sent_tokens = [tokenize(s, language) for s in sents]
    # 计算 IDF/TF
    idf = build_idf([t for t in sent_tokens if t])
    scores = []
    for idx, toks in enumerate(sent_tokens):
        if not toks:
            scores.append((0.0, idx)); continue
        tf = Counter(toks)
        length = len(toks)
        # 句子得分：∑(tf*idf) / sqrt(length) 兼顾覆盖与长度惩罚
        score = sum((tf[w] * idf[w]) for w in tf) / math.sqrt(length)
        scores.append((score, idx))
    scores.sort(reverse=True, key=lambda x: x[0])

    keep = max(1, int(len(sents) * target_ratio))
    chosen_idx = sorted([idx for _, idx in scores[:keep]])
    picked = [sents[i] for i in chosen_idx]
    return " ".join(picked), picked

def chunk_pages_to_text(pages: List[str]) -> str:
    return "\n".join(pages)

# -------------------- 主流程 --------------------
def main():
    uploaded = st.file_uploader("📄 Upload a PDF file (book, report, or notes)", type="pdf")
    if not uploaded:
        return

    st.info("✅ File uploaded. Extracting text…")
    raw = uploaded.read()
    pages_text: List[str] = []
    try:
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            total_pages = len(pdf.pages)
            st.write(f"Total pages detected: **{total_pages}**")
            prog = st.progress(0.0)
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    t = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                except Exception:
                    t = ""
                pages_text.append(t)
                if i % 10 == 0 or i == total_pages:
                    prog.progress(i / total_pages)
    except Exception as e:
        st.error(f"❌ Failed to parse PDF: {e}")
        return

    full_text = "\n".join([p for p in pages_text if p.strip()]).strip()
    if not full_text:
        st.error("❌ No readable text found. It may be a scanned (image-only) PDF.")
        return

    # 分块摘要
    st.divider()
    st.subheader("📖 Chapter-like Summaries")
    chunk_size = int(custom_pages)
    chunk_summaries: List[str] = []
    sent_pool: List[str] = []

    prog2 = st.progress(0.0)
    total_chunks = max(1, (len(pages_text) + chunk_size - 1) // chunk_size)

    for ci in range(0, len(pages_text), chunk_size):
        block = chunk_pages_to_text(pages_text[ci:ci+chunk_size])
        chunk_summary, sent_list = summarize_chunk(block, lang, target_ratio)
        chunk_summaries.append(chunk_summary if chunk_summary else "(empty)")
        sent_pool.extend(sent_list)
        st.markdown(f"### 📘 Part {len(chunk_summaries)}")
        st.write(chunk_summary if chunk_summary else "_(This part had little extractable text.)_")
        prog2.progress(min(1.0, len(chunk_summaries) / total_chunks))

    # 最终总摘要：从所有选句里再打分一次，选出 N 句
    st.divider()
    st.subheader("📙 Final Book Summary")
    joined = " ".join(sent_pool)
    final_summary, picked = summarize_chunk(joined, lang, target_ratio=0.08)
    # 如果用户设置了固定句子数，则裁剪
    if picked:
        picked2 = picked[: int(final_sentences)]
        final_text = (" " if lang == "chinese" else " ").join(picked2)
    else:
        final_text = final_summary
    st.write(final_text if final_text else "(No final summary could be produced.)")

    st.caption("🚀 Torch-free · Works on 700+ page PDFs · Frequency/IDF sentence scoring · Chunk-wise summarization")

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        st.error("App crashed with an exception:")
        st.exception(ex)
