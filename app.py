# app.py — ReadLess Pro (Py3.13 + Torch 2.5.x workaround)

import os
# —— 关键规避：在任何 torch/transformers 导入前设置 —— #
os.environ.setdefault("PYTORCH_JIT", "0")            # 禁用 JIT，绕过 torch.classes 的注册路径问题
os.environ.setdefault("TORCH_DISABLE_JIT", "1")      # 双保险
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # 强制 CPU
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import io
import sys
import warnings
from typing import List

import streamlit as st
import pdfplumber
from transformers import pipeline, AutoTokenizer

warnings.filterwarnings("ignore", category=SyntaxWarning)

# ----------------- 页面设置 -----------------
st.set_page_config(page_title="📘 ReadLess Pro – Book Summarizer", page_icon="📘", layout="wide")
st.title("📚 ReadLess Pro – AI Book Summarizer")
st.caption("Upload a long PDF (even full books!) and get automatic chapter summaries powered by AI (T5-small).")

# ----------------- 会员与登录 -----------------
REAL_CODE = os.getenv("ACCESS_CODE") or st.secrets.get("ACCESS_CODE", "")
BUY_LINK = "https://readlesspro.lemonsqueezy.com/buy/d0a09dc2-f156-4b4b-8407-12a87943bbb6"

# ----------------- 控制面板（傻瓜模式 + 高级设置） -----------------
with st.sidebar:
    st.header("🔒 Member Login")
    code = st.text_input("Enter access code (for paid users)", type="password")
    st.markdown(f"若没有：💳 [点击订阅 ReadLess Pro]({BUY_LINK})")

    st.divider()
    st.header("⚙️ Controls")
    mode = st.radio(
        "Summary mode",
        ["快速（最短）", "标准（推荐）", "详细（更长）"],
        index=1,
        help="选择一档就好；需要微调再展开高级设置"
    )
    presets = {
        "快速（最短）":  dict(sections=16, sec_max=140, sec_min=60, final_max=260, final_min=120),
        "标准（推荐）":  dict(sections=20, sec_max=180, sec_min=70, final_max=320, final_min=140),
        "详细（更长）":  dict(sections=26, sec_max=220, sec_min=90, final_max=420, final_min=200),
    }
    P = presets[mode]

    est_pages = st.number_input("估计页数（可选）", min_value=1, value=200, step=50,
                                help="填写后会自动调节分段数，更贴合书本长度")
    if est_pages:
        P["sections"] = min(30, max(10, int(est_pages / 15)))  # 约15页一段

    # 传递给下游变量（可被高级设置覆盖）
    max_sections = P["sections"]
    per_section_max_len = P["sec_max"]
    per_section_min_len = P["sec_min"]
    final_max_len = P["final_max"]
    final_min_len = P["final_min"]

    with st.expander("高级设置（可选）", expanded=False):
        max_sections   = st.number_input("Max sections to summarize", 5, 100, max_sections, 1)
        per_section_max_len = st.slider("Per-section max length", 80, 300, per_section_max_len, 10)
        per_section_min_len = st.slider("Per-section min length", 30, 200, per_section_min_len, 10)
        final_max_len  = st.slider("Final summary max length", 150, 500, final_max_len, 10)
        final_min_len  = st.slider("Final summary min length", 80, 300, final_min_len, 10)

    st.divider()
    if st.button("♻️ Reset app cache"):
        try: st.cache_data.clear()
        except Exception: pass
        try: st.cache_resource.clear()
        except Exception: pass
        st.success("Cache cleared. Rerun the app.")

    # 不再 import torch（避免触发报错）；只显示 Python 版本
    st.caption(f"Python: {sys.version.split()[0]}")

if code != REAL_CODE:
    st.warning("请输入有效的访问码继续使用。")
    st.stop()

# ----------------- 模型（懒加载） -----------------
@st.cache_resource(show_spinner=True)
def load_summarizer_and_tokenizer():
    tok = AutoTokenizer.from_pretrained("t5-small", use_fast=True)
    # 仅在这里延迟导入 torch，并限制线程，继续规避 JIT/并发问题
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    summarizer = pipeline(
        "summarization",
        model="t5-small",
        tokenizer=tok,
        framework="pt",   # 用 PyTorch，但已强制 CPU + 关闭 JIT
        device=-1,
    )
    return summarizer, tok

# ----------------- Token 级分块 -----------------
def chunk_by_tokens(tokenizer: AutoTokenizer, text: str, max_tokens: int = 480, overlap: int = 40) -> List[str]:
    if not text.strip():
        return []
    paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    buf_tokens = 0

    def tok_len(t: str) -> int:
        return len(tokenizer.encode(t, add_special_tokens=False))

    for p in paras:
        p_tokens = tok_len(p)
        if p_tokens > max_tokens:
            sentences = []
            tmp = []
            for seg in p.replace("。", "。|").replace("！", "！|").replace("？", "？|").split("|"):
                s = seg.strip()
                if s:
                    tmp.append(s)
                    if s[-1:] in "。！？.!?":
                        sentences.append("".join(tmp)); tmp = []
            if tmp: sentences.append("".join(tmp))
            for s in sentences:
                s_tokens = tok_len(s)
                if buf_tokens + s_tokens <= max_tokens:
                    buf.append(s); buf_tokens += s_tokens
                else:
                    if buf:
                        piece = " ".join(buf); chunks.append(piece)
                        tail = piece[-overlap * 2 :] if overlap > 0 else ""
                        buf = [tail] if tail else []; buf_tokens = tok_len(" ".join(buf)) if buf else 0
                    if s_tokens <= max_tokens:
                        buf.append(s); buf_tokens = tok_len(" ".join(buf))
                    else:
                        ids = tokenizer.encode(s, add_special_tokens=False)
                        for i in range(0, len(ids), max_tokens):
                            chunks.append(tokenizer.decode(ids[i : i + max_tokens]))
                        buf, buf_tokens = [], 0
        else:
            if buf_tokens + p_tokens <= max_tokens:
                buf.append(p); buf_tokens += p_tokens
            else:
                piece = " ".join(buf); chunks.append(piece)
                tail = piece[-overlap * 2 :] if overlap > 0 else ""
                buf = [tail, p] if tail else [p]
                buf_tokens = tok_len(" ".join(buf))
    if buf:
        chunks.append(" ".join(buf))
    return [c.strip() for c in chunks if c.strip()]

# ----------------- 文件上传 -----------------
def main():
    uploaded = st.file_uploader("📄 Upload a PDF file (book, report, or notes)", type="pdf")
    if not uploaded:
        return

    # ----------------- PDF 解析 -----------------
    st.info("✅ File uploaded successfully. Extracting text…")
    text_parts: List[str] = []
    try:
        raw = uploaded.read()
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            total_pages = len(pdf.pages)
            st.write(f"Total pages detected: **{total_pages}**")
            progress_pages = st.progress(0.0)
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    t = page.extract_text(x_tolerance=2, y_tolerance=2)
                except Exception:
                    t = ""
                if t:
                    text_parts.append(t)
                if i % 10 == 0 or i == total_pages:
                    progress_pages.progress(i / total_pages)
    except Exception as e:
        st.error(f"❌ Failed to parse PDF: {e}")
        return

    full_text = "\n".join(text_parts).strip()
    if not full_text:
        st.error("❌ No readable text found in PDF. It may be scanned images.")
        return

    # ----------------- 分块与摘要 -----------------
    summarizer, tokenizer = load_summarizer_and_tokenizer()
    token_chunks = chunk_by_tokens(tokenizer, full_text, max_tokens=480, overlap=40)
    st.write(f"🔍 Split into **{len(token_chunks)}** sections for summarization.")

    token_chunks = token_chunks[: int(max_sections)]
    progress = st.progress(0.0)
    chapter_summaries: List[str] = []

    for i, chunk in enumerate(token_chunks, start=1):
        inp = "summarize: " + chunk
        result = summarizer(
            inp,
            max_length=int(per_section_max_len),
            min_length=int(per_section_min_len),
            do_sample=False,
            clean_up_tokenization_spaces=True,
        )
        chapter_summary = result[0]["summary_text"].strip()
        chapter_summaries.append(f"### 📖 Chapter {i}\n{chapter_summary}")
        progress.progress(i / len(token_chunks))

    # ----------------- 输出章节摘要 -----------------
    st.success("✅ Chapter Summaries Generated!")
    for ch in chapter_summaries:
        st.markdown(ch)

    # ----------------- 全书综合摘要 -----------------
    st.divider()
    st.subheader("📙 Final Book Summary")
    combined = " ".join([s.replace("### 📖 Chapter", "Chapter") for s in chapter_summaries])
    final = summarizer(
        "summarize: " + combined[:12000],
        max_length=int(final_max_len),
        min_length=int(final_min_len),
        do_sample=False,
        clean_up_tokenization_spaces=True,
    )[0]["summary_text"].strip()
    st.write(final)

    st.caption("🚀 Powered by T5-small • Token-aware chunking • Optimized for long PDFs")

# —— 捕获顶层异常并在页面显示（避免白屏） —— #
try:
    main()
except Exception as ex:
    st.error("App crashed with an exception:")
    st.exception(ex)
