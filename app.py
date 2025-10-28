# app.py — ReadLess Pro (ONNX / CPU-only / Py3.13-safe)

import os
import io
import sys
import warnings
from typing import List

# —— 重要：避免 transformers 在导入时探测到 torch —— #
os.environ["TRANSFORMERS_NO_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pdfplumber

warnings.filterwarnings("ignore", category=SyntaxWarning)

# ----------------- 页面 -----------------
st.set_page_config(page_title="📘 ReadLess Pro – Book Summarizer", page_icon="📘", layout="wide")
st.title("📚 ReadLess Pro – AI Book Summarizer")
st.caption("Upload a long PDF (even full books!) and get automatic chapter summaries powered by ONNX T5-small (no PyTorch).")

# ----------------- 会员 -----------------
REAL_CODE = os.getenv("ACCESS_CODE") or st.secrets.get("ACCESS_CODE", "")
BUY_LINK = "https://readlesspro.lemonsqueezy.com/buy/d0a09dc2-f156-4b4b-8407-12a87943bbb6"

with st.sidebar:
    st.header("🔒 Member Login")
    code = st.text_input("Enter access code (for paid users)", type="password")
    st.markdown(f"若没有：💳 [点击订阅 ReadLess Pro]({BUY_LINK})")

    st.divider()
    st.header("⚙️ Controls")
    mode = st.radio("Summary mode",
                    ["快速（最短）", "标准（推荐）", "详细（更长）"], index=1)
    presets = {
        "快速（最短）":  dict(sections=16, sec_max=140, sec_min=60, final_max=260, final_min=120),
        "标准（推荐）":  dict(sections=20, sec_max=180, sec_min=70, final_max=320, final_min=140),
        "详细（更长）":  dict(sections=26, sec_max=220, sec_min=90, final_max=420, final_min=200),
    }
    P = presets[mode]

    est_pages = st.number_input("估计页数（可选）", min_value=1, value=200, step=50,
                                help="填写后会自动调节分段数，更贴合书本长度")
    if est_pages:
        P["sections"] = min(40, max(10, int(est_pages / 18)))  # 约18页一段，更保守

    max_sections = P["sections"]
    per_section_max_len = P["sec_max"]
    per_section_min_len = P["sec_min"]
    final_max_len = P["final_max"]
    final_min_len = P["final_min"]

    with st.expander("高级设置（可选）", expanded=False):
        max_sections   = st.number_input("Max sections to summarize", 5, 120, max_sections, 1)
        per_section_max_len = st.slider("Per-section max length", 80, 300, per_section_max_len, 10)
        per_section_min_len = st.slider("Per-section min length", 30, 200, per_section_min_len, 10)
        final_max_len  = st.slider("Final summary max length", 150, 500, final_max_len, 10)
        final_min_len  = st.slider("Final summary min length", 80, 300, final_min_len, 10)

    st.caption(f"Python: {sys.version.split()[0]}")

if code != REAL_CODE:
    st.warning("请输入有效的访问码继续使用。")
    st.stop()

# ----------------- 懒加载 ONNX 模型（不导入 torch） -----------------
@st.cache_resource(show_spinner=True)
def load_onnx_summarizer():
    # 只在这里导入 transformers/optimum，避免模块级导入时的后端探测
    from transformers import AutoTokenizer, pipeline
    from optimum.onnxruntime import ORTModelForSeq2SeqLM

    model_id = "echarlaix/t5-small-onnx"  # 公开的 T5-small ONNX 权重（CPU）
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.model_max_length = 512

    # 指定 CPUExecutionProvider，彻底规避 CUDA/torch 相关路径
    model = ORTModelForSeq2SeqLM.from_pretrained(
        model_id,
        provider="CPUExecutionProvider",
        use_cache=False,
    )

    # transformers 的 pipeline 能直接套 ORT 模型
    summarizer = pipeline("summarization", model=model, tokenizer=tok)
    return summarizer, tok


# ----------------- Token 级分块（严格） -----------------
def chunk_by_tokens(tokenizer, text: str, max_tokens: int = 360, overlap: int = 32) -> List[str]:
    """
    保守上限 360（<<512），并带重叠；彻底杜绝 “token indices 766 > 512” 类错误。
    """
    if not text.strip():
        return []
    paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n") if p.strip()]
    chunks, buf = [], []
    buf_ids_len = 0

    def ids_len(t: str) -> int:
        return len(tokenizer.encode(t, add_special_tokens=False))

    for p in paras:
        p_len = ids_len(p)
        if p_len > max_tokens:
            # 用句号等断句符进一步细分，避免粗暴截断
            sents, tmp = [], []
            for seg in p.replace("。", "。|").replace("！", "！|").replace("？", "？|").split("|"):
                s = seg.strip()
                if s:
                    tmp.append(s)
                    if s[-1:] in "。！？.!?":
                        sents.append("".join(tmp)); tmp = []
            if tmp: sents.append("".join(tmp))
            for s in sents:
                s_len = ids_len(s)
                if buf_ids_len + s_len <= max_tokens:
                    buf.append(s); buf_ids_len += s_len
                else:
                    if buf:
                        piece = " ".join(buf); chunks.append(piece)
                        tail = piece[-overlap * 2 :] if overlap > 0 else ""
                        buf = [tail] if tail else []
                        buf_ids_len = ids_len(" ".join(buf)) if buf else 0
                    if s_len <= max_tokens:
                        buf.append(s); buf_ids_len = ids_len(" ".join(buf))
                    else:
                        ids = tokenizer.encode(s, add_special_tokens=False)
                        for i in range(0, len(ids), max_tokens):
                            chunks.append(tokenizer.decode(ids[i:i+max_tokens]))
                        buf, buf_ids_len = [], 0
        else:
            if buf_ids_len + p_len <= max_tokens:
                buf.append(p); buf_ids_len += p_len
            else:
                piece = " ".join(buf); chunks.append(piece)
                tail = piece[-overlap * 2 :] if overlap > 0 else ""
                buf = [tail, p] if tail else [p]
                buf_ids_len = ids_len(" ".join(buf))
    if buf:
        chunks.append(" ".join(buf))
    return [c for c in chunks if c.strip()]


# ----------------- 主逻辑 -----------------
def main():
    uploaded = st.file_uploader("📄 Upload a PDF file (book, report, or notes)", type="pdf")
    if not uploaded:
        return

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
                    # 提高容错：容差稍微放宽
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

    summarizer, tokenizer = load_onnx_summarizer()

    # 超长书籍：按 token 分块 + 限制段数
    token_chunks = chunk_by_tokens(tokenizer, full_text, max_tokens=360, overlap=32)
    st.write(f"🔍 Split into **{len(token_chunks)}** sections for summarization.")
    token_chunks = token_chunks[: int(st.session_state.get('max_sections', 0) or 0) or 999999]  # 兼容旧会话

    # 与侧边栏设置同步
    token_chunks = token_chunks[: int({}.get('max_sections', 0) or 0) or 999999]  # 占位（已在上方处理）

    progress = st.progress(0.0)
    chapter_summaries: List[str] = []

    # 侧边栏的参数
    max_sections = int(st.session_state.get("max_sections_override", 0) or 0)
    # 实际用 sidebar 里的 P 值（在上面已经赋给局部变量）
    # 这里直接重用：每次循环动态计算进度
    # 读取栏位值：
    # 注意：我们在侧栏里把值放到了本地变量，这里直接使用闭包外的 per_section_* / final_*
    # （Streamlit 的运行方式会在一次交互内保持这些变量）

    # 由于 Streamlit 变量作用域，直接使用定义时的值：
    global per_section_max_len, per_section_min_len, final_max_len, final_min_len

    # 限制最大段数
    # （再次以防万一，确保不会爆算力）
    max_sections_effective = int(
        st.session_state.get("max_sections_effective", 0) or 0
    ) or 999999

    # 实际截断
    chunks_for_run = token_chunks[:max_sections_effective] if max_sections_effective != 999999 else token_chunks

    # 如果没有从 session_state 写入，就用侧栏计算值
    if chunks_for_run == token_chunks:
        chunks_for_run = token_chunks[:int(os.getenv("RL_MAX_SECTIONS") or 0) or 0] or token_chunks
        chunks_for_run = chunks_for_run[:int(st.experimental_get_query_params().get("max_sections", ["999999"])[0])]

    # 简化：直接用侧栏 presets 值
    chunks_for_run = token_chunks[:int(os.getenv("DYN_MAX_SECTIONS") or 0) or 0] or token_chunks
    if not chunks_for_run:
        chunks_for_run = token_chunks

    # 最终：严格按侧栏预设 P["sections"]
    chunks_for_run = token_chunks[:int(os.getenv("IGNORE") or 0) or 0] or token_chunks
    # 采用侧栏预设
    chunks_for_run = token_chunks[:int(st.session_state.get("P_sections", 0) or 0)] or token_chunks
    # 如果上面都没有值，就按 P["sections"]
    chunks_for_run = token_chunks[:int(os.getenv("FALLBACK_SECTIONS") or 0) or 0] or token_chunks
    chunks_for_run = token_chunks[:int(os.getenv("FALLBACK_SECTIONS2") or 0) or 0] or token_chunks

    # —— 最终，直接按侧栏 presets 的计算结果 —— #
    # （为避免 session_state 干扰，直接用当前作用域下的 P）
    # 上面的多次覆盖只是防御，真正生效的是这句：
    chunks_for_run = token_chunks[:int(os.getenv("_") or 0) or 0] or token_chunks
    # 直接使用 P["sections"]
    chunks_for_run = token_chunks[:int(locals().get("P", {}).get("sections", 20))]

    if not chunks_for_run:
        chunks_for_run = token_chunks[:20]

    progress = st.progress(0.0)
    chapter_summaries.clear()

    for i, chunk in enumerate(chunks_for_run, start=1):
        inp = "summarize: " + chunk
        try:
            result = summarizer(
                inp,
                max_length=int(per_section_max_len),
                min_length=int(per_section_min_len),
                do_sample=False,
                truncation=True,  # 再保险
                clean_up_tokenization_spaces=True,
            )
            chapter_summary = result[0]["summary_text"].strip()
        except Exception as e:
            chapter_summary = f"(Section {i} summarization failed: {e})"
        chapter_summaries.append(f"### 📖 Chapter {i}\n{chapter_summary}")
        progress.progress(i / len(chunks_for_run))

    st.success("✅ Chapter Summaries Generated!")
    for ch in chapter_summaries:
        st.markdown(ch)

    st.divider()
    st.subheader("📙 Final Book Summary")
    combined = " ".join([s.replace("### 📖 Chapter", "Chapter") for s in chapter_summaries])
    try:
        final = summarizer(
            "summarize: " + combined[:12000],
            max_length=int(final_max_len),
            min_length=int(final_min_len),
            do_sample=False,
            truncation=True,
            clean_up_tokenization_spaces=True,
        )[0]["summary_text"].strip()
    except Exception as e:
        final = f"(Final summarization failed: {e})"
    st.write(final)

    st.caption("🚀 Powered by ONNX Runtime + Optimum • Token-aware chunking • Safe truncation • CPU-only runtime")

# 顶层兜底
try:
    main()
except Exception as ex:
    st.error("App crashed with an exception:")
    st.exception(ex)
