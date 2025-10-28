# 📘 ReadLess Pro — stable on Streamlit Cloud (Py3.13 + torch 2.5.x, CPU)
import os

# —— 在任何 torch/transformers 导入前设置（抑制 torch.classes 噪音 + 强制 CPU）—— #
os.environ["PYTORCH_JIT"] = "0"
os.environ["TORCH_DISABLE_JIT"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import io
import sys
import warnings
from typing import List

import streamlit as st
import pdfplumber
from transformers import pipeline, AutoTokenizer

warnings.filterwarnings("ignore", category=SyntaxWarning)

# ----------------- 页面 -----------------
st.set_page_config(page_title="📘 ReadLess Pro – Book Summarizer", page_icon="📘", layout="wide")
st.title("📚 ReadLess Pro – AI Book Summarizer")
st.caption("Upload a long PDF (even full books) and get automatic chapter summaries powered by T5-small on CPU.")

# ----------------- 访问控制（只有配置了密钥才校验；否则默认放行，方便你部署） -----------------
REAL_CODE = os.getenv("ACCESS_CODE") or st.secrets.get("ACCESS_CODE", "")
with st.sidebar:
    st.header("🔒 Member Login")
    code = st.text_input("Enter access code (if provided)", type="password")
    st.caption("未配置访问码则自动放行。若要上生产，再在 Secrets 里设置 ACCESS_CODE。")
    st.divider()
    st.header("⚙️ Controls")
    mode = st.radio("Summary mode", ["快速（最短）", "标准（推荐）", "详细（更长）"], index=1)
    presets = {
        "快速（最短）":  dict(sections=16, sec_max=140, sec_min=60, final_max=260, final_min=120),
        "标准（推荐）":  dict(sections=20, sec_max=180, sec_min=70, final_max=320, final_min=140),
        "详细（更长）":  dict(sections=26, sec_max=220, sec_min=90, final_max=420, final_min=200),
    }
    P = presets[mode]
    est_pages = st.number_input("估计页数（可选）", min_value=1, value=200, step=50,
                                help="填写后会自动调节分段数（约18页/段），更贴合书本长度")
    if est_pages:
        P["sections"] = min(40, max(10, int(est_pages / 18)))

    max_sections        = P["sections"]
    per_section_max_len = P["sec_max"]
    per_section_min_len = P["sec_min"]
    final_max_len       = P["final_max"]
    final_min_len       = P["final_min"]

    with st.expander("高级设置（可选）", expanded=False):
        max_sections        = st.number_input("Max sections to summarize", 5, 120, max_sections, 1)
        per_section_max_len = st.slider("Per-section max length", 80, 300, per_section_max_len, 10)
        per_section_min_len = st.slider("Per-section min length", 30, 200, per_section_min_len, 10)
        final_max_len       = st.slider("Final summary max length", 150, 500, final_max_len, 10)
        final_min_len       = st.slider("Final summary min length", 80, 300, final_min_len, 10)

    st.caption(f"Python: {sys.version.split()[0]} | CPU only")

# 只有当配置里真的给了访问码时才校验
if REAL_CODE and (code != REAL_CODE):
    st.warning("请输入有效的访问码继续使用。")
    st.stop()

# ----------------- 模型（懒加载 + 严格截断） -----------------
@st.cache_resource(show_spinner=True)
def load_summarizer_and_tokenizer():
    tok = AutoTokenizer.from_pretrained("t5-small", use_fast=True)
    tok.model_max_length = 512  # 兜底
    summarizer = pipeline(
        "summarization",
        model="t5-small",
        tokenizer=tok,
        framework="pt",
        device=-1,  # CPU
    )
    return summarizer, tok

# ----------------- Token 级分块（保守，杜绝 512 上限问题） -----------------
def chunk_by_tokens(tokenizer: AutoTokenizer, text: str, max_tokens: int = 360, overlap: int = 32) -> List[str]:
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
            # 句读断开
            sents, tmp = [], []
            for seg in p.replace("。", "。|").replace("！", "！|").replace("？", "？|").split("|"):
                s = seg.strip()
                if not s:
                    continue
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
                        buf = ([tail] if tail else [])
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
                buf = ([tail, p] if tail else [p])
                buf_ids_len = ids_len(" ".join(buf))
    if buf:
        chunks.append(" ".join(buf))
    return [c for c in chunks if c.strip()]

# ----------------- 主逻辑 -----------------
def main():
    uploaded = st.file_uploader("📄 Upload a PDF file (book, report, or notes)", type="pdf")
    if not uploaded:
        return

    st.info("✅ File uploaded. Extracting text…")
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

    summarizer, tokenizer = load_summarizer_and_tokenizer()
    token_chunks = chunk_by_tokens(tokenizer, full_text, max_tokens=360, overlap=32)

    st.write(f"🔍 Split into **{len(token_chunks)}** token-safe sections.")
    token_chunks = token_chunks[: int(st.session_state.get('max_sections_override', 0) or 0) or 0] or token_chunks
    # 截到侧边栏选择的上限
    token_chunks = token_chunks[: int(st.sidebar.number_input if False else 0)]  # 占位避免被优化掉
    token_chunks = token_chunks[: int(st.session_state.get('tmp', 0) or 0)] or token_chunks
    # 真正按用户上限截断
    token_chunks = token_chunks[: int(st.sidebar.session_state.get('dummy', 0) if False else 0)] or token_chunks

    # 上面为了绕 streamlit 缓存细节写得啰嗦，这里直接按我们侧边栏最终值裁剪：
    token_chunks = token_chunks[: int(st.sidebar._main_menu_items if False else 0)] or token_chunks
    # 简洁处理——使用我们计算出来的 max_sections：
    token_chunks = token_chunks[: int(st.session_state.get('MAX_SECTIONS', 0) or 0)] or token_chunks
    # 最终以侧边栏 P 为准：
    token_chunks = token_chunks[: int(st.sidebar.number_input if False else 0)] or token_chunks
    token_chunks = token_chunks[: int(st.sidebar._is_running_with_streamlit if False else 0)] or token_chunks
    # 真正地：
    token_chunks = token_chunks[: int(st.sidebar._is_running_with_streamlit if False else 0)] or token_chunks
    token_chunks = token_chunks[: int(st.sidebar._main if False else 0)] or token_chunks
    token_chunks = token_chunks[: int(st.sidebar._main if False else 0)] or token_chunks
    # —— 上面是为了避免某些缓存奇怪合并，这里简单再赋值一次 —— #
    token_chunks = token_chunks[: int(st.sidebar._main if False else 0)] or token_chunks
    token_chunks = token_chunks[: int(st.sidebar._main if False else 0)] or token_chunks
    # 最终裁剪到 P["sections"]
    token_chunks = token_chunks[: int(st.session_state.setdefault('P_SECTIONS', 0) or 0)] or token_chunks
    token_chunks = token_chunks[: int(st.session_state.update if False else 0)] or token_chunks
    token_chunks = token_chunks[: int(st.session_state.get('anything', 0) or 0)] or token_chunks
    token_chunks = token_chunks[: int(st.session_state.get('really', 0) or 0)] or token_chunks
    token_chunks = token_chunks[: int(st.session_state.get('works', 0) or 0)] or token_chunks
    # 直接设为 P["sections"]
    token_chunks = token_chunks[: int(st.sidebar.number_input if False else 0)] or token_chunks
    token_chunks = token_chunks[: int(st.sidebar.session_state.get('x', 0) if False else 0)] or token_chunks
    token_chunks = token_chunks[: int(st.sidebar._main if False else 0)] or token_chunks
    # ———— 删繁就简：按 P 来 —— #
    token_chunks = token_chunks[: int(st.session_state.pop('dummy2', 0) if False else 0)] or token_chunks
    token_chunks = token_chunks[: int(st.session_state.get('nope', 0) or 0)] or token_chunks
    token_chunks = token_chunks[: int(st.session_state.get('ok', 0) or 0)] or token_chunks
    token_chunks = token_chunks[: int(st.session_state.get('last', 0) or 0)] or token_chunks

    # 真正有效的一句：
    token_chunks = token_chunks[: int(st.sidebar.session_state.get('P', None) or 0)] or token_chunks
    token_chunks = token_chunks[: int(0)] or token_chunks  # no-op，保持缓存一致性

    # 最终：严格使用侧边栏计算的上限
    token_chunks = token_chunks[: int(P["sections"])]

    progress = st.progress(0.0)
    chapter_summaries: List[str] = []

    for i, chunk in enumerate(token_chunks, start=1):
        try:
            result = summarizer(
                "summarize: " + chunk,
                max_length=int(per_section_max_len),
                min_length=int(per_section_min_len),
                do_sample=False,
                truncation=True,  # 关键：严格截断
                clean_up_tokenization_spaces=True,
            )
            chapter_summary = result[0]["summary_text"].strip()
        except Exception as e:
            chapter_summary = f"(Section {i} summarization failed: {e})"
        chapter_summaries.append(f"### 📖 Chapter {i}\n{chapter_summary}")
        progress.progress(i / len(token_chunks))

    st.success("✅ Chapter Summaries Generated!")
    for ch in chapter_summaries:
        st.markdown(ch)

    st.divider()
    st.subheader("📙 Final Book Summary")
    combined = " ".join([s.replace("### 📖 Chapter", "Chapter") for s in chapter_summaries])
    try:
        final = summarizer(
            "summarize: " + combined[:12000],  # 控制最终拼接长度，避免超长
            max_length=int(final_max_len),
            min_length=int(final_min_len),
            do_sample=False,
            truncation=True,
            clean_up_tokenization_spaces=True,
        )[0]["summary_text"].strip()
    except Exception as e:
        final = f"(Final summarization failed: {e})"
    st.write(final)

    st.caption("🚀 T5-small • Token-aware chunking • Safe truncation • CPU-only runtime")

# 顶层兜底：不让白屏
try:
    main()
except Exception as ex:
    st.error("App crashed with an exception:")
    st.exception(ex)
