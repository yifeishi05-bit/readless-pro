# ReadLess Pro — 分层摘要（20页一段 × 二级汇总）【稳定可跑版】
# 纯 Python / 不用 transformers / 不用 torch，适合大 PDF

import io
import re
import time
import textwrap
from collections import Counter
from typing import List, Tuple

import streamlit as st
import pdfplumber

# ---------------- UI ----------------
st.set_page_config(page_title="📘 ReadLess Pro — 分层摘要（稳定可跑版）", page_icon="📘", layout="wide")
st.title("📚 ReadLess Pro — 分层摘要（稳定可跑版）")
st.caption("上传任意大的**文本型**PDF：每20页摘要→二级汇总→最终总览（参数可调）。")

with st.sidebar:
    st.header("⚙️ 参数")
    pages_per_chunk = st.number_input("每段包含的页数（一级）", 5, 50, 20, 1)
    sents_per_chunk = st.number_input("每段保留句数（一级）", 3, 20, 6, 1)
    chunks_per_super = st.number_input("多少段合并为一组（二级）", 5, 50, 20, 1)
    sents_per_super = st.number_input("每组保留句数（二级）", 3, 30, 8, 1)
    sents_final = st.number_input("最终总览句数", 5, 40, 14, 1)

    st.divider()
    hard_caps = st.toggle("开启长度限幅（更稳）", value=True)
    debug = st.toggle("显示调试信息", value=False)

    st.caption("提示：扫描/图片型PDF无法直接提取文字，本工具会提示先做OCR。")

uploaded = st.file_uploader("📄 上传 PDF（书籍/报告/讲义，非扫描）", type=["pdf"])
if not uploaded:
    st.stop()

# ---------------- 工具函数 ----------------
_CJK_RANGE = (
    ("\u4e00", "\u9fff"),  # CJK Unified Ideographs
    ("\u3400", "\u4dbf"),  # CJK Extension A
)

def is_cjk(ch: str) -> bool:
    if len(ch) != 1:
        return False
    o = ord(ch)
    for a, b in _CJK_RANGE:
        if ord(a) <= o <= ord(b):
            return True
    return False

def split_sentences(text: str) -> List[str]:
    # 中英文通用句子切分
    text = re.sub(r"[ \t]+", " ", text)
    # 先处理中文标点，再处理英文
    parts = re.split(r"(?<=[。！？…])\s*|\s*(?<=[!?])\s+", text)
    sents = [s.strip() for s in parts if s and s.strip()]
    return sents

_EN_STOP = set("""
a an the of to in for on with at from into during including until against among throughout despite toward upon
I you he she it we they me him her them my your his their our ours yours mine
and or but if while though although as than so because since unless until whereas whether nor
is am are was were be being been do does did done doing have has had having can could may might must shall should will would
""".split())
# 常见中文虚词/停用词（简版）
_ZH_STOP = set(list("的了呢吧啊嘛哦呀呀着过也很都就并而及与把被对于不是没有还是"))

def tokenize(text: str) -> List[str]:
    # 对中文：按单字（去停用）；对英文：\w+ 小写（去停用）
    if any(is_cjk(c) for c in text):
        toks = [c for c in text if is_cjk(c) and c not in _ZH_STOP]
    else:
        toks = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", text)]
        toks = [w for w in toks if w not in _EN_STOP and len(w) > 1]
    return toks

def summarize_extractive(text: str, keep: int, cap_chars: int = 40000) -> str:
    """
    纯抽取式摘要（频次 + 位置微调）
    - cap_chars: 截断上限防爆内存/超长
    """
    if not text.strip():
        return ""

    if hard_caps and len(text) > cap_chars:
        text = text[:cap_chars]

    sents = split_sentences(text)
    if not sents:
        return ""

    # 统计词频
    freq = Counter()
    for s in sents:
        for t in tokenize(s):
            freq[t] += 1
    if not freq:
        # 没法统计就取开头若干句
        return " ".join(sents[:keep])

    maxf = max(freq.values())
    for k in list(freq.keys()):
        freq[k] = freq[k] / maxf

    # 句子打分：词频和 + 位置奖励（靠前略高）
    scored: List[Tuple[int, float, str]] = []
    n = len(sents)
    for i, s in enumerate(sents):
        tokens = tokenize(s)
        base = sum(freq.get(t, 0) for t in tokens)
        # 位置奖励：前 20% 稍微加分
        pos_bonus = 0.15 if i < max(1, int(0.2 * n)) else 0.0
        length_norm = (len(tokens) ** 0.5) or 1.0
        score = (base / length_norm) + pos_bonus
        scored.append((i, score, s))

    # 选 topK，但保持原顺序
    scored.sort(key=lambda x: x[1], reverse=True)
    top = sorted(scored[:max(1, keep)], key=lambda x: x[0])
    return " ".join(s for (_, _, s) in top)

def chunk_pages_text(pages: List[str], group: int) -> List[str]:
    chunks = []
    for i in range(0, len(pages), group):
        part = "\n".join(pages[i:i+group])
        chunks.append(part)
    return chunks

def safe_extract_text(file_bytes: bytes) -> Tuple[List[str], List[int]]:
    pages_text, empty_pages = [], []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        total = len(pdf.pages)
        pbar = st.progress(0.0, text="解析PDF页面中…")
        for i, page in enumerate(pdf.pages, start=1):
            try:
                t = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            except Exception:
                t = ""
            if not t.strip():
                empty_pages.append(i)
            pages_text.append(t)
            if i % 5 == 0 or i == total:
                pbar.progress(i / total, text=f"解析PDF页面中…（{i}/{total}）")
    return pages_text, empty_pages

# ---------------- 主流程 ----------------
try:
    raw = uploaded.read()
except Exception as e:
    st.error(f"读取文件失败：{e}")
    st.stop()

t0 = time.time()
pages_text, empty_pages = safe_extract_text(raw)
total_pages = len(pages_text)
st.success(f"✅ 已解析页数：{total_pages} 页")
if empty_pages:
    st.warning(f"有 {len(empty_pages)} 页几乎没有可读文字（可能是扫描/图片页）。示例：{empty_pages[:10]}…")

# 一级：每 N 页一段 → 抽取若干句
level1_chunks = chunk_pages_text(pages_text, pages_per_chunk)
st.write(f"🔹 一级分段数：**{len(level1_chunks)}**（每段约 {pages_per_chunk} 页）")

l1_summaries: List[str] = []
pb1 = st.progress(0.0, text="一级摘要生成中…")
for i, chunk in enumerate(level1_chunks, start=1):
    try:
        summ = summarize_extractive(chunk, keep=sents_per_chunk, cap_chars=60000)
    except Exception as e:
        summ = f"(第 {i} 段摘要失败：{e})"
    l1_summaries.append(summ)
    if i % 2 == 0 or i == len(level1_chunks):
        pb1.progress(i / len(level1_chunks), text=f"一级摘要生成中…（{i}/{len(level1_chunks)}）")

st.subheader("📖 一级摘要（按段）")
for idx, s in enumerate(l1_summaries, start=1):
    st.markdown(f"**段 {idx}**：{s}")

# 二级：每 M 段合并 → 再抽取
l2_inputs = chunk_pages_text(l1_summaries, chunks_per_super)
st.write(f"🔹 二级汇总组数：**{len(l2_inputs)}**（每组 {chunks_per_super} 段）")

l2_summaries: List[str] = []
pb2 = st.progress(0.0, text="二级汇总生成中…")
for i, group_text in enumerate(l2_inputs, start=1):
    try:
        summ = summarize_extractive(group_text, keep=sents_per_super, cap_chars=40000)
    except Exception as e:
        summ = f"(第 {i} 组汇总失败：{e})"
    l2_summaries.append(summ)
    if i % 1 == 0 or i == len(l2_inputs):
        pb2.progress(i / len(l2_inputs), text=f"二级汇总生成中…（{i}/{len(l2_inputs)}）")

st.subheader("📚 二级汇总（按组）")
for idx, s in enumerate(l2_summaries, start=1):
    st.markdown(f"**组 {idx}**：{s}")

# 最终总览
st.subheader("🧭 最终总览")
final_input = "\n".join(l2_summaries) if l2_summaries else "\n".join(l1_summaries)
final_summary = summarize_extractive(final_input, keep=sents_final, cap_chars=50000)
st.write(final_summary)

# 导出
st.divider()
export_txt = []
export_txt.append("# 最终总览\n" + textwrap.fill(final_summary, width=100))
export_txt.append("\n# 二级汇总\n" + "\n\n".join(f"【组{i+1}】{s}" for i, s in enumerate(l2_summaries)))
export_txt.append("\n# 一级摘要\n" + "\n\n".join(f"【段{i+1}】{s}" for i, s in enumerate(l1_summaries)))
txt_bytes = "\n\n".join(export_txt).encode("utf-8", errors="ignore")
st.download_button("📥 下载摘要（TXT）", data=txt_bytes, file_name="readless_summary.txt", mime="text/plain")

# 调试信息
if debug:
    st.divider()
    st.caption(f"⏱️ 用时：{time.time()-t0:.2f}s | 页数：{total_pages} | 一级段数：{len(level1_chunks)} | 二级组数：{len(l2_inputs)}")
    st.caption(f"参数：pages_per_chunk={pages_per_chunk}, sents_per_chunk={sents_per_chunk}, chunks_per_super={chunks_per_super}, sents_per_super={sents_per_super}, sents_final={sents_final}")
