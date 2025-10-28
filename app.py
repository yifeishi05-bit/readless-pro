# requirements.txt

```
streamlit==1.37.1
pdfplumber==0.11.4
```

---

# app.py

```python
# 🟦 ReadLess Pro — 分层摘要稳定版（20页一级摘要 → 每20段再做二级摘要 → 最终摘要）
# 纯Python抽取式算法，无Torch/Transformers；适配 Python 3.13 / Streamlit Cloud；
# 专治大PDF：全量异常捕获 + 分层分块 + 字符长度上限 + 渐进式内存占用。

import io
import re
import time
import traceback
from collections import Counter
from typing import List

import streamlit as st
import pdfplumber

# ================= 页面与侧边栏 =================
st.set_page_config(page_title="📘 ReadLess Pro – Hierarchical PDF Summarizer", page_icon="📘", layout="wide")
st.title("📚 ReadLess Pro – 分层摘要（大PDF稳如老狗）")
st.caption("20页一段做一级摘要 → 每20段再汇总做二级摘要 → 再汇总成全书摘要。纯Python，无外部大模型依赖。")

with st.sidebar:
    st.header("⚙️ 摘要参数（可调，默认已很稳）")
    CHUNK_PAGES = st.number_input("一级分块：每段包含的页数", min_value=10, max_value=60, value=20, step=1,
                                  help="建议 15~30，越大越稳；20 与你提出的方案一致")
    GROUP_SUMMARIES = st.number_input("二级分块：每组包含的一级段数", min_value=5, max_value=60, value=20, step=1,
                                      help="20 表示把 20 段一级摘要再合并做一次摘要（约 400 页/组）")

    top_k_lvl1 = st.slider("一级摘要：每段保留关键句数", 2, 12, 6)
    top_k_lvl2 = st.slider("二级摘要：每组保留关键句数", 2, 12, 8)
    top_k_final = st.slider("最终摘要：全书保留关键句数", 4, 30, 14)

    show_debug = st.checkbox("显示调试信息（字符数/句子数/用时）", value=True)

    st.divider()
    st.caption("如果仍然崩：① 把‘每段页数’调大；② 关键句数调小；③ 关闭调试输出。")

# ================= 摘要工具（纯Python抽取式） =================
WHITES = re.compile(r"\s+")
CN_PUNCS = "。！？；："
EN_PUNCS = r"\.\!\?\;\:"
SPLIT_REGEX = re.compile(rf"(?<=[{CN_PUNCS}])|(?<=[{EN_PUNCS}])")
STOPWORDS = set("""
的 了 和 与 及 或 而 被 将 把 在 之 其 这 那 本 该 并 对 于 从 中 等 比 更 很 非 非常 我们 他们 你们 因此 所以 但是 然而 以及 通过 通过
the a an and or but so of in on at to for with from by this that these those is are was were be been being it they we you he she as if than then
""".split())

# 安全上限，防极端长页面/句子/拼接导致崩溃
MAX_CHARS_PER_PAGE = 15000
MAX_CHARS_PER_SENT = 1000
MAX_JOINED_LEN = 1_200_000


def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\x00", " ").replace("\u200b", " ").replace("\ufeff", " ")
    t = WHITES.sub(" ", t).strip()
    if len(t) > MAX_CHARS_PER_PAGE:
        t = t[:MAX_CHARS_PER_PAGE]
    return t


def split_sentences(text: str) -> List[str]:
    raw = [s.strip() for s in SPLIT_REGEX.split(text) if s and s.strip()]
    # 合并过短片段，限制超长句
    sents, buf = [], ""
    for s in raw:
        if len(s) < 8:
            buf += s
            continue
        if buf:
            s = (buf + s)[:MAX_CHARS_PER_SENT]
            buf = ""
        sents.append(s[:MAX_CHARS_PER_SENT])
    if buf:
        sents.append(buf[:MAX_CHARS_PER_SENT])
    # 若几乎无标点 → 硬切
    if len(sents) <= 1 and len(text) > 0:
        sents = [text[i:i + 300] for i in range(0, len(text), 300)]
    return sents


def tokenize_mixed(sent: str) -> List[str]:
    parts = []
    for w in WHITES.split(sent):
        w = w.strip().lower()
        if not w:
            continue
        if re.search(r"[a-z]", w):
            parts.append(w)
        else:
            for ch in w:
                if re.match(r"[\u4e00-\u9fff]", ch):
                    parts.append(ch)
    return [p for p in parts if p and p not in STOPWORDS and not p.isdigit()]


def score_and_pick(text: str, top_k: int) -> str:
    text = clean_text(text)
    if not text:
        return "(空段)"
    sents = split_sentences(text)
    if len(sents) <= top_k:
        return " ".join(sents)

    toks_per_sent = []
    freq = Counter()
    for s in sents:
        toks = tokenize_mixed(s)
        toks_per_sent.append(toks)
        freq.update(toks)
    if not freq:
        return " ".join(sents[:top_k])

    maxf = max(freq.values())
    weights = {w: v / maxf for w, v in freq.items()}

    scores = []
    n = len(sents)
    for i, toks in enumerate(toks_per_sent):
        if not toks:
            scores.append(0.0)
            continue
        base = sum(weights.get(t, 0.0) for t in toks) / len(toks)
        # 位置微调：中间略高 + 首段稍高，避免只取开头
        pos_boost = 1.0 + 0.15 * (1 - abs((i + 1) - (n / 2)) / (n / 2 + 1e-9))
        scores.append(base * pos_boost)

    idx = sorted(range(n), key=lambda i: (-scores[i], i))[:top_k]
    idx.sort()
    return " ".join(sents[i] for i in idx)


# ================= 核心流程（分层摘要） =================

def summarize_pages_to_level1(page_texts: List[str], pages_per_chunk: int, top_k: int):
    chunks = []
    buf, cnt = [], 0
    for i, t in enumerate(page_texts, start=1):
        if t:
            buf.append(t)
        cnt += 1
        if cnt % pages_per_chunk == 0 or i == len(page_texts):
            ct = "\n".join(buf).strip()
            if ct:
                chunks.append(ct)
            buf, cnt = [], 0
    summaries = []
    prog = st.progress(0.0)
    for idx, ch in enumerate(chunks, start=1):
        t0 = time.time()
        s = score_and_pick(ch, top_k=top_k)
        summaries.append(s)
        if show_debug:
            st.markdown(f"### 📖 一级摘要 第 {idx} 段")
            st.write(s)
            st.caption(f"chunk_chars={len(ch):,} | sum_chars={len(s):,} | time={time.time()-t0:.2f}s")
        else:
            with st.expander(f"📖 一级摘要 第 {idx} 段", expanded=False):
                st.write(s)
        prog.progress(idx / len(chunks))
    return summaries


def summarize_level1_to_level2(level1_summaries: List[str], group_size: int, top_k: int):
    groups = []
    for i in range(0, len(level1_summaries), group_size):
        joined = " ".join(level1_summaries[i:i + group_size])
        if len(joined) > MAX_JOINED_LEN:
            joined = joined[:MAX_JOINED_LEN]
        groups.append(joined)

    if not groups:
        return []

    st.divider()
    st.subheader("📘 二级摘要（按一级摘要每组 %d 段再次归纳）" % group_size)
    summaries = []
    prog = st.progress(0.0)
    for idx, g in enumerate(groups, start=1):
        t0 = time.time()
        s = score_and_pick(g, top_k=top_k)
        summaries.append(s)
        if show_debug:
            st.markdown(f"### 🧩 二级摘要 第 {idx} 组")
            st.write(s)
            st.caption(f"group_chars={len(g):,} | sum_chars={len(s):,} | time={time.time()-t0:.2f}s")
        else:
            with st.expander(f"🧩 二级摘要 第 {idx} 组", expanded=False):
                st.write(s)
        prog.progress(idx / len(groups))
    return summaries


def render_final_summary(level2_summaries: List[str], top_k: int) -> str:
    st.divider()
    st.subheader("📙 全书最终摘要")
    if not level2_summaries:
        st.info("只有一级摘要，直接对一级摘要进行最终提炼。")
        joined = " ".join(level2_summaries)
    joined = " ".join(level2_summaries) if level2_summaries else ""
    if len(joined) > MAX_JOINED_LEN:
        joined = joined[:MAX_JOINED_LEN]
    final = score_and_pick(joined, top_k=top_k) if joined else "(没有可供最终摘要的文本)"
    st.write(final)
    return final


# ================= 主程序（全量异常捕获） =================

def main():
    uploaded = st.file_uploader("📄 上传PDF（任意大小，文本版最佳）", type="pdf")
    if not uploaded:
        return

    st.info("✅ 文件已上传，开始逐页解析…")
    t0 = time.time()

    # 逐页提取文本（边读边清洗，限制每页长度）
    page_texts: List[str] = []
    try:
        raw = uploaded.read()
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            total_pages = len(pdf.pages)
            st.write(f"检测到总页数：**{total_pages}**")
            bar = st.progress(0.0)
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    t = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                except Exception as e:
                    t = ""
                    if show_debug:
                        st.write(f"⚠️ 第 {i} 页解析异常：{e}")
                page_texts.append(clean_text(t))
                if i % 10 == 0 or i == total_pages:
                    bar.progress(i / total_pages)
    except Exception as e:
        st.error("❌ 解析PDF失败（外层打开/读取阶段）")
        st.exception(e)
        return

    non_empty = sum(1 for t in page_texts if t)
    if non_empty == 0:
        st.error("❌ 没有读到可用文本：可能是扫描/图片型PDF，请先做OCR再上传。")
        return

    # 一级摘要（每 CHUNK_PAGES 页一段）
    st.divider()
    st.subheader("📖 一级摘要（每 %d 页一段）" % CHUNK_PAGES)
    lvl1 = summarize_pages_to_level1(page_texts, pages_per_chunk=CHUNK_PAGES, top_k=top_k_lvl1)

    # 二级摘要（每 GROUP_SUMMARIES 段一级摘要再做一次）
    lvl2 = summarize_level1_to_level2(lvl1, group_size=GROUP_SUMMARIES, top_k=top_k_lvl2)

    # 最终摘要
    final = render_final_summary(lvl2 if lvl2 else lvl1, top_k=top_k_final)

    # 下载按钮（包含一级/二级/最终摘要）
    st.divider()
    st.subheader("⬇️ 导出摘要")
    export_lines = ["# 最终摘要\n", final, "\n\n## 二级摘要\n"]
    export_lines += [f"- {s}" for s in (lvl2 if lvl2 else [])]
    export_lines.append("\n\n## 一级摘要\n")
    for i, s in enumerate(lvl1, start=1):
        export_lines.append(f"### 段 {i}\n{s}\n")
    export_text = "\n".join(export_lines)

    st.download_button(
        label="下载 Markdown 摘要",
        data=export_text.encode("utf-8"),
        file_name="readless_summary.md",
        mime="text/markdown",
    )

    if show_debug:
        st.caption(f"总用时：{time.time()-t0:.2f}s（纯CPU）")


# 顶层全量捕获，避免‘Oh no’只给红屏
try:
    main()
except Exception as ex:
    st.error("❌ 程序异常（已捕获），请将以下堆栈发我排查：")
    st.exception(ex)
    st.code(traceback.format_exc())
```
