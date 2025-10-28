# 🟦 ReadLess Pro — 分层摘要稳定版（20页一级摘要 → 每20段再做二级摘要 → 最终摘要）


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