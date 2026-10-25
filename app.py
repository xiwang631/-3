# -*- coding: utf-8 -*-
"""财报问答知识库 - Streamlit 页面 (第 5 步交付物)
启动: streamlit run app.py
检索: 向量(bge-m3) + BM25 双路 RRF 合并;  作答: DeepSeek, 答案带出处
"""
import io
import json
import os
import pickle
import sys

import numpy as np
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "scripts"))

st.set_page_config(page_title="医药年报问答知识库", page_icon="💊", layout="wide")

# ---------- 资源加载(缓存) ----------

@st.cache_resource
def load_all():
    import ask  # noqa: 复用 scripts/ask.py 的函数
    records = [json.loads(l) for l in io.open(os.path.join(DATA_DIR, "chunks.jsonl"), encoding="utf-8")]
    embs = np.load(os.path.join(DATA_DIR, "embeddings.npy"))
    with open(os.path.join(DATA_DIR, "bm25_index.pkl"), "rb") as f:
        obj = pickle.load(f)
    return ask, records, embs, obj


DATA_DIR = os.path.join(BASE, "data")
ask, records, embs, bm25obj = load_all()

# ---------- 页面 ----------
st.title("💊 医药上市公司年报问答知识库")
st.caption("10 家医药公司 2025 年报 · 10,408 知识块 · 向量(bge-m3)+BM25 双路检索 · DeepSeek 带出处作答")

with st.sidebar:
    st.header("📚 知识库概况")
    st.markdown("""
| 项目 | 规模 |
|---|---|
| 公司 | 恒瑞/复星/华东/云南白药/片仔癀/药明康德/凯莱英/迈瑞/联影/智飞 |
| 报告 | 2025 年年度报告(全文) |
| 正文 | 约 470 万字 |
| 表格 | 3,583 张(行列已还原) |
| 知识块 | 10,408 (800字/重合120) |
| 索引 | bge-m3 向量 + BM25 |
    """)
    st.divider()
    st.markdown("**试试这些问题**")
    examples = [
        "恒瑞医药和复星医药2025年谁的研发费用投入更大？",
        "迈瑞医疗2025年营业收入是多少？",
        "云南白药和片仔癀谁的毛利率更高？",
        "药明康德2025年研发费用率是多少？",
        "联影医疗2025年经营活动现金流量净额是多少？",
    ]
    clicked = None
    for e in examples:
        if st.button(e, use_container_width=True):
            clicked = e

question = st.session_state.get("q", "")
if clicked:
    st.session_state["q"] = clicked
    question = clicked

q = st.text_input("你的问题（答案会标注出自哪家公司哪份报告哪一页）",
                  value=question, key="input_box")
go = st.button("🔍 提问", type="primary")

if go or (clicked and q):
    with st.spinner("检索中…"):
        pre = ask.preprocess(q)
        hits = ask.retrieve(q, pre)

    st.markdown(f"**问题**：{q}")
    st.caption(f"检索预处理 → 语义: {pre['refined']} ｜ 关键词: {', '.join(pre['keywords'])}")

    with st.spinner("DeepSeek 作答中…"):
        ans = ask.answer(q, hits)

    st.markdown("### 💡 回答")
    st.markdown(ans)

    st.markdown("### 📎 命中的知识块（出处）")
    for k, h in enumerate(hits, 1):
        src = f"{h['company']} · 2025年报 · {h['section']} · 第{h['page']}页"
        vec_info = f"向量路第{h['vec_rank']}名" if h['vec_rank'] else "BM25路"
        with st.expander(f"[片段{k}] {src}  （{vec_info}，RRF={h['score']}）"):
            st.text(h["text"][:2000])
            st.caption(f"chunk_id: {h['chunk_id']}")
else:
    st.info("输入问题开始提问，或点击左侧示例问题。")
