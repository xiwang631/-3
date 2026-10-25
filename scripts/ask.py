# -*- coding: utf-8 -*-
"""第 3+4+5 步: 混合检索 + DeepSeek 带出处问答
用法: python ask.py "你的问题"
流程(按课件):
  1) DeepSeek 问题侧预处理 -> 精炼问题(走向量) + 关键词(走BM25)
  2) 双路检索各取 Top20 -> RRF 合并 -> Top5
  3) DeepSeek 作答, 答案须标注 公司/报告/页码, 不得编造
"""
import io
import json
import os
import pickle
import sys
import urllib.request

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

# ---- 加载 key ----
KEYS = {}
for line in io.open(os.path.join(BASE, ".env"), encoding="utf-8"):
    if "=" in line:
        k, v = line.strip().split("=", 1)
        KEYS[k] = v


def _post(url, key, payload, timeout=120):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


# ---- 1) 问题侧预处理 (DeepSeek) ----
def preprocess(question):
    sys_p = ("你是检索查询预处理器。对用户关于上市公司年报的问题, 输出JSON: "
             '{"refined": "适合语义检索的完整精炼问题", "keywords": ["BM25关键词1", "关键词2", ...]}'
             "。只输出JSON。")
    r = _post("https://api.deepseek.com/chat/completions", KEYS["DEEPSEEK_API_KEY"], {
        "model": "deepseek-flash",
        "messages": [{"role": "system", "content": sys_p},
                     {"role": "user", "content": question}],
        "max_tokens": 2000, "temperature": 0})
    txt = r["choices"][0]["message"]["content"].strip()
    try:
        txt2 = txt[txt.find("{"): txt.rfind("}") + 1]
        return json.loads(txt2)
    except Exception:
        # 兜底: 预处理失败时直接用原问题, 不中断链路
        return {"refined": question, "keywords": [question]}


# ---- 2) 双路检索 ----
def embed_query(text):
    r = _post("https://api.siliconflow.cn/v1/embeddings", KEYS["SILICONFLOW_API_KEY"],
              {"model": "BAAI/bge-m3", "input": [text]})
    v = np.array(r["data"][0]["embedding"], dtype=np.float32)
    return v / (np.linalg.norm(v) or 1.0)


def retrieve(question, pre, topk=5):
    records = [json.loads(l) for l in io.open(os.path.join(DATA, "chunks.jsonl"), encoding="utf-8")]
    embs = np.load(os.path.join(DATA, "embeddings.npy"))

    # A 路: 向量
    qv = embed_query(pre["refined"])
    vec_scores = embs @ qv
    vec_top = set(np.argsort(-vec_scores)[:20].tolist())

    # B 路: BM25
    import jieba
    with open(os.path.join(DATA, "bm25_index.pkl"), "rb") as f:
        obj = pickle.load(f)
    bm25, ids = obj["bm25"], obj["ids"]
    id2idx = {cid: i for i, cid in enumerate(ids)}
    toks = []
    for t in pre["keywords"]:
        toks.extend([w for w in jieba.cut_for_search(t) if w.strip()])
    bm_scores = bm25.get_scores(toks)
    bm_top = set(np.argsort(-bm_scores)[:20].tolist())

    # RRF 合并
    rrf = {}
    for rank, i in enumerate(vec_top):
        rrf[i] = rrf.get(i, 0) + 1.0 / (60 + rank + 1)
    for rank, i in enumerate(bm_top):
        j = id2idx.get(ids[i], i)
        rrf[j] = rrf.get(j, 0) + 1.0 / (60 + rank + 1)
    merged = sorted(rrf.items(), key=lambda x: -x[1])[:topk]

    hits = []
    for i, score in merged:
        r = records[i]
        hits.append({"company": r["company"], "section": r["section"],
                     "page": r["page"], "chunk_id": r["chunk_id"],
                     "score": round(float(score), 4),
                     "vec_rank": (sorted(vec_top).index(i) + 1) if i in vec_top else None,
                     "text": r["text"]})
    return hits


# ---- 3) DeepSeek 作答 ----
def answer(question, hits):
    ctx = ""
    for k, h in enumerate(hits, 1):
        ctx += (f"[片段{k}] 出处: {h['company']} 2025年报 {h['section']} "
                f"第{h['page']}页 (chunk {h['chunk_id']})\n{h['text'][:1500]}\n\n")
    sys_p = ("你是上市公司财报问答助手。只根据给定片段回答, 每个关键数字/结论后用[片段k]标注出处。"
             "若片段不足以回答, 明确说\"提供的片段中没有足够信息\", 不得编造。")
    r = _post("https://api.deepseek.com/chat/completions", KEYS["DEEPSEEK_API_KEY"], {
        "model": "deepseek-flash",
        "messages": [{"role": "system", "content": sys_p},
                     {"role": "user", "content": f"问题: {question}\n\n片段:\n{ctx}"}],
        "max_tokens": 8000, "temperature": 0.2})
    return r["choices"][0]["message"]["content"]


def main():
    question = " ".join(sys.argv[1:])
    if not question:
        print("usage: python ask.py \"问题\"")
        return
    pre = preprocess(question)
    print(f"[预处理] refined={pre['refined']}\n[预处理] keywords={pre['keywords']}\n")
    hits = retrieve(question, pre)
    print("[检索命中]")
    for k, h in enumerate(hits, 1):
        print(f"  {k}. {h['company']} {h['section']} p{h['page']} "
              f"(vec#{h['vec_rank']}, rrf={h['score']})")
    print()
    print("=" * 60)
    print(answer(question, hits))


if __name__ == "__main__":
    main()
