# -*- coding: utf-8 -*-
"""第 3 步 A 路(最终版): 硅基流动 BAAI/bge-m3 API 向量化
- 全量 10,408 块 (本地 Qwen3 已算的作废, 两模型向量不可混用)
- bge-m3 支持长文本(8192 token), 覆盖所有切块
- 批量 32 / 断点续算 / 429 限速自动退避
- 归一化后余弦检索等价点积
"""
import io
import json
import os
import time
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS = os.path.join(BASE, "data", "chunks.jsonl")
OUT = os.path.join(BASE, "data", "embeddings.npy")
PROGRESS = os.path.join(BASE, "data", "_embed_api_progress.txt")

API = "https://api.siliconflow.cn/v1/embeddings"
MODEL = "BAAI/bge-m3"
BATCH = 32
# key 从 .env 读取, 不硬编码
KEY = ""
for line in io.open(os.path.join(BASE, ".env"), encoding="utf-8"):
    if line.startswith("SILICONFLOW_API_KEY="):
        KEY = line.strip().split("=", 1)[1]
assert KEY, "SILICONFLOW_API_KEY missing in .env"

HEADERS = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}


def call_api(texts, retry=6):
    body = json.dumps({"model": MODEL, "input": texts}).encode()
    for attempt in range(retry):
        try:
            req = urllib.request.Request(API, data=body, headers=HEADERS)
            r = json.loads(urllib.request.urlopen(req, timeout=120).read())
            return [d["embedding"] for d in r["data"]]
        except Exception as e:
            wait = min(2 ** attempt * 2, 60)
            with io.open(PROGRESS, "a", encoding="utf-8") as f:
                f.write(f"retry {attempt+1}: {str(e)[:80]} wait {wait}s\n")
            time.sleep(wait)
    raise RuntimeError("API failed after retries")


def main():
    records = [json.loads(l) for l in io.open(CHUNKS, encoding="utf-8")]
    n = len(records)
    print(f"total chunks: {n}", flush=True)

    # 断点: 已算好的部分保留在 npy 里
    dim = 1024
    embs = None
    done = 0
    if os.path.exists(OUT):
        old = __import__("numpy").load(OUT)
        if old.shape[0] <= n and old.shape[1] == dim and os.path.exists(PROGRESS):
            # 只有上次也是 bge-m3 才复用(进度文件里记了模型名)
            head = io.open(PROGRESS, encoding="utf-8").read(200)
            if "bge-m3" in head:
                embs = old
                done = old.shape[0]
    if embs is None:
        import numpy as np
        embs = np.zeros((n, dim), dtype=np.float32)
        with io.open(PROGRESS, "w", encoding="utf-8") as f:
            f.write("model=BAAI/bge-m3 start\n")

    print(f"resuming from {done}", flush=True)
    t0 = time.time()
    while done < n:
        batch = [r["text"] for r in records[done:done + BATCH]]
        vs = call_api(batch)
        for j, v in enumerate(vs):
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            embs[done + j] = [x / norm for x in v]
        done += len(vs)
        with io.open(PROGRESS, "w", encoding="utf-8") as f:
            f.write(f"model=BAAI/bge-m3 done={done}/{n}\n")
        if done % 1024 < BATCH:
            import numpy as np
            np.save(OUT, embs[:done])
            print(f"{done}/{n}  {done/(time.time()-t0):.1f}/s", flush=True)

    import numpy as np
    np.save(OUT, embs)
    print(f"ALL DONE: {n} vectors -> {OUT}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
