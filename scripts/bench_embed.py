# -*- coding: utf-8 -*-
"""基准测试: dtype x batch x max_len, 找最快配置"""
import io
import json
import time

BASE = r"C:\Users\Lenovo\Downloads\financial-qa-kb"
MODEL = r"D:\workbuddy\models\Qwen3-Embedding-0.6B"
CHUNKS = BASE + r"\data\chunks.jsonl"

import torch
from sentence_transformers import SentenceTransformer

rows = [json.loads(l) for l in io.open(CHUNKS, encoding="utf-8")]
# 取有代表性的样本: 8 个短 + 8 个长
rows_sorted = sorted(rows, key=lambda r: len(r["text"]))
sample = [r["text"] for r in rows_sorted[:8] + rows_sorted[-8:]]

out = io.open(BASE + r"\data\_bench.txt", "w", encoding="utf-8")
for dtype in ["float32", "bfloat16"]:
    for bs, ml in [(8, 512), (16, 512), (8, 1024)]:
        try:
            m = SentenceTransformer(MODEL, device="cpu",
                                    model_kwargs={"dtype": dtype})
            m.max_seq_length = ml
            t0 = time.time()
            v = m.encode(sample, batch_size=bs, normalize_embeddings=True,
                         show_progress_bar=False)
            dt = time.time() - t0
            line = f"dtype={dtype} batch={bs} maxlen={ml}: {dt:.1f}s for 16 chunks -> {16/dt:.2f} ch/s, dim={v.shape[1]}"
        except Exception as e:
            line = f"dtype={dtype} batch={bs} maxlen={ml}: FAIL {str(e)[:120]}"
        print(line, flush=True)
        out.write(line + "\n")
        out.flush()
        del m
out.close()
