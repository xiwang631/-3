# -*- coding: utf-8 -*-
"""基准测试 2: int8 动态量化 vs fp32, 按长度排序后的真实混合负载"""
import io
import json
import time

import torch
from sentence_transformers import SentenceTransformer

BASE = r"C:\Users\Lenovo\Downloads\financial-qa-kb"
MODEL = r"D:\workbuddy\models\Qwen3-Embedding-0.6B"
CHUNKS = BASE + r"\data\chunks.jsonl"

rows = [json.loads(l) for l in io.open(CHUNKS, encoding="utf-8")]
# 真实负载: 随机 16 条(非极端长度)
sample = [rows[i]["text"] for i in range(0, len(rows), len(rows) // 16)][:16]

out = io.open(BASE + r"\data\_bench2.txt", "w", encoding="utf-8")


def bench(tag, m, bs, ml):
    m.max_seq_length = ml
    t0 = time.time()
    m.encode(sample, batch_size=bs, normalize_embeddings=True,
             show_progress_bar=False)
    dt = time.time() - t0
    line = f"{tag} batch={bs} maxlen={ml}: {dt:.1f}s for 16 -> {16/dt:.2f} ch/s"
    print(line, flush=True)
    out.write(line + "\n")
    out.flush()


# fp32 基准
m = SentenceTransformer(MODEL, device="cpu", model_kwargs={"dtype": "float32"})
bench("fp32", m, 8, 512)

# int8 动态量化
try:
    q = torch.ao.quantization.quantize_dynamic(
        m[0].auto_model, {torch.nn.Linear}, dtype=torch.qint8)
    m[0].auto_model = q
    bench("int8", m, 8, 512)
    bench("int8", m, 16, 512)
except Exception as e:
    line = f"int8 FAIL: {str(e)[:150]}"
    print(line, flush=True)
    out.write(line + "\n")
out.close()
