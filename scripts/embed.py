# -*- coding: utf-8 -*-
"""
第 3 步(向量路) · 用本地 Qwen3-Embedding-0.6B 给 chunks.jsonl 全部块算向量
- 模型路径: D:/workbuddy/models/Qwen3-Embedding-0.6B (ModelScope 下载)
- 断点续算: 已有 embeddings.npy 与进度文件则跳过已完成块
输出:
  data/embeddings.npy   (N, 1024) float32 矩阵, 行序与 chunks.jsonl 一致
  data/_embed_progress.txt  已完成条数
用法: python embed.py
"""
import io
import json
import os
import time

BASE = r"C:\Users\Lenovo\Downloads\financial-qa-kb"
MODEL = r"D:\workbuddy\models\Qwen3-Embedding-0.6B"
CHUNKS = os.path.join(BASE, "data", "chunks.jsonl")
EMB = os.path.join(BASE, "data", "embeddings.npy")
PROG = os.path.join(BASE, "data", "_embed_progress.txt")

BATCH = 8           # 基准测试最优: fp32/batch8 (见 data/_bench.txt)
MAX_LEN = 512       # 平均块长 744 字, 512 token 覆盖绝大多数块, 截断尾部

def main():
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(max(1, os.cpu_count() - 2))
    rows = [json.loads(l) for l in io.open(CHUNKS, encoding="utf-8")]
    texts = [r["text"] for r in rows]

    done = 0
    if os.path.exists(EMB) and os.path.exists(PROG):
        done = int(io.open(PROG, encoding="utf-8").read().strip())
        emb = np.load(EMB)
    else:
        emb = None

    # CPU 上 bfloat16 推理极慢, 转 float32
    model = SentenceTransformer(MODEL, device="cpu",
                                model_kwargs={"dtype": "float32"})
    model.max_seq_length = MAX_LEN  # 截断超长块
    t0 = time.time()
    log = io.open(os.path.join(BASE, "data", "_embed_log.txt"), "a", encoding="utf-8")
    while done < len(texts):
        batch = texts[done:done + BATCH]
        v = model.encode(batch, normalize_embeddings=True,
                         show_progress_bar=False)
        if emb is None:
            emb = np.zeros((len(texts), v.shape[1]), dtype="float32")
        emb[done:done + len(batch)] = v
        done += len(batch)
        if done % (BATCH * 20) == 0 or done >= len(texts):
            np.save(EMB, emb)
            io.open(PROG, "w", encoding="utf-8").write(str(done))
            speed = done / (time.time() - t0)
            eta = (len(texts) - done) / speed / 60 if speed else 0
            msg = f"{done}/{len(texts)} speed={speed:.1f}ch/s eta={eta:.0f}min"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
    log.write("ALL DONE\n")
    log.close()
    print("ALL DONE")


if __name__ == "__main__":
    main()
