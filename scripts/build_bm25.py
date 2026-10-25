# -*- coding: utf-8 -*-
"""第 3 步 B 路: BM25 关键词检索索引
读 chunks.jsonl -> jieba 分词 -> BM25Okapi -> 存 pickle
查询时用同样的 tokenize() 处理问题即可检索。
"""
import io
import json
import os
import pickle
import time

import jieba
from rank_bm25 import BM25Okapi

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS = os.path.join(BASE, "data", "chunks.jsonl")
OUT = os.path.join(BASE, "data", "bm25_index.pkl")

# 金融领域自定义词, 避免被切碎
for w in ["毛利率", "净利率", "归母净利润", "研发费用", "销售费用", "管理费用",
          "财务费用", "经营活动", "现金流量表", "资产负债表", "利润表",
          "合并报表", "母公司", "集采", "License-out", "CXO", "营业总收入",
          "扣非净利润", "研发投入", "同比", "环比", "营业收入", "应收账款",
          "存货周转", "商誉", "带息负债", "有息负债", "云南白药", "片仔癀",
          "恒瑞医药", "复星医药", "华东医药", "药明康德", "凯莱英",
          "迈瑞医疗", "联影医疗", "智飞生物", "三一重工", "徐工机械",
          "中联重科", "柳工", "潍柴动力", "恒立液压", "艾迪精密",
          "浙江鼎力", "安徽合力", "杭叉集团"]:
    jieba.add_word(w)


def tokenize(text):
    return [t for t in jieba.cut_for_search(text) if t.strip() and t.strip() not in "，。；：、（）()[]{} \t\n"]


def main():
    t0 = time.time()
    records = []
    for line in io.open(CHUNKS, encoding="utf-8"):
        records.append(json.loads(line))
    print(f"loaded {len(records)} chunks")

    corpus = [tokenize(r["text"]) for r in records]
    print(f"tokenized in {time.time()-t0:.1f}s")

    bm25 = BM25Okapi(corpus)
    with open(OUT, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": [r["chunk_id"] for r in records]}, f)
    print(f"BM25 index built & saved -> {OUT}  ({time.time()-t0:.1f}s total)")

    # 冒烟测试
    for q in ["恒瑞医药研发费用率", "毛利率对比", "经营活动产生的现金流量净额"]:
        scores = bm25.get_scores(tokenize(q))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:3]
        print(f"\nQ: {q}")
        for i in top:
            r = records[i]
            print(f"  [{scores[i]:.1f}] {r['company']} {r['section']} p{r['page']} :: {r['text'][:60]}")


if __name__ == "__main__":
    main()
