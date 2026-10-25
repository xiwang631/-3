# 财报问答知识库（医药行业 · 方向 A）

《人工智能与数据分析》第 3 次作业：把 10 家医药上市公司 2025 年年报做成**能提问、能带出处回答**的问答知识库。

## 仓库结构

```
financial-qa-kb/
├── app.py                  # Streamlit 问答页面（检索过程 + 答案 + 命中块出处展示）
├── scripts/
│   ├── extract.py          # 第 1 步 PDF 提取（PyMuPDF + pdfplumber 混合，含段错误隔离与数字修正）
│   ├── repair_tables.py    # 表格修复（无边框报表 text 策略二级回退）
│   ├── chunk.py            # 第 2 步切块（800 字/重合 120/句末收尾，块带公司·章节·页码·口径标签）
│   ├── build_bm25.py       # 第 3 步 BM25 索引（jieba 金融词表）
│   ├── embed_api.py        # 第 3 步 向量索引（硅基流动 BAAI/bge-m3，批量+断点续算）
│   ├── ask.py              # 第 3-5 步 检索+作答链路（问题预处理 → 双路 RRF 合并 → DeepSeek 带出处作答）
│   └── eval.py             # 第 6 步 10 题评估（含跨公司/口径陷阱/换词陷阱题）
├── data/
│   ├── report/             # 10 份年报 PDF 原始材料（未入库，体积原因，见 data/report/README）
│   ├── extracted/          # 提取产物：正文按页 + 表格行列 JSON + 提取日志
│   ├── chunks.jsonl        # 10,408 个知识块
│   ├── bm25_index.pkl      # BM25 索引
│   ├── embeddings.npy      # bge-m3 向量 (10408 × 1024, 已归一化)
│   ├── eval_results.md     # 10 题评估记录（逐题召回块/答案/错因）
│   └── _extract_log.md     # 提取统计日志
└── output/
    ├── conclusion.html     # 一页结论（网页版）
    └── conclusion.pdf      # 一页结论（提交版）
```

## 快速开始

```bash
pip install -r requirements.txt
# 配置 .env（不会被提交）：
#   SILICONFLOW_API_KEY=...   # bge-m3 向量化
#   DEEPSEEK_API_KEY=...      # 问题改写 + 作答
streamlit run app.py          # 打开 http://localhost:8501
python scripts/ask.py "恒瑞医药2025年营业收入是多少？"   # 命令行版
python scripts/eval.py       # 重跑 10 题评估
```

## 系统流程

```
年报 PDF → 提取(正文+表格行列还原) → 切块(带公司/章节/页码/口径标签)
        → 向量索引(bge-m3) + BM25(jieba)
提问 → LLM 问题预处理(精炼+关键词) → 双路检索 Top20 → RRF 合并 Top5
    → DeepSeek 作答(带[片段k]出处, 片段不足则拒答) → Streamlit 展示命中块
```

## 关键工程决策（详见 output/conclusion.pdf）

1. **表格数字保真**：PyMuPDF 会搅乱千分位数字（`7,816,273,687.17` → `781627368717 , , , .`），检测到即用 pdfplumber 整页重取；无边框财务报表用 text 对齐策略二级回退。
2. **口径陷阱防护**：六大报表切块强制携带「合并/母公司 + 报表名」标签——评测中「恒瑞母公司货币资金」题准确命中母公司口径。
3. **零编造约束**：模型被要求片段不足即拒答。10 题评估 6 对 4 拒答、**0 编造**，失败模式是「漏检索→诚实拒答」而非瞎说。

## 数据说明

年报 PDF 来自巨潮资讯网（cninfo.com.cn），共 10 家：恒瑞医药、复星医药、华东医药、云南白药、片仔癀、药明康德、凯莱英、迈瑞医疗、联影医疗、智飞生物。
