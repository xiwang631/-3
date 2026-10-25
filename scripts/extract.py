# -*- coding: utf-8 -*-
"""
第 1 步 · 财报提取脚本(带进程隔离, 防 PyMuPDF 段错误拖垮全局)
- 正文: PyMuPDF 按页提取
- 表格: PyMuPDF find_tables 按行列还原
- 隔离: 每份 PDF 一个子进程; 若段错误, 自动降级为逐页子进程, 崩溃页标记 skip
输出:
  data/extracted/<code>_pages.json
  data/extracted/<code>_tables.json
  data/extracted/_extract_log.md
用法:
  python extract.py            # 全部
  python extract.py 300122     # 只处理指定代码
"""
import io
import json
import os
import re
import subprocess
import sys
import time

BASE = r"C:\Users\Lenovo\Downloads\financial-qa-kb"
SRC = os.path.join(BASE, "data", "report")
OUT = os.path.join(BASE, "data", "extracted")
PY = sys.executable
os.makedirs(OUT, exist_ok=True)

FNAME_RE = re.compile(r"^(?P<code>\d{6})_(?P<company>.+?)_(?P<period>\d{4}年报)\.pdf$")

# PyMuPDF 会把表格里的千分位数字搅乱(如 7,816,273,687.17 -> 781627368717 , , , .)
# 特征: 出现 8 位以上连续数字。检测到即对该页改用 pdfplumber 重取表格。
SCRAMBLE_RE = re.compile(r"\d{8,}")


def clean_rows(rows):
    clean = [[(c or "").replace("\n", "").strip() for c in row] for row in rows]
    return [r for r in clean if any(r)]


def extract_page(path, page_no):
    """单页提取(在子进程里跑): 返回 (page_text, tables)"""
    import pymupdf
    doc = pymupdf.open(path)
    page = doc[page_no - 1]
    text = page.get_text("text", sort=True).strip()
    tables = []
    need_plumber = False
    try:
        found = page.find_tables()
        for k, tb in enumerate(found.tables):
            try:
                rows = clean_rows(tb.extract())
            except Exception as e:
                tables.append({"page": page_no, "table_id": f"p{page_no}t{k}",
                               "error": str(e)[:120], "rows": []})
                continue
            joined = " ".join(" ".join(r) for r in rows)
            if SCRAMBLE_RE.search(joined):
                need_plumber = True
            if len(rows) >= 2:
                tables.append({"page": page_no, "table_id": f"p{page_no}t{k}",
                               "n_rows": len(rows),
                               "n_cols": max(len(r) for r in rows), "rows": rows})
    except Exception as e:
        tables.append({"page": page_no, "error": str(e)[:120], "rows": []})
    doc.close()

    # 数字被搅乱的页 -> pdfplumber 整页重取(行列与数字均按坐标还原, 质量更高)
    if need_plumber:
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                pl_page = pdf.pages[page_no - 1]
                pl_tables = pl_page.extract_tables()  # 一级: 线框策略
                src_tag = "pdfplumber"
                if not pl_tables:
                    # 二级: 无边框报表(如迈瑞)用 text 对齐策略
                    found2 = pl_page.find_tables(table_settings={
                        "vertical_strategy": "text", "horizontal_strategy": "text"})
                    pl_tables = [tb.extract() for tb in found2]
                    src_tag = "pdfplumber-text"
            if pl_tables:
                tables = []
                for k, rows in enumerate(pl_tables):
                    rows = clean_rows(rows)
                    if len(rows) >= 2:
                        tables.append({"page": page_no, "table_id": f"p{page_no}t{k}",
                                       "n_rows": len(rows),
                                       "n_cols": max(len(r) for r in rows),
                                       "rows": rows, "src": src_tag})
        except Exception as e:
            tables.append({"page": page_no, "error": "pdfplumber " + str(e)[:100],
                           "rows": []})
    return text, tables


def run_worker_file(fname):
    """整本提取(子进程入口): 全部成功返回 0"""
    m = FNAME_RE.match(fname)
    code, company, period = m.group("code"), m.group("company"), m.group("period")
    path = os.path.join(SRC, fname)
    import pymupdf
    doc = pymupdf.open(path)
    n = doc.page_count
    doc.close()
    pages, tables = [], []
    for i in range(1, n + 1):
        text, tabs = extract_page(path, i)
        if text:
            pages.append({"page": i, "text": text})
        tables.extend(tabs)
    _save(code, company, period, n, pages, tables)
    return 0


def run_pagescan(fname):
    """逐页隔离模式(段错误后的降级): 每页一个子进程, 崩溃页跳过"""
    m = FNAME_RE.match(fname)
    code, company, period = m.group("code"), m.group("company"), m.group("period")
    path = os.path.join(SRC, fname)
    import pymupdf
    doc = pymupdf.open(path)
    n = doc.page_count
    doc.close()
    pages, tables, skipped = [], [], []
    for i in range(1, n + 1):
        r = subprocess.run([PY, __file__, "--page", fname, str(i)],
                           capture_output=True, timeout=120)
        if r.returncode != 0:
            skipped.append(i)
            continue
        obj = json.loads(r.stdout.decode("utf-8"))
        if obj["text"]:
            pages.append({"page": i, "text": obj["text"]})
        tables.extend(obj["tables"])
    _save(code, company, period, n, pages, tables)
    return skipped


def _save(code, company, period, n_pages, pages, tables):
    with io.open(os.path.join(OUT, f"{code}_pages.json"), "w", encoding="utf-8") as f:
        json.dump({"code": code, "company": company, "period": period,
                   "n_pages": n_pages, "pages": pages}, f, ensure_ascii=False)
    with io.open(os.path.join(OUT, f"{code}_tables.json"), "w", encoding="utf-8") as f:
        json.dump(tables, f, ensure_ascii=False)


def main():
    argv = sys.argv[1:]
    if not argv:
        only = None
    elif argv[0] == "--page":
        fname, pno = argv[1], int(argv[2])
        text, tabs = extract_page(os.path.join(SRC, fname), pno)
        sys.stdout.write(json.dumps({"text": text, "tables": tabs}, ensure_ascii=False))
        return
    elif argv[0] == "--file":
        run_worker_file(argv[1])
        return
    else:
        only = argv[0]

    pdfs = sorted(f for f in os.listdir(SRC) if f.endswith(".pdf") and FNAME_RE.match(f))
    if only:
        pdfs = [f for f in pdfs if f.startswith(only)]
    log = ["# 第 1 步提取日志\n", "| 文件 | 页数 | 正文字数 | 表格数 | 模式 | 跳过页 | 耗时 |",
           "|---|---|---|---|---|---|---|"]
    for fname in pdfs:
        code = FNAME_RE.match(fname).group("code")
        t0 = time.time()
        r = subprocess.run([PY, __file__, "--file", fname],
                           capture_output=True, timeout=1800)
        mode, skipped = "whole", []
        if r.returncode != 0:
            # 段错误或其他崩溃 -> 逐页隔离重跑
            skipped = run_pagescan(fname)
            mode = "page-isolated"
        pfile = os.path.join(OUT, f"{code}_pages.json")
        with io.open(pfile, encoding="utf-8") as f:
            obj = json.load(f)
        with io.open(os.path.join(OUT, f"{code}_tables.json"), encoding="utf-8") as f:
            tabs = json.load(f)
        n_chars = sum(len(p["text"]) for p in obj["pages"])
        n_tables = sum(1 for t in tabs if t.get("rows"))
        dt = time.time() - t0
        log.append(f"| {fname} | {obj['n_pages']} | {n_chars:,} | {n_tables} | {mode} "
                   f"| {','.join(map(str, skipped)) or '-'} | {dt:.1f}s |")
        print(f"done {code} mode={mode} pages={obj['n_pages']} chars={n_chars:,} "
              f"tables={n_tables} skipped={skipped or '-'} {dt:.1f}s", flush=True)
    with io.open(os.path.join(OUT, "_extract_log.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("ALL DONE")


if __name__ == "__main__":
    main()
