# -*- coding: utf-8 -*-
"""
修复脚本: 只重取仍被搅乱的表格页(不重跑全本)
搅乱特征(严格): 单元格里 5 位以上数字后跟 >=2 个游离逗号, 如 '17690372308, , ,'
修复方式: pdfplumber 线框策略 -> 空则 text 策略; 成功则整页替换该页表格
"""
import glob
import io
import json
import os
import re

BASE = r"C:\Users\Lenovo\Downloads\financial-qa-kb"
SRC = os.path.join(BASE, "data", "report")
OUT = os.path.join(BASE, "data", "extracted")

SCRAMBLE = re.compile(r"\d{5,}\s*(?:,\s*){2,}")


def clean_rows(rows):
    clean = [[(c or "").replace("\n", "").strip() for c in row] for row in rows]
    return [r for r in clean if any(r)]


def replumb(path, page_no):
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        pl = pdf.pages[page_no - 1]
        tabs = pl.extract_tables()
        tag = "pdfplumber"
        if not tabs:
            found = pl.find_tables(table_settings={
                "vertical_strategy": "text", "horizontal_strategy": "text"})
            tabs = [tb.extract() for tb in found]
            tag = "pdfplumber-text"
    out = []
    for k, rows in enumerate(tabs):
        rows = clean_rows(rows)
        if len(rows) >= 2:
            out.append({"page": page_no, "table_id": f"p{page_no}t{k}",
                        "n_rows": len(rows),
                        "n_cols": max(len(r) for r in rows),
                        "rows": rows, "src": tag})
    return out


def main():
    log = io.open(os.path.join(OUT, "_repair_log.txt"), "w", encoding="utf-8")
    for f in sorted(os.listdir(OUT)):
        if not f.endswith("_tables.json"):
            continue
        code = f[:6]
        tabs = json.load(io.open(os.path.join(OUT, f), encoding="utf-8"))
        bad_pages = sorted({t["page"] for t in tabs
                            if t.get("rows") and any(SCRAMBLE.search(c) for r in t["rows"] for c in r)})
        if not bad_pages:
            continue
        pdf = glob.glob(os.path.join(SRC, code + "_*.pdf"))[0]
        log.write(f"{code}: repair pages {bad_pages}\n")
        log.flush()
        new_tabs = [t for t in tabs if t["page"] not in bad_pages]
        for pno in bad_pages:
            try:
                fixed = replumb(pdf, pno)
                new_tabs.extend(fixed)
                n_scram = sum(1 for t in fixed
                              if any(SCRAMBLE.search(c) for r in t["rows"] for c in r))
                log.write(f"  p{pno}: {len(fixed)} tables (still scrambled: {n_scram})\n")
            except Exception as e:
                log.write(f"  p{pno}: FAILED {str(e)[:100]}\n")
        new_tabs.sort(key=lambda t: (t["page"], t["table_id"]))
        json.dump(new_tabs, io.open(os.path.join(OUT, f), "w", encoding="utf-8"),
                  ensure_ascii=False)
    log.write("done\n")
    log.close()
    print("repair finished")


if __name__ == "__main__":
    main()
