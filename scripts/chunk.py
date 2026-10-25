# -*- coding: utf-8 -*-
"""
第 2 步 · 切块脚本(课件口径: 800 字/重合 120/句末收尾)
- 正文: 各页正文拼接后按句切, 滑窗 800 字、重合 120 字
- 表格: 整表成块(行数 > 40 时按行切分, 表头随块携带)
- 每块标签: 公司/代码/报告期/章节/页码/口径(六大报表名), 公司名与章节名直接拼进块文本
           (课件 slide 25: 检索时才分得清是谁的数)
输出: data/chunks.jsonl  {chunk_id, company, code, period, section, page, type, calibre, text}
"""
import io
import json
import os
import re

BASE = r"C:\Users\Lenovo\Downloads\financial-qa-kb"
EXT = os.path.join(BASE, "data", "extracted")
OUT = os.path.join(BASE, "data", "chunks.jsonl")

CHUNK = 800
OVERLAP = 120

SECTION_RE = re.compile(r"第[一二三四五六七八九十]+节[\s、.]*[^\n]{0,30}")
CALIBRE_KEYS = ["合并资产负债表", "母公司资产负债表", "合并利润表", "母公司利润表",
                "合并现金流量表", "母公司现金流量表"]

SENT_SPLIT = re.compile(r"(?<=[。；！？!?])|\n+")


def split_sentences(text):
    return [s for s in SENT_SPLIT.split(text) if s and s.strip()]


def window_chunks(text, size=CHUNK, overlap=OVERLAP):
    """句末收尾滑窗切分"""
    sents = split_sentences(text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        cur.append(s)
        cur_len += len(s)
        if cur_len >= size:
            chunks.append("".join(cur))
            # 回退凑出 ~overlap 字的句子作为下一窗口开头
            tail, tl = [], 0
            for t in reversed(cur):
                tail.insert(0, t)
                tl += len(t)
                if tl >= overlap:
                    break
            cur, cur_len = tail, tl
    if cur and cur_len > 30:
        chunks.append("".join(cur))
    return chunks


def build_section_map(pages):
    """逐页扫描章节标题, 得到 页码->当前章节"""
    smap, cur = {}, "卷首"
    for p in pages:
        for m in SECTION_RE.finditer(p["text"]):
            cur = re.sub(r"\s+", " ", m.group(0)).strip()[:30]
        smap[p["page"]] = cur
    return smap


def detect_calibre(page_text):
    hits = [k for k in CALIBRE_KEYS if k in page_text]
    return hits[0] if hits else ""


def fmt_table_rows(rows):
    return "\n".join(" | ".join(r) for r in rows)


def main():
    n_out = 0
    with io.open(OUT, "w", encoding="utf-8") as fo:
        for f in sorted(os.listdir(EXT)):
            if not f.endswith("_pages.json"):
                continue
            code = f[:6]
            meta = json.load(io.open(os.path.join(EXT, f), encoding="utf-8"))
            company, period = meta["company"], meta["period"]
            pages = meta["pages"]
            tables = json.load(io.open(os.path.join(EXT, code + "_tables.json"),
                                       encoding="utf-8"))
            smap = build_section_map(pages)
            ptext = {p["page"]: p["text"] for p in pages}

            xseq = tseq = 0

            # --- 正文块 ---
            full = "\n".join(f"【第{p['page']}页】{p['text']}" for p in pages)
            for piece in window_chunks(full):
                pgs = sorted({int(m) for m in re.findall(r"【第(\d+)页】", piece)})
                section = smap.get(pgs[0], "未知") if pgs else "未知"
                # 去掉页码标记, 成品文本
                body = re.sub(r"【第\d+页】", "", piece)
                xseq += 1
                rec = {"chunk_id": f"{code}-x{xseq:04d}", "company": company,
                       "code": code, "period": period, "section": section,
                       "page": pgs, "type": "text", "calibre": "",
                       "text": f"【{company}·{section}】\n{body}"}
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1

            # --- 表格块 ---
            for t in tables:
                if not t.get("rows"):
                    continue
                pno = t["page"]
                section = smap.get(pno, "未知")
                calibre = detect_calibre(ptext.get(pno, ""))
                prefix = f"【{company}·{period}·{section}·第{pno}页"
                if calibre:
                    prefix += f"·口径:{calibre}"
                prefix += "】\n"
                head, rows = t["rows"][0], t["rows"]
                body_rows = rows if len(rows) <= 40 else rows
                if len(rows) <= 40:
                    tseq += 1
                    rec = {"chunk_id": f"{code}-t{tseq:04d}", "company": company,
                           "code": code, "period": period, "section": section,
                           "page": [pno], "type": "table", "calibre": calibre,
                           "text": prefix + fmt_table_rows(rows)}
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_out += 1
                else:
                    # 长表按 40 行一段切, 表头(前 2 行)随段携带
                    step, h = 40, rows[:2]
                    for i in range(0, len(rows), step):
                        seg = rows[i:i + step] if i == 0 else h + rows[i:i + step]
                        tseq += 1
                        rec = {"chunk_id": f"{code}-t{tseq:04d}", "company": company,
                               "code": code, "period": period, "section": section,
                               "page": [pno], "type": "table", "calibre": calibre,
                               "text": prefix + fmt_table_rows(seg)}
                        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_out += 1
    print(f"chunks total: {n_out} -> {OUT}")


if __name__ == "__main__":
    main()
