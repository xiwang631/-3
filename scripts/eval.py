# -*- coding: utf-8 -*-
"""第 6 步: 10 题正式评估
- 含 2 道跨公司比较题(作业要求) + 口径陷阱题 + 换词陷阱题(课件)
- 每题记录: 检索召回了哪块 / 答对了没 / 错在哪 / 出处对不对
- 人工判分项留空, 由人工填写 -> 标注 [人工评估]
"""
import io
import json
import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "scripts"))
import ask  # noqa

QUESTIONS = [
    # --- 单公司事实题 ---
    {"q": "恒瑞医药2025年营业收入是多少？", "type": "单公司事实",
     "expect": "恒瑞医药年报主营业务收入/利润表, 约300亿量级(以原文为准)"},
    {"q": "迈瑞医疗2025年归属于上市公司股东的净利润是多少？", "type": "单公司事实",
     "expect": "迈瑞医疗 主要会计数据"},
    {"q": "云南白药2025年经营活动产生的现金流量净额是多少？", "type": "单公司事实",
     "expect": "云南白药 主要会计数据/现金流量表(合并口径)"},
    {"q": "药明康德2025年研发投入占营业收入比例是多少？", "type": "单公司事实",
     "expect": "药明康德 管理层讨论与分析"},
    {"q": "智飞生物2025年销售费用是多少？", "type": "单公司事实",
     "expect": "智飞生物 利润表(合并)或费用分析"},
    # --- 跨公司比较题(作业要求>=2道) ---
    {"q": "恒瑞医药和复星医药2025年谁的研发费用投入更大？", "type": "跨公司比较",
     "expect": "恒瑞约87.24亿研发投入合计; 复星同业比较表口径约59.13亿; 结论恒瑞更大"},
    {"q": "迈瑞医疗和联影医疗2025年谁的营业收入更高？", "type": "跨公司比较",
     "expect": "迈瑞约350亿+ vs 联影约100亿量级, 迈瑞更高"},
    # --- 口径陷阱题(课件: 母公司 vs 合并) ---
    {"q": "恒瑞医药母公司2025年货币资金是多少？", "type": "口径陷阱",
     "expect": "应命中母公司资产负债表而非合并资产负债表, 金额明显小于合并口径"},
    # --- 换词陷阱题(课件: 同义换词看检索鲁棒性) ---
    {"q": "片仔癀2025年自营渠道的收入规模如何？", "type": "换词陷阱",
     "expect": "年报原文多用'直销', 看检索能否把'自营'对齐到'直销'"},
    # --- 否定/开放题(看会不会编造) ---
    {"q": "凯莱英2025年海外业务收入占比是多少？", "type": "单公司事实",
     "expect": "凯莱英年报分地区收入, 以原文为准; 若未披露应如实说"},
]

OUT = os.path.join(BASE, "data", "eval_results.md")


def main():
    lines = ["# 10 题评估记录", "",
             "评估口径: 每题记录【检索召回】与【模型作答】, 人工核对列留空待填。",
             ""]
    for i, item in enumerate(QUESTIONS, 1):
        t0 = time.time()
        print(f"[{i}/10] {item['q']}", flush=True)
        lines.append(f"## 题 {i} · {item['type']}")
        lines.append(f"**问题**: {item['q']}")
        lines.append(f"**预期要点**: {item['expect']}")
        try:
            pre = ask.preprocess(item["q"])
            hits = ask.retrieve(item["q"], pre)
            ans = ask.answer(item["q"], hits)
            lines.append("**检索召回 Top5**:")
            for k, h in enumerate(hits, 1):
                v = f"vec#{h['vec_rank']}" if h["vec_rank"] else "bm25"
                lines.append(f"  {k}. {h['company']} {h['section']} 第{h['page']}页 ({v}, rrf={h['score']})")
            lines.append("**模型作答**:")
            lines.append(ans)
            lines.append("")
            lines.append("| 人工核对项 | 结论(待填) |")
            lines.append("|---|---|")
            lines.append("| 答对了吗 | [人工评估] |")
            lines.append("| 出处对吗 | [人工评估] |")
            lines.append("| 错因分析 | [人工评估] |")
        except Exception as e:
            lines.append(f"**执行失败**: {str(e)[:200]}")
        lines.append(f"*(耗时 {time.time()-t0:.0f}s)*")
        lines.append("")
        io.open(OUT, "w", encoding="utf-8").write("\n".join(lines))
    print(f"done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
