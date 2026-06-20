"""
paper-reading: 决策性阅读
分析论文的 claim、证据强度、复现价值、工程落地风险。

输入：arxiv URL 或 arxiv ID
输出：结构化分析结果 + 飞书消息

Usage:
    python reading.py --url https://arxiv.org/abs/2210.03629
    python reading.py --id 2210.03629
    python reading.py --url https://arxiv.org/abs/2210.03629 --dry-run
    python reading.py --url https://arxiv.org/abs/2210.03629 --use-ocr  # OCR 读全文以提取实验
"""

import os
import sys
import json
import argparse
import urllib.request
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    llm_chat, get_llm_model,
    send_feishu_message, build_reading_result_card,
    ocr_arxiv_pdf, extract_experiment_section,
)


# ─── Arxiv Paper Fetcher (single paper) ──────────────────────────────────────

def fetch_single_paper(arxiv_id: str) -> dict:
    """Fetch a single paper's metadata from arxiv API."""
    arxiv_id = arxiv_id.strip()
    # Strip URL if given
    for prefix in ["https://arxiv.org/abs/", "http://arxiv.org/abs/", "arxiv.org/abs/"]:
        if arxiv_id.startswith(prefix):
            arxiv_id = arxiv_id[len(prefix):]
            break
    # Strip version suffix
    if arxiv_id.endswith("v1") or arxiv_id.endswith("v2") or arxiv_id.endswith("v3"):
        arxiv_id = arxiv_id[:-2]

    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
    req = urllib.request.Request(url, headers={"User-Agent": "ArxivDigest/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        xml_data = resp.read().decode("utf-8")

    root = ET.fromstring(xml_data)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    entry = root.find("atom:entry", ns)
    if entry is None:
        raise ValueError(f"Paper not found: {arxiv_id}")

    title_el = entry.find("atom:title", ns)
    summary_el = entry.find("atom:summary", ns)

    title = " ".join(title_el.text.split()) if title_el is not None and title_el.text else ""
    summary = " ".join(summary_el.text.split()) if summary_el is not None and summary_el.text else ""

    authors = [
        " ".join(a.find("atom:name", ns).text.split())
        for a in entry.findall("atom:author", ns)
        if a.find("atom:name", ns) is not None
    ]

    return {
        "title": title,
        "summary": summary,
        "arxiv_id": arxiv_id,
        "abstract_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "authors": authors,
    }


# ─── Decision-level Reading ───────────────────────────────────────────────────

READING_SYSTEM_PROMPT = """你是一个严苛的论文审稿人。你的任务是做决策性阅读——回答读者真正关心的问题，而不是复述摘要。

**核心原则**：
1. 严格区分三类断言：
   - "论文明确说"：论文原文直接陈述的内容
   - "合理推断"：基于论文内容的合理推论
   - "未支撑"：论文没有提供证据的说法（即使看起来像结论）
2. 不要编造任何实验数字、消融结论、数据集细节、开源状态。不确定就写"未知"。
3. 对声明保持怀疑：高调声明 + 弱实验 = 红旗。

输出 JSON 格式：
{
  "title": "论文标题",
  "core_claim": "论文的核心 claim（一句话）",
  "claims": [
    {"statement": "具体 claim", "type": "explicit|inference|unsupported", "evidence": "支撑证据或缺失说明"}
  ],
  "method_summary": "方法概述（200 字以内，只描述论文说了什么）",
  "key_figure": "最核心的实验图/表是什么，承载了什么信息",
  "evidence_level": "strong|moderate|weak|insufficient",
  "evidence_assessment": "对实验设计的整体评价",
  "reproducibility": {
    "difficulty": "low|medium|high|unknown",
    "risks": "复现时可能的坑",
    "open_source": "true|false|unknown",
    "code_url": "如果有的话"
  },
  "engineering_risks": "工程落地可能的崩点",
  "verdict": "值得复现|值得速读|可跳过",
  "reading_priority": "值得精读|值得速读|可暂缓",
  "verdict_reason": "一句话判定理由"
}
"""


def analyze_paper(paper: dict, model: str = None, full_text_experiment: str = None) -> dict:
    """对单篇论文进行决策性阅读分析。

    Args:
        paper: 论文基础信息
        model: LLM 模型
        full_text_experiment: OCR 提取的全文实验部分（可选）
    """
    if not model:
        model = get_llm_model()

    user_prompt = f"""请分析以下论文：

标题：{paper['title']}
作者：{', '.join(paper['authors'][:5])}
摘要：{paper['summary']}

arxiv: {paper['abstract_url']}"""

    if full_text_experiment:
        user_prompt += f"""

⚠️ 以下是从论文 PDF 全文 OCR 提取的**实验部分**（可能包含格式噪音）：
---
{full_text_experiment}
---

请结合摘要和上述实验内容进行全面分析。注意：OCR 可能有识别错误，数字和表格可能不准确。"""

    user_prompt += "\n\n请严格按照系统 prompt 的要求进行分析。"

    try:
        result = llm_chat(
            messages=[
                {"role": "system", "content": READING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.1,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        analysis = json.loads(result)
        analysis["arxiv_id"] = paper["arxiv_id"]
        analysis["abstract_url"] = paper["abstract_url"]
        return analysis
    except Exception as e:
        print(f"[reading error] {e}")
        return {
            "title": paper["title"],
            "arxiv_id": paper["arxiv_id"],
            "abstract_url": paper["abstract_url"],
            "error": str(e),
            "core_claim": "分析失败",
            "claims": [],
            "evidence_level": "unknown",
            "verdict": "分析失败，请重试",
            "reading_priority": "可暂缓",
            "verdict_reason": f"LLM 调用失败: {e}",
        }


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_reading(
    arxiv_id: str = None,
    arxiv_url: str = None,
    dry_run: bool = False,
    model: str = None,
    use_ocr: bool = False,
):
    """Run decision-level reading for a paper."""
    if arxiv_url:
        paper_id = arxiv_url
    elif arxiv_id:
        paper_id = arxiv_id
    else:
        print("[reading] No paper ID or URL provided")
        return None

    print(f"[reading] Fetching paper: {paper_id}")
    paper = fetch_single_paper(paper_id)
    print(f"[reading] Title: {paper['title']}")

    # OCR 全文提取实验部分
    full_text_exp = None
    if use_ocr:
        print("[reading] Running OCR on PDF...")
        try:
            ocr_result = ocr_arxiv_pdf(paper["abstract_url"], output_dir="output/ocr")
            if ocr_result:
                full_text_exp = extract_experiment_section(ocr_result)
                print(f"[reading] OCR done, experiment section: {len(full_text_exp)} chars")
        except Exception as e:
            print(f"[reading] OCR failed: {e}")

    print(f"[reading] Analyzing...")
    analysis = analyze_paper(paper, model=model, full_text_experiment=full_text_exp)

    if dry_run:
        print("\n[DRY RUN] Analysis result:")
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
        return analysis

    # Push result to Feishu
    card = build_reading_result_card(analysis)
    card_json = json.dumps(card, ensure_ascii=False)

    # 从环境变量获取接收者 ID（先尝试 FEISHU_RECEIVE_ID，默认用 webhook）
    receive_id = os.environ.get("FEISHU_RECEIVE_ID", "")
    if receive_id:
        ok = send_feishu_message(receive_id, "interactive", card_json)
    else:
        # Fallback: print result
        print("[reading] FEISHU_RECEIVE_ID not set. Printing result:")
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
        ok = True

    if ok:
        print(f"[reading] ✅ Result sent")
    else:
        print("[reading] ❌ Failed to send result")

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Paper Reading — Decision-level analysis")
    parser.add_argument("--url", type=str, help="Arxiv URL")
    parser.add_argument("--id", type=str, help="Arxiv paper ID")
    parser.add_argument("--dry-run", action="store_true", help="Print without pushing to Feishu")
    parser.add_argument("--use-ocr", action="store_true", help="Use PaddleOCR to extract full text from PDF")
    args = parser.parse_args()

    if not args.url and not args.id:
        print("Please provide --url or --id")
        sys.exit(1)

    run_reading(
        arxiv_id=args.id,
        arxiv_url=args.url,
        dry_run=args.dry_run,
        use_ocr=args.use_ocr,
    )


if __name__ == "__main__":
    main()
