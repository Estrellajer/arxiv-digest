"""
experiment-setup: 单篇论文的实验配置 / 复现要素抽取

传入 arxiv 链接，OCR 全文后抽取：数据集、超参、模型规模、训练硬件、评测协议、消融、复现要点。
与 reading.py（整体精读）、benchmark.py（多篇对比）互补，只聚焦"实验配置"。

Usage:
    python extract_setup.py --url https://arxiv.org/abs/2210.03629
    python extract_setup.py --id 2210.03629 --dry-run
    python extract_setup.py --url https://arxiv.org/abs/2210.03629 --push-to-feishu
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    llm_chat, get_llm_model,
    send_feishu_message, build_setup_result_card,
    ocr_arxiv_pdf, extract_experiment_section,
)
from paper_context import build_ocr_evidence, load_prompt

SETUP_SYSTEM_PROMPT = load_prompt("experiment_setup")


def extract_setup(paper: dict, ocr_evidence: dict = None, model: str = None) -> dict:
    """从论文（优先 OCR 全文）抽取结构化实验配置。"""
    if not model:
        model = get_llm_model()

    if ocr_evidence and ocr_evidence.get("experiment"):
        evidence_block = (
            f"首页 OCR：\n{ocr_evidence.get('first_page', '')[:5000]}\n\n"
            f"实验/实现细节 OCR：\n{ocr_evidence.get('experiment', '')[:9000]}\n\n"
            f"论文证据中的代码链接：{ocr_evidence.get('code_urls', [])}"
        )
    else:
        evidence_block = f"摘要（无 OCR 全文，多数实验配置可能未知）：\n{paper.get('summary', '')[:2000]}"

    user_prompt = f"""论文标题：{paper.get('title', '')}

{evidence_block}

请抽取实验配置 / 复现要素。"""

    result = llm_chat(
        messages=[
            {"role": "system", "content": SETUP_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=0.1,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    data = json.loads(result)
    if not data.get("title"):
        data["title"] = paper.get("title", "")
    data["arxiv_id"] = paper.get("arxiv_id", "")
    data["abstract_url"] = paper.get("abstract_url", "")
    return data


def run_extract_setup(
    arxiv_id: str = None,
    arxiv_url: str = None,
    dry_run: bool = False,
    model: str = None,
    push_to_feishu: bool = False,
):
    """Run experiment-setup extraction for a paper."""
    from reading import fetch_single_paper

    paper_id = arxiv_url or arxiv_id
    if not paper_id:
        print("[setup] No paper ID or URL provided")
        return None

    print(f"[setup] Fetching paper: {paper_id}")
    paper = fetch_single_paper(paper_id)
    print(f"[setup] Title: {paper['title']}")

    # OCR 全文（实验配置必须基于全文，否则抽不准）
    ocr_evidence = None
    print("[setup] Running OCR on PDF...")
    try:
        ocr_result = ocr_arxiv_pdf(paper["abstract_url"], output_dir="output/ocr")
        if ocr_result:
            ocr_evidence = build_ocr_evidence(ocr_result, paper["abstract_url"])
            ocr_evidence["ocr_source"] = ocr_result.get("ocr_source", "")
            print(f"[setup] OCR done via {ocr_result.get('ocr_source', '?')}, "
                  f"experiment section: {len(ocr_evidence['experiment'])} chars")
    except Exception as e:
        print(f"[setup] OCR failed, falling back to abstract: {e}")

    print("[setup] Extracting experiment setup...")
    setup = extract_setup(paper, ocr_evidence=ocr_evidence, model=model)
    setup["input_coverage"] = "OCR 全文（前20页）" if ocr_evidence else "仅摘要"
    setup["ocr_status"] = (ocr_evidence or {}).get("ocr_status", "failed")

    if dry_run:
        print("\n[DRY RUN] Experiment setup:")
        print(json.dumps(setup, ensure_ascii=False, indent=2))
        return setup

    if push_to_feishu:
        card_json = json.dumps(build_setup_result_card(setup), ensure_ascii=False)
        receive_id = os.environ.get("FEISHU_RECEIVE_ID", "")
        if receive_id:
            if send_feishu_message(receive_id, "interactive", card_json):
                print("[setup] ✅ Result pushed via Feishu app")
            else:
                print("[setup] ❌ Failed to push via Feishu app")
        else:
            print("[setup] FEISHU_RECEIVE_ID not set. Printing result:")
            print(json.dumps(setup, ensure_ascii=False, indent=2))

    print(json.dumps(setup, ensure_ascii=False))  # always print JSON for downstream

    try:
        os.makedirs("output", exist_ok=True)
        with open("output/setup.json", "w", encoding="utf-8") as f:
            json.dump(setup, f, ensure_ascii=False, indent=2)
        print("[setup] Setup saved to output/setup.json")
    except Exception as e:
        print(f"[setup] Failed to save setup.json: {e}")

    return setup


def main():
    parser = argparse.ArgumentParser(description="Experiment Setup Extractor")
    parser.add_argument("--url", type=str, help="Arxiv URL")
    parser.add_argument("--id", type=str, help="Arxiv paper ID")
    parser.add_argument("--dry-run", action="store_true", help="Print without pushing to Feishu")
    parser.add_argument("--push-to-feishu", action="store_true", help="Push result via Feishu app API")
    args = parser.parse_args()

    if not args.url and not args.id:
        print("Please provide --url or --id")
        sys.exit(1)

    run_extract_setup(
        arxiv_id=args.id,
        arxiv_url=args.url,
        dry_run=args.dry_run,
        push_to_feishu=args.push_to_feishu,
    )


if __name__ == "__main__":
    main()
